# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Demucs: Waveform-based Source Separation U-Net

WHAT: A convolutional encoder-decoder that separates mixed audio into individual stems
      (drums, bass, vocals, other) by operating directly on raw waveform samples.
WHY:  Waveform-level avoids STFT phase reconstruction issues; end-to-end learning.
HOW:  Encoder downsamples with strided conv → bottleneck (optional LSTM) → decoder
      upsamples with transpose conv. Skip connections + optional DConv residual branches.
"""

import math
import typing as tp

import julius
import torch
from torch import nn
from torch.nn import functional as F

from .states import capture_init
from .utils import center_trim, unfold
from .transformer import LayerScale


class BLSTM(nn.Module):
    """
    Bidirectional LSTM over the time dimension.

    WHAT: Processes (B, C, T) along T with BiLSTM; outputs same shape.
    WHY:  Adds sequential/long-range modeling at the bottleneck; bi-directional sees past+future.
    HOW:  Permute to (T, B, C) for LSTM, then linear projects 2*dim→dim to merge directions.
          If max_steps set, splits long sequences into overlapping chunks to avoid OOM.
    """
    def __init__(self, dim, layers=1, max_steps=None, skip=False):
        super().__init__()
        assert max_steps is None or max_steps % 4 == 0  # Must align with overlap logic
        self.max_steps = max_steps  # If set, chunk long sequences
        self.lstm = nn.LSTM(bidirectional=True, num_layers=layers, hidden_size=dim, input_size=dim)
        self.linear = nn.Linear(2 * dim, dim)  # Merge fwd+bwd outputs back to dim
        self.skip = skip  # Residual: add input to output

    def forward(self, x):
        B, C, T = x.shape
        y = x  # Save for skip connection
        framed = False
        # WHAT: Chunk long sequences into overlapping windows. WHY: LSTM O(T) memory; long T → OOM.
        # HOW: unfold into frames (width, stride), run LSTM per frame, overlap-add to reconstruct.
        if self.max_steps is not None and T > self.max_steps:
            width = self.max_steps
            stride = width // 2  # 50% overlap
            frames = unfold(x, width, stride)  # (B, C, nframes, width)
            nframes = frames.shape[2]
            framed = True
            x = frames.permute(0, 2, 1, 3).reshape(-1, C, width)  # (B*nframes, C, width)

        # LSTM expects (seq_len, batch, input_size)
        x = x.permute(2, 0, 1)  # (T, B, C)

        x = self.lstm(x)[0]  # (T, B, 2*dim)
        x = self.linear(x)    # (T, B, dim)
        x = x.permute(1, 2, 0)  # (B, C, T)

        # Overlap-add: take center region of each frame, concatenate (discard overlaps)
        if framed:
            out = []
            frames = x.reshape(B, -1, C, width)
            limit = stride // 2  # Overlap region to trim
            for k in range(nframes):
                if k == 0:
                    out.append(frames[:, k, :, :-limit])
                elif k == nframes - 1:
                    out.append(frames[:, k, :, limit:])
                else:
                    out.append(frames[:, k, :, limit:-limit])
            out = torch.cat(out, -1)
            out = out[..., :T]  # Trim to original length
            x = out
        if self.skip:
            x = x + y
        return x


def rescale_conv(conv, reference):
    """
    WHAT: Rescale conv weights so their std ≈ reference (via square-root scaling).
    WHY:  Empirically stabilizes training; deep U-Nets can have exploding activations.
    HOW:  Divide weights by sqrt(actual_std/reference); scale bias the same.
    """
    std = conv.weight.std().detach()
    scale = (std / reference)**0.5
    conv.weight.data /= scale
    if conv.bias is not None:
        conv.bias.data /= scale


def rescale_module(module, reference):
    """
    Apply rescale_conv to all Conv/ConvTranspose layers in module.
    """
    for sub in module.modules():
        if isinstance(sub, (nn.Conv1d, nn.ConvTranspose1d, nn.Conv2d, nn.ConvTranspose2d)):
            rescale_conv(sub, reference)


class DConv(nn.Module):
    """
    DConv: Dilated Convolution residual branch.

    WHAT: A residual block with dilated convs (optionally + LSTM + attention) that runs
          in parallel to the main encoder path. Compresses channels internally.
    WHY:  Increases capacity without blowing up params; dilated convs expand receptive
          field; LSTM/attn add sequential modeling in deeper layers.
    HOW:  For each sub-layer: Conv(ch→hidden) → [opt LSTM] [opt LocalState] →
          Conv(hidden→2ch) → GLU → LayerScale; residual add. Dilation doubles per layer.
    """
    def __init__(self, channels: int, compress: float = 4, depth: int = 2, init: float = 1e-4,
                 norm=True, attn=False, heads=4, ndecay=4, lstm=False, gelu=True,
                 kernel=3, dilate=True):
        """
        Args:
            channels: input/output channels for residual branch.
            compress: amount of channel compression inside the branch.
            depth: number of layers in the residual branch. Each layer has its own
                projection, and potentially LSTM and attention.
            init: initial scale for LayerNorm.
            norm: use GroupNorm.
            attn: use LocalAttention.
            heads: number of heads for the LocalAttention.
            ndecay: number of decay controls in the LocalAttention.
            lstm: use LSTM.
            gelu: Use GELU activation.
            kernel: kernel size for the (dilated) convolutions.
            dilate: if true, use dilation, increasing with the depth.
        """

        super().__init__()
        assert kernel % 2 == 1  # Odd kernel for symmetric padding
        self.channels = channels
        self.compress = compress
        self.depth = abs(depth)
        dilate = depth > 0

        norm_fn: tp.Callable[[int], nn.Module]
        norm_fn = lambda d: nn.Identity()  # noqa
        if norm:
            norm_fn = lambda d: nn.GroupNorm(1, d)  # noqa

        hidden = int(channels / compress)  # Compress: fewer channels in branch (cheaper)

        act: tp.Type[nn.Module]
        if gelu:
            act = nn.GELU
        else:
            act = nn.ReLU

        self.layers = nn.ModuleList([])
        for d in range(self.depth):
            dilation = 2 ** d if dilate else 1  # 1, 2, 4, ... → exponential receptive field
            padding = dilation * (kernel // 2)   # Keep same spatial size
            # Structure: conv→norm→act → [LSTM or attn] → conv→GLU → LayerScale
            mods = [
                nn.Conv1d(channels, hidden, kernel, dilation=dilation, padding=padding),
                norm_fn(hidden), act(),
                nn.Conv1d(hidden, 2 * channels, 1),
                norm_fn(2 * channels), nn.GLU(1),  # GLU halves channels; gate mechanism
                LayerScale(channels, init),        # Small init for residual branch
            ]
            if attn:
                mods.insert(3, LocalState(hidden, heads=heads, ndecay=ndecay))
            if lstm:
                mods.insert(3, BLSTM(hidden, layers=2, max_steps=200, skip=True))
            layer = nn.Sequential(*mods)
            self.layers.append(layer)

    def forward(self, x):
        # Residual: x + branch(x) for each sub-layer
        for layer in self.layers:
            x = x + layer(x)
        return x


class LocalState(nn.Module):
    """
    Local attention: data-dependent attention with optional time-decay bias.

    WHAT: Multi-head self-attention over time, with optional decay term that penalizes
          attending to distant positions. No positional embedding; position comes from
          delta = t - s and a learned decay.
    WHY:  Standard attention is O(T²); decay lets model focus on local context while
          still allowing long-range if needed. Cheaper than full transformer.
    HOW:  Q,K from 1x1 convs → dots = K^T Q / sqrt(d) + decay_bias(|t-s|) → softmax →
          output = attention_weights @ content. Residual add.
    """
    def __init__(self, channels: int, heads: int = 4, nfreqs: int = 0, ndecay: int = 4):
        super().__init__()
        assert channels % heads == 0, (channels, heads)
        self.heads = heads
        self.nfreqs = nfreqs
        self.ndecay = ndecay
        self.content = nn.Conv1d(channels, channels, 1)  # Values for attention
        self.query = nn.Conv1d(channels, channels, 1)
        self.key = nn.Conv1d(channels, channels, 1)
        if nfreqs:
            self.query_freqs = nn.Conv1d(channels, heads * nfreqs, 1)
        if ndecay:
            self.query_decay = nn.Conv1d(channels, heads * ndecay, 1)
            # Initialize decay close to zero (sigmoid), for maximum initial window.
            self.query_decay.weight.data *= 0.01
            assert self.query_decay.bias is not None  # stupid type checker
            self.query_decay.bias.data[:] = -2
        self.proj = nn.Conv1d(channels + heads * nfreqs, channels, 1)

    def forward(self, x):
        B, C, T = x.shape
        heads = self.heads
        indexes = torch.arange(T, device=x.device, dtype=x.dtype)
        # delta[t,s] = t - s (key index - query index)
        delta = indexes[:, None] - indexes[None, :]

        queries = self.query(x).view(B, heads, -1, T)
        keys = self.key(x).view(B, heads, -1, T)
        # Attention logits: dots[b,h,t,s] = sum over key/query dim of K[b,h,:,t]*Q[b,h,:,s]
        dots = torch.einsum("bhct,bhcs->bhts", keys, queries)
        dots /= keys.shape[2]**0.5  # Scale by sqrt(d_k)
        if self.nfreqs:
            periods = torch.arange(1, self.nfreqs + 1, device=x.device, dtype=x.dtype)
            freq_kernel = torch.cos(2 * math.pi * delta / periods.view(-1, 1, 1))
            freq_q = self.query_freqs(x).view(B, heads, -1, T) / self.nfreqs ** 0.5
            dots += torch.einsum("fts,bhfs->bhts", freq_kernel, freq_q)
        if self.ndecay:
            # Decay bias: -decay*|delta| → farther = more negative = less attention
            decays = torch.arange(1, self.ndecay + 1, device=x.device, dtype=x.dtype)
            decay_q = self.query_decay(x).view(B, heads, -1, T)
            decay_q = torch.sigmoid(decay_q) / 2
            decay_kernel = - decays.view(-1, 1, 1) * delta.abs() / self.ndecay**0.5
            dots += torch.einsum("fts,bhfs->bhts", decay_kernel, decay_q)

        # Prevent attending to self (would dominate softmax)
        dots.masked_fill_(torch.eye(T, device=dots.device, dtype=torch.bool), -100)
        weights = torch.softmax(dots, dim=2)

        content = self.content(x).view(B, heads, -1, T)
        result = torch.einsum("bhts,bhct->bhcs", weights, content)
        if self.nfreqs:
            time_sig = torch.einsum("bhts,fts->bhfs", weights, freq_kernel)
            result = torch.cat([result, time_sig], 2)
        result = result.reshape(B, -1, T)
        return x + self.proj(result)


class Demucs(nn.Module):
    """
    Demucs: Waveform U-Net for source separation.

    WHAT: Encoder (stride conv) + optional LSTM bottleneck + decoder (transpose conv)
          with skip connections. Input (B, C, T) mixed → output (B, S, C, T) stems.
    WHY:  End-to-end on waveform; no STFT phase issues. Skip connections preserve detail.
    HOW:  Build encoder/decoder stacks layer-by-layer; channels grow by `growth` in encoder.
    """

    @capture_init
    def __init__(self,
                 sources,
                 # Channels
                 audio_channels=2,
                 channels=64,
                 growth=2.,
                 # Main structure
                 depth=6,
                 rewrite=True,
                 lstm_layers=0,
                 # Convolutions
                 kernel_size=8,
                 stride=4,
                 context=1,
                 # Activations
                 gelu=True,
                 glu=True,
                 # Normalization
                 norm_starts=4,
                 norm_groups=4,
                 # DConv residual branch
                 dconv_mode=1,
                 dconv_depth=2,
                 dconv_comp=4,
                 dconv_attn=4,
                 dconv_lstm=4,
                 dconv_init=1e-4,
                 # Pre/post processing
                 normalize=True,
                 resample=True,
                 # Weight init
                 rescale=0.1,
                 # Metadata
                 samplerate=44100,
                 segment=4 * 10):
        """
        Args:
            sources (list[str]): list of source names
            audio_channels (int): stereo or mono
            channels (int): first convolution channels
            depth (int): number of encoder/decoder layers
            growth (float): multiply (resp divide) number of channels by that
                for each layer of the encoder (resp decoder)
            depth (int): number of layers in the encoder and in the decoder.
            rewrite (bool): add 1x1 convolution to each layer.
            lstm_layers (int): number of lstm layers, 0 = no lstm. Deactivated
                by default, as this is now replaced by the smaller and faster small LSTMs
                in the DConv branches.
            kernel_size (int): kernel size for convolutions
            stride (int): stride for convolutions
            context (int): kernel size of the convolution in the
                decoder before the transposed convolution. If > 1,
                will provide some context from neighboring time steps.
            gelu: use GELU activation function.
            glu (bool): use glu instead of ReLU for the 1x1 rewrite conv.
            norm_starts: layer at which group norm starts being used.
                decoder layers are numbered in reverse order.
            norm_groups: number of groups for group norm.
            dconv_mode: if 1: dconv in encoder only, 2: decoder only, 3: both.
            dconv_depth: depth of residual DConv branch.
            dconv_comp: compression of DConv branch.
            dconv_attn: adds attention layers in DConv branch starting at this layer.
            dconv_lstm: adds a LSTM layer in DConv branch starting at this layer.
            dconv_init: initial scale for the DConv branch LayerScale.
            normalize (bool): normalizes the input audio on the fly, and scales back
                the output by the same amount.
            resample (bool): upsample x2 the input and downsample /2 the output.
            rescale (float): rescale initial weights of convolutions
                to get their standard deviation closer to `rescale`.
            samplerate (int): stored as meta information for easing
                future evaluations of the model.
            segment (float): duration of the chunks of audio to ideally evaluate the model on.
                This is used by `demucs.apply.apply_model`.
        """

        super().__init__()
        # Store config for inference / apply_model
        self.audio_channels = audio_channels
        self.sources = sources
        self.kernel_size = kernel_size
        self.context = context
        self.stride = stride
        self.depth = depth
        self.resample = resample
        self.channels = channels
        self.normalize = normalize
        self.samplerate = samplerate
        self.segment = segment
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.skip_scales = nn.ModuleList()

        # Activation: GLU doubles effective channels (gating); ReLU does not.
        if glu:
            activation = nn.GLU(dim=1)
            ch_scale = 2
        else:
            activation = nn.ReLU()
            ch_scale = 1
        if gelu:
            act2 = nn.GELU
        else:
            act2 = nn.ReLU

        in_channels = audio_channels  # Start with 2 (stereo) or 1 (mono)
        padding = 0
        # Build encoder and decoder pairwise. Decoder is built in reverse order (insert 0).
        for index in range(depth):
            # GroupNorm only in deeper layers (norm_starts); avoids over-smoothing early layers
            norm_fn = lambda d: nn.Identity()  # noqa
            if index >= norm_starts:
                norm_fn = lambda d: nn.GroupNorm(norm_groups, d)  # noqa

            # --- ENCODER: stride conv downsamples time, expands channels ---
            encode = []
            encode += [
                nn.Conv1d(in_channels, channels, kernel_size, stride),  # Main downsampling
                norm_fn(channels),
                act2(),
            ]
            attn = index >= dconv_attn  # Add attn in deeper layers only
            lstm = index >= dconv_lstm
            if dconv_mode & 1:  # 1 or 3: DConv in encoder
                encode += [DConv(channels, depth=dconv_depth, init=dconv_init,
                                 compress=dconv_comp, attn=attn, lstm=lstm)]
            if rewrite:  # 1x1 conv to mix channels before passing to decoder
                encode += [
                    nn.Conv1d(channels, ch_scale * channels, 1),
                    norm_fn(ch_scale * channels), activation]
            self.encoder.append(nn.Sequential(*encode))

            # --- DECODER: transpose conv upsamples. Built backwards (decoder[0] = last layer) ---
            decode = []
            if index > 0:
                out_channels = in_channels  # Intermediates: match encoder input of that layer
            else:
                out_channels = len(self.sources) * audio_channels  # Final: S stems × C channels
            if rewrite:
                decode += [
                    nn.Conv1d(channels, ch_scale * channels, 2 * context + 1, padding=context),
                    norm_fn(ch_scale * channels), activation]
            if dconv_mode & 2:  # 2 or 3: DConv in decoder
                decode += [DConv(channels, depth=dconv_depth, init=dconv_init,
                                 compress=dconv_comp, attn=attn, lstm=lstm)]
            decode += [nn.ConvTranspose1d(channels, out_channels,
                       kernel_size, stride, padding=padding)]
            if index > 0:
                decode += [norm_fn(out_channels), act2()]
            self.decoder.insert(0, nn.Sequential(*decode))

            in_channels = channels
            channels = int(growth * channels)  # Double (or growth) channels per layer

        channels = in_channels  # Back to bottleneck channels
        if lstm_layers:
            self.lstm = BLSTM(channels, lstm_layers)
        else:
            self.lstm = None

        if rescale:
            rescale_module(self, reference=rescale)

    def valid_length(self, length):
        """
        WHAT: Compute the minimum length L such that convolutions don't drop samples.
        WHY:  Strided conv: out_len = ceil((in_len - k + 1) / s). For transposed conv
              to invert exactly, in_len must satisfy certain constraints.
        HOW:  Simulate encoder (downsample depth times) then decoder (upsample).
              Resample 2× before if enabled, ½ after.
        """
        if self.resample:
            length *= 2  # Upsample first in forward

        # Encoder: each layer (length - kernel_size) / stride + 1
        for _ in range(self.depth):
            length = math.ceil((length - self.kernel_size) / self.stride) + 1
            length = max(1, length)

        # Decoder: each layer (length - 1) * stride + kernel_size
        for idx in range(self.depth):
            length = (length - 1) * self.stride + self.kernel_size

        if self.resample:
            length = math.ceil(length / 2)
        return int(length)

    def forward(self, mix):
        """
        WHAT: Separate mixed audio into stems. Input (B, C, T) → output (B, S, C, T).
        WHY:  Normalize for stable training; pad to valid length; skip connections for detail.
        HOW:  Normalize → pad → [opt resample 2×] → encoder (save skips) → LSTM →
              decoder (add skips) → [opt resample ½] → denormalize → reshape.
        """
        x = mix
        length = x.shape[-1]

        # Normalize: zero mean, unit std (per track). Denormalize at output.
        if self.normalize:
            mono = mix.mean(dim=1, keepdim=True)
            mean = mono.mean(dim=-1, keepdim=True)
            std = mono.std(dim=-1, keepdim=True)
            x = (x - mean) / (1e-5 + std)
        else:
            mean = 0
            std = 1

        # Pad to valid length so convs don't drop samples at boundaries
        delta = self.valid_length(length) - length
        x = F.pad(x, (delta // 2, delta - delta // 2))

        # Resample 2×: more samples → finer temporal resolution for separation
        if self.resample:
            x = julius.resample_frac(x, 1, 2)

        saved = []
        for encode in self.encoder:
            x = encode(x)
            saved.append(x)  # Skip connection: encoder output → decoder input

        if self.lstm:
            x = self.lstm(x)

        for decode in self.decoder:
            skip = saved.pop(-1)
            skip = center_trim(skip, x)  # Align skip to current x (may differ by 1 from striding)
            x = decode(x + skip)  # U-Net skip: add encoder feature before upsample

        if self.resample:
            x = julius.resample_frac(x, 2, 1)
        x = x * std + mean
        x = center_trim(x, length)
        x = x.view(x.size(0), len(self.sources), self.audio_channels, x.size(-1))
        return x

    def load_state_dict(self, state, strict=True):
        """
        WHAT: Load checkpoint; remap old key names to new if needed.
        WHY:  Older Demucs had different layer indexing (e.g. rewrite at idx 2 vs 3).
        HOW:  If old key exists and new does not, copy state to new key.
        """
        for idx in range(self.depth):
            for a in ['encoder', 'decoder']:
                for b in ['bias', 'weight']:
                    new = f'{a}.{idx}.3.{b}'
                    old = f'{a}.{idx}.2.{b}'
                    if old in state and new not in state:
                        state[new] = state.pop(old)
        super().load_state_dict(state, strict=strict)
