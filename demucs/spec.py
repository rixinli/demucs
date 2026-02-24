# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Conveniance wrapper to perform STFT and iSTFT"""

import torch as th


def spectro(x, n_fft=512, hop_length=None, pad=0):
    # ── Shape handling: support arbitrary leading dims (batch, channels, etc.) ──
    # *other = all dims except last; length = last dim (samples).
    # E.g. (2, 1, 44100) → other=(2,1), length=44100
    *other, length = x.shape

    # Flatten to (N, length) so stft sees (num_signals, samples).
    # stft expects (..., length); flattening allows batch+channel in one go.
    x = x.reshape(-1, length)

    # ── Device workaround: MPS (Apple M1+) and XPU (Intel) have buggy stft ──
    # Demucs moves to CPU for stft on these devices, then back in the model.
    is_mps_xpu = x.device.type in ['mps', 'xpu']
    if is_mps_xpu:
        x = x.cpu()

    z = th.stft(x,
                # n_fft * (1 + pad): FFT size. pad>0 → zero-pad window for finer freq resolution.
                # E.g. pad=1 → n_fft*2 bins, no change in time resolution.
                n_fft * (1 + pad),

                # hop_length: samples between consecutive frames. Default 75% overlap (n_fft//4).
                # Smaller hop = smoother time, more frames, higher cost.
                hop_length or n_fft // 4,

                # window: taper applied to each frame before FFT.
                # Hann = 0.5*(1 - cos(2πn/N)). Reduces spectral leakage vs rectangular.
                window=th.hann_window(n_fft).to(x),

                # win_length: actual window length. Can be < n_fft (then zero-padded).
                # Here win_length=n_fft → no zero-padding in window.
                win_length=n_fft,

                # normalized=True: scale output by 1/sqrt(n_fft).
                # Keeps total energy similar to input; helps reconstruction & training stability.
                normalized=True,

                # center=True: pad input so frame t is centered at t*hop_length.
                # First/last frames are centered; avoids truncating edges.
                center=True,

                # return_complex=True: output is complex dtype (standard since PyTorch 1.7).
                return_complex=True,

                # pad_mode='reflect': for center padding, use reflection at boundaries.
                # 'reflect' = [a,b,c,d] → [c,b,a,b,c,d,d,c,b]. Often better than 'zeros'.
                pad_mode='reflect')
    _, freqs, frame = z.shape
    # Restore original batch/channel layout: (N, freqs, frame) → (*other, freqs, frame)
    return z.view(*other, freqs, frame)


def ispectro(z, hop_length=None, length=None, pad=0):
    *other, freqs, frames = z.shape
    n_fft = 2 * freqs - 2
    z = z.view(-1, freqs, frames)
    win_length = n_fft // (1 + pad)
    is_mps_xpu = z.device.type in ['mps', 'xpu']
    if is_mps_xpu:
        z = z.cpu()
    x = th.istft(z,
                 n_fft,
                 hop_length,
                 window=th.hann_window(win_length).to(z.real),
                 win_length=win_length,
                 normalized=True,
                 length=length,
                 center=True)
    _, length = x.shape
    return x.view(*other, length)
