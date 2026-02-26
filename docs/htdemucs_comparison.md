# HTDemucs: What It Is and How It Differs from Demucs and HDemucs

## What is HTDemucs?

**HTDemucs** (Hybrid Transformer Demucs) is the v4 model in the Demucs family. It builds on HDemucs by adding a **CrossTransformerEncoder** at the bottleneck to enable information exchange between the frequency (spectrogram) and time (waveform) branches via cross-attention.

From the docstring in `demucs/htdemucs.py`:

> "Spectrogram and hybrid Demucs model... Hybrid model have a parallel time branch. At some layer, the time branch has the same stride as the frequency branch and then the two are combined."

The key addition: **unlike HDemucs, after the encoder, a Transformer block processes both branches and lets them attend to each other**.

---

## First Principles: What, Why, How

### What (Domain & Building Blocks)

| Model        | Input Domain                    | Bottleneck       | Output                               |
| ------------ | ------------------------------- | ---------------- | ------------------------------------ |
| **Demucs**   | Raw waveform                    | BiLSTM           | Waveform                             |
| **HDemucs**  | Spectrogram + waveform (hybrid) | Zero (skip-only) | Spectrogram mask + waveform residual |
| **HTDemucs** | Spectrogram + waveform (hybrid) | CrossTransformer | Spectrogram mask + waveform residual |

### Why (Design Rationale)

- **Demucs**: Pure waveform to avoid STFT phase issues.
- **HDemucs**: Spectrogram captures harmonics; waveform preserves phase; both run in parallel and merge at the bottleneck.
- **HTDemucs**: HDemucs only *merges* the two branches (adds time features into freq); there is no *interaction*. The Transformer adds **cross-attention** so that:
  - Freq branch can attend to time branch (global temporal context).
  - Time branch can attend to freq branch (harmonic context).

This improves long-range modeling at the bottleneck.

### How (Architecture Flow)

```
Demucs:     Raw waveform → Conv Encoder → BiLSTM → Conv Decoder

HDemucs:    STFT → Freq Encoder ──┐
            Waveform → Time Encoder ──→ Zero Bottleneck → Decoders with Skip

HTDemucs:   STFT → Freq Encoder ──┐
            Waveform → Time Encoder ──→ CrossTransformer → Decoders with Skip
```

---

## Concrete Differences

### 1. Bottleneck: Zero vs Transformer

|                   | HDemucs                                                | HTDemucs                                                                  |
| ----------------- | ------------------------------------------------------ | ------------------------------------------------------------------------- |
| Bottleneck        | `x = torch.zeros_like(x)` (information only via skips) | `x, xt = self.crosstransformer(x, xt)`                                    |
| Cross-domain flow | Simple `inject` at merge layer                         | Alternate **self-attention** and **cross-attention** over multiple layers |

In `demucs/htdemucs.py` lines 436-450:

```python
if self.crosstransformer:
    ...
    x, xt = self.crosstransformer(x, xt)
```

The transformer receives:

- **Freq**: `(B, C, Fr, T)` reshaped to sequence `(B, Fr*T, C)`
- **Time**: `(B, C, T)` as sequence `(B, T, C)`

### 2. CrossTransformerEncoder

From `demucs/transformer.py` lines 418-455:

- Alternating layers (parity controlled by `cross_first`):
  - **Odd indices**: `MyTransformerEncoderLayer` — self-attention within each branch.
  - **Even indices**: `CrossTransformerEncoderLayer` — cross-attention:
    - Freq branch: `q = freq, k = time`
    - Time branch: `q = time, k = freq`

This enables bidirectional information flow between frequency and time representations.

### 3. DConv: Simpler in HTDemucs

HDemucs DConv has LSTM and attention in deeper layers (`dconv_lstm`, `dconv_attn`). HTDemucs omits them:

- `demucs/hdemucs.py` lines 439-442: passes `lstm`, `attn` to `dconv_kw`
- `demucs/htdemucs.py` lines 284-289: only `depth`, `compress`, `init`, `gelu`

The Transformer replaces long-range modeling previously done by LSTM/attention in DConv.

### 4. Depth and Segment

|                 | HDemucs          | HTDemucs                                               |
| --------------- | ---------------- | ------------------------------------------------------ |
| Default `depth` | 6                | 4                                                      |
| `segment`       | Fixed per config | `use_train_segment`: uses training length at inference |

HTDemucs is designed to work with a fixed training segment length during inference (e.g. 10s default). See `valid_length()` and `use_train_segment` in `demucs/htdemucs.py` lines 392-403 and 413-419.

### 5. MultiWrap and Other Shared Components

Both share from HDemucs:

- `HEncLayer`, `HDecLayer`, `MultiWrap`, `ScaledEmbedding`
- Same STFT, CaC, Wiener, freq embedding
- Same merge point (when `freq` collapses to 1)

---

## Summary Table

| Aspect              | Demucs                                        | HDemucs                       | HTDemucs                                        |
| ------------------- | --------------------------------------------- | ----------------------------- | ----------------------------------------------- |
| Domain              | Waveform                                      | Hybrid (spec + waveform)      | Hybrid (spec + waveform)                        |
| Encoder             | Conv1d U-Net                                  | Conv2d (freq) + Conv1d (time) | Same as HDemucs                                 |
| Bottleneck          | BiLSTM                                        | Zero + skip                   | CrossTransformer                                |
| DConv extras        | LSTM, attn                                    | LSTM, attn                    | None (simpler)                                  |
| Default depth       | 6                                             | 6                             | 4                                               |
| Long-range modeling | LSTM                                          | LSTM in DConv                 | Transformer cross-attn                          |
| Paper               | [Demucs v2](https://arxiv.org/abs/1911.13254) | HDemucs (v3)                  | [HTDemucs v4](https://arxiv.org/abs/2211.08553) |

---

## Key Takeaway

**HTDemucs = HDemucs + CrossTransformer at bottleneck.**

- Keeps the hybrid dual-branch design (spectrogram + waveform).
- Replaces the zero bottleneck and LSTM/attention in DConv with a Transformer that:
  1. Does self-attention within each branch.
  2. Does cross-attention between branches.

This improves cross-domain information flow and long-range modeling, at the cost of more compute (transformer layers) and stricter segment handling.
