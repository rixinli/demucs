# Demucs Learning Plan: From First Principles to Your Own Audio Model

**Goal:** Build foundational understanding of the Demucs framework to design your own audio processing model.  
**Constraint:** NVIDIA 3070 Ti laptop GPU (~8GB VRAM)  
**Approach:** What → Why → How (first principles)

---

## Part I: What — Core Concepts

### 1. Music Source Separation (MSS)
**What:** Given a mixed audio recording (e.g., a song), predict the individual sources (drums, bass, vocals, other).

**Why it matters:** Enables karaoke, remixing, mastering, and is the foundation for many audio ML tasks.

**Key insight:** Supervised learning. We need `(mix, sources)` pairs. The model learns: `sources = f_θ(mix)`.

### 2. Waveform vs Spectrogram Domain
| Domain | What | Pros | Cons |
|--------|------|------|------|
| **Waveform** | Raw samples (time series) | No info loss, phase-preserving | Long sequences, heavy compute |
| **Spectrogram** | STFT magnitude/phase (time-freq) | Compact, interpretable | Phase reconstruction artifacts |

**Demucs insight:** Use BOTH — hybrid models. Frequency branch captures global structure; time branch captures fine temporal detail.

### 3. Demucs Model Evolution
| Version | Model | Domain | Key idea |
|---------|-------|--------|----------|
| v1/v2 | Demucs | Waveform | U-Net + LSTM on raw audio |
| v3 | HDemucs | Hybrid | Dual U-Net: time + spectrogram branches |
| v4 | HTDemucs | Hybrid + Transformer | Cross-domain Transformer between encoders |

**HTDemucs in depth:** See [docs/htdemucs_comparison.md](docs/htdemucs_comparison.md) for a detailed comparison of Demucs, HDemucs, and HTDemucs — bottlenecks, DConv differences, and why the Transformer was added.

---

## Part II: Why — Design Decisions

### Why U-Net?
- **Encoder:** Compresses spatial/temporal resolution, increases channels → hierarchical features
- **Decoder:** Upsamples back, with skip connections from encoder
- **Skip connections:** Preserve fine details lost in downsampling (critical for audio)

### Why Hybrid (Time + Frequency)?
- Spectrogram: good for harmonic structure, efficient for long-range dependencies
- Waveform: preserves phase, avoids STFT artifacts
- Combining both: best of both worlds

### Why Transformer in v4?
- **Self-attention:** global context within each branch (freq and time)
- **Cross-attention:** information flow *between* branches — freq attends to time, time attends to freq
- HDemucs only merges branches (simple inject); HTDemucs enables true interaction via cross-attention
- Replaces zero bottleneck + LSTM/attention in DConv → better long-range modeling

### Why Chunking (Segment) at Inference?
- Full-song forward pass exceeds GPU memory
- Process overlapping chunks, blend at boundaries
- Trade-off: smaller segment = less memory, worse quality at boundaries

---

## Part III: How — Demucs Workflow

### Inference Pipeline
```
Audio file → Load & resample → Chunk (overlap) → Model forward → Blend chunks → Save stems
```

### Training Pipeline
```
MusDB HQ dataset → Sample segment → Augment → Model forward → L1/MSE loss → Backprop
```

### Key Hyperparameters (3070 Ti Friendly)
| Param | Default | 3070 Ti Suggested | Why |
|-------|---------|-------------|-----|
| `batch_size` | 64 | 4–8 | VRAM limit |
| `segment` | 11s | 6–8s | Shorter = less memory |
| `channels` | 48–64 | 32 | Smaller model |
| `depth` | 4–6 | 4 | Fewer layers |

*Note: For notebooks, ensure `sys.path` includes the demucs repo root (e.g. `D:\demucs`). Run notebooks from repo root or adjust the path in the first cell.*

---

## Part IV: Step-by-Step Learning Path

### Phase 1: Foundations (Notebooks 0–1)
1. **0_setup_and_audio_basics.ipynb**  
   - Environment setup, PyTorch/CUDA check  
   - What is STFT? Visualize time vs frequency  
   - Mix = sum of sources (supervised setup)

2. **1_audio_io_and_preprocessing.ipynb**  
   - Demucs audio loading (`demucs.audio`)  
   - Resampling, channel handling  
   - Segment extraction logic

### Phase 2: Model Understanding (Notebooks 2–3)
3. **2_model_architectures.ipynb**  
   - Instantiate Demucs, HDemucs, HTDemucs  
   - Compare parameter counts, input/output shapes  
   - Trace one forward pass

4. **3_inference_pipeline.ipynb**  
   - `apply_model` with chunking  
   - Shift trick for equivariance  
   - Run separation on a short file

### Phase 3: Training & Customization (Notebooks 4–5)
5. **4_training_on_3070ti.ipynb**  
   - Minimal training config for 3070 Ti  
   - Single-epoch sanity check  
   - Memory profiling

6. **5_minimal_custom_model.ipynb**  
   - Simplified 2-source toy task  
   - Modify architecture (channels, depth)  
   - End-to-end: data → train → separate

---

## Part V: Building Your Own Model — Design Checklist

When designing your audio model:

1. **Task:** What do you separate? (sources, number)
2. **Domain:** Waveform, spectrogram, or hybrid?
3. **Architecture:** U-Net backbone? Add Transformer?
4. **Memory budget:** Segment length × batch size × model size
5. **Data:** Do you have (mix, sources) pairs?
6. **Loss:** L1 vs MSE; per-source weighting

---

## References

- [docs/htdemucs_comparison.md](docs/htdemucs_comparison.md) — Demucs vs HDemucs vs HTDemucs (bottlenecks, DConv, Transformer)
- [Hybrid Demucs Paper](https://arxiv.org/abs/2111.03600)
- [HTDemucs Paper](https://arxiv.org/abs/2211.08553)
- [Wave-U-Net](https://github.com/f90/Wave-U-Net) (Demucs inspiration)
- `docs/training.md` — full training guide
