# Demucs Learning Experiments

Step-by-step notebooks for learning the Demucs framework with first principles (what-why-how). Designed for **NVIDIA 3070 Ti (8GB VRAM)**.

## Prerequisites

- Python 3.8+
- PyTorch with CUDA
- Demucs installed: `pip install -e .` from repo root, or `pip install demucs`

## Notebooks (in order)

| # | Notebook | Focus |
|---|----------|-------|
| 0 | `0_setup_and_audio_basics.ipynb` | Environment check, waveform vs spectrogram, STFT |
| 1 | `1_audio_io_and_preprocessing.ipynb` | Loading, resampling, segment extraction |
| 2 | `2_model_architectures.ipynb` | Demucs, HDemucs, HTDemucs — compare params and shapes |
| 3 | `3_inference_pipeline.ipynb` | Pretrained separation, apply_model, chunking |
| 4 | `4_training_on_3070ti.ipynb` | Memory-efficient training, Dora overrides, synthetic sanity check |
| 5 | `5_minimal_custom_model.ipynb` | 2-source toy model, end-to-end from scratch |

## Running on Windows

1. Open Anaconda Prompt or PowerShell.
2. `cd D:\demucs`
3. `conda activate demucs` (or your env)
4. `jupyter notebook learning_experiments\` or open in VS Code/Cursor.

If `sys.path` fails, edit the first code cell to point to your demucs root:

```python
sys.path.insert(0, r'D:\demucs')  # Adjust path if needed
```

## Main Plan

See [LEARNING_PLAN.md](../LEARNING_PLAN.md) in the repo root for the full what-why-how roadmap.
