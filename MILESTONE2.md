# Milestone 2: MotionGPT pretrained inference in Colab

Reproduction of text-to-motion inference for MotionGPT (Jiang et al., NeurIPS 2023) using the authors' pretrained
Base (220M) checkpoint. Everything runs from `MotionGPT_Milestone2_Colab_v2.ipynb`.

## What was reproduced

- Pretrained MotionGPT-Base loaded from Hugging Face (`OpenMotionLab/MotionGPT-base`) on a Colab T4 GPU
- Text-to-motion for 5 instruction prompts with the repo's `demo.py`
- Output: `(1, frames, 22, 3)` joint-position arrays, rendered to skeleton GIFs with matplotlib

| Prompt | Frames | Length at 20 fps |
|---|---|---|
| Generate a person walks forward and then turns around. | 196 | 9.8 s |
| Generate a person waves with their right hand. | 84 | 4.2 s |
| Generate a person jumps up and down. | 56 | 2.8 s |
| Generate a person walks in a circle. | 196 | 9.8 s |
| Generate a person kicks with their left leg. | 196 | 9.8 s |

Timing from the final run: 13.6 s total including model loading, and the generation step for all 5 prompts took
about 3.4 s. 196 frames matches `MAX_MOTION_LEN: 196` in the repo's config, so those three may be capped.

Environment: Colab, T4 GPU, Python 3.13, `transformers` 5.20.2 (the repo was written and tested on Python 3.10.6,
PyTorch 2.0.0, and a much older `transformers`). Fill in the PyTorch version from the environment cell.

## Problems we hit and how they were fixed

Almost all of the work was making 2023 code run on a 2026 Colab.

| # | Symptom | Cause | Fix |
|---|---|---|---|
| 1 | `No module named 'pytorch_lightning'` | `pip install -r requirements.txt` aborts at the first failure (`chumpy` does not build), leaving everything uninstalled | Install each package separately; `chumpy` is not needed for the demo |
| 2 | `FileNotFoundError ... mean.npy` | Evaluator statistics script had been skipped; the demo needs it for mean/std | Run `prepare/download_t2m_evaluators.sh` |
| 3 | Same error again | Download extracts into `deps/t2m/t2m/t2m/`, code expects `deps/t2m/t2m/` | Copy the folders up one level |
| 4 | `deps/glove/our_vab_data.npy` missing | Word vectors extract to `deps/t2m/glove/` | Copy to `deps/glove/` |
| 5 | `datasets/humanml3d/test.txt` missing | `HumanML3D.py` loads part of the full dataset only to read the feature count | Set `nfeats = 263` directly (the demo needs no dataset) |
| 6 | `No module named 'bert_score'` | Imported by the metrics module, missing from `requirements.txt` | `pip install bert_score` |
| 7 | `SafetensorError: header too large` | The repo's scripts use `git clone` + `git lfs`; without git-lfs the model files are tiny text stubs | Download with `huggingface_hub.snapshot_download` |
| 8 | Outputs only 4 frames; raw output was a few repeated token IDs | Newer `transformers` ties `lm_head` to the input embeddings; loading the checkpoint overwrote one with the other (the checkpoint stores them as different tensors) | Untie after loading and restore both from the checkpoint |

Defensive changes also made: `model.eval()`, attention mask passed to `generate`, `weights_only=False` for
`torch.load`.

## How to reproduce

1. Open `MotionGPT_Milestone2_Colab_v2.ipynb` in Google Colab and select a T4 GPU runtime.
2. Run all cells top to bottom (about 10 to 15 minutes, mostly downloads).
3. Outputs are written to `results/mgpt/.../samples_<time>/`; the notebook converts them to GIFs.

## Not done / next steps

- Motion-to-text, prediction and in-between tasks: example inputs exist in `demos/` but were not run
- Full-body mesh rendering (needs Blender and SMPL models)
- Milestone 3: serve the model behind an API (the repo's `app.py` is a Gradio UI) and build a client. Note that `app.py`
  builds the same data module, so it will need the same patches, plus a Whisper model for its audio input.
