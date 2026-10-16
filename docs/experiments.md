# Experiment history

The model was developed through many iterations of a single training notebook (about 40 notebooks in total, differing mostly in a few lines: dropout, whether OCR text is used, base encoder, how video embeddings are fused). Their logic is now in `app/src/` and `app/train.py`.

Kept in `experiments/` because they explore ideas that are not in `app/src/`:

| Notebook | Idea |
|---|---|
| `curriculum.ipynb` | Curriculum-style training variant |
| `separate_branches_fallback.ipynb` | Variant with separate branches and a fallback path (see the notebook for details) |
| `augmented_data.ipynb` | Variant trained with augmented data |
| `feed_forward_head.ipynb` | Variant with a feed-forward head |

These notebooks have no saved outputs and were not re-run after the repository cleanup.
