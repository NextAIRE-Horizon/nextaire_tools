# Examples

Runnable examples for `nextaire_tools`.

| File | What it shows | Needs network? |
|------|---------------|----------------|
| [`end_to_end.py`](end_to_end.py) | Full offline workflow: synthetic data → cleaning + feature pipeline → time-series CV → figures. | No |

## Run

```bash
pip install -e ".[deep]"      # from the repo root
python examples/end_to_end.py
```

`end_to_end.py` writes figures to `examples/figures/` and prints a per-fold
cross-validation report (MAE, RMSE, R², index of agreement, FAC2, …).

## Copernicus extraction

Downloading real ERA5 / CAMS / ERA5-Land data requires the `extract` extra and
Copernicus credentials — see the
[extractors guide](../docs/user-guide/extractors.md). The original
[`scripts/Era5Extractor.py`](../scripts/Era5Extractor.py) shows the same idea as
a standalone script; the packaged equivalent is
`nextaire_tools.extractors.ERA5Extractor.extract_to_frames(...)`.
