# Deployment artifact bundle

Lightweight manifest and config for inference. **Model weights are not committed** — copy from a training run:

```bash
RUN=runs/20260522_181848_v3_nifty100_from_nifty500  # or your hosted run
mkdir -p backend/artifacts/default_bundle/{checkpoints,artifacts}
cp "$RUN/checkpoints/model_final.pt" backend/artifacts/default_bundle/checkpoints/
cp "$RUN/artifacts/scalers.pkl" backend/artifacts/default_bundle/artifacts/
cp "$RUN/artifacts/calibrators.pkl" backend/artifacts/default_bundle/artifacts/
cp "$RUN/config.yaml" backend/artifacts/default_bundle/
cp "$RUN/traits.json" backend/artifacts/default_bundle/ 2>/dev/null || true
```

Used by local inference and mirrored in **StockXpert_Web** for the FastAPI backend.
