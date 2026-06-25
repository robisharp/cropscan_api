# CropScan API — Backend

REST API for crop disease detection. Deployed on Render.  
Frontend: [cropscan-app](https://robisharp.github.io/cropscan_api/)

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Service info |
| GET | `/health` | Health check |
| GET | `/diseases` | List detectable diseases |
| POST | `/predict` | Upload leaf image → full diagnosis |

## POST /predict

Send a `multipart/form-data` POST with field `file` (PNG or JPG).

**Response:**
```json
{
  "disease":              { "id", "name", "plant", "icon" },
  "confidence":           0.87,
  "severity_pct":         34.2,
  "risk_level":           "High",
  "urgency":              "danger",
  "urgency_message":      "...",
  "treatments":           ["..."],
  "feature_importances":  [{ "feature", "shap_value", "importance" }],
  "model_votes":          { "xgboost": {...}, "random_forest": {...}, "svm": {...} },
  "segmentation_b64":     "<base64 PNG overlay>",
  "inference_ms":         142
}
```

## ML Pipeline

- **Feature extraction:** HOG · LBP · HSV Color Stats · GLCM Texture · Pixel Ratios · Edge Density (24 features)
- **Classifiers:** XGBoost (50%) + Random Forest (30%) + SVM-RBF (20%) ensemble
- **Explainability:** SHAP TreeExplainer (top-5 features per prediction)
- **Severity:** HSV pixel thresholding → infected area ratio

## Run locally

```bash
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

## Deploy to Render

1. Push this repo to GitHub
2. New Web Service on render.com → connect repo
3. Build: `pip install -r requirements.txt`
4. Start: `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Free tier works fine for portfolio use
