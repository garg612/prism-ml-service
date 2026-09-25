# PRism ML Inference Service

This is the production inference boundary between the Python ML model and the PRism Node/Express backend.

## Files

```text
prism-ml-service/
├── model/
│   └── prism_exp2_ensemble_v1.0.joblib   # copy from Colab
├── app/
│   ├── __init__.py
│   └── main.py
├── prism_preprocessing.py
├── requirements.txt
├── test_payload.json
└── node-client/
    ├── ml.service.ts
    └── ml.types.ts
```

## Put the model artifact in place

Copy:

```text
prism_exp2_ensemble_v1.0.joblib
```

into:

```text
model/prism_exp2_ensemble_v1.0.joblib
```

`prism_preprocessing.py` must stay at the project root because the serialized sklearn pipelines reference the custom `PRismCodeSanitizer` class.

## Install

Use Python 3.13.x and the versions in `requirements.txt`.

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Start

From the `prism-ml-service` directory:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Health check

```http
GET http://localhost:8000/health
```

## Prediction endpoint

```http
POST http://localhost:8000/api/v1/predict
Content-Type: application/json
```

The request body is the same raw feature object used during model inference. Node should NOT calculate TF-IDF, one-hot encoding, scaling, calibration, ensemble math, or thresholding.

The Python service returns:

```json
{
  "finding_id": "MOCK-NODE-001",
  "risk_score": 0.87,
  "decision": "SURFACE",
  "threshold": 0.2,
  "model_version": "prism-exp2-ensemble-v1.0",
  "dataset_version": "v0.2",
  "feature_version": "structured-plus-code-tfidf-sanitized-v1",
  "component_scores": {
    "lr": 0.84,
    "rf": 0.91,
    "xgb": 0.86
  }
}
```

## cURL smoke test

```bash
curl -X POST http://localhost:8000/api/v1/predict ^
  -H "Content-Type: application/json" ^
  --data @test_payload.json
```

PowerShell:

```powershell
Invoke-RestMethod `
  -Uri "http://localhost:8000/api/v1/predict" `
  -Method Post `
  -ContentType "application/json" `
  -Body (Get-Content .\test_payload.json -Raw)
```

## Node integration

Copy:

```text
node-client/ml.types.ts
node-client/ml.service.ts
```

into the Node backend's service layer.

Set:

```env
ML_SERVICE_URL=http://localhost:8000
```

Then:

```ts
const prediction = await predictFinding(finding, requestId);

if (prediction.decision === "SURFACE") {
  // continue to LLM
} else {
  // filter
}
```

Node only orchestrates the workflow. The Python service owns the ML artifact and inference logic.

## Important

Do not send the dataset or TF-IDF vectors from Node.

Send the raw finding features and raw `pr_change_code`.

The deployed `.joblib` artifact contains the learned preprocessing/model pipelines, calibrators, ensemble configuration and selected threshold.
