from pathlib import Path
import logging
from typing import Dict

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

# Required so joblib can resolve the custom transformer class
# stored inside the serialized sklearn pipelines.
from prism_preprocessing import PRismCodeSanitizer  # noqa: F401


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "model" / "prism_exp2_ensemble_v1.0.joblib"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("prism-ml")


if not MODEL_PATH.exists():
    raise RuntimeError(
        f"Model artifact not found: {MODEL_PATH}. "
        "Copy prism_exp2_ensemble_v1.0.joblib into model/."
    )

try:
    BUNDLE = joblib.load(MODEL_PATH)
except Exception as exc:
    raise RuntimeError(
        f"Failed to load PRism model artifact: {exc}"
    ) from exc


REQUIRED_MODELS = {"LR", "RF", "XGB"}
REQUIRED_CALIBRATORS = {"LR", "RF", "XGB"}

if set(BUNDLE.get("models", {}).keys()) != REQUIRED_MODELS:
    raise RuntimeError(
        "Invalid model artifact: expected LR, RF and XGB."
    )

if set(BUNDLE.get("calibrators", {}).keys()) != REQUIRED_CALIBRATORS:
    raise RuntimeError(
        "Invalid model artifact: expected three Platt calibrators."
    )


APP_VERSION = BUNDLE["model_version"]


class FindingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    finding_id: str = Field(min_length=1)

    language: str = Field(min_length=1)
    rule_id: str = Field(min_length=1)
    severity: str = Field(min_length=1)

    function_length: float = Field(ge=0)
    cyclomatic_complexity: float = Field(ge=0)
    file_size_lines: float = Field(ge=0)
    pr_changed_lines: float = Field(ge=0)

    historical_rule_fp_rate: float = Field(ge=0, le=1)
    file_churn_90d: float = Field(ge=0)
    repo_finding_density_per_1k_loc: float = Field(ge=0)

    pr_change_code: str


class ComponentScores(BaseModel):
    lr: float
    rf: float
    xgb: float


class PredictionResponse(BaseModel):
    finding_id: str
    risk_score: float
    decision: str
    threshold: float
    model_version: str
    dataset_version: str
    feature_version: str
    component_scores: ComponentScores


app = FastAPI(
    title="PRism ML Inference API",
    version=APP_VERSION,
)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_version": BUNDLE["model_version"],
        "dataset_version": BUNDLE["dataset_version"],
        "feature_version": BUNDLE["feature_version"],
        "threshold": BUNDLE["threshold"],
    }


@app.get("/")
def root() -> dict:
    return {
        "service": "prism-ml",
        "status": "ok",
        "model_version": BUNDLE["model_version"],
    }


@app.post(
    "/api/v1/predict",
    response_model=PredictionResponse,
)
def predict(
    finding: FindingRequest,
    request: Request,
) -> PredictionResponse:
    request_id = request.headers.get("x-request-id", "-")

    try:
        input_data = finding.model_dump()

        # Each saved sklearn pipeline contains:
        # code sanitization + TF-IDF + structured preprocessing + classifier.
        X = pd.DataFrame([input_data])

        component_scores: Dict[str, float] = {}

        for model_name in ("LR", "RF", "XGB"):
            raw_probability = float(
                BUNDLE["models"][model_name].predict_proba(X)[0, 1]
            )

            calibrated_probability = float(
                BUNDLE["calibrators"][model_name]
                .predict_proba([[raw_probability]])[0, 1]
            )

            component_scores[model_name] = calibrated_probability

        risk_score = float(
            np.mean(list(component_scores.values()))
        )

        threshold = float(BUNDLE["threshold"])

        decision = (
            "SURFACE"
            if risk_score >= threshold
            else "FILTER"
        )

        logger.info(
            "ML prediction request_id=%s finding_id=%s "
            "risk_score=%.6f decision=%s model=%s",
            request_id,
            finding.finding_id,
            risk_score,
            decision,
            BUNDLE["model_version"],
        )

        return PredictionResponse(
            finding_id=finding.finding_id,
            risk_score=round(risk_score, 6),
            decision=decision,
            threshold=threshold,
            model_version=BUNDLE["model_version"],
            dataset_version=BUNDLE["dataset_version"],
            feature_version=BUNDLE["feature_version"],
            component_scores=ComponentScores(
                lr=round(component_scores["LR"], 6),
                rf=round(component_scores["RF"], 6),
                xgb=round(component_scores["XGB"], 6),
            ),
        )

    except Exception:
        logger.exception(
            "ML prediction failed request_id=%s finding_id=%s",
            request_id,
            finding.finding_id,
        )

        raise HTTPException(
            status_code=500,
            detail="ML prediction failed",
        )
