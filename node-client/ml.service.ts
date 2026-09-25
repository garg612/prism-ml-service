import type {
  MLFindingInput,
  MLPrediction,
} from "./ml.types";

const ML_SERVICE_URL =
  process.env.ML_SERVICE_URL ?? "http://localhost:8000";

export async function predictFinding(
  finding: MLFindingInput,
  requestId?: string,
): Promise<MLPrediction> {
  const response = await fetch(
    `${ML_SERVICE_URL}/api/v1/predict`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(requestId
          ? { "X-Request-ID": requestId }
          : {}),
      },
      body: JSON.stringify(finding),
    },
  );

  if (!response.ok) {
    const body = await response.text();

    throw new Error(
      `PRism ML service error ${response.status}: ${body}`,
    );
  }

  return (await response.json()) as MLPrediction;
}

export async function checkMLHealth(): Promise<{
  status: string;
  model_version: string;
  dataset_version: string;
  feature_version: string;
  threshold: number;
}> {
  const response = await fetch(
    `${ML_SERVICE_URL}/health`,
  );

  if (!response.ok) {
    throw new Error(
      `PRism ML health check failed: ${response.status}`,
    );
  }

  return response.json() as Promise<{
    status: string;
    model_version: string;
    dataset_version: string;
    feature_version: string;
    threshold: number;
  }>;
}
