export interface MLFindingInput {
  finding_id: string;

  language: string;
  rule_id: string;
  severity: string;

  function_length: number;
  cyclomatic_complexity: number;
  file_size_lines: number;
  pr_changed_lines: number;

  historical_rule_fp_rate: number;
  file_churn_90d: number;
  repo_finding_density_per_1k_loc: number;

  pr_change_code: string;
}

export interface MLPrediction {
  finding_id: string;
  risk_score: number;
  decision: "SURFACE" | "FILTER";
  threshold: number;
  model_version: string;
  dataset_version: string;
  feature_version: string;

  component_scores: {
    lr: number;
    rf: number;
    xgb: number;
  };
}
