import re
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class PRismCodeSanitizer(BaseEstimator, TransformerMixin):
    """Normalize row-specific synthetic identifiers in PRism code snippets."""

    def fit(self, X, y=None):
        return self

    @staticmethod
    def _sanitize_row(row):
        text = "" if pd.isna(row.get("pr_change_code")) else str(row["pr_change_code"])
        finding_id = str(row.get("finding_id", ""))

        # Remove direct synthetic trace IDs.
        text = re.sub(r"trace-F\d+", "trace-FID", text)

        # Normalize identifiers whose numeric suffix equals this row's finding number.
        match = re.search(r"(\d+)$", finding_id)
        if match:
            finding_num = int(match.group(1))
            pattern = (
                rf"(?<![A-Za-z0-9_$])"
                rf"([A-Za-z_$][A-Za-z0-9_$]*?)"
                rf"(?:_)?{finding_num}"
                rf"(?![A-Za-z0-9_$])"
            )
            text = re.sub(pattern, r"\1ID", text)

            # Normalize a literal finding ID if it appears anywhere else.
            text = text.replace(finding_id, "FINDING_ID")

        return text

    def transform(self, X):
        X = X.copy()
        X["pr_change_code_clean"] = X.apply(self._sanitize_row, axis=1)
        return X
