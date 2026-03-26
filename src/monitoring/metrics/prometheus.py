"""Prometheus custom metrics for the PredMaint API.

All metric objects are module-level singletons so they are registered once
with the default Prometheus registry and reused across requests.
"""

from prometheus_client import Counter, Histogram

PREDICTIONS_TOTAL = Counter(
    name="predmaint_predictions_total",
    documentation="Number of predictions served, labelled by outcome.",
    labelnames=["outcome"],  # "failure" | "normal"
)

MODEL_CONFIDENCE_SCORE = Histogram(
    name="predmaint_model_confidence_score",
    documentation="Distribution of the model failure-probability output (0–1).",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

DATA_DRIFT_DETECTED_TOTAL = Counter(
    name="predmaint_data_drift_detected_total",
    documentation="Number of drift-detection runs where drift was detected.",
)
