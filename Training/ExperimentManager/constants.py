"""Shared constants for the JimmyTorch ExperimentManager."""

DEFAULT_SERVER_HOST = "127.0.0.1"
DEFAULT_SERVER_PORT = 9000

RUN_DB_FILENAME = "status_and_log.sqlite"
GLOBAL_DB_FILENAME = "Experiment_GUI_Status.sqlite"

# Reservoir-sampling defaults per scalar tag.
DEFAULT_SCALAR_CAP = 10000
DEFAULT_SCALAR_RECENT_KEEP = 200

# Cap per figure tag (blobs are large).
DEFAULT_FIGURE_CAP = 100

# Status transitions the trainer emits.
STATUS_TRAINING = "training"
STATUS_EVALUATING = "evaluating"
STATUS_IDLE = "idle"
STATUS_DONE = "done"
STATUS_ERROR = "error"
