class ProjectConfigError(ValueError):
    """Raised when required project configuration is missing or invalid."""


class ModelLoadError(RuntimeError):
    """Raised when a saved model artifact cannot be loaded."""
