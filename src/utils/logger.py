import logging


def get_logger(name: str = "churn_project") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    return logger


def log_prediction_event(event: str, payload: dict | None = None) -> None:
    logger = get_logger("prediction.events")
    message = event
    if payload is not None:
        message = f"{event} | payload={payload}"
    logger.info(message)
