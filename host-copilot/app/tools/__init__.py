"""Host Copilot tools. Each returns {"component": ..., "data": ...}."""
from app.tools.pricing import predict_price, what_if

__all__ = ["predict_price", "what_if"]
