"""Host Copilot tools. Each returns {"component": ..., "data": ...}."""
from app.tools.costs import get_amenity_cost
from app.tools.planning import goal_seek, roi_rank
from app.tools.pricing import predict_price, what_if

__all__ = ["predict_price", "what_if", "goal_seek", "roi_rank", "get_amenity_cost"]
