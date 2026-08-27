"""API entrypoint: uvicorn app.api_main:app"""

from retailpulse.api.app import app

__all__ = ["app"]
