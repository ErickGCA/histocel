"""Training engines: the only part of histocel that knows a training library.

An engine turns a built dataset into weights and, later, tiles into detections.
The rest of the package talks to engines through :class:`Engine` only, so adding
or replacing one does not touch anything else.
"""

from histocel.engines.base import (
    Engine,
    EngineError,
    TrainResult,
    TrainSettings,
    available_engines,
    get_engine,
    register_engine,
)

__all__ = [
    "Engine",
    "EngineError",
    "TrainResult",
    "TrainSettings",
    "available_engines",
    "get_engine",
    "register_engine",
]
