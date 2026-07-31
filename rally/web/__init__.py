"""Web UI and persistent job service for the rally trimmer.

The subpackage composes the core pipeline with upload/gallery/job management, live
progress, review, human labeling, editable segments, and re-export.

Run it with::

    python -m rally.web            # or: rally-web
"""

# NB: don't ``from .app import app`` here — binding the name ``app`` on the
# package would shadow the ``rally.web.app`` submodule. Import the module and
# read ``app`` off it instead (``rally.web.app:app`` is what uvicorn loads).
from .app import main

__all__ = ["main"]
