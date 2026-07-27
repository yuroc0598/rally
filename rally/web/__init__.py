"""Web UI for the rally trimmer.

A thin, self-contained FastAPI layer over :func:`rally.pipeline.trim`. The core
``rally`` package is used unchanged and treated as a black box — this subpackage
adds upload/gallery/job management, live progress, side-by-side review, an
editable segment list, and re-export.

Run it with::

    python -m rally.web            # or: rally-web
"""

# NB: don't ``from .app import app`` here — binding the name ``app`` on the
# package would shadow the ``rally.web.app`` submodule. Import the module and
# read ``app`` off it instead (``rally.web.app:app`` is what uvicorn loads).
from .app import main

__all__ = ["main"]
