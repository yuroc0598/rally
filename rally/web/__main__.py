"""Allow ``python -m rally.web`` to launch the server."""

from .app import main

if __name__ == "__main__":
    raise SystemExit(main())
