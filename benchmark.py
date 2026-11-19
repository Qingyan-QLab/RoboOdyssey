"""Compatibility entry point for the existing CI workflow in source checkouts."""

from roboodyssey.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
