from __future__ import annotations

import pytest


def pytest_sessionstart(session):
    try:
        from openmagnets import reload_backend
    except Exception as exc:
        raise pytest.UsageError(
            "OpenMagnets could not be imported. Build the native backend first with: python scripts/build_native.py"
        ) from exc

    reload_backend()
