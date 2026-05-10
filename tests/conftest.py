import os

import pytest


@pytest.fixture(autouse=True)
def _quiet_logging(monkeypatch):
    monkeypatch.setenv("IOMMI_LSP_LOG", os.environ.get("IOMMI_LSP_LOG", "WARNING"))
