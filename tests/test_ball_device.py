from types import SimpleNamespace

from rally.signals.ball import resolve_device


class _Device:
    def __init__(self, name):
        self.type = name

    def __str__(self):
        return self.type


def _torch(*, cuda=False, mps=False):
    return SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: cuda),
        backends=SimpleNamespace(mps=SimpleNamespace(
            is_built=lambda: mps, is_available=lambda: mps)),
        device=lambda name: _Device(name),
    )


def test_resolve_device_prefers_cuda_then_mps(monkeypatch):
    monkeypatch.delenv("RALLY_DEVICE", raising=False)
    assert str(resolve_device(_torch(cuda=True, mps=True))) == "cuda"
    assert str(resolve_device(_torch(mps=True))) == "mps"
    assert str(resolve_device(_torch())) == "cpu"


def test_unavailable_explicit_mps_falls_back_to_cpu(monkeypatch):
    monkeypatch.setenv("RALLY_DEVICE", "mps")
    assert str(resolve_device(_torch())) == "cpu"
