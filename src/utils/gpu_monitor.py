"""
Lightweight NVIDIA GPU monitor for training-time telemetry.

Uses the ``nvidia-ml-py3`` package (which exposes the ``pynvml`` module)
to read GPU name, utilization, VRAM usage, and temperature without
spawning any shell commands.

Design principles:
- Import is lazy + optional: if the library is missing or NVML cannot be
  initialised (no driver, no GPU, container without access), the monitor
  degrades to a no-op logger and training continues uninterrupted.
- All reads are cached for one second so calling once per epoch is cheap.
- No background threads, no subprocesses — pure synchronous NVML calls.

Usage::

    from src.utils.gpu_monitor import GPUMonitor

    monitor = GPUMonitor()              # picks GPU 0 by default
    monitor.log_once()                   # one-shot summary line

    for epoch in range(epochs):
        ...
        monitor.log_epoch(epoch + 1)    # "GPU: ... | VRAM: ... | Util: ... | Temp: ..."
"""

from __future__ import annotations

import time
from typing import Optional

from src.utils.logger import get_logger

log = get_logger(__name__)

# Try to import the NVIDIA management library.  ``nvidia-ml-py3`` is the
# actively-maintained PyPI package; it registers itself under the legacy
# ``pynvml`` module name so the import below works for both.
try:  # pragma: no cover - exercised only when the lib is present
    import pynvml as _nvml  # type: ignore
    _NVML_AVAILABLE = True
except Exception:  # ImportError, OSError (no libnvidia-driver), etc.
    _nvml = None
    _NVML_AVAILABLE = False


class GPUMonitor:
    """Best-effort GPU telemetry reader.

    If NVML cannot be initialised (driver missing, no GPU, container
    without /dev/nvidia*), every method becomes a no-op and ``available``
    is set to ``False``.  Training code can therefore call this freely
    without guarding.
    """

    def __init__(self, device_index: int = 0) -> None:
        self.device_index = device_index
        self._handle = None
        self._name: Optional[str] = None
        self._total_mem_mb: Optional[float] = None
        self._last_read_ts: float = 0.0
        self._cache_ttl_seconds: float = 1.0
        self._cached_stats: Optional[dict] = None

        if not _NVML_AVAILABLE:
            log.debug("pynvml not available; GPU monitoring disabled.")
            return

        try:
            _nvml.nvmlInit()
            # Make sure the requested GPU exists; fall back to 0 otherwise.
            count = _nvml.nvmlDeviceGetCount()
            if count == 0:
                log.debug("NVML reports zero GPUs; monitoring disabled.")
                return
            idx = min(device_index, count - 1)
            self._handle = _nvml.nvmlDeviceGetHandleByIndex(idx)
            self._name = _nvml.nvmlDeviceGetName(self._handle)
            if isinstance(self._name, bytes):
                self._name = self._name.decode("utf-8", errors="replace")
            mem = _nvml.nvmlDeviceGetMemoryInfo(self._handle)
            self._total_mem_mb = mem.total / (1024 ** 2)
            self.device_index = idx
            log.debug(
                "GPU monitor attached to %s (%.0f MiB total).",
                self._name, self._total_mem_mb,
            )
        except Exception as e:  # NVML_ERROR_*, driver issues, etc.
            log.debug("NVML init failed; GPU monitoring disabled: %s", e)
            self._handle = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def available(self) -> bool:
        """True iff NVML is initialised and a GPU handle is bound."""
        return self._handle is not None

    def log_once(self) -> None:
        """Print a single header line with the GPU identity (or skip)."""
        if not self.available:
            return
        stats = self._read_stats(force=True)
        if stats is None:
            return
        log.info(
            "GPU: %s | VRAM: %.1f / %.1f GB | Utilization: %d%% | Temperature: %d°C",
            stats["name"],
            stats["vram_used_gb"],
            stats["vram_total_gb"],
            stats["util_pct"],
            stats["temp_c"],
        )

    def log_epoch(self, epoch: int, total_epochs: int) -> None:
        """Per-epoch one-liner with a small cache to avoid NVML spam."""
        if not self.available:
            return
        stats = self._read_stats(force=False)
        if stats is None:
            return
        log.info(
            "Epoch %d/%d | GPU: %s | VRAM: %.1f / %.1f GB | "
            "Utilization: %d%% | Temperature: %d°C",
            epoch, total_epochs,
            stats["name"],
            stats["vram_used_gb"],
            stats["vram_total_gb"],
            stats["util_pct"],
            stats["temp_c"],
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _read_stats(self, force: bool = False) -> Optional[dict]:
        if not self.available:
            return None
        now = time.monotonic()
        if (
            not force
            and self._cached_stats is not None
            and (now - self._last_read_ts) < self._cache_ttl_seconds
        ):
            return self._cached_stats

        try:
            util = _nvml.nvmlDeviceGetUtilizationRates(self._handle)
            mem = _nvml.nvmlDeviceGetMemoryInfo(self._handle)
            try:
                temp_c = _nvml.nvmlDeviceGetTemperature(
                    self._handle, _nvml.NVML_TEMPERATURE_GPU,
                )
            except Exception:
                temp_c = -1
        except Exception as e:
            log.debug("NVML read failed: %s", e)
            return None

        total_mb = self._total_mem_mb or (mem.total / (1024 ** 2)) or 1.0
        stats = {
            "name": self._name or "NVIDIA GPU",
            "vram_used_gb": mem.used / (1024 ** 3),
            "vram_total_gb": total_mb / 1024.0,
            "util_pct": int(util.gpu),
            "temp_c": int(temp_c),
        }
        self._cached_stats = stats
        self._last_read_ts = now
        return stats


def shutdown() -> None:  # pragma: no cover - safe no-op when unavailable
    """Release NVML.  Call once at process exit if you initialised it manually."""
    if _NVML_AVAILABLE:
        try:
            _nvml.nvmlShutdown()
        except Exception:
            pass
