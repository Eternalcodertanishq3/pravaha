"""Adaptive Load Balancer — CPU/GPU/Hybrid dispatch.

Rules:
- CPU memory > 70% AND GPU available → shift inference to GPU
- GPU memory > 85% → offload KV cache blocks to CPU
- CPU > 70% AND GPU > 70% → split batch across devices
- All metrics polled every 2 seconds, decisions applied immediately.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

import psutil

logger = logging.getLogger(__name__)


class ComputeTarget(Enum):
    """Where to dispatch inference workloads."""

    CPU = auto()
    GPU = auto()
    HYBRID = auto()


@dataclass
class LoadSnapshot:
    """Point-in-time snapshot of system load and recommendation."""

    cpu_pct: float
    ram_pct: float
    gpu_pct: float
    vram_pct: float
    recommended: ComputeTarget
    reason: str
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        if self.timestamp == 0.0:
            self.timestamp = time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "cpu_pct": round(self.cpu_pct, 1),
            "ram_pct": round(self.ram_pct, 1),
            "gpu_pct": round(self.gpu_pct, 1),
            "vram_pct": round(self.vram_pct, 1),
            "recommended": self.recommended.name,
            "reason": self.reason,
        }


class AdaptiveLoadBalancer:
    """Monitor CPU/GPU load and recommend compute target.

    Plugs into AsyncPravahaEngine. Engine calls
    get_compute_target() before each generation step.
    """

    CPU_PRESSURE_THRESHOLD = 70.0   # % RAM usage → move to GPU
    GPU_PRESSURE_THRESHOLD = 85.0   # % VRAM usage → offload to CPU
    CPU_CORE_THRESHOLD = 80.0       # % CPU utilization → flag overload

    def __init__(self) -> None:
        self._current: ComputeTarget = ComputeTarget.CPU
        self._snapshot: LoadSnapshot | None = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._callbacks: list[Any] = []
        self._has_gpu = self._detect_gpu()
        self._transition_count = 0
        self._history: list[LoadSnapshot] = []

    @staticmethod
    def _detect_gpu() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def start(self) -> None:
        """Start the background monitoring loop."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._monitor_loop, daemon=True, name="LoadBalancer"
        )
        self._thread.start()
        logger.info(
            f"AdaptiveLoadBalancer started (GPU={'detected' if self._has_gpu else 'none'})"
        )

    def stop(self) -> None:
        """Stop the monitoring loop."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)

    def _monitor_loop(self) -> None:
        while self._running:
            try:
                self._poll()
            except Exception as e:
                logger.warning(f"LoadBalancer poll error: {e}")
            time.sleep(2.0)

    def _poll(self) -> None:
        cpu_pct = psutil.cpu_percent(interval=None)
        ram_pct = psutil.virtual_memory().percent
        gpu_pct = 0.0
        vram_pct = 0.0

        if self._has_gpu:
            try:
                import torch
                if torch.cuda.is_available():
                    props = torch.cuda.get_device_properties(0)
                    alloc = torch.cuda.memory_allocated(0)
                    reserved = torch.cuda.memory_reserved(0)
                    vram_pct = (reserved / props.total_memory) * 100 if props.total_memory > 0 else 0
                    gpu_pct = (alloc / props.total_memory) * 100 if props.total_memory > 0 else 0
            except Exception:
                pass

        # ── Decision logic ──
        if not self._has_gpu:
            target = ComputeTarget.CPU
            reason = "No GPU available"
        elif ram_pct > self.CPU_PRESSURE_THRESHOLD and vram_pct < self.GPU_PRESSURE_THRESHOLD:
            target = ComputeTarget.GPU
            reason = f"RAM at {ram_pct:.0f}% → shifting to GPU"
        elif vram_pct > self.GPU_PRESSURE_THRESHOLD and ram_pct < self.CPU_PRESSURE_THRESHOLD:
            target = ComputeTarget.CPU
            reason = f"VRAM at {vram_pct:.0f}% → offloading to CPU"
        elif ram_pct > self.CPU_PRESSURE_THRESHOLD and vram_pct > self.GPU_PRESSURE_THRESHOLD:
            target = ComputeTarget.HYBRID
            reason = f"Both under pressure (RAM={ram_pct:.0f}%, VRAM={vram_pct:.0f}%) → hybrid split"
        elif cpu_pct > self.CPU_CORE_THRESHOLD and self._has_gpu:
            target = ComputeTarget.GPU
            reason = f"CPU cores at {cpu_pct:.0f}% → dispatch to GPU"
        else:
            target = ComputeTarget.GPU if self._has_gpu else ComputeTarget.CPU
            reason = "Normal load"

        snapshot = LoadSnapshot(
            cpu_pct=cpu_pct,
            ram_pct=ram_pct,
            gpu_pct=gpu_pct,
            vram_pct=vram_pct,
            recommended=target,
            reason=reason,
        )

        with self._lock:
            old = self._current
            self._current = target
            self._snapshot = snapshot
            self._history.append(snapshot)
            # Keep last 100 snapshots
            if len(self._history) > 100:
                self._history = self._history[-100:]

        if old != target:
            self._transition_count += 1
            logger.info(f"LoadBalancer: {old.name} → {target.name} ({reason})")
            for cb in self._callbacks:
                try:
                    cb(snapshot)
                except Exception as e:
                    logger.warning(f"LoadBalancer callback error: {e}")

    def get_compute_target(self) -> ComputeTarget:
        """Get current recommended compute target."""
        with self._lock:
            return self._current

    def get_snapshot(self) -> LoadSnapshot | None:
        """Get the most recent load snapshot."""
        with self._lock:
            return self._snapshot

    def get_device_string(self) -> str:
        """Return torch device string for current recommended target."""
        target = self.get_compute_target()
        if target == ComputeTarget.GPU and self._has_gpu:
            return "cuda"
        return "cpu"

    def get_stats(self) -> dict[str, Any]:
        """Return load balancer statistics."""
        with self._lock:
            snapshot = self._snapshot
        return {
            "current_target": self._current.name,
            "has_gpu": self._has_gpu,
            "transitions": self._transition_count,
            "snapshot": snapshot.to_dict() if snapshot else {},
        }

    def register_callback(self, cb: Any) -> None:
        """Register a callback for load change events."""
        self._callbacks.append(cb)
