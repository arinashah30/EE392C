from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TensorResidency:
    home_level: str
    resident_level: str | None = None
    last_home_touch_ns: float | None = None
    corrupt: bool = False


@dataclass
class FastBufferState:
    """Tracks what is cached in volatile fast levels (PSUM/SBUF) for one core."""

    occupants: dict[str, int] = field(default_factory=dict)
    used_bytes: int = 0

    def clear(self) -> None:
        self.occupants.clear()
        self.used_bytes = 0


@dataclass
class LevelPoolState:
    capacity_bytes: int
    used_bytes: int = 0
    occupants: dict[str, int] = field(default_factory=dict)

    def can_fit(self, nbytes: int) -> bool:
        return self.used_bytes + nbytes <= self.capacity_bytes

    def install(self, tensor_id: str, nbytes: int) -> None:
        if tensor_id in self.occupants:
            self.used_bytes -= self.occupants[tensor_id]
        self.occupants[tensor_id] = nbytes
        self.used_bytes += nbytes

    def remove(self, tensor_id: str) -> None:
        if tensor_id not in self.occupants:
            return
        self.used_bytes -= self.occupants[tensor_id]
        del self.occupants[tensor_id]
