"""Typed experiment configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(slots=True)
class ExperimentConfig:
    mechanism: str = "reinforced"
    seed: int = 42
    epochs: int = 24
    continual_epochs: int = 16
    hidden_size: int = 32
    learning_rate: float = 0.08
    batch_size: int = 64
    forgetting_rate: float = 0.004
    reinforcement_strength: float = 8.0
    activation_threshold: float = 0.12
    idle_steps: int = 30
    test_size: float = 0.25
    output_dir: Path = Path("results/runs/latest")

    def __post_init__(self) -> None:
        if self.mechanism not in {"baseline", "passive", "reinforced"}:
            raise ValueError("mechanism must be baseline, passive, or reinforced")
        if self.epochs < 1 or self.continual_epochs < 1:
            raise ValueError("epoch counts must be positive")
        if not 0 <= self.forgetting_rate < 1:
            raise ValueError("forgetting_rate must be in [0, 1)")
        if self.batch_size < 1 or self.hidden_size < 1:
            raise ValueError("batch_size and hidden_size must be positive")
        self.output_dir = Path(self.output_dir)

    def to_dict(self) -> dict[str, object]:
        values = asdict(self)
        values["output_dir"] = str(self.output_dir)
        return values
