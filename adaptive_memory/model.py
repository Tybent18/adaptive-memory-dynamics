"""Small transparent MLP with explicit adaptive forgetting dynamics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class TrainSnapshot:
    loss: float
    accuracy: float
    mean_usage: float
    weight_norm: float


class AdaptiveMLP:
    """A NumPy reference model whose memory mechanism is directly inspectable.

    Forgetting is applied to hidden-unit pathways after gradient updates. Passive
    decay treats every pathway equally. Reinforced decay protects pathways in
    proportion to their exponentially averaged activation usage.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        mechanism: str = "baseline",
        forgetting_rate: float = 0.004,
        reinforcement_strength: float = 8.0,
        activation_threshold: float = 0.12,
        seed: int = 42,
    ) -> None:
        if mechanism not in {"baseline", "passive", "reinforced"}:
            raise ValueError("unknown memory mechanism")
        rng = np.random.default_rng(seed)
        self.w1 = rng.normal(0, np.sqrt(2 / input_size), (input_size, hidden_size))
        self.b1 = np.zeros(hidden_size)
        self.w2 = rng.normal(0, np.sqrt(2 / hidden_size), (hidden_size, output_size))
        self.b2 = np.zeros(output_size)
        self.mechanism = mechanism
        self.forgetting_rate = forgetting_rate
        self.reinforcement_strength = reinforcement_strength
        self.activation_threshold = activation_threshold
        self.usage = np.zeros(hidden_size)
        self.inactive_steps = np.zeros(hidden_size, dtype=np.int64)
        self.step = 0

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=1, keepdims=True)

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        hidden = np.maximum(0.0, x @ self.w1 + self.b1)
        return self._softmax(hidden @ self.w2 + self.b2), hidden

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)[0].argmax(axis=1)

    def accuracy(self, x: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(x) == y))

    def _observe_usage(self, hidden: np.ndarray) -> None:
        activity = np.mean(hidden, axis=0)
        scale = max(float(activity.max()), 1e-12)
        normalized = activity / scale
        self.usage = 0.92 * self.usage + 0.08 * normalized
        active = normalized >= self.activation_threshold
        self.inactive_steps = np.where(active, 0, self.inactive_steps + 1)

    def apply_forgetting(self, steps: int = 1) -> None:
        if steps <= 0 or self.mechanism == "baseline" or self.forgetting_rate == 0:
            return
        elapsed = np.maximum(1, self.inactive_steps + steps)
        rates = np.full(self.usage.shape, self.forgetting_rate)
        if self.mechanism == "reinforced":
            rates = rates / (1.0 + self.reinforcement_strength * self.usage)
        factors = np.exp(-rates * elapsed)
        self.w1 *= factors[np.newaxis, :]
        self.b1 *= factors
        self.w2 *= factors[:, np.newaxis]
        self.inactive_steps += steps

    def train_batch(self, x: np.ndarray, y: np.ndarray, learning_rate: float) -> TrainSnapshot:
        probs, hidden = self.forward(x)
        n = len(x)
        loss = -float(np.mean(np.log(probs[np.arange(n), y] + 1e-12)))
        grad_logits = probs.copy()
        grad_logits[np.arange(n), y] -= 1
        grad_logits /= n
        grad_w2 = hidden.T @ grad_logits
        grad_b2 = grad_logits.sum(axis=0)
        grad_hidden = (grad_logits @ self.w2.T) * (hidden > 0)
        grad_w1 = x.T @ grad_hidden
        grad_b1 = grad_hidden.sum(axis=0)
        self.w1 -= learning_rate * grad_w1
        self.b1 -= learning_rate * grad_b1
        self.w2 -= learning_rate * grad_w2
        self.b2 -= learning_rate * grad_b2
        self._observe_usage(hidden)
        self.apply_forgetting()
        self.step += 1
        return TrainSnapshot(
            loss=loss,
            accuracy=float(np.mean(probs.argmax(axis=1) == y)),
            mean_usage=float(self.usage.mean()),
            weight_norm=self.weight_norm(),
        )

    def weight_norm(self) -> float:
        return float(np.sqrt(np.sum(self.w1**2) + np.sum(self.w2**2)))

    def clone(self) -> AdaptiveMLP:
        duplicate = AdaptiveMLP(
            self.w1.shape[0],
            self.w1.shape[1],
            self.w2.shape[1],
            self.mechanism,
            self.forgetting_rate,
            self.reinforcement_strength,
            self.activation_threshold,
        )
        for name in ("w1", "b1", "w2", "b2", "usage", "inactive_steps"):
            setattr(duplicate, name, getattr(self, name).copy())
        duplicate.step = self.step
        return duplicate
