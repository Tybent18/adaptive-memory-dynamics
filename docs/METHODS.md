# Methods

[← Project home](../README.md) · [Evidence →](EVIDENCE.md) · [Architecture →](ARCHITECTURE.md)

## Research question

Does controlled, use-dependent forgetting improve retention–adaptation trade-offs relative to persistent memory and indiscriminate temporal decay?

## Mechanisms

For hidden pathway \(j\), the post-update decay factor is:

\[
W_j \leftarrow W_j \exp(-\lambda_j \Delta t_j)
\]

where \(\Delta t_j\) is the pathway's inactivity duration. The conditions are:

1. **Persistent baseline:** \(\lambda_j = 0\).
2. **Passive decay:** \(\lambda_j = \lambda\).
3. **Reinforced retention:** \(\lambda_j = \lambda / (1 + \alpha u_j)\), where \(u_j\) is an exponential moving average of normalized activation and \(\alpha\) is reinforcement strength.

This makes reinforcement local and activity-dependent. It is not equivalent to applying the same weight-decay coefficient to every parameter.

## Stage One protocol

The reference engine is a transparent one-hidden-layer NumPy MLP. Stage One uses the deterministic `sklearn-digits-8x8` dataset because it is available offline and makes the entire experiment auditable in CI.

Each seed runs three protocols:

1. **Supervised learning:** stratified train/test split with all ten classes.
2. **Idle retention:** evaluate before and after decay-only intervals with no gradient updates.
3. **Continual learning:** train classes 0–4 (Task A), then classes 5–9 (Task B), and measure Task A loss.

Default evidence uses five independent seeds. All mechanisms receive the same split, initialization seed, architecture, learning rate, batch size, and epoch budget within each seed.

## Outcomes

| Outcome | Definition |
|---|---|
| Generalization gap | train accuracy − test accuracy |
| Retention loss | final test accuracy − post-idle accuracy |
| Catastrophic forgetting | max(0, Task A before B − Task A after B) |
| Adaptation epoch | first Task B epoch reaching 85% of that run's best Task B accuracy |

## Interpretation rules

- A mechanism is not “better” based on one metric or one seed.
- Reduced forgetting accompanied by failed Task B learning is not successful adaptation.
- Stage One supports pipeline and feasibility conclusions only.
- Dataset-scale claims require the planned PyTorch experiments and statistical reporting.

