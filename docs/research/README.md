# Adaptive Memory Dynamics research library

[← Project home](../../README.md) · [Stage One results](../../results/stage_one_baseline/README.md) · [Methods](../METHODS.md) · [Roadmap](../ROADMAP.md)

This library organizes the current evidence report, the next preregistered experiment, and the three-tier research progression. It deliberately separates what Stage One measured from what later studies propose.

## Evidence and next experiment

| Document | Role | Evidence status |
|---|---|---|
| [Stage One Technical Report](stage-one-technical-report.pdf) | Frozen five-seed comparison of persistent, passive-decay, and reinforced-retention mechanisms | Measured Stage One evidence |
| [Pre-Registered Experimental Protocol](experimental-protocol.pdf) | Forgetting-rate, reinforcement-strength, threshold, and task-sequence ablations | Planned confirmatory experiment; outcomes not yet claimed |

## Research progression

| Tier | Document | Scope |
|---|---|---|
| 1 | [Capstone: Adaptive Memory Dynamics](tier-1-capstone.pdf) | Controlled forgetting, passive decay, and reinforcement-sensitive retention |
| 2 | [Master's: Adaptive Retention Under Continual Learning](tier-2-masters.pdf) | Rate, reinforcement-strength, and task-sequence ablations |
| 3 | [Doctoral Agenda: Formal and Scalable Adaptive Memory Dynamics](tier-3-doctoral-agenda.pdf) | Memory allocation, selective decay, retrieval, and resource constraints |

## Claim boundary

Stage One establishes a reproducible experimental pipeline and measured behavior on `sklearn-digits-8x8`. Reinforced retention approximately matches the persistent control on ordinary accuracy and Task B adaptation; passive decay reduces accuracy; neither forgetting mechanism improves Task A retention in the frozen configuration.

These results do not establish a universal benefit from forgetting, a biologically faithful model of memory, or performance on MNIST, Fashion-MNIST, CIFAR-10, or real-world continual-learning systems. The doctoral formulation is a falsifiable research scaffold—not a discovered law of memory.

