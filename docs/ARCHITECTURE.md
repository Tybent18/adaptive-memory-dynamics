# Architecture

[← Project home](../README.md) · [Methods](METHODS.md) · [Roadmap →](ROADMAP.md)

```mermaid
flowchart TD
    A[Experiment configuration] --> B[Deterministic dataset split]
    B --> C[Persistent baseline]
    B --> D[Passive temporal decay]
    B --> E[Reinforced retention]
    C --> F[Protocol runner]
    D --> F
    E --> F
    F --> G[CSV and JSON evidence]
    F --> H[Comparison charts]
```

## Module map

| Module | Responsibility |
|---|---|
| `data.py` | deterministic dataset loading and continual-task construction |
| `model.py` | transparent MLP, usage memory, and decay operators |
| `experiment.py` | controlled training, retention, continual learning, exports |
| `visualization.py` | fixed-style evidence charts |
| `cli.py` | scripted/reproducible execution |
| `gui.py` | one-click local experiment collection |
| `demo.py` | deterministic README demonstration asset |

The model and experimental protocol are deliberately separate. Future PyTorch models can implement the same mechanism contract without changing metrics, exports, or chart generation.

