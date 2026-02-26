# SLM Text-to-SQL Module Documentation

Detailed documentation for each module in the SLM Text-to-SQL pipeline.

## Modules

| Module | Description | Documentation |
|--------|-------------|---------------|
| **config** | Configuration management with Pydantic | [config.md](./config.md) |
| **data** | Data pipeline (download, clean, format, split) | [data.md](./data.md) |
| **training** | SFT training with Unsloth + LoRA | [training.md](./training.md) |
| **evaluation** | Model evaluation and metrics | [evaluation.md](./evaluation.md) |
| **inference** | SQL generation and MLflow tracing | [inference.md](./inference.md) |
| **utils** | Shared utilities (logging, schema, SQL) | [utils.md](./utils.md) |
| **cli** | Command-line interface | [cli.md](./cli.md) |

## Quick Links

- [Main README](../../README.md)
- [Configuration Guide](./config.md)
- [Training Guide](./training.md)
- [Production Deployment](../../README.md#production-deployment)
