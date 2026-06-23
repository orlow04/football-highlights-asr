"""Wrapper fino e OPCIONAL do Weights & Biases.

Os scripts de experimento (E1/E2) sempre salvam resultados locais (CSV/JSON).
O wandb é só uma camada de visualização/comparação por cima. Se ele não estiver
instalado, ou se WANDB_MODE=disabled, estas funções viram no-op silencioso — a
reprodutibilidade nunca depende de ter conta/login no wandb.

Instalação (à parte do pin do torch, p/ não mexer no NeMo):
    pip install wandb && wandb login
"""
from __future__ import annotations

import os
from typing import Any


def disponivel() -> bool:
    if os.environ.get("WANDB_MODE") == "disabled":
        return False
    try:
        import wandb  # noqa: F401
        return True
    except ImportError:
        return False


def init(project: str, name: str | None = None, config: dict | None = None,
         group: str | None = None, job_type: str | None = None):
    if not disponivel():
        return None
    import wandb

    return wandb.init(project=project, name=name, config=config or {},
                      group=group, job_type=job_type, reinit=True)


def summary(run, data: dict[str, Any]) -> None:
    if run is not None:
        run.summary.update(data)


def log(run, data: dict[str, Any]) -> None:
    if run is not None:
        run.log(data)


def finish(run) -> None:
    if run is not None:
        run.finish()
