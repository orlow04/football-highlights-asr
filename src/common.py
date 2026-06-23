"""Utilitários compartilhados: config, normalização de texto, grade de 1 s."""
from __future__ import annotations

import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# Raiz do repositório (este arquivo está em src/).
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "params.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normaliza(t: str) -> str:
    """minúsculas + remoção de acentos (NFD) e de pontuação, para casar léxico/negadores.

    O checkpoint Parakeet é pontuado, então os tokens vêm com pontuação grudada
    ('gol,', 'rede!', 'Obrigado.'). Removê-la torna o casamento robusto — em especial
    o multi-palavra por tokens exatos ('na rede!' → ('na','rede') == chave). Espaços
    são preservados, pois as chaves multi-palavra do léxico/negadores dependem deles.
    """
    t = t.lower()
    t = "".join(
        c for c in unicodedata.normalize("NFD", t) if unicodedata.category(c) != "Mn"
    )
    return "".join(c for c in t if not unicodedata.category(c).startswith("P"))


# Conversão dígito→extenso (PT, sem acento) para WER justo: a transcrição de
# referência costuma escrever números por extenso ("seis", "onze") e o ASR usa
# dígitos ("6", "11"); sem unificar, cada número vira um erro de substituição.
_UNI = ["zero", "um", "dois", "tres", "quatro", "cinco", "seis", "sete",
        "oito", "nove"]
_DEZ = ["dez", "onze", "doze", "treze", "quatorze", "quinze", "dezesseis",
        "dezessete", "dezoito", "dezenove"]
_DEZENAS = {2: "vinte", 3: "trinta", 4: "quarenta", 5: "cinquenta",
            6: "sessenta", 7: "setenta", 8: "oitenta", 9: "noventa"}


def _num_extenso(n: int) -> str:
    if n < 10:
        return _UNI[n]
    if n < 20:
        return _DEZ[n - 10]
    if n < 100:
        d, u = divmod(n, 10)
        return _DEZENAS[d] + (" e " + _UNI[u] if u else "")
    return str(n)  # ≥100 (raro em narração): deixa como dígito


def converte_numeros(texto: str) -> str:
    """Troca dígitos 0–99 por extenso, para unificar a convenção antes do WER."""
    import re

    return re.sub(r"\d+", lambda m: _num_extenso(int(m.group()))
                  if int(m.group()) < 100 else m.group(), texto)


def zscore(x: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(np.asarray(x, dtype=float))
    return (x - x.mean()) / (x.std() + 1e-8)


def n_windows(duracao_s: float, janela: float) -> int:
    return int(np.ceil(duracao_s / janela))


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
