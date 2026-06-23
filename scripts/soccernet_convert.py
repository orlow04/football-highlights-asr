"""Converte Labels-v2.json (SoccerNet) → eventos.json no formato do projeto.

No SoccerNet, cada anotação traz:
  - gameTime: "1 - 00:53"  (metade - mm:ss)
  - position: "53000"       (MILISSEGUNDOS desde o início daquela metade)
  - label:    "Goal" / "Penalty" / ...

Como o áudio é extraído por metade (1_224p.mkv → início da metade 1), filtra-se
UMA metade e usa-se position/1000 como t_seg. O resultado sai no formato PT que o
pipeline já entende ({"eventos": [{"t_seg", "tipo"}]}), alinhado com o .wav da
mesma metade.

Uso:
    python scripts/soccernet_convert.py \
        --labels data/soccernet/<liga>/<jogo>/Labels-v2.json \
        --half 1 --out data/soccernet/eventos.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Labels de "highlight" (alto interesse). Sobrescreva com --labels-keep se quiser.
RELEVANTES_DEFAULT = [
    "Goal", "Penalty", "Yellow card", "Red card", "Yellow->red card",
    "Shots on target", "Shots off target",
]


def converte(labels_path: str, half: int, relevantes) -> list[dict]:
    data = json.loads(Path(labels_path).read_text(encoding="utf-8"))
    keep = set(relevantes) if relevantes else None
    eventos = []
    for a in data.get("annotations", []):
        gt = a.get("gameTime", "")
        try:
            h = int(gt.split("-")[0].strip())
        except (ValueError, IndexError):
            continue
        if h != half:
            continue
        label = a.get("label", "")
        if keep is not None and label not in keep:
            continue
        try:
            t_seg = float(a["position"]) / 1000.0
        except (KeyError, ValueError):
            continue
        eventos.append({"t_seg": round(t_seg, 1), "tipo": label})
    eventos.sort(key=lambda e: e["t_seg"])
    return eventos


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, help="Labels-v2.json")
    ap.add_argument("--half", type=int, default=1, choices=[1, 2])
    ap.add_argument("--out", required=True)
    ap.add_argument("--labels-keep", nargs="*", default=None,
                    help="labels a manter (default: highlights). Vazio = todos.")
    args = ap.parse_args()

    relevantes = (RELEVANTES_DEFAULT if args.labels_keep is None
                  else args.labels_keep)
    eventos = converte(args.labels, args.half, relevantes)
    if not eventos:
        sys.exit("Nenhum evento após o filtro — confira --half e --labels-keep.")

    out = {"fonte": "soccernet", "labels": args.labels, "half": args.half,
           "eventos": eventos}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False),
                              encoding="utf-8")
    tipos = {}
    for e in eventos:
        tipos[e["tipo"]] = tipos.get(e["tipo"], 0) + 1
    print(f"OK: {args.out} | {len(eventos)} eventos (metade {args.half}) | {tipos}")


if __name__ == "__main__":
    main()
