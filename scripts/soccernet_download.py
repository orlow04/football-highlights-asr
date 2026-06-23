"""Baixa do SoccerNet o necessário para o Exp. 2b (generalização do mecanismo).

Conforme docs/estrategia_validacao.md e docs/soccernet_2b.md, o 2b roda detecção
LÉXICA sobre as transcrições Echoes — NÃO há ASR nem áudio aqui. Portanto baixa-se
apenas dados LEVES e textuais, NÃO vídeo/áudio (centenas de GB, exige NDA, inútil):

  1. Labels-v2.json  — ground-truth de evento (humano). Via cliente pip SoccerNet.
  2. Transcrições Echoes (Whisper) — via repositório git `SoccerNet/sn-echoes`
     (os JSONs ficam versionados em Dataset/whisper_v*/...; não passam pelo pip).

Uso:
    # 1) labels (público, sem senha)
    python scripts/soccernet_download.py --dir data/soccernet --split test
    # 2) transcrições Echoes (clonar uma vez; é leve, só texto)
    git clone --depth 1 https://github.com/SoccerNet/sn-echoes data/sn-echoes
    #    → use --echoes-root data/sn-echoes/Dataset/whisper_v3 no soccernet_echoes
"""
from __future__ import annotations

import argparse
import sys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/soccernet", help="diretório dos labels")
    ap.add_argument("--split", default="test",
                    choices=["train", "valid", "test", "challenge"])
    args = ap.parse_args()

    try:
        from SoccerNet.Downloader import SoccerNetDownloader
    except ImportError:
        sys.exit("Cliente não instalado. Rode: pip install SoccerNet")

    d = SoccerNetDownloader(LocalDirectory=args.dir)
    print(f"→ Labels-v2.json (público) | split={args.split}")
    d.downloadGames(files=["Labels-v2.json"], split=[args.split])

    print(f"OK: labels em {args.dir}")
    print("\nTranscrições Echoes (texto, sem áudio) — clone o repo uma vez:")
    print("  git clone --depth 1 https://github.com/SoccerNet/sn-echoes data/sn-echoes")
    print("\nDepois rode o Exp. 2b (detecção léxica, sem SER):")
    print("  python -m src.soccernet_echoes \\")
    print("      --echoes-root data/sn-echoes/Dataset/whisper_v3 \\")
    print(f"      --labels-root {args.dir} --half 1 --out out/e2b/")
    print("\nNUNCA baixe vídeo/áudio para o 2b — não há ASR a rodar sobre o SoccerNet.")


if __name__ == "__main__":
    main()
