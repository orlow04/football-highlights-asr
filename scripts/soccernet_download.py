"""Baixa do SoccerNet o necessário para o fallback EN (estratégia de dados, §3).

  - Labels-v2.json (anotações de evento): PÚBLICO, sem senha.
  - 1_224p.mkv / 2_224p.mkv (vídeo LQ, já com áudio): requer SENHA NDA.

A senha NDA sai preenchendo o formulário acadêmico do SoccerNet (instantâneo).
Sem ela, baixe só os labels e use o áudio do SoccerNet-Echoes, ou rode em outro
jogo cujo vídeo você já tenha.

Instale o cliente à parte do pin do torch:  pip install SoccerNet

Uso:
    python scripts/soccernet_download.py --dir data/soccernet --split test
    python scripts/soccernet_download.py --dir data/soccernet --split test \
        --video --password SUA_SENHA_NDA
"""
from __future__ import annotations

import argparse
import sys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/soccernet", help="diretório local")
    ap.add_argument("--split", default="test",
                    choices=["train", "valid", "test", "challenge"])
    ap.add_argument("--video", action="store_true",
                    help="baixar também o vídeo LQ (requer --password)")
    ap.add_argument("--password", default=None, help="senha NDA do SoccerNet")
    args = ap.parse_args()

    try:
        from SoccerNet.Downloader import SoccerNetDownloader
    except ImportError:
        sys.exit("Cliente não instalado. Rode: pip install SoccerNet")

    d = SoccerNetDownloader(LocalDirectory=args.dir)

    print(f"→ Labels-v2.json (público) | split={args.split}")
    d.downloadGames(files=["Labels-v2.json"], split=[args.split])

    if args.video:
        if not args.password:
            sys.exit("--video requer --password (NDA).")
        d.password = args.password
        print(f"→ 1_224p.mkv / 2_224p.mkv (NDA) | split={args.split}")
        d.downloadGames(files=["1_224p.mkv", "2_224p.mkv"], split=[args.split])

    print(f"OK: dados em {args.dir}")
    print("Próximo: extrair o áudio de uma metade e converter os labels —")
    print("  ffmpeg -i <jogo>/1_224p.mkv -vn -ac 1 -ar 16000 -sample_fmt s16 "
          "data/soccernet/jogo.wav")
    print("  python scripts/soccernet_convert.py --labels <jogo>/Labels-v2.json "
          "--half 1 --out data/soccernet/eventos.json")


if __name__ == "__main__":
    main()
