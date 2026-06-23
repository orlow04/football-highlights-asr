"""E1 — WER/CER e recall do léxico de domínio: ASR base vs fine-tune (§9.2).

Para cada modelo (base e fine-tune PT-BR), transcreve o áudio, recorta as
palavras dentro das janelas anotadas em gt_transcricao.json e compara contra o
texto de referência da MESMA janela (alinhamento temporal — sem isso o WER não
faz sentido). Reporta WER, CER (pós-normalização: minúsculas, sem acento/
pontuação) e o RECALL DO LÉXICO de domínio, que é a métrica que revela o gap de
jargão esportivo entre os dois modelos.

Saída: out/e1_results.{csv,json}, ref.txt e hyp_<modelo>.txt (sempre) + um run
por modelo no wandb (opcional).

Uso:
    python -m src.e1_asr --audio data/pt/jogo_corte.wav \
        --gt data/pt/gt_transcricao.json --out out/e1/
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import asr as asr_mod  # noqa: E402
import wandb_log  # noqa: E402
from common import load_config, normaliza  # noqa: E402
from evaluate import metricas_asr, wer_lexico  # noqa: E402


def _hyp_por_janela(word_ts, janelas) -> str:
    """Concatena as palavras do ASR que caem em cada janela [t_ini, t_fim)."""
    partes = []
    for jan in janelas:
        ini, fim = jan["t_ini"], jan["t_fim"]
        ws = [w["word"] for w in word_ts
              if w.get("start") is not None and ini <= w["start"] < fim]
        partes.append(" ".join(ws))
    return " ".join(partes)


def _libera_modelo(model) -> None:
    try:
        import torch
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        del model
        gc.collect()


def run(audio, gt_path, out_dir, config=None):
    cfg = load_config(config)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    gt = json.loads(Path(gt_path).read_text(encoding="utf-8"))
    janelas = gt["janelas"]
    ref = " ".join(j["texto"].strip() for j in janelas)
    (out / "ref.txt").write_text(ref, encoding="utf-8")

    exp = cfg.get("experiments", {})
    modelos = exp.get("e1", {}).get("models")
    if not modelos:  # fallback se o config não trouxer a lista
        modelos = [{"name": "fine-tune",
                    "model_name": cfg["asr"]["model_name"],
                    "model_file": cfg["asr"].get("model_file")}]
    project = exp.get("wandb_project", "football-highlights")

    resultados = []
    for m in modelos:
        nome = m["name"]
        print(f"\n=== Modelo: {nome} ({m['model_name']}) ===")
        model = asr_mod.carregar_modelo(
            m["model_name"], m.get("model_file"),
            local_attn=cfg["asr"].get("local_attn", False),
            att_context_size=cfg["asr"].get("att_context_size", (128, 128)))
        trans = asr_mod.transcrever(model, audio)
        _libera_modelo(model)

        hyp = _hyp_por_janela(trans["word_ts"], janelas)
        (out / f"hyp_{nome}.txt").write_text(hyp, encoding="utf-8")

        asr_m = metricas_asr(normaliza(ref), normaliza(hyp))  # WER/CER normalizados
        lex = wer_lexico(ref, hyp, cfg)                        # recall do léxico
        reg = {"modelo": nome, "model_name": m["model_name"],
               **{k: round(v, 4) if isinstance(v, float) else v
                  for k, v in asr_m.items()},
               **lex}
        resultados.append(reg)
        print(f"  WER={reg['wer']} CER={reg['cer']} "
              f"léxico_recall={reg.get('lexico_recall')} "
              f"({reg.get('lexico_acertos')}/{reg.get('lexico_total')})")

        w = wandb_log.init(project, name=f"e1_{nome}", group="e1",
                           job_type="e1",
                           config={"modelo": nome, "model_name": m["model_name"]})
        wandb_log.summary(w, {k: v for k, v in reg.items()
                              if k not in ("modelo", "model_name")})
        wandb_log.finish(w)

    (out / "e1_results.json").write_text(
        json.dumps(resultados, indent=2, ensure_ascii=False), encoding="utf-8")
    with open(out / "e1_results.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(resultados[0].keys()))
        wr.writeheader()
        wr.writerows(resultados)
    print(f"\nOK: {out/'e1_results.csv'}  ({len(resultados)} modelos)")
    return resultados


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--gt", required=True, help="gt_transcricao.json (janelas)")
    ap.add_argument("--out", default="out/e1")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    run(args.audio, args.gt, args.out, config=args.config)


if __name__ == "__main__":
    main()
