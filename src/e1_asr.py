"""E1 — WER/CER e recall do léxico de domínio: ASR base vs fine-tune (§9.2).

Para cada modelo (base e fine-tune PT-BR), transcreve o áudio, recorta as
palavras dentro das janelas anotadas em gt_transcricao.json e compara contra o
texto de referência da MESMA janela (alinhamento temporal — sem isso o WER não
faz sentido). Reporta WER, CER e o RECALL DO LÉXICO de domínio.

WER/CER são pós-normalização: minúsculas, sem acento/pontuação E com números
unificados (extenso↔dígito), senão "seis"≠"6" infla o erro de ambos os modelos.

MÚLTIPLOS VÍDEOS: passe vários --audio/--gt (pareados). Reporta-se uma linha por
vídeo + uma linha AGREGADA (WER sobre o texto concatenado de todos — não média
de WERs), que é o número com poder estatístico.

Saída: out/e1_results.{csv,json}, ref.txt e hyp_<modelo>_<video>.txt + wandb.

Uso:
    python -m src.e1_asr --audio data/pt/jogo_corte.wav data/pt/jogo_corte-msn.wav \
        --gt data/pt/gt_transcricao.json data/pt/gt_transcricao-msn.json --out out/e1/
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
from common import converte_numeros, load_config, normaliza  # noqa: E402
from evaluate import metricas_asr, wer_lexico  # noqa: E402


def _norm_wer(t: str) -> str:
    """Normalização para WER: extenso↔dígito unificados + lower/sem acento/pont."""
    return normaliza(converte_numeros(t))


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


def _metricas(ref, hyp, cfg) -> dict:
    am = metricas_asr(_norm_wer(ref), _norm_wer(hyp))
    return {**{k: round(v, 4) for k, v in am.items()},
            **wer_lexico(ref, hyp, cfg)}


def run(audios, gts, out_dir, config=None):
    cfg = load_config(config)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # carrega janelas + ref por vídeo
    videos = []
    for audio, gtp in zip(audios, gts):
        gt = json.loads(Path(gtp).read_text(encoding="utf-8"))
        videos.append({"audio": audio, "stem": Path(audio).stem,
                       "janelas": gt["janelas"],
                       "ref": " ".join(j["texto"].strip() for j in gt["janelas"])})
    (out / "ref.txt").write_text("\n\n".join(v["ref"] for v in videos),
                                 encoding="utf-8")

    exp = cfg.get("experiments", {})
    modelos = exp.get("e1", {}).get("models") or [
        {"name": "fine-tune", "model_name": cfg["asr"]["model_name"],
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

        refs, hyps = [], []
        for v in videos:
            trans = asr_mod.transcrever(model, v["audio"])
            hyp = _hyp_por_janela(trans["word_ts"], v["janelas"])
            (out / f"hyp_{nome}_{v['stem']}.txt").write_text(hyp, encoding="utf-8")
            refs.append(v["ref"])
            hyps.append(hyp)
            reg = {"modelo": nome, "model_name": m["model_name"],
                   "escopo": v["stem"], **_metricas(v["ref"], hyp, cfg)}
            resultados.append(reg)
            print(f"  [{v['stem']}] WER={reg['wer']} CER={reg['cer']} "
                  f"léxico={reg.get('lexico_acertos')}/{reg.get('lexico_total')}")
        _libera_modelo(model)

        # agregado: concatena ref/hyp de todos os vídeos (WER ponderado por palavras)
        REF, HYP = " ".join(refs), " ".join(hyps)
        agg = {"modelo": nome, "model_name": m["model_name"],
               "escopo": "AGREGADO", **_metricas(REF, HYP, cfg)}
        resultados.append(agg)
        print(f"  [AGREGADO] WER={agg['wer']} CER={agg['cer']} "
              f"léxico_recall={agg.get('lexico_recall')} "
              f"({agg.get('lexico_acertos')}/{agg.get('lexico_total')})")

        w = wandb_log.init(project, name=f"e1_{nome}", group="e1", job_type="e1",
                           config={"modelo": nome, "model_name": m["model_name"],
                                   "n_videos": len(videos)})
        wandb_log.summary(w, {k: v for k, v in agg.items()
                              if k not in ("modelo", "model_name", "escopo")})
        wandb_log.finish(w)

    (out / "e1_results.json").write_text(
        json.dumps(resultados, indent=2, ensure_ascii=False), encoding="utf-8")
    with open(out / "e1_results.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(resultados[0].keys()))
        wr.writeheader()
        wr.writerows(resultados)
    print(f"\nOK: {out/'e1_results.csv'}  ({len(videos)} vídeo(s), "
          f"{len(modelos)} modelo(s))")
    return resultados


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", nargs="+", required=True)
    ap.add_argument("--gt", nargs="+", required=True,
                    help="1+ gt_transcricao.json (pareados com --audio)")
    ap.add_argument("--out", default="out/e1")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    if len(args.audio) != len(args.gt):
        ap.error("--audio e --gt precisam ter o mesmo número de itens.")
    run(args.audio, args.gt, args.out, config=args.config)


if __name__ == "__main__":
    main()
