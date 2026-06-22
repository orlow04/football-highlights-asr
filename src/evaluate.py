"""Etapa 5 — métricas: WER/CER (ASR) e P/R/F1 com tolerância (detecção) (§8).

Uso (detecção contra GT de eventos):
    python -m src.evaluate --pred out/highlights.json --events data/pt/eventos.json

Uso (ASR):
    python -m src.evaluate --ref ref.txt --hyp hyp.txt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_config, normaliza  # noqa: E402


# ---------- ASR ----------
def metricas_asr(referencia: str, hipotese: str) -> dict:
    from jiwer import cer, wer

    return {"wer": wer(referencia, hipotese), "cer": cer(referencia, hipotese)}


def wer_lexico(referencia: str, hipotese: str, cfg) -> dict:
    """Recall das palavras-chave de domínio: fração das ocorrências do léxico
    na referência que também aparecem na hipótese (revela o gap de domínio)."""
    lexico = [normaliza(k) for k in cfg["lexicon"]]
    ref = normaliza(referencia)
    hyp = normaliza(hipotese)
    total = acertos = 0
    for chave in lexico:
        c_ref = ref.count(chave)
        if c_ref == 0:
            continue
        total += c_ref
        acertos += min(c_ref, hyp.count(chave))
    rec = acertos / total if total else None
    return {"lexico_recall": rec, "lexico_total": total, "lexico_acertos": acertos}


# ---------- Detecção ----------
def avalia_deteccao(picos_s, eventos_s, tol: float = 10.0) -> dict:
    eventos = list(eventos_s)
    tp = 0
    for p in picos_s:
        for e in list(eventos):
            if abs(p - e) <= tol:
                tp += 1
                eventos.remove(e)
                break
    fp = len(picos_s) - tp
    fn = len(eventos)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return dict(precision=prec, recall=rec, f1=f1, tp=tp, fp=fp, fn=fn, tol=tol)


def curva_f1(picos_s, eventos_s, tolerancias) -> list[dict]:
    return [avalia_deteccao(picos_s, eventos_s, tol=t) for t in tolerancias]


def _load_times(path: str):
    """Aceita JSON: lista de números, ou {peaks_s:[...]}, ou [{t:..}/{time:..}]."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("peaks_s") or data.get("events_s") or data.get("events")
    out = []
    for x in data:
        out.append(float(x) if not isinstance(x, dict)
                   else float(x.get("t", x.get("time", x.get("start")))))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", help="JSON de highlights (peaks_s)")
    ap.add_argument("--events", help="JSON de eventos GT")
    ap.add_argument("--ref", help="arquivo .txt de referência (ASR)")
    ap.add_argument("--hyp", help="arquivo .txt de hipótese (ASR)")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    res: dict = {}

    if args.ref and args.hyp:
        ref = Path(args.ref).read_text(encoding="utf-8")
        hyp = Path(args.hyp).read_text(encoding="utf-8")
        res["asr"] = {**metricas_asr(ref, hyp), **wer_lexico(ref, hyp, cfg)}

    if args.pred and args.events:
        picos = _load_times(args.pred)
        eventos = _load_times(args.events)
        res["deteccao"] = curva_f1(picos, eventos, cfg["evaluate"]["tolerances_s"])

    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
