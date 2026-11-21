"""Calibração: transformar o score (que ordena) numa chance que vale como número.

A logística usa class_weight="balanced": ordena bem, mas infla a probabilidade (prevê várias vezes a taxa
real). A isotônica por fase, ajustada numa safra que o modelo não viu no treino, conserta o número sem mudar
a ordem dentro da fase. O valor esperado (chance × ticket) precisa do número certo.
"""
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

GLOBAL = "__GLOBAL__"


def _iso(score, y) -> IsotonicRegression:
    return IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(score, y)


def ajustar_calibracao(score, fase, y, min_pagadores: int = 30) -> dict[str, IsotonicRegression]:
    """Uma isotônica por fase com pelo menos `min_pagadores`; as demais usam a global."""
    score, fase, y = np.asarray(score, float), np.asarray(fase), np.asarray(y)
    cal = {GLOBAL: _iso(score, y)}
    for f in np.unique(fase):
        m = fase == f
        if y[m].sum() >= min_pagadores:
            cal[str(f)] = _iso(score[m], y[m])
    return cal


def aplicar_calibracao(cal: dict, score, fase) -> np.ndarray:
    score, fase = np.asarray(score, float), np.asarray(fase)
    p = cal[GLOBAL].predict(score)
    for f, iso in cal.items():
        if f != GLOBAL:
            m = fase == f
            if m.any():
                p[m] = iso.predict(score[m])
    return p


def brier(y, p) -> float:
    return float(np.mean((np.asarray(p, float) - np.asarray(y)) ** 2))


def previu_pagou(y, p, bins: int = 10) -> pd.DataFrame:
    """Por decil de p (1 = maior), a chance média prevista contra a taxa observada."""
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p, float)})
    df["decil"] = pd.qcut(df["p"].rank(method="first", ascending=False), bins, labels=range(1, bins + 1)).astype(int)
    t = df.groupby("decil").agg(n=("y", "size"), previsto=("p", "mean"), observado=("y", "mean"))
    return t.reset_index()


def ece(y, p, bins: int = 10) -> float:
    """Expected Calibration Error com bins de mesma frequência: média ponderada de |previsto − observado|."""
    t = previu_pagou(y, p, bins)
    return float((t["n"] * (t["previsto"] - t["observado"]).abs()).sum() / t["n"].sum())
