"""Métricas que o negócio de cobrança usa para julgar um score, e PSI para monitorar estabilidade."""
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import average_precision_score, roc_auc_score

CHAVES = ["AUC", "Gini", "KS", "PR-AUC", "captura_top20"]


def metricas(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """AUC, Gini, KS, PR-AUC e % dos pagadores capturados nos 20% de maior score."""
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    if y.min(initial=1) == y.max(initial=0):   # sem as duas classes (ex.: safra pequena sem pagador)
        return dict.fromkeys(CHAVES, float("nan"))
    auc = roc_auc_score(y, p)
    top20 = pd.Series(p).rank(method="first", ascending=False).to_numpy() <= 0.2 * len(p)
    return {"AUC": auc, "Gini": 2 * auc - 1,
            "KS": ks_2samp(p[y == 1], p[y == 0]).statistic,
            "PR-AUC": average_precision_score(y, p),
            "captura_top20": y[top20].sum() / y.sum()}


def captura_por_decil(y, p, valor=None) -> pd.DataFrame:
    """Decil 1 = 10% de maior score. Mostra quanto dos pagadores (e do valor) se acumula até cada decil."""
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p, dtype=float)})
    if valor is not None:
        df["valor"] = np.asarray(valor, dtype=float)
    df["decil"] = pd.qcut(df["p"].rank(method="first", ascending=False), 10, labels=range(1, 11)).astype(int)
    g = df.groupby("decil").sum(numeric_only=True)
    out = pd.DataFrame({"decil": g.index,
                        "pct_carteira": np.arange(1, 11) / 10,
                        "pct_pagadores_acum": g["y"].cumsum() / g["y"].sum()})
    if valor is not None:
        out["pct_valor_acum"] = g["valor"].cumsum() / g["valor"].sum()
    return out.reset_index(drop=True)


def psi(referencia: np.ndarray, atual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index. Regra usual: < 0,10 estável; 0,10–0,25 atenção; > 0,25 mudou."""
    ref = pd.Series(referencia, dtype=float).dropna().to_numpy()
    atu = pd.Series(atual, dtype=float).dropna().to_numpy()
    cortes = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    cortes[0], cortes[-1] = -np.inf, np.inf
    r = np.maximum(np.histogram(ref, cortes)[0] / len(ref), 1e-4)
    a = np.maximum(np.histogram(atu, cortes)[0] / len(atu), 1e-4)
    return float(np.sum((a - r) * np.log(a / r)))
