"""Backtest de PRODUÇÃO: a faixa que o modelo publicou em cada dia contra o pagamento que de fato veio.

Não é métrica de treino. Só entram os dias maduros (D+30 <= hoje), porque antes disso o alvo não existe.
A ordenação resume se as faixas se comportam na ordem esperada (Diamante paga mais que Bronze):
    perfeita  = monotônica com lift relevante;
    fraca     = na ordem, mas com lift baixo;
    INVERTIDA = alguma faixa inferior supera uma superior: pior que não ranquear, porque dá falsa confiança;
    sem dados = recorte sem evidência (sem esta guarda, zero pagamentos passaria como "perfeita").
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

LIFT_MIN_RELEVANTE = 1.10
ORDEM = ["Diamante", "Ouro", "Prata", "Bronze"]
TODAS = "__TODAS__"


def classificar_ordenacao(taxas: list) -> tuple[str, float | None]:
    validas = [t for t in taxas if t is not None and not pd.isna(t)]
    if len(validas) < 2:
        return "sem dados", None
    base = float(np.mean(validas))
    if base <= 0:
        return "sem dados", None
    lift = validas[0] / base
    if any(validas[i] < validas[i + 1] for i in range(len(validas) - 1)):
        return "INVERTIDA", lift
    return ("fraca" if lift < LIFT_MIN_RELEVANTE else "perfeita"), lift


def _r(x, n=6):
    return None if x is None or pd.isna(x) else round(float(x), n)


def _taxas(sub: pd.DataFrame) -> list:
    t = sub.groupby("faixa")["pagou_30d"].mean()
    return [float(t[f]) if f in t.index else None for f in ORDEM]


def _recortes(m: pd.DataFrame):
    for fase in sorted(m["fase"].unique()):
        yield fase, m[m["fase"] == fase]
    yield TODAS, m


def backtest_producao(esc: pd.DataFrame) -> list[dict]:
    m = esc[esc["maduro"]]
    out = []
    for fase, sub in _recortes(m):
        y = sub["pagou_30d"].to_numpy()
        taxas = _taxas(sub)
        ordenacao, lift = classificar_ordenacao(taxas)
        auc = roc_auc_score(y, sub["score"]) if 0 < y.sum() < len(y) else None
        out.append({"fase": fase, "n": int(len(sub)), "taxa_geral": _r(y.mean()) if len(y) else None,
                    "auc": _r(auc, 4), "taxa_diamante": _r(taxas[0]), "taxa_bronze": _r(taxas[-1]),
                    "lift": _r(lift, 3), "ordenacao": ordenacao,
                    "motivo": None if y.sum() else "nenhum pagamento na janela"})
    return out


def _semana(dia: pd.Series) -> pd.Series:
    return dia - pd.to_timedelta(dia.dt.weekday, unit="D")   # segunda-feira da semana


def estabilidade_semanal(esc: pd.DataFrame) -> pd.DataFrame:
    m = esc[esc["maduro"]].assign(semana=lambda d: _semana(d["dia"]))
    partes = []
    for fase, sub in _recortes(m):
        t = sub.groupby(["faixa", "semana"])["pagou_30d"].mean().groupby("faixa")
        g = t.agg(media="mean", desvio="std", melhor="max", pior="min", semanas="count").reset_index()
        partes.append(g.assign(fase=fase))
    out = pd.concat(partes, ignore_index=True)
    out["faixa"] = pd.Categorical(out["faixa"], ORDEM, ordered=True)
    return out.sort_values(["fase", "faixa"])[["fase", "faixa", "media", "desvio", "melhor", "pior", "semanas"]] \
        .astype({"faixa": str}).reset_index(drop=True)


def inversoes(esc: pd.DataFrame) -> pd.DataFrame:
    m = esc[esc["maduro"]].assign(semana=lambda d: _semana(d["dia"]))
    linhas = []
    for (semana, fase), sub in m.groupby(["semana", "fase"]):
        taxas = _taxas(sub)
        ordenacao, _ = classificar_ordenacao(taxas)
        if ordenacao == "INVERTIDA":
            linhas.append({"semana": semana, "fase": fase, "ordenacao": ordenacao, "taxas": [_r(t) for t in taxas]})
    return pd.DataFrame(linhas, columns=["semana", "fase", "ordenacao", "taxas"])


def lift_por_fase(esc: pd.DataFrame) -> pd.DataFrame:
    m = esc[esc["maduro"]]
    taxa_fase = m.groupby("fase")["pagou_30d"].mean()
    t = m.groupby(["fase", "faixa"])["pagou_30d"].mean().reset_index(name="taxa")
    t["lift"] = t["taxa"] / t["fase"].map(taxa_fase)
    t["faixa"] = pd.Categorical(t["faixa"], ORDEM, ordered=True)
    return t.sort_values(["fase", "faixa"]).astype({"faixa": str})[["fase", "faixa", "lift"]].reset_index(drop=True)
