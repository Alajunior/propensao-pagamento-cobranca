"""Escada de candidatos com gate: vence o MAIS SIMPLES que passa, não o de maior AUC.

Degraus: régua (sem modelo) → regressão logística → LightGBM. A régua é o primeiro campeão; cada degrau
desafia o campeão da vez e só o substitui se passar nos 4 critérios do gate:
    1. captura_top10_ic: % do valor pago nos 10% de maior score (por safra) maior que a do campeão,
       com o limite inferior do IC bootstrap 95% da diferença acima de zero (reamostragem por cliente);
    2. auc_por_fase: nenhuma fase com AUC pior que a do campeão por mais de 0,01;
    3. sem_inversao: nenhuma fase com as faixas do desafiante fora de ordem (Diamante ≥ Ouro ≥ Prata ≥ Bronze);
    4. acima_da_regua: captura no top 10% acima da régua.
Avaliação nas safras de teste empilhadas, cada modelo retreinado todo mês com as 3 safras anteriores fechadas.
"""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

from src.backtest import X, safras_de_treino
from src.base_analitica import ALVO, montar_base, safras
from src.faixas import atribuir_faixas, faixa_atraso
from src.modelos import criar_modelos
from src.qualidade import ORDEM, classificar_ordenacao

NOMES = {"regua": "Régua", "logistica": "Logística", "lightgbm": "LightGBM"}
TOL_AUC = 0.01


def _pp(x: float) -> str:
    return f"{x * 100:+.1f}".replace(".", ",")


def score_regua(base: pd.DataFrame) -> np.ndarray:
    """Sem modelo: quem teve contato com a pessoa certa nos últimos 30 dias primeiro; depois, o menor atraso."""
    return (1000 * (base["qtd_cpc_30d"] > 0) - base["dias_atraso"]).to_numpy(dtype=float)


def _top(score, safra, pct: float) -> np.ndarray:
    """True nos `pct` de maior score de cada safra."""
    s = pd.Series(np.asarray(score, float))
    g = pd.Series(np.asarray(safra))
    pos = s.groupby(g).rank(method="first", ascending=False)
    return (pos <= np.maximum(1, (pct * g.map(g.value_counts())).round())).to_numpy()


def captura_valor_top(valor_pago, score, safra, pct: float = 0.10) -> float:
    v = np.asarray(valor_pago, float)
    return float(v[_top(score, safra, pct)].sum() / v.sum())


def ic_bootstrap_diff(df: pd.DataFrame, col_a: str, col_b: str, n: int = 1000, seed: int = 42,
                      pct: float = 0.10) -> tuple[float, float, float]:
    """Diferença de captura (b − a) e o IC 95% reamostrando CLIENTES (contratos do mesmo cliente andam juntos).

    shortcut: o corte do top 10% é fixado na amostra original (não recalculado a cada reamostragem);
    suficiente para medir a incerteza da diferença, recalcular se a decisão ficar na fronteira.
    """
    v = df["valor_pago_30d"].to_numpy(float)
    cli = pd.factorize(df["id_cliente"])[0]
    k = cli.max() + 1
    V = np.bincount(cli, v, k)
    A = np.bincount(cli, v * _top(df[col_a], df["safra"], pct), k)
    B = np.bincount(cli, v * _top(df[col_b], df["safra"], pct), k)
    rng = np.random.default_rng(seed)
    difs = np.empty(n)
    for i in range(n):
        w = np.bincount(rng.integers(0, k, k), minlength=k)
        difs[i] = (w @ B - w @ A) / (w @ V)
    lo, hi = np.percentile(difs, [2.5, 97.5])
    return float((B.sum() - A.sum()) / V.sum()), float(lo), float(hi)


def _auc_fase(df, col) -> dict:
    out = {}
    for f, g in df.groupby("fase"):
        y = g[ALVO]
        out[f] = roc_auc_score(y, g[col]) if 0 < y.sum() < len(y) else None
    return out


def _ordenacao_fases(df, col) -> dict:
    faixas = pd.concat(atribuir_faixas(g.rename(columns={col: "_s"}), "_s") for _, g in df.groupby("safra"))
    d = df.assign(_fx=faixas.reindex(df.index))
    out = {}
    for f, g in d.groupby("fase"):
        t = g.groupby("_fx")[ALVO].mean()
        out[f] = classificar_ordenacao([float(t[x]) if x in t.index else None for x in ORDEM])[0]
    return out


def gate(df: pd.DataFrame, campeao: str, desafiante: str) -> dict:
    dif, lo, hi = ic_bootstrap_diff(df, campeao, desafiante)
    auc_c, auc_d = _auc_fase(df, campeao), _auc_fase(df, desafiante)
    piores = [f for f in auc_d if auc_d[f] is not None and auc_c.get(f) is not None and auc_d[f] < auc_c[f] - TOL_AUC]
    ordem = _ordenacao_fases(df, desafiante)
    invertidas = [f for f, o in ordem.items() if o == "INVERTIDA"]
    cap = {c: captura_valor_top(df["valor_pago_30d"], df[c], df["safra"]) for c in (desafiante, "regua")}
    crit = [
        {"nome": "captura_top10_ic", "ok": bool(lo > 0),
         "detalhe": f"diferença {_pp(dif)} pp (IC 95% {_pp(lo)} a {_pp(hi)})"},
        {"nome": "auc_por_fase", "ok": not piores,
         "detalhe": "nenhuma fase pior" if not piores else "pior em " + ", ".join(piores)},
        {"nome": "sem_inversao", "ok": not invertidas,
         "detalhe": "faixas em ordem em todas as fases" if not invertidas else "invertida em " + ", ".join(invertidas)},
        {"nome": "acima_da_regua", "ok": bool(cap[desafiante] > cap["regua"]) or desafiante == "regua",
         "detalhe": f"{cap[desafiante] * 100:.1f}% contra {cap['regua'] * 100:.1f}% da régua".replace(".", ",")},
    ]
    return {"campeao": campeao, "desafiante": desafiante, "criterios": crit,
            "passou": all(c["ok"] for c in crit), "diferenca": dif, "ic": [lo, hi]}


def pilha_oot(tabelas: dict, inicio="2025-12-31", fim="2026-10-31") -> pd.DataFrame:
    todas = safras()
    bases = {d: montar_base(d, tabelas) for d in todas}
    modelos = {k: v for k, v in criar_modelos().items() if k in ("logistica", "lightgbm")}
    partes = []
    for d in [d for d in todas if pd.Timestamp(inicio) <= d <= pd.Timestamp(fim)]:
        treino = pd.concat(bases[s] for s in safras_de_treino(d, todas, 3))
        t = bases[d]
        p = t[["id_cliente", "dias_atraso", ALVO, "valor_pago_30d"]].assign(safra=d, fase=faixa_atraso(t["dias_atraso"]),
                                                                              regua=score_regua(t))
        for nome, m in modelos.items():
            p[nome] = clone(m).fit(treino[X], treino[ALVO]).predict_proba(t[X])[:, 1]
        partes.append(p)
    return pd.concat(partes, ignore_index=True)


def avaliar_escada(tabelas: dict, inicio="2025-12-31", fim="2026-10-31", df: pd.DataFrame | None = None) -> dict:
    df = pilha_oot(tabelas, inicio, fim) if df is None else df
    candidatos = [{"nome": c, "rotulo": NOMES[c],
                   "captura_top10": captura_valor_top(df["valor_pago_30d"], df[c], df["safra"]),
                   "auc": float(roc_auc_score(df[ALVO], df[c]))} for c in ("regua", "logistica", "lightgbm")]
    campeao, etapas = "regua", []
    for desafiante in ("logistica", "lightgbm"):
        g = gate(df, campeao, desafiante)
        etapas.append(g)
        if g["passou"]:
            campeao = desafiante
    ult = etapas[-1]
    lo, hi = (_pp(x) for x in ult["ic"])
    if ult["passou"]:
        frase = (f"{NOMES[ult['desafiante']]} passou no gate (IC {lo} a {hi} pp): é o novo campeão e entra em "
                 f"modo sombra antes de substituir o modelo em produção.")
    else:
        frase = (f"{NOMES[ult['desafiante']]} sem ganho claro sobre a {NOMES[ult['campeao']]} "
                 f"(IC {lo} a {hi} pp): mantém a {NOMES[campeao]}.")
    return {"candidatos": candidatos, "etapas": etapas, "vencedor": campeao, "frase": frase,
            "safras": [f"{d:%Y-%m}" for d in sorted(df["safra"].unique())]}
