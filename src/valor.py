"""Do score ao R$: ticket típico por fase, valor esperado e concentração da recuperação.

valor_esperado = chance calibrada × ticket, onde ticket = valor da dívida × a razão típica (mediana) entre o
que os pagadores pagaram em 30 dias e o valor da dívida, na mesma fase de atraso, na janela de treino.
A faixa (Diamante..Bronze) continua sendo cortada pela chance; a FILA de trabalho usa o valor esperado.
"""
import numpy as np
import pandas as pd

from src.faixas import faixa_atraso

GLOBAL = "__GLOBAL__"


def ticket_por_fase(treino: pd.DataFrame) -> dict[str, float]:
    pag = treino[treino["pagou_30d"] == 1]
    razao = pag["valor_pago_30d"] / pag["valor_divida"]
    out = razao.groupby(faixa_atraso(pag["dias_atraso"])).median().to_dict()
    out[GLOBAL] = float(razao.median())
    return {str(k): float(v) for k, v in out.items()}


def aplicar_ticket(df: pd.DataFrame, razao: dict) -> pd.Series:
    r = faixa_atraso(df["dias_atraso"]).map(razao).fillna(razao[GLOBAL])
    return df["valor_divida"] * r.to_numpy()


def _curva(ordem_por: np.ndarray, acumular: np.ndarray, n_pontos: int) -> list[list[float]]:
    ordem = np.argsort(-np.asarray(ordem_por, float), kind="stable")
    acum = np.concatenate([[0.0], np.cumsum(np.asarray(acumular, float)[ordem])])
    total = acum[-1]
    n = len(ordem)
    xs = np.linspace(0, 1, n_pontos)
    return [[round(float(x), 6), round(float(acum[int(np.ceil(x * n))] / total), 6) if total > 0 else 0.0]
            for x in xs]


def curva_concentracao(valor_esperado, n_pontos: int = 60) -> list[list[float]]:
    """[% do estoque, % da recuperação esperada], acionando do maior valor esperado para o menor."""
    return _curva(valor_esperado, valor_esperado, n_pontos)


def curva_realizada(valor_esperado, valor_pago, n_pontos: int = 60) -> list[list[float]]:
    """Mesma fila (valor esperado), mas acumulando o que de fato foi pago: a promessa contra a entrega."""
    return _curva(valor_esperado, valor_pago, n_pontos)


def top_concentra(valor_esperado, pct: float = 0.10) -> float:
    v = np.sort(np.asarray(valor_esperado, float))[::-1]
    k = max(1, round(pct * len(v)))
    return float(v[:k].sum() / v.sum()) if v.sum() > 0 else 0.0


def comparar_filas(df: pd.DataFrame, capacidades=(0.1, 0.2, 0.3)) -> pd.DataFrame:
    """Com capacidade para X% da carteira: fila pela chance × fila pelo valor esperado."""
    linhas = []
    for cap in capacidades:
        k = round(cap * len(df))
        for nome, col in (("chance", "p_cal"), ("valor", "valor_esperado")):
            topo = df.sort_values(col, ascending=False, kind="stable").head(k)
            linhas.append({"capacidade": cap, "fila": nome,
                           "pct_pagadores": topo["pagou_30d"].sum() / df["pagou_30d"].sum(),
                           "pct_valor": topo["valor_pago_30d"].sum() / df["valor_pago_30d"].sum()})
    return pd.DataFrame(linhas)
