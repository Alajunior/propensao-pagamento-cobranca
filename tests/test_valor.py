import numpy as np
import pandas as pd
import pytest

from src.valor import (aplicar_ticket, comparar_filas, curva_concentracao, curva_realizada, ticket_por_fase,
                       top_concentra)


def test_ticket_mediana_por_fase():
    df = pd.DataFrame({"dias_atraso": [10, 20, 30, 500], "valor_divida": [100, 200, 100, 1000],
                       "valor_pago_30d": [50, 40, 0, 300], "pagou_30d": [1, 1, 0, 1]})
    r = ticket_por_fase(df)
    assert r["0001-0360"] == pytest.approx(0.35) and r["0361-0720"] == pytest.approx(0.30)
    assert "__GLOBAL__" in r
    t = aplicar_ticket(pd.DataFrame({"dias_atraso": [5, 2000], "valor_divida": [1000, 1000]}), r)
    assert t.iloc[0] == pytest.approx(350) and t.iloc[1] == pytest.approx(1000 * r["__GLOBAL__"])


def test_curva_concentracao_monotonica():
    c = curva_concentracao(np.array([10., 5, 1, 1, 1, 0]))
    assert c[0] == [0.0, 0.0] and c[-1] == [1.0, 1.0]
    assert all(b[1] >= a[1] for a, b in zip(c, c[1:]))
    assert top_concentra(np.array([90.] + [1.] * 9), 0.10) == pytest.approx(0.909, abs=1e-3)


def test_curva_realizada_segue_a_ordem_do_esperado():
    ve = np.arange(100, 0, -1, dtype=float)
    pago = np.zeros(100)
    pago[0] = 500.0                       # só o primeiro da fila (maior valor esperado) pagou
    c = curva_realizada(ve, pago)
    assert c[0] == [0.0, 0.0] and c[1][1] == pytest.approx(1.0) and c[-1] == [1.0, 1.0]


def test_fila_por_valor_traz_mais_reais():
    df = pd.DataFrame({"p_cal": [.9, .8, .1, .1], "valor_esperado": [9, 8, 50, 40],
                       "pagou_30d": [1, 1, 1, 0], "valor_pago_30d": [10, 10, 500, 0]})
    t = comparar_filas(df, capacidades=(0.5,)).set_index("fila")
    assert t.loc["chance", "pct_pagadores"] > t.loc["valor", "pct_pagadores"]
    assert t.loc["valor", "pct_valor"] > t.loc["chance", "pct_valor"]
