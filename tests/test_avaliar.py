import math

import numpy as np

from src.avaliar import captura_por_decil, metricas, psi

rng = np.random.default_rng(0)


def test_metricas_score_perfeito():
    y = np.array([0] * 90 + [1] * 10)
    m = metricas(y, y.astype(float))
    assert m["AUC"] == 1 and m["KS"] == 1 and m["Gini"] == 1


def test_metricas_sem_positivos():
    m = metricas(np.zeros(50, dtype=int), rng.random(50))
    assert set(m) == {"AUC", "Gini", "KS", "PR-AUC", "captura_top20"}
    assert all(math.isnan(v) for v in m.values())


def test_captura_monotona():
    y = (rng.random(1000) < 0.05).astype(int)
    p = rng.random(1000) + y
    c = captura_por_decil(y, p, valor=y * 100.0)
    assert list(c["decil"]) == list(range(1, 11))
    assert c["pct_pagadores_acum"].is_monotonic_increasing
    assert c["pct_pagadores_acum"].iloc[-1] == 1.0
    assert c["pct_valor_acum"].iloc[-1] == 1.0


def test_psi_identico_zero():
    x = rng.normal(0, 1, 5000)
    assert psi(x, x) < 0.01


def test_psi_deslocado_alto():
    assert psi(rng.normal(0, 1, 5000), rng.normal(1, 1, 5000)) > 0.25
