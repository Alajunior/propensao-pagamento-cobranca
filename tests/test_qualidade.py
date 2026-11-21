import numpy as np
import pandas as pd

from src.qualidade import (ORDEM, backtest_producao, classificar_ordenacao, estabilidade_semanal, inversoes,
                           lift_por_fase)

TAXAS = {"Diamante": .40, "Ouro": .20, "Prata": .10, "Bronze": .05}


def _esc_sintetico(pagadores: int, inverter: str | None = None, seed: int = 0) -> pd.DataFrame:
    """2 fases × 4 faixas × 15 dias úteis × 100 contratos/dia por faixa. `pagadores` = 0 zera o alvo."""
    rng = np.random.default_rng(seed)
    linhas = []
    for dia in pd.bdate_range("2026-03-02", periods=15):
        for fase in ("0001-0360", "0361-0720"):
            for i, fx in enumerate(ORDEM):
                taxa = TAXAS[fx]
                if fase == inverter and fx == "Ouro":
                    taxa = .60
                y = (rng.random(100) < taxa).astype(float) if pagadores else np.zeros(100)
                linhas.append(pd.DataFrame({"dia": dia, "fase": fase, "faixa": fx, "score": 1 - i * 0.2 - rng.random(100) * 0.1,
                                            "maduro": True, "pagou_30d": y}))
    return pd.concat(linhas, ignore_index=True)


def test_classificar_ordenacao():
    assert classificar_ordenacao([.4, .2, .1, .05])[0] == "perfeita"
    assert classificar_ordenacao([.11, .10, .10, .09])[0] == "fraca"
    assert classificar_ordenacao([.2, .3, .1, .05])[0] == "INVERTIDA"
    assert classificar_ordenacao([.3, None, None, None]) == ("sem dados", None)


def test_sem_pagador_e_sem_dados():
    assert classificar_ordenacao([0, 0, 0, 0]) == ("sem dados", None)
    linha = [r for r in backtest_producao(_esc_sintetico(pagadores=0)) if r["fase"] == "__TODAS__"][0]
    assert linha["ordenacao"] == "sem dados" and linha["motivo"] and linha["auc"] is None


def test_dia_nao_maduro_fica_fora():
    esc = _esc_sintetico(pagadores=50)
    esc.loc[esc.dia == esc.dia.max(), ["maduro", "pagou_30d"]] = [False, np.nan]
    total = [r for r in backtest_producao(esc) if r["fase"] == "__TODAS__"][0]
    assert total["n"] == esc.maduro.sum()
    assert total["ordenacao"] == "perfeita" and 0.5 < total["auc"] <= 1


def test_inversao_detectada():
    esc = _esc_sintetico(pagadores=50, inverter="0361-0720")
    inv = inversoes(esc)
    assert (inv.fase == "0361-0720").any() and not (inv.fase == "0001-0360").any()
    assert all(len(t) == 4 for t in inv.taxas)
    fases = {r["fase"]: r["ordenacao"] for r in backtest_producao(esc)}
    assert fases["0361-0720"] == "INVERTIDA" and fases["0001-0360"] == "perfeita"


def test_estabilidade_e_lift():
    esc = _esc_sintetico(pagadores=50)
    est = estabilidade_semanal(esc)
    assert set(est.columns) >= {"fase", "faixa", "media", "desvio", "melhor", "pior", "semanas"}
    assert (est.semanas == 3).all() and (est.melhor >= est.pior).all()
    lf = lift_por_fase(esc).set_index(["fase", "faixa"])["lift"]
    assert lf[("0001-0360", "Diamante")] > 1 > lf[("0001-0360", "Bronze")]
