import numpy as np
import pandas as pd

from src.faixas import FAIXAS_ATRASO, atribuir_faixas, comparar_estrategias, faixa_atraso

rng = np.random.default_rng(0)


def _carteira(n=4000):
    df = pd.DataFrame({"dias_atraso": rng.integers(1, 1200, n), "score": rng.random(n)})
    df["pagou_30d"] = (rng.random(n) < df["score"] * 0.1).astype(int)
    df["valor_pago_30d"] = df["pagou_30d"] * rng.uniform(100, 1000, n)
    return df


def test_proporcoes_da_piramide():
    df = _carteira()
    df["faixa"] = atribuir_faixas(df)
    df["atraso"] = faixa_atraso(df["dias_atraso"])
    for _, g in df.groupby("atraso"):
        if len(g) >= 100:
            prop = g["faixa"].value_counts(normalize=True)
            for nome, alvo in {"Diamante": .10, "Ouro": .20, "Prata": .30, "Bronze": .40}.items():
                assert abs(prop[nome] - alvo) <= 0.01


def test_diamante_tem_maior_score():
    df = _carteira()
    df["faixa"] = atribuir_faixas(df)
    df["atraso"] = faixa_atraso(df["dias_atraso"])
    for _, g in df.groupby("atraso"):
        assert g.loc[g.faixa == "Diamante", "score"].min() >= g.loc[g.faixa == "Ouro", "score"].max()


def test_faixas_grupo_pequeno_e_empates():
    df = pd.DataFrame({"dias_atraso": [10, 12, 15], "score": [0.5, 0.5, 0.5]})
    faixas = atribuir_faixas(df)
    assert faixas.notna().all()
    assert set(faixas) <= {"Diamante", "Ouro", "Prata", "Bronze"}


def test_capacidade_respeitada():
    df = _carteira(1001)
    est = comparar_estrategias(df, capacidade=0.30)
    assert list(est["estrategia"]) == ["ranking_global", "por_faixa"]
    assert (est["contratos_contatados"] == round(0.30 * 1001)).all()


def test_faixas_atraso_cobrem_tudo():
    assert list(FAIXAS_ATRASO) == ["01-30", "31-90", "91-360", "361+"]
    assert faixa_atraso(pd.Series([1, 30, 31, 90, 91, 360, 361, 5000])).tolist() == \
        ["01-30", "01-30", "31-90", "31-90", "91-360", "91-360", "361+", "361+"]


def test_tabela_faixas_taxa_e_participacao():
    from src.faixas import tabela_faixas
    t = tabela_faixas(_carteira())
    assert len(t) == 16
    assert t["faixa"].tolist()[:4] == ["Diamante", "Ouro", "Prata", "Bronze"]
    assert t["faixa_atraso"].tolist()[0] == "01-30" and t["faixa_atraso"].tolist()[-1] == "361+"
    assert abs(t["pct_carteira"].sum() - 1) < 1e-9 and abs(t["pct_pagadores"].sum() - 1) < 1e-9
