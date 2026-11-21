import pandas as pd
import pytest

from src.base_analitica import ALVO, carregar, montar_base, safras

D = pd.Timestamp("2026-04-30")


@pytest.fixture(scope="module")
def tabelas(pasta_pequena):
    return carregar(pasta_pequena)


@pytest.fixture(scope="module")
def base(tabelas):
    return montar_base(D, tabelas)


def test_grao_um_contrato_por_linha(base):
    assert base["id_contrato"].is_unique


def test_populacao_na_carteira_ativa(base):
    assert (base["data_entrada"] <= D).all()
    assert (D < base["data_entrada"] + pd.Timedelta(days=180)).all()
    assert (base["data_vencimento"] < D).all()


def test_features_nao_olham_o_futuro(base, tabelas):
    assert (base["dias_desde_ult_cpc"].dropna() >= 0).all()
    acion = tabelas["acionamentos"]
    amostra = base.sample(20, random_state=0)
    for _, linha in amostra.iterrows():
        a = acion[(acion["id_contrato"] == linha["id_contrato"]) & (acion["data_hora"] < D)]
        assert linha["qtd_acion_total"] == len(a)


def test_alvo_binario_e_valor_coerente(base):
    assert set(base[ALVO].unique()) <= {0, 1}
    assert ((base["valor_pago_30d"] > 0) == (base[ALVO] == 1)).all()


def test_safras():
    s = safras()
    assert len(s) == 17
    assert s[0] == pd.Timestamp("2025-07-31")
    assert s[-1] == pd.Timestamp("2026-11-30")


def test_todas_as_faixas_de_atraso_tem_volume(base):
    from src.faixas import FAIXAS_ATRASO, faixa_atraso
    dist = faixa_atraso(base["dias_atraso"]).value_counts(normalize=True)
    for faixa in FAIXAS_ATRASO:
        assert dist.get(faixa, 0) >= 0.10, (faixa, dist.to_dict())
