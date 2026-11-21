import pandas as pd
import pytest

from dados.gerar_carteira import gerar


@pytest.fixture(scope="module")
def tabelas(tmp_path_factory):
    return gerar(3000, tmp_path_factory.mktemp("dados"))


def test_periodo_24_meses(tabelas):
    entrada = tabelas["contratos"]["data_entrada"]
    assert entrada.min() >= pd.Timestamp("2025-01-01")
    assert entrada.max() <= pd.Timestamp("2026-12-31")
    assert entrada.dt.to_period("M").nunique() == 24


def test_relacoes_fecham(tabelas):
    t = tabelas
    assert t["contratos"]["id_cliente"].isin(t["clientes"]["id_cliente"]).all()
    assert t["acordos"]["id_contrato"].isin(t["contratos"]["id_contrato"]).all()
    assert t["pagamentos"]["id_acordo"].isin(t["acordos"]["id_acordo"]).all()


def test_renda_cai_na_etapa_1(tabelas):
    ct = tabelas["contratos"].merge(tabelas["clientes"], on="id_cliente")
    novos = ct["data_entrada"] >= pd.Timestamp("2026-02-01")
    antes, depois = ct.loc[~novos, "renda_estimada"].median(), ct.loc[novos, "renda_estimada"].median()
    assert depois < 0.8 * antes


def test_acionamento_dentro_da_janela(tabelas):
    a = tabelas["acionamentos"].merge(tabelas["contratos"][["id_contrato", "data_entrada"]], on="id_contrato")
    assert (a["data_hora"] >= a["data_entrada"]).all()
    assert (a["data_hora"] < a["data_entrada"] + pd.Timedelta(days=180)).all()
