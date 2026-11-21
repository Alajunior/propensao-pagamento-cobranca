import json
import re

import pytest

from src.base_analitica import carregar
from src.montar_painel import CHARTJS, FONTES, montar_dados, montar_html


@pytest.fixture(scope="module")
def dados(pasta_pequena, tmp_path_factory):
    return montar_dados(carregar(pasta_pequena), inicio="2026-09-01", relatorios=tmp_path_factory.mktemp("rel"))


def test_payload_tem_as_chaves(dados):
    assert {"meta", "diario", "conc", "conc_real", "top10", "perfil", "fila", "filas_cmp", "qualidade"} <= set(dados)
    assert dados["meta"]["dias"][-1] == "2026-11-30" and len(dados["meta"]["faixas"]) == 4
    d = dados["diario"][dados["meta"]["dias"][-1]]["__TODAS__"]
    assert set(d) == {"Diamante", "Ouro", "Prata", "Bronze"} and len(d["Diamante"]) == 8
    assert "escada" not in dados["qualidade"] and "tempo" not in dados["qualidade"]   # relatórios ausentes
    assert len(next(iter(dados["conc"].values()))["__TODAS__"]) == 60
    json.dumps(dados)


def test_nada_depois_de_hoje(dados):
    assert max(dados["diario"]) <= "2026-11-30" and dados["meta"]["maduro_ate"] <= "2026-10-31"
    assert max(dados["conc_real"]) <= dados["meta"]["maduro_ate"]
    assert len(dados["fila"]) == 5 and all(len(v["__TODAS__"]) <= 200 for v in dados["fila"].values())
    # boleto pago no caixa nunca supera o que existe até hoje: soma do caixa = soma da coorte (mesmos boletos)
    tot = dados["diario"]
    caixa = sum(v[fx][4] for dia in tot.values() for f, v in dia.items() if f == "__TODAS__" for fx in v)
    coorte = sum(v[fx][6] for dia in tot.values() for f, v in dia.items() if f == "__TODAS__" for fx in v)
    assert caixa <= coorte


def test_todas_e_soma_das_fases(dados):
    dia = dados["meta"]["dias"][-1]
    dd = dados["diario"][dia]
    for fx in ("Diamante", "Bronze"):
        assert dd["__TODAS__"][fx][0] == sum(dd[f][fx][0] for f in dados["meta"]["fases"] if f in dd)


def test_html_autocontido(dados):
    html = montar_html(dados)
    assert "const DADOS = {" in html and "fetch(" not in html and 'type="module"' not in html
    assert set(re.findall(r"https://[^\s\"'<>)]+", html)) <= {CHARTJS, FONTES, "https://fonts.googleapis.com"}
    assert len(html.encode("utf-8")) < 5 * 1024 * 1024


def test_evento_de_fim_de_semana_vai_para_a_segunda():
    import pandas as pd
    from src.montar_painel import dia_util_seguinte
    dias = pd.DatetimeIndex(pd.bdate_range("2026-11-20", "2026-11-30"))
    datas = pd.Series(pd.to_datetime(["2026-11-21", "2026-11-23", None, "2026-12-01"]))
    r = dia_util_seguinte(dias, datas)
    assert r.iloc[0] == pd.Timestamp("2026-11-23") and r.iloc[1] == pd.Timestamp("2026-11-23")
    assert pd.isna(r.iloc[2]) and pd.isna(r.iloc[3])   # vazio e depois do último dia ficam de fora


def test_tempo_nao_usa_safra_com_alvo_depois_de_hoje(pasta_pequena, tmp_path):
    import pandas as pd
    from src.montar_painel import _qualidade
    from src.escoragem import escorar_periodo
    linhas = [{"safra_teste": s, "modelo": "logistica", "modo": "mensal", "AUC": .8, "auc_validacao": .8,
               "psi_score": 0, "psi_renda": 0, "psi_dias_atraso": 0} for s in ("2026-10-31", "2026-11-30")]
    pd.DataFrame(linhas).to_csv(tmp_path / "backtest.csv", index=False)
    esc = escorar_periodo(carregar(pasta_pequena), inicio="2026-11-23")
    q = _qualidade(esc, [], tmp_path, hoje=pd.Timestamp("2026-11-30"))
    assert [r["safra_teste"] for r in q["tempo"]] == ["2026-10-31"]   # alvo de nov/2026 só fecha em 30/12
