import json
import re

import pytest

from src.base_analitica import carregar
from src.montar_painel import CHARTJS, FONTES, montar_dados, montar_html


@pytest.fixture(scope="module")
def dados(pasta_pequena, tmp_path_factory):
    rel = tmp_path_factory.mktemp("rel")
    (rel / "validacao.json").write_text(json.dumps({"testes": [], "resumo": "x"}), encoding="utf-8")
    return montar_dados(carregar(pasta_pequena), inicio="2026-09-01", relatorios=rel, lacuna="2026-11-25")


def test_payload_tem_as_chaves(dados):
    assert {"meta", "diario", "conc", "conc_real", "top10", "perfil", "fila", "filas_cmp", "qualidade"} <= set(dados)
    assert dados["meta"]["dias"][-1] == "2026-11-30" and len(dados["meta"]["faixas"]) == 4
    d = dados["diario"][dados["meta"]["dias"][-1]]["__TODAS__"]
    assert set(d) == {"Diamante", "Ouro", "Prata", "Bronze"} and len(d["Diamante"]) == 8
    assert "escada" not in dados["qualidade"] and "tempo" not in dados["qualidade"]   # relatórios ausentes
    json.dumps(dados, allow_nan=False)
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


def test_lacuna_copia_foto_anterior(dados):
    d = dados["diario"]
    assert dados["meta"]["lacunas"][0]["dia"] == "2026-11-25" and dados["meta"]["lacunas"][0]["foto_usada"] == "2026-11-24"
    assert "2026-11-25" in d and "2026-11-25" in dados["meta"]["dias"]
    for fx in ("Diamante", "Bronze"):
        assert d["2026-11-25"]["__TODAS__"][fx][:2] == d["2026-11-24"]["__TODAS__"][fx][:2]
    assert dados["conc"]["2026-11-25"] == dados["conc"]["2026-11-24"]
    assert dados["perfil"]["2026-11-25"] == dados["perfil"]["2026-11-24"]
    assert "2026-11-25" not in dados["fila"]
    bol_dia = sum(v[2] for v in d["2026-11-25"]["__TODAS__"].values())
    assert bol_dia == sum(d["2026-11-25"][f][fx][2] for f in dados["meta"]["fases"] if f in d["2026-11-25"]
                          for fx in d["2026-11-25"][f])   # os boletos do dia continuam (e fecham com as fases)


def test_execucao_mostra_a_lacuna(dados):
    ex = dados["execucao"]
    assert len(ex) == 6 and [e["codigo"] for e in ex if e["dia"] == "2026-11-25"] == [3]
    assert all(e["codigo"] == 0 for e in ex if e["dia"] != "2026-11-25")
    assert [e["dia"] for e in ex] == sorted(e["dia"] for e in ex)


def test_chaves_novas(dados):
    assert {"validacao", "sinal", "tempo_exec"} <= set(dados)
    assert all(len(r) == 7 and isinstance(r[6], list) for v in dados["fila"].values() for r in v["__TODAS__"])
    assert len(dados["tempo_exec"]["conclusoes"]) == 4 and dados["tempo_exec"]["serie"]
    assert {"medir", "importancia", "perfil_tipico"} <= set(dados["sinal"])
