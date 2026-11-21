import numpy as np
import pandas as pd

from src.backtest import X
from src.base_analitica import ALVO, NUM, carregar, montar_base
from src.faixas import atribuir_faixas
from src.modelos import criar_modelos
from src.sinal import NOMES, contribuicoes, importancia, medir_sinal, motivos, perfil_tipico, referencia


def test_contribuicoes_fecham_o_logodds(pasta_pequena):
    t = carregar(pasta_pequena)
    b = montar_base("2026-08-31", t)
    m = criar_modelos()["logistica"].fit(b[X], b[ALVO])
    ref = referencia(m, b[X])
    c = contribuicoes(m, b[X].head(50), ref)
    lr = m[-1]
    zref = ref.to_numpy() @ lr.coef_[0] + lr.intercept_[0]
    assert np.allclose(c.sum(axis=1).to_numpy() + zref, m.decision_function(b[X].head(50)), atol=1e-6)
    assert set(c.columns) <= set(NUM) | {"produto", "uf"}
    imp = importancia(c)
    assert list(imp.columns) == ["variavel", "nome", "media_abs"] and imp.media_abs.is_monotonic_decreasing


def test_motivos_so_positivos():
    c = pd.DataFrame({"qtd_cpc_30d": [1.2, -0.5], "dias_atraso": [0.4, -0.1], "idade": [-0.3, -0.2]})
    assert motivos(c) == [[NOMES["qtd_cpc_30d"], NOMES["dias_atraso"]], []]


def test_medir_sinal_formato(pasta_pequena):
    s = medir_sinal(carregar(pasta_pequena), inicio="2026-08-31", fim="2026-10-31")
    assert {"__TODAS__"} <= set(s.fase) and set(s.variavel) == set(NUM)
    assert (s.auc >= 0.5).all() and s.captura_top40.between(0, 1).all() and s.captura_modelo.between(0, 1).all()
    assert set(s.direcao) <= {"+", "−"}


def test_perfil_tipico(pasta_pequena):
    b = montar_base("2026-10-31", carregar(pasta_pequena))
    b["score"] = np.random.default_rng(0).random(len(b))
    b["faixa"] = atribuir_faixas(b)
    p = perfil_tipico(b)
    assert set(p["__TODAS__"]) == {"Diamante", "Ouro", "Prata", "Bronze"}
    assert set(p["__TODAS__"]["Ouro"]) == {"dias_atraso", "valor_divida", "idade", "renda_estimada", "qtd_acion_30d",
                                           "qtd_cpc_30d", "dias_desde_ult_cpc", "tem_acordo_antes"}


def test_perfil_tipico_sem_nan():
    import json
    b = pd.DataFrame({"dias_atraso": [10, 20], "faixa": ["Ouro", "Ouro"], "valor_divida": [1.0, 2.0], "idade": [30, 40],
                      "renda_estimada": [np.nan, np.nan], "qtd_acion_30d": [1, 2], "qtd_cpc_30d": [0, 1],
                      "dias_desde_ult_cpc": [np.nan, np.nan], "tem_acordo_antes": [0, 1]})
    p = perfil_tipico(b)
    assert p["__TODAS__"]["Ouro"]["renda_estimada"] is None
    json.dumps(p, allow_nan=False)
