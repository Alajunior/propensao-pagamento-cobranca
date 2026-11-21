import numpy as np
import pandas as pd
import pytest

from src.base_analitica import carregar, montar_base, safras
from src.escoragem import HOJE, boletos, escorar_dia, escorar_periodo, treinar_vigente


@pytest.fixture(scope="module")
def tab(pasta_pequena):
    return carregar(pasta_pequena)


@pytest.fixture(scope="module")
def bases(tab):
    return {d: montar_base(d, tab) for d in safras()}


def test_vigente_usa_safras_fechadas(bases):
    v = treinar_vigente(pd.Timestamp("2025-12-01"), bases)
    assert v.safra_calib == pd.Timestamp("2025-10-31")
    assert v.safras_treino == [pd.Timestamp(x) for x in ("2025-07-31", "2025-08-31", "2025-09-30")]
    assert treinar_vigente(pd.Timestamp("2025-11-28"), bases) is None
    v2 = treinar_vigente(pd.Timestamp("2026-03-16"), bases)
    assert v2.safra_calib + pd.Timedelta(days=30) <= pd.Timestamp("2026-03-16")
    assert all(s < v2.safra_calib for s in v2.safras_treino)


def test_escorar_dia_faixas_e_maturidade(tab, bases):
    D = pd.Timestamp("2026-11-16")
    e = escorar_dia(D, tab, treinar_vigente(D, bases))
    assert not e["maduro"].any() and e["pagou_30d"].isna().all() and e["valor_pago_30d"].isna().all()
    assert set(e.faixa) <= {"Diamante", "Ouro", "Prata", "Bronze"}
    assert np.allclose(e.valor_esperado, e.p_cal * e.ticket)
    assert e.p_cal.between(0, 1).all()
    D2 = pd.Timestamp("2026-10-15")
    e2 = escorar_dia(D2, tab, treinar_vigente(D2, bases))
    assert e2["maduro"].all() and e2["pagou_30d"].notna().all()


def test_boletos_nao_passam_de_hoje(tab):
    esc = escorar_periodo(tab, inicio="2026-10-01")
    b = boletos(tab, esc)
    assert len(b) > 0
    assert (b.dia_acordo <= HOJE).all() and (b.dia_pago.dropna() <= HOJE).all()
    assert (b.dia_foto <= b.dia_acordo).all()
    assert esc.groupby("dia").versao.nunique().eq(1).all()
    assert esc.dia.min() >= pd.Timestamp("2026-10-01") and esc.dia.max() <= HOJE
