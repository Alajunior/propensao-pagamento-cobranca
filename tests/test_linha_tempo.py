import numpy as np
import pandas as pd

from src.linha_tempo import cenarios_mensais, conclusoes


def _esc_par(seed: int = 0):
    """2 meses × 5 dias úteis × 100 contratos. O mensal ordena pelo sinal verdadeiro; o congelado, quase ao acaso."""
    rng = np.random.default_rng(seed)
    partes = []
    for dia in list(pd.bdate_range("2026-07-01", periods=5)) + list(pd.bdate_range("2026-08-03", periods=5)):
        verdade = rng.random(100)
        y = (rng.random(100) < verdade * 0.3).astype(float)
        partes.append(pd.DataFrame({"dia": dia, "id_contrato": np.arange(100), "score": verdade + rng.normal(0, .05, 100),
                                    "maduro": True, "pagou_30d": y, "valor_pago_30d": y * 100,
                                    "qtd_cpc_30d": rng.integers(0, 3, 100), "dias_atraso": rng.integers(1, 1500, 100)}))
    esc_m = pd.concat(partes, ignore_index=True)
    esc_c = esc_m.assign(score=rng.random(len(esc_m)))
    return esc_m, esc_c


def test_cenarios_e_capacidade():
    esc_m, esc_c = _esc_par()
    df = cenarios_mensais(esc_m, esc_c)
    assert set(df.cenario) == {"regua", "congelado", "mensal"} and df.mes.nunique() == 2
    assert (df.acionados == 30 * 5).all()
    m = df.set_index(["mes", "cenario"])
    assert (m.xs("mensal", level=1).pct_pagadores >= m.xs("regua", level=1).pct_pagadores).all()
    assert df.pct_valor.between(0, 1).all() and (df.pag_por_mil >= 0).all()


def test_nao_usa_dia_nao_maduro():
    esc_m, esc_c = _esc_par()
    esc_m.loc[esc_m.dia >= "2026-08-01", ["maduro", "pagou_30d", "valor_pago_30d"]] = [False, np.nan, np.nan]
    assert cenarios_mensais(esc_m, esc_c).mes.unique().tolist() == ["2026-07"]


def test_conclusao_muda_com_o_numero():
    df = pd.DataFrame([{"mes": "2026-07", "cenario": c, "pct_pagadores": v, "pct_valor": v, "pag_por_mil": 1,
                        "acionados": 1, "pagadores": 1} for c, v in (("regua", .5), ("congelado", .795), ("mensal", .80))])
    t = conclusoes(df, 0.002, 0.10, 0.45)
    assert len(t) == 4 and all({"titulo", "texto", "numero"} <= set(x) for x in t)
    assert "só" in t[1]["titulo"] and t[3]["titulo"].startswith("Diamante pequeno")
    df.loc[df.cenario == "congelado", "pct_pagadores"] = .75
    assert "perde" in conclusoes(df, 0.002, 0.10, 0.45)[1]["titulo"]
