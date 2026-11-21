import numpy as np
import pandas as pd

from src.faixas import faixa_atraso
from src.candidatos import captura_valor_top, gate, ic_bootstrap_diff, score_regua


def _pilha(efeito: float, seed: int = 0) -> pd.DataFrame:
    """3 safras × 3.000 contratos (2 por cliente). `a` tem sinal fraco; `b` = `a` + efeito × sinal verdadeiro."""
    rng = np.random.default_rng(seed)
    partes = []
    for s, safra in enumerate(pd.to_datetime(["2026-01-31", "2026-02-28", "2026-03-31"])):
        n = 3000
        verdade = rng.normal(size=n)
        y = (rng.random(n) < 1 / (1 + np.exp(-(verdade * 1.5 - 2.5)))).astype(int)
        a = 0.3 * verdade + rng.normal(size=n)
        partes.append(pd.DataFrame({
            "safra": safra, "id_cliente": s * 10_000 + np.arange(n) // 2,
            "dias_atraso": (atraso := rng.integers(1, 1500, n)), "fase": faixa_atraso(pd.Series(atraso)),
            "pagou_30d": y, "valor_pago_30d": y * rng.uniform(100, 1000, n),
            "a": a, "b": a + efeito * 3 * verdade + rng.normal(scale=0.01, size=n),
            "regua": rng.random(n)}))
    return pd.concat(partes, ignore_index=True)


def test_regua_prioriza_contato_recente_e_menor_atraso():
    b = pd.DataFrame({"qtd_cpc_30d": [0, 2, 1], "dias_atraso": [10, 400, 50]})
    assert list(np.argsort(-score_regua(b))) == [2, 1, 0]


def test_captura_por_safra():
    df = pd.DataFrame({"safra": [1] * 10 + [2] * 10, "valor_pago_30d": [100] + [0] * 9 + [0] * 9 + [100],
                       "s": list(range(10, 0, -1)) * 2})
    assert captura_valor_top(df["valor_pago_30d"], df["s"], df["safra"], 0.10) == 0.5


def test_gate_reprova_desafiante_igual():
    g = gate(_pilha(efeito=0.0), "a", "b")
    assert not g["passou"] and g["ic"][0] <= 0
    assert [c["nome"] for c in g["criterios"]] == ["captura_top10_ic", "auc_por_fase", "sem_inversao", "acima_da_regua"]


def test_gate_aprova_desafiante_claramente_melhor():
    g = gate(_pilha(efeito=1.0), "a", "b")
    assert g["passou"] and g["ic"][0] > 0


def test_bootstrap_reprodutivel():
    df = _pilha(efeito=0.5)
    assert ic_bootstrap_diff(df, "a", "b", n=200) == ic_bootstrap_diff(df, "a", "b", n=200)
