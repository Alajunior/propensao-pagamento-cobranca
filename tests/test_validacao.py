import json

from src.base_analitica import carregar
from src.validacao import rodar_bateria

IDS = ["walk_forward", "aleatorio_vs_cliente", "alvo_embaralhado", "gap_treino_teste", "curva_aprendizado", "ic_bootstrap"]


def test_bateria_usa_safra_seguinte(pasta_pequena, tmp_path):
    r = rodar_bateria(carregar(pasta_pequena), n_boot=50, relatorios=tmp_path)
    assert [t["id"] for t in r["testes"]] == IDS
    d = r["testes"][2]["detalhe"]
    assert d["safra_teste"] == "2026-06-30" and d["safras_treino"] == ["2026-03-31", "2026-04-30", "2026-05-31"]
    assert all(t["veredito"] in {"ok", "atencao", "falhou"} for t in r["testes"])
    assert all(t["titulo"] and t["resultado"] for t in r["testes"])
    assert 0.40 < d["auc"] < 0.60
    lo, hi = r["testes"][5]["detalhe"]["ic"]
    assert lo < r["testes"][5]["detalhe"]["auc"] < hi
    assert r["resumo"]
    json.dumps(r)


def test_walk_forward_so_safras_maduras(pasta_pequena, tmp_path):
    import pandas as pd
    from src.validacao import _walk_forward
    linhas = [{"safra_teste": s, "modelo": "logistica", "modo": "mensal", "AUC": a}
              for s, a in (("2026-09-30", .80), ("2026-10-31", .82), ("2026-11-30", .50))]
    pd.DataFrame(linhas).to_csv(tmp_path / "backtest.csv", index=False)
    r = _walk_forward(carregar(pasta_pequena), tmp_path)
    assert r["detalhe"]["safras"] == 2 and r["detalhe"]["pior"] == .80   # nov/2026 só fecha em 30/12
