import pandas as pd

from src.backtest import COLUNAS, rodar_backtest, safras_de_treino
from src.base_analitica import carregar, safras
from src.modelos import criar_modelos


def test_treino_nao_toca_o_teste():
    todas = safras()
    for d_teste in todas:
        treino = safras_de_treino(d_teste, todas, janela=3)
        if treino:
            assert len(treino) == 3
            assert max(treino) + pd.Timedelta(days=30) <= d_teste


def test_formato_do_backtest(pasta_pequena):
    bt = rodar_backtest(carregar(pasta_pequena), janela=3,
                        modelos={"logistica": criar_modelos()["logistica"]})
    assert list(bt.columns) == COLUNAS
    testaveis = [d for d in safras() if safras_de_treino(d, safras(), 3)]
    assert len(bt) == 2 * len(testaveis)
    assert set(bt["modo"]) == {"congelado", "mensal"}
    assert not bt.duplicated(["safra_teste", "modelo", "modo"]).any()


def test_split_por_cliente_nao_repete_cliente():
    """Safras empilhadas repetem o mesmo contrato (e cliente): validar ao acaso põe o mesmo cliente nos dois lados."""
    from src.backtest import _separar
    base = pd.DataFrame({"id_cliente": [i // 3 for i in range(600)], "pagou_30d": [i % 7 == 0 for i in range(600)]})
    tr, va = _separar(base)
    assert not set(tr.id_cliente) & set(va.id_cliente)
    assert 0.2 < len(va) / len(base) < 0.4
