import numpy as np
import pytest

from src.base_analitica import ALVO, CAT, NUM, carregar, montar_base
from src.modelos import criar_modelos


@pytest.fixture(scope="module")
def treino_teste(pasta_pequena):
    t = carregar(pasta_pequena)
    return montar_base("2026-03-31", t), montar_base("2026-04-30", t)


@pytest.mark.parametrize("nome", ["logistica", "lightgbm", "mlp"])
def test_todos_preveem_probabilidade(treino_teste, nome):
    treino, teste = treino_teste
    m = criar_modelos()[nome].fit(treino[NUM + CAT], treino[ALVO])
    p = m.predict_proba(teste[NUM + CAT])
    assert p.shape == (len(teste), 2)
    assert np.all((p >= 0) & (p <= 1))


def test_categoria_nova_no_oot(treino_teste):
    treino, teste = treino_teste
    teste = teste.copy()
    teste.loc[teste.index[:5], "uf"] = "ZZ"
    for m in criar_modelos().values():
        m.fit(treino[NUM + CAT], treino[ALVO]).predict_proba(teste[NUM + CAT])
