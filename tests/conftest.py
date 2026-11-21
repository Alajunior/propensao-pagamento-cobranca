import pytest

from dados.gerar_carteira import gerar


@pytest.fixture(scope="session")
def pasta_pequena(tmp_path_factory):
    """Base reduzida (3 mil clientes), gerada uma vez para toda a suíte."""
    pasta = tmp_path_factory.mktemp("carteira")
    gerar(3000, pasta)
    return pasta
