from src.treinar import ARQUIVOS, main


def test_main_gera_relatorios(pasta_pequena, tmp_path):
    main(pasta_pequena, tmp_path)
    for nome in ARQUIVOS:
        assert (tmp_path / nome).stat().st_size > 0, nome
    assert len(ARQUIVOS) == 7
