"""Guarda de publicação: nada que identifique empresa, cliente ou sistema real pode ir para o GitHub.

A lista de termos proibidos NÃO fica neste repositório (publicá-la seria o próprio vazamento).
Ela é lida de um arquivo local, um termo por linha:
    - caminho na variável de ambiente PORTFOLIO_TERMOS, ou
    - ../_specs/termos_proibidos.txt (ao lado do repositório).
Sem o arquivo (CI, clone de outra pessoa), a guarda real é pulada; os testes da lógica rodam sempre.
"""
import os
import re
import subprocess
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parent.parent
LISTA_PADRAO = RAIZ.parent / "_specs" / "termos_proibidos.txt"


def compilar(termos: list[str]) -> list[re.Pattern]:
    """Sempre como palavra inteira (assim 'cobrança' não acusa um termo 'cobr').
    Siglas curtas (até 3 letras ou com dígito) com caixa exata; o resto ignora a caixa."""
    padroes = []
    for t in termos:
        curto = len(t) <= 3 or any(c.isdigit() for c in t)
        regex = (r"\b" if t[0].isalnum() else "") + re.escape(t) + (r"\b" if t[-1].isalnum() else "")
        padroes.append(re.compile(regex, 0 if curto else re.IGNORECASE))
    return padroes


def texto_de(caminho: Path) -> str:
    try:
        return caminho.read_text(encoding="utf-8")
    except UnicodeDecodeError:   # binário (png): só trechos legíveis, como o comando `strings`
        return "\n".join(t.decode() for t in re.findall(rb"[\x20-\x7e]{6,}", caminho.read_bytes()))


def achar_tracos(padroes: list[re.Pattern], textos: dict[str, str]) -> list[str]:
    """Devolve 'origem: termo nº i' para cada termo encontrado (sem repetir o termo na mensagem)."""
    return [f"{origem}: termo proibido nº {i + 1}"
            for origem, texto in textos.items() for i, p in enumerate(padroes) if p.search(texto)]


# ---------- a lógica da guarda (roda sempre, inclusive na CI) ----------

def test_pega_termo_longo_em_qualquer_caixa():
    assert achar_tracos(compilar(["Empresa Exemplo"]), {"a.py": "x = 'EMPRESA EXEMPLO'"})


def test_sigla_curta_so_como_palavra():
    p = compilar(["XYZ"])
    assert achar_tracos(p, {"a.py": "cliente XYZ"})
    assert not achar_tracos(p, {"a.py": "XYZW e xyz"})


def test_pega_prefixo_de_ip():
    assert achar_tracos(compilar(["10.99."]), {"a.py": "host = '10.99.1.2'"})


# ---------- a guarda real (precisa da lista local) ----------

def _arquivos_e_commits() -> dict[str, str]:
    """Arquivos que iriam para o GitHub (versionados + novos não ignorados) e os metadados de todos os commits."""
    def git(*args):
        return subprocess.run(["git", *args], cwd=RAIZ, capture_output=True, text=True, check=True).stdout
    textos = {}
    for nome in git("ls-files", "--cached", "--others", "--exclude-standard").splitlines():
        caminho = RAIZ / nome
        if caminho.is_file():
            textos[nome] = nome + "\n" + texto_de(caminho)   # o nome do arquivo também conta
    textos["(commits: autor, e-mail, mensagem)"] = git("log", "--all", "--format=%an %ae %cn %ce %B")
    return textos


def test_sem_tracos_da_empresa():
    lista = Path(os.environ.get("PORTFOLIO_TERMOS", LISTA_PADRAO))
    if not lista.is_file():
        pytest.skip(f"lista de termos não encontrada ({lista}); guarda real pulada")
    if not (RAIZ / ".git").exists():
        pytest.skip("fora de um repositório git")
    termos = [t.strip() for t in lista.read_text(encoding="utf-8").splitlines() if t.strip()]
    achados = achar_tracos(compilar(termos), _arquivos_e_commits())
    assert not achados, "\n".join(achados)


def test_termo_dentro_de_palavra_nao_acusa():
    assert not achar_tracos(compilar(["cobr"]), {"a.py": "assessoria de cobrança"})
