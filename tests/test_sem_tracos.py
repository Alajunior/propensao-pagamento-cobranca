"""Guarda de publicação: o repositório não pode conter nenhum traço de empresa, cliente ou sistema real.

A lista fica em base64 para este arquivo não acusar a si mesmo nem expor os nomes em texto puro.
"""
import base64
import re
import subprocess
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
_LISTA = ""   # lista removida do histórico
_CURTOS = set()


def _padroes():
    for termo in base64.b64decode(_LISTA).decode().split("|"):
        borda = r"\b" if termo in _CURTOS else ""
        yield termo, re.compile(borda + re.escape(termo) + borda)


def _arquivos():
    """Arquivos que iriam para o GitHub: versionados + novos não ignorados."""
    saida = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"],
                           cwd=RAIZ, capture_output=True, text=True, check=True).stdout
    for nome in saida.splitlines():
        caminho = RAIZ / nome
        if caminho.is_file() and caminho.resolve() != Path(__file__).resolve():
            yield caminho


def achar_tracos() -> list[str]:
    achados = []
    padroes = list(_padroes())
    for caminho in _arquivos():
        try:
            texto = caminho.read_text(encoding="utf-8")
        except UnicodeDecodeError:   # binário (png): só os trechos de texto legível, como o comando `strings`
            texto = "\n".join(t.decode() for t in re.findall(rb"[\x20-\x7e]{6,}", caminho.read_bytes()))
        texto += "\n" + caminho.relative_to(RAIZ).as_posix()  # o nome do arquivo também conta
        achados += [f"{caminho.relative_to(RAIZ)}: termo proibido nº {i + 1}"
                    for i, (_, p) in enumerate(padroes) if p.search(texto)]
    return achados


def test_sem_tracos_da_empresa():
    achados = achar_tracos()
    assert not achados, "\n".join(achados)
