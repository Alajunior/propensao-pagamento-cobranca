from types import SimpleNamespace

import pandas as pd
import pytest

from src.base_analitica import carregar, montar_base, safras
from src.escoragem import treinar_vigente
from src.producao import (DIA_PASSADO, FOTO_AUSENTE, HASH_DIVERGENTE, OK, VOLUME_BAIXO, carregar_modelo,
                          escorar_dia_guardado, salvar_modelo)


@pytest.fixture(scope="module")
def base_modelo(pasta_pequena, tmp_path_factory):
    t = carregar(pasta_pequena)
    vig = treinar_vigente("2026-11-16", {d: montar_base(d, t) for d in safras()})
    return t, vig


@pytest.fixture
def ctx(base_modelo, tmp_path):
    t, vig = base_modelo
    mod = tmp_path / "modelo"
    salvar_modelo(vig, mod)
    return SimpleNamespace(t=t, mod=mod, out=tmp_path / "saida", vazia=montar_base("2026-11-17", t).iloc[0:0])


def test_manifest_confere_o_hash(ctx, base_modelo):
    vig, cod = carregar_modelo(ctx.mod)
    assert cod == OK and vig.safra_calib == base_modelo[1].safra_calib == pd.Timestamp("2026-09-30")   # out ainda aberto em 16/11


def test_dia_normal_grava_e_e_idempotente(ctx):
    r1 = escorar_dia_guardado("2026-11-16", ctx.t, ctx.mod, ctx.out)
    r2 = escorar_dia_guardado("2026-11-16", ctx.t, ctx.mod, ctx.out)
    assert r1["codigo"] == r2["codigo"] == OK and len(list(ctx.out.glob("escoragem_*.csv"))) == 1
    assert r1["piramide_ok"] and r1["sha_ok"] and r1["linhas"] > 0
    assert not list(ctx.out.glob("*.tmp"))


def test_foto_ausente_e_volume_baixo(ctx):
    escorar_dia_guardado("2026-11-16", ctx.t, ctx.mod, ctx.out)
    r = escorar_dia_guardado("2026-11-17", ctx.t, ctx.mod, ctx.out, foto=ctx.vazia)
    assert r["codigo"] == FOTO_AUSENTE and not (ctx.out / "escoragem_20261117.csv").exists()
    meia = montar_base("2026-11-18", ctx.t).sample(frac=0.5, random_state=1)
    r = escorar_dia_guardado("2026-11-18", ctx.t, ctx.mod, ctx.out, foto=meia)
    assert r["codigo"] == VOLUME_BAIXO and r["volume_vs_anterior"] < 0.85


def test_hash_divergente_e_dia_passado(ctx):
    escorar_dia_guardado("2026-11-18", ctx.t, ctx.mod, ctx.out)
    assert escorar_dia_guardado("2026-11-16", ctx.t, ctx.mod, ctx.out)["codigo"] == DIA_PASSADO
    (ctx.mod / "modelo.pkl").write_bytes(b"adulterado")
    r = escorar_dia_guardado("2026-11-19", ctx.t, ctx.mod, ctx.out)
    assert r["codigo"] == HASH_DIVERGENTE and r["sha_ok"] is False
