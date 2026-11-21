"""Calcula tudo o que o painel mostra e grava UM HTML com os dados embutidos (docs/index.html).

    python -m src.treinar          # antes: backtest.csv e veredito.json em relatorios/
    python -m src.montar_painel    # escoragem diária (cache em dados/escoragem/), agregados e HTML

Contrato com o front: chave ausente = card escondido. O front nunca recorta faixa nem recalcula modelo;
só soma dias e divide. Nada posterior a HOJE entra; alvo só de dia maduro (D+30 <= HOJE).
"""
import json
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest import X
from src.base_analitica import carregar, montar_base, safras
from src.calibracao import brier, ece, previu_pagou
from src.escoragem import HOJE, INICIO, _elegiveis, boletos, escorar_periodo, treinar_vigente
from src.linha_tempo import cenarios_mensais, conclusoes
from src.producao import NOMES_CODIGO, escorar_dia_guardado, salvar_modelo
from src.sinal import contribuicoes, direcoes, importancia, medir_sinal, motivos, perfil_tipico, referencia
from src.faixas import FAIXAS_ATRASO
from src.qualidade import ORDEM, TODAS, backtest_producao, estabilidade_semanal, inversoes, lift_por_fase
from src.valor import comparar_filas, curva_concentracao, curva_realizada, top_concentra

RAIZ = Path(__file__).resolve().parent.parent
TEMPLATE = Path(__file__).resolve().parent / "painel.html"
CHARTJS = "https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"
FONTES = ("https://fonts.googleapis.com/css2?family=Sora:wght@700;800&family=Inter:wght@400;600;700;800"
          "&family=JetBrains+Mono:wght@600&display=swap")
FAIXAS = [{"key": "Diamante", "nome": "Diamante", "icone": "💎"}, {"key": "Ouro", "nome": "Ouro", "icone": "🥇"},
          {"key": "Prata", "nome": "Prata", "icone": "🥈"}, {"key": "Bronze", "nome": "Bronze", "icone": "🥉"}]
N_FILA, DIAS_FILA, PONTOS, DIAS_EXEC = 200, 5, 60, 6
LACUNA = "2026-11-25"   # dia em que a foto do estoque "não chega" (demonstra a trava e o aviso do painel)


def _iso(d) -> str:
    return f"{pd.Timestamp(d):%Y-%m-%d}"


def _r(x, n=6):
    return None if x is None or pd.isna(x) else round(float(x), n)


def _recs(df: pd.DataFrame) -> list[dict]:
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = df[c].dt.strftime("%Y-%m-%d")
    return json.loads(df.to_json(orient="records", double_precision=6, force_ascii=False))


def _com_todas(df: pd.DataFrame) -> pd.DataFrame:
    """Duplica as linhas com fase = "__TODAS__" para os agregados por fase saírem com o total junto."""
    return pd.concat([df, df.assign(fase=TODAS)], ignore_index=True)


def dia_util_seguinte(dias: pd.DatetimeIndex, datas: pd.Series) -> pd.Series:
    """Primeiro dia útil do painel >= a data (evento de sábado/domingo cai na segunda, como na operação)."""
    pos = dias.searchsorted(datas, side="left")
    ok = datas.notna().to_numpy() & (pos < len(dias))
    out = pd.Series(pd.NaT, index=datas.index, dtype="datetime64[ns]")
    out[ok] = dias[pos[ok]]
    return out


def _diario(esc, bol, dias, copia: dict | None = None) -> dict:
    """`copia` = {dia sem foto: dia da foto usada}: estoque e recuperação esperada vêm da última foto."""
    dias_ts = pd.DatetimeIndex(pd.to_datetime(dias))
    est = _com_todas(esc[["dia", "fase", "faixa", "valor_esperado"]].astype({"fase": str, "faixa": str}))
    g_est = est.groupby(["dia", "fase", "faixa"]).agg(n=("valor_esperado", "size"), er=("valor_esperado", "sum"))
    for sem_foto, usada in (copia or {}).items():
        extra = g_est.xs(usada, level="dia", drop_level=False).rename(index={usada: sem_foto}, level="dia")
        g_est = pd.concat([g_est, extra]).sort_index()
    b = bol.astype({"fase": str, "faixa": str}).copy()
    # a faixa vem da foto até a emissão; a contagem vai para o dia útil da emissão / do pagamento (fim de semana → segunda)
    b["dia_painel"] = dia_util_seguinte(dias_ts, b["dia_emit"])
    b["dia_caixa"] = dia_util_seguinte(dias_ts, b["dia_pago"])
    b = _com_todas(b[b["dia_painel"].notna()])
    pago = b["dia_pago"].notna()
    emit = b.groupby(["dia_painel", "fase", "faixa"]).agg(bol=("rs_emit", "size"), rs_emit=("rs_emit", "sum"))
    coorte = b[pago].groupby(["dia_painel", "fase", "faixa"]).agg(pag=("rs_pago", "size"), rs=("rs_pago", "sum"))
    caixa = b[b["dia_caixa"].notna()].groupby(["dia_caixa", "fase", "faixa"]).agg(pag=("rs_pago", "size"),
                                                                                  rs=("rs_pago", "sum"))
    for t in (emit, coorte, caixa):
        t.index = t.index.set_names(["dia", "fase", "faixa"])
    tab = g_est.join(emit, how="left").join(caixa.add_suffix("_caixa"), how="left") \
        .join(coorte.add_suffix("_coorte"), how="left").fillna(0)
    out: dict = {}
    for (dia, fase, fx), r in tab.iterrows():
        out.setdefault(_iso(dia), {}).setdefault(fase, {})[fx] = [
            int(r.n), round(r.er, 2), int(r.bol), round(r.rs_emit, 2), int(r.pag_caixa), round(r.rs_caixa, 2),
            int(r.pag_coorte), round(r.rs_coorte, 2)]
    return out


def _por_dia_fase(esc, fn) -> dict:
    out: dict = {}
    for dia, g in esc.groupby("dia"):
        d = out.setdefault(_iso(dia), {})
        for fase, gf in g.groupby("fase"):
            d[str(fase)] = fn(gf)
        d[TODAS] = fn(g)
    return out


def _perfil(g: pd.DataFrame) -> dict:
    out = {}
    for fx, gf in g.groupby("faixa"):
        out[str(fx)] = {"produto": {k: _r(v, 4) for k, v in gf["produto"].astype(str).value_counts(normalize=True).items()},
                        "atraso_med": _r(gf["dias_atraso"].median(), 0), "valor_med": _r(gf["valor_divida"].median(), 2),
                        "cpc30_pct": _r((gf["qtd_cpc_30d"] > 0).mean(), 4), "p_med": _r(gf["p_cal"].mean(), 4)}
    return out


def _fila(g: pd.DataFrame) -> list:
    t = g.nlargest(N_FILA, "valor_esperado")
    return [[int(r.id_contrato), str(r.fase), str(r.faixa), _r(r.p_cal, 4), _r(r.ticket, 2), _r(r.valor_esperado, 2)]
            for r in t.itertuples()]


def _qualidade(esc_m, bt_congelado, relatorios: Path, hoje=HOJE) -> dict:
    m = esc_m[esc_m["maduro"]]
    y = m["pagou_30d"].to_numpy()
    q = {"backtest": {"mensal": backtest_producao(esc_m), "congelado": bt_congelado},
         "lift_fase": _recs(lift_por_fase(esc_m)), "estab": _recs(estabilidade_semanal(esc_m)),
         "inv": _recs(inversoes(esc_m))}
    if len(m) >= 10:   # sem dia maduro não há o que comparar: a chave fica ausente e o card some
        q["previu_pagou"] = {"antes": _recs(previu_pagou(y, m["score"])), "depois": _recs(previu_pagou(y, m["p_cal"])),
                             "brier": {"antes": _r(brier(y, m["score"])), "depois": _r(brier(y, m["p_cal"]))},
                             "ece": {"antes": _r(ece(y, m["score"])), "depois": _r(ece(y, m["p_cal"]))},
                             "taxa_real": _r(y.mean())}
    vered = Path(relatorios) / "veredito.json"
    if vered.exists():
        q["escada"] = json.loads(vered.read_text(encoding="utf-8"))
    bt = Path(relatorios) / "backtest.csv"
    if bt.exists():
        t = pd.read_csv(bt, parse_dates=["safra_teste"])
        # safra cujo alvo (d, d+30] ainda não fechou em "hoje" fica de fora: o painel não conhece esse futuro
        t = t[t["modelo"].isin(["logistica", "lightgbm"]) & (t["safra_teste"] + pd.Timedelta(days=30) <= pd.Timestamp(hoje))]
        q["tempo"] = _recs(t[["safra_teste", "modelo", "modo", "AUC", "auc_validacao", "psi_score", "psi_renda",
                              "psi_dias_atraso"]])
    return q


def _execucao(tabelas, bases, dias_exec, lacuna) -> list[dict]:
    """Roda de verdade a escoragem guardada nos últimos dias úteis, numa pasta temporária, com a lacuna injetada."""
    out = []
    with tempfile.TemporaryDirectory() as tmp:
        mod, pub = Path(tmp) / "modelo", Path(tmp) / "publicado"
        calib = None
        for D in dias_exec:
            vig = treinar_vigente(D, bases)
            if vig.safra_calib != calib:             # virada do mês: o modelo novo é publicado com manifest
                salvar_modelo(vig, mod)
                calib = vig.safra_calib
            foto = montar_base(D, tabelas).iloc[0:0] if lacuna is not None and D == lacuna else None
            r = escorar_dia_guardado(D, tabelas, mod, pub, foto=foto)
            out.append({**r, "nome_codigo": NOMES_CODIGO[r["codigo"]], "versao": f"mensal-{calib:%Y-%m}"})
    return out


def _motivos_fila(fila: dict, tabelas, bases) -> dict:
    """Acrescenta a cada linha da fila os 2 motivos que mais empurraram a chance para cima (modelo vigente do dia)."""
    refs = {}
    for dia, por_fase in fila.items():
        D = pd.Timestamp(dia)
        vig = treinar_vigente(D, bases)
        if vig.safra_calib not in refs:
            treino = pd.concat(bases[s] for s in vig.safras_treino)
            refs[vig.safra_calib] = referencia(vig.modelo, treino[X])
        ids = {r[0] for linhas in por_fase.values() for r in linhas}
        base = montar_base(D, tabelas)
        base = base[base["id_contrato"].isin(ids)].set_index("id_contrato")
        ref = refs[vig.safra_calib]
        mot = dict(zip(base.index, motivos(contribuicoes(vig.modelo, base[X], ref), direcoes(vig.modelo, base[X], ref))))
        for linhas in por_fase.values():
            for r in linhas:
                r.append(mot.get(r[0], []))
    return fila


def _sinal(tabelas, bases, esc_hoje, hoje) -> dict:
    vig = treinar_vigente(hoje, bases)
    treino = pd.concat(bases[s] for s in vig.safras_treino)
    ref = referencia(vig.modelo, treino[X])
    base = montar_base(hoje, tabelas).merge(esc_hoje[["id_contrato", "faixa"]].astype({"faixa": str}), on="id_contrato")
    return {"medir": _recs(medir_sinal(tabelas)),
            "importancia": _recs(importancia(contribuicoes(vig.modelo, treino[X], ref))),
            "perfil_tipico": perfil_tipico(base), "foto": _iso(hoje)}


def _tempo_exec(esc, esc_c, ece_depois) -> dict | None:
    m = esc[esc["maduro"]]
    if not len(m) or ece_depois is None:
        return None
    serie = cenarios_mensais(esc, esc_c)
    diam = m["faixa"].astype(str) == "Diamante"
    pct_est = float(diam.mean())
    pct_pag = float(m.loc[diam, "pagou_30d"].sum() / m["pagou_30d"].sum())
    return {"serie": _recs(serie), "conclusoes": conclusoes(serie, ece_depois, pct_est, pct_pag), "capacidade": 0.30}


def montar_dados(tabelas: dict, hoje=HOJE, inicio=INICIO, cache=None, relatorios=RAIZ / "relatorios",
                 lacuna=LACUNA) -> dict:
    hoje = pd.Timestamp(hoje)
    esc_c = escorar_periodo(tabelas, "congelado", inicio, hoje, cache, hoje)
    esc = escorar_periodo(tabelas, "mensal", inicio, hoje, cache, hoje)
    dias_painel = sorted(esc["dia"].unique())
    lac = pd.Timestamp(lacuna) if lacuna is not None and pd.Timestamp(lacuna) in set(dias_painel[1:]) else None
    copia, lacunas = {}, []
    if lac is not None:   # a foto do dia não chegou: sai da escoragem; o painel segue com a foto anterior
        usada = dias_painel[dias_painel.index(lac) - 1]
        esc, esc_c = esc[esc["dia"] != lac], esc_c[esc_c["dia"] != lac]
        copia = {lac: usada}
        lacunas = [{"dia": _iso(lac), "motivo": "foto do estoque não chegou (código 3)", "foto_usada": _iso(usada)}]
    dias = sorted(esc["dia"].unique())
    bol = boletos(tabelas, esc, hoje)
    maduro = esc[esc["maduro"]]
    el = _elegiveis(hoje, safras())
    ultimos = set(dias[-DIAS_FILA:])
    bases = {d: montar_base(d, tabelas) for d in safras()}

    def com_lacuna(d: dict) -> dict:
        for sem_foto, usada in copia.items():
            if _iso(usada) in d:
                d[_iso(sem_foto)] = d[_iso(usada)]
        return d

    qualidade = _qualidade(esc, backtest_producao(esc_c), relatorios, hoje)
    dados = {
        "meta": {"hoje": _iso(hoje), "gerado_em": datetime.now().strftime("%d/%m/%Y %H:%M"),
                 "dias": [_iso(d) for d in dias_painel], "fases": list(FAIXAS_ATRASO), "faixas": FAIXAS,
                 "maduro_ate": _iso(maduro["dia"].max()) if len(maduro) else None,
                 "versao_vigente": str(esc.loc[esc["dia"] == dias[-1], "versao"].iloc[0]),
                 "safra_calib": _iso(el[-1]), "safras_treino": [_iso(d) for d in el[-4:-1]],
                 "pontos_conc": PONTOS, "lacunas": lacunas},
        "diario": _diario(esc, bol, dias_painel, copia),
        "conc": com_lacuna(_por_dia_fase(esc, lambda g: [y for _, y in curva_concentracao(g["valor_esperado"], PONTOS)])),
        "conc_real": _por_dia_fase(maduro, lambda g: [y for _, y in curva_realizada(g["valor_esperado"],
                                                                                   g["valor_pago_30d"], PONTOS)]),
        "top10": com_lacuna(_por_dia_fase(esc, lambda g: _r(top_concentra(g["valor_esperado"]), 4))),
        "perfil": com_lacuna(_por_dia_fase(esc, _perfil)),
        "fila": _motivos_fila(_por_dia_fase(esc[esc["dia"].isin(ultimos)], _fila), tabelas, bases),
        "filas_cmp": _recs(comparar_filas(maduro)),
        "qualidade": qualidade,
        "execucao": _execucao(tabelas, bases, dias_painel[-DIAS_EXEC:], lac),
        "sinal": _sinal(tabelas, bases, esc[esc["dia"] == dias[-1]], dias[-1]),
    }
    te = _tempo_exec(esc, esc_c, (qualidade.get("previu_pagou") or {}).get("ece", {}).get("depois"))
    if te:
        dados["tempo_exec"] = te
    vali = Path(relatorios) / "validacao.json"
    if vali.exists():
        dados["validacao"] = json.loads(vali.read_text(encoding="utf-8"))
    return dados


def montar_html(dados: dict) -> str:
    js = json.dumps(dados, ensure_ascii=False, separators=(",", ":"), allow_nan=False)   # NaN quebraria o JS
    return (TEMPLATE.read_text(encoding="utf-8").replace("/*__DADOS__*/", js)
            .replace("__CHARTJS__", CHARTJS).replace("__FONTES__", FONTES))


def main(saida=RAIZ / "docs" / "index.html"):
    dados = montar_dados(carregar(), cache=RAIZ / "dados" / "escoragem")
    (RAIZ / "dados" / "escoragem" / "payload.json").write_text(json.dumps(dados, ensure_ascii=False), encoding="utf-8")
    html = montar_html(dados)
    Path(saida).parent.mkdir(parents=True, exist_ok=True)
    Path(saida).write_text(html, encoding="utf-8")
    print(f"painel gravado em {saida} ({len(html.encode('utf-8')) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
