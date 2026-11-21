"""Bateria de validação: seis perguntas que um revisor faria antes de confiar no modelo.

Janela fixa: as 3 safras que terminam em `janela_fim` são o treino; a safra SEGUINTE é o teste fora do tempo
(o alvo dela fecha antes de HOJE). Cada teste devolve um veredito ok / atencao / falhou com regra explícita:
    walk_forward          a AUC fora do tempo, safra a safra, não despenca em nenhum mês
    aleatorio_vs_cliente  validação aleatória × agrupada por cliente: mede o otimismo do vazamento
    alvo_embaralhado      com o alvo sorteado, o modelo tem que virar moeda (AUC ~ 0,50)
    gap_treino_teste      o modelo não decorou o treino
    curva_aprendizado     mais dados ainda ajudariam?
    ic_bootstrap          quanto a AUC fora do tempo pode variar só pelo acaso (reamostrando clientes)
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, KFold, cross_val_score

from src.backtest import X, rodar_backtest
from src.base_analitica import ALVO, montar_base, safras
from src.modelos import criar_modelos

FRACOES = (0.10, 0.25, 0.50, 0.75, 1.00)
MESES = ["jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez"]


def _f(x, c=3):
    return f"{x:.{c}f}".replace(".", ",").replace("-", "−")


def _mes(d) -> str:
    d = pd.Timestamp(d)
    return f"{MESES[d.month - 1]}/{d:%y}"


def _iso(d) -> str:
    return f"{pd.Timestamp(d):%Y-%m-%d}"


def _walk_forward(tabelas, relatorios: Path) -> dict:
    arq = Path(relatorios) / "backtest.csv"
    if arq.exists():
        bt = pd.read_csv(arq, parse_dates=["safra_teste"])
    else:
        bt = rodar_backtest(tabelas, modelos={"logistica": criar_modelos()["logistica"]})
    x = bt[(bt["modelo"] == "logistica") & (bt["modo"] == "mensal")].dropna(subset=["AUC"])
    media, pior = float(x["AUC"].mean()), float(x["AUC"].min())
    mes_pior = x.loc[x["AUC"].idxmin(), "safra_teste"]
    return {"id": "walk_forward", "titulo": "Validação fora do tempo, safra a safra",
            "veredito": "ok" if pior >= media - 0.03 else "atencao",
            "resultado": f"AUC média {_f(media)} em {len(x)} safras; a pior foi {_f(pior)} ({_mes(mes_pior)})",
            "detalhe": {"media": media, "pior": pior, "safras": int(len(x))}}


def _aleatorio_vs_cliente(tr: pd.DataFrame, seed: int) -> dict:
    gaps, det = [], {}
    for nome in ("logistica", "lightgbm"):
        m = criar_modelos()[nome]
        rand = cross_val_score(clone(m), tr[X], tr[ALVO], cv=KFold(5, shuffle=True, random_state=seed),
                               scoring="roc_auc").mean()
        grp = cross_val_score(clone(m), tr[X], tr[ALVO], cv=GroupKFold(5), groups=tr["id_cliente"],
                              scoring="roc_auc").mean()
        det[nome] = {"aleatorio": float(rand), "por_cliente": float(grp), "gap": float(rand - grp)}
        gaps.append(rand - grp)
    g = max(gaps)
    veredito = "ok" if g < 0.005 else "atencao" if g <= 0.02 else "falhou"
    return {"id": "aleatorio_vs_cliente", "titulo": "Validação aleatória × separada por cliente",
            "veredito": veredito,
            "resultado": (f"o sorteio aleatório superestima a AUC em até {_f(g * 100, 1)} ponto(s) "
                          f"(logística {_f(det['logistica']['gap'] * 100, 1)}, LightGBM {_f(det['lightgbm']['gap'] * 100, 1)})"),
            "detalhe": det}


def rodar_bateria(tabelas: dict, janela_fim="2026-05-31", n_boot: int = 1000, seed: int = 42,
                  relatorios=Path("relatorios")) -> dict:
    todas = safras()
    i = todas.index(pd.Timestamp(janela_fim))
    treino_ds, teste_d = todas[i - 2:i + 1], todas[i + 1]
    tr = pd.concat(montar_base(d, tabelas) for d in treino_ds)
    te = montar_base(teste_d, tabelas)
    rng = np.random.default_rng(seed)
    janela = {"safras_treino": [_iso(d) for d in treino_ds], "safra_teste": _iso(teste_d)}
    testes = [_walk_forward(tabelas, relatorios), _aleatorio_vs_cliente(tr, seed)]

    # alvo embaralhado (y-scrambling): com o alvo sorteado, a validação cruzada contra esse mesmo alvo tem que dar ~0,50.
    # Se der mais, o pipeline vaza (preparo ajustado na base toda, linha duplicada entre dobras...).
    # Avaliar contra o alvo VERDADEIRO não serve: uma direção aleatória dos coeficientes pode se alinhar ao sinal real.
    auc_emb = float(cross_val_score(clone(criar_modelos()["logistica"]), tr[X], rng.permutation(tr[ALVO].to_numpy()),
                                    cv=KFold(5, shuffle=True, random_state=seed), scoring="roc_auc").mean())
    testes.append({"id": "alvo_embaralhado", "titulo": "Alvo embaralhado (o modelo tem que virar moeda)",
                   "veredito": "ok" if 0.48 <= auc_emb <= 0.52 else "falhou",
                   "resultado": f"AUC {_f(auc_emb)} na validação cruzada com o alvo sorteado (esperado entre 0,48 e 0,52)",
                   "detalhe": {**janela, "auc": auc_emb}})

    # gap treino × teste
    m = clone(criar_modelos()["logistica"]).fit(tr[X], tr[ALVO])
    p_te = m.predict_proba(te[X])[:, 1]
    auc_tr, auc_te = float(roc_auc_score(tr[ALVO], m.predict_proba(tr[X])[:, 1])), float(roc_auc_score(te[ALVO], p_te))
    gap = auc_tr - auc_te
    testes.append({"id": "gap_treino_teste", "titulo": "Treino × fora do tempo (o modelo decorou?)",
                   "veredito": "ok" if gap < 0.03 else "atencao" if gap <= 0.05 else "falhou",
                   "resultado": f"AUC {_f(auc_tr)} no treino e {_f(auc_te)} em {_mes(teste_d)} (diferença {_f(gap * 100, 1)} ponto(s))",
                   "detalhe": {**janela, "auc_treino": auc_tr, "auc_teste": auc_te, "gap": gap}})

    # curva de aprendizado
    curva = []
    for fr in FRACOES:
        amostra = tr.sample(frac=fr, random_state=seed) if fr < 1 else tr
        mf = clone(criar_modelos()["logistica"]).fit(amostra[X], amostra[ALVO])
        curva.append({"fracao": fr, "linhas": int(len(amostra)),
                      "auc": float(roc_auc_score(te[ALVO], mf.predict_proba(te[X])[:, 1]))})
    ganho = curva[-1]["auc"] - curva[2]["auc"]
    testes.append({"id": "curva_aprendizado", "titulo": "Curva de aprendizado (mais dados ajudariam?)",
                   "veredito": "ok" if ganho < 0.005 else "atencao",
                   "resultado": (f"de 50% para 100% do treino a AUC muda {_f(ganho * 100, 1)} ponto(s): "
                                 + ("platô, mais linhas não ajudam" if ganho < 0.005 else "ainda melhora com mais dados")),
                   "detalhe": {"curva": curva, "ganho_50_100": ganho}})

    # IC bootstrap da AUC fora do tempo, reamostrando clientes (contratos do mesmo cliente andam juntos)
    cli = pd.factorize(te["id_cliente"])[0]
    k = cli.max() + 1
    y = te[ALVO].to_numpy()
    aucs = []
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, k, k), minlength=k)[cli]
        if 0 < (w * y).sum() < w.sum():
            aucs.append(roc_auc_score(y, p_te, sample_weight=w))
    lo, hi = (float(v) for v in np.percentile(aucs, [2.5, 97.5]))
    testes.append({"id": "ic_bootstrap", "titulo": "Intervalo de confiança da AUC (bootstrap por cliente)",
                   "veredito": "ok" if hi - lo < 0.02 else "atencao",
                   "resultado": f"AUC {_f(auc_te)} em {_mes(teste_d)}, IC 95% de {_f(lo)} a {_f(hi)}",
                   "detalhe": {"auc": auc_te, "ic": [lo, hi], "reamostragens": len(aucs)}})

    n = {v: sum(t["veredito"] == v for t in testes) for v in ("ok", "atencao", "falhou")}
    resumo = f"{n['ok']} de {len(testes)} testes ok, {n['atencao']} em atenção, {n['falhou']} com falha"
    return {"testes": testes, "resumo": resumo, **janela}
