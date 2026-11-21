"""Backtest out-of-time mês a mês: o que teria acontecido se o modelo estivesse em produção.

Dois modos para cada modelo:
    - congelado: treinado uma vez nas primeiras safras e nunca mais mexido;
    - mensal:    retreinado todo mês com as `janela` safras mais recentes que já têm o alvo fechado.

Para cada safra de teste também registramos:
    - auc_validacao: a AUC num holdout aleatório das safras de treino (o número que "aparece na validação");
    - PSI do score e de duas features contra a janela de treino do modelo congelado (o alarme de monitoramento).
"""
import time

import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.avaliar import metricas, psi
from src.base_analitica import ALVO, CAT, NUM, JANELA_ALVO, montar_base, safras
from src.modelos import criar_modelos

COLUNAS = ["safra_teste", "modelo", "modo", "AUC", "Gini", "KS", "PR-AUC", "captura_top20",
           "auc_validacao", "psi_score", "psi_renda", "psi_dias_atraso", "segundos_treino"]
X = NUM + CAT


def safras_de_treino(d_teste, todas, janela: int = 3) -> list[pd.Timestamp]:
    """As `janela` safras anteriores cujo alvo (D, D+30] já fechou na data de teste. Vazio se não houver."""
    elegiveis = [d for d in todas if d + pd.Timedelta(days=JANELA_ALVO) <= pd.Timestamp(d_teste)]
    return elegiveis[-janela:] if len(elegiveis) >= janela else []


def _treinar(modelo, base: pd.DataFrame):
    """Treina em 70% (estratificado) e devolve o modelo, a AUC no holdout de 30% e o tempo de treino."""
    tr, va = train_test_split(base, test_size=0.3, stratify=base[ALVO], random_state=42)
    t0 = time.perf_counter()
    m = clone(modelo).fit(tr[X], tr[ALVO])
    seg = time.perf_counter() - t0
    return m, roc_auc_score(va[ALVO], m.predict_proba(va[X])[:, 1]), seg


def rodar_backtest(tabelas: dict, janela: int = 3, modelos: dict | None = None) -> pd.DataFrame:
    modelos = modelos or criar_modelos()
    todas = safras()
    bases = {d: montar_base(d, tabelas) for d in todas}
    testes = [d for d in todas if safras_de_treino(d, todas, janela)]
    ref_base = pd.concat(bases[d] for d in safras_de_treino(testes[0], todas, janela))

    linhas = []
    for nome, modelo in modelos.items():
        congelado, auc_val_cong, seg_cong = _treinar(modelo, ref_base)
        ref_score = congelado.predict_proba(ref_base[X])[:, 1]
        for d in testes:
            teste = bases[d]
            score_cong = congelado.predict_proba(teste[X])[:, 1]
            monitor = {"psi_score": psi(ref_score, score_cong),
                       "psi_renda": psi(ref_base["renda_estimada"], teste["renda_estimada"]),
                       "psi_dias_atraso": psi(ref_base["dias_atraso"], teste["dias_atraso"])}
            mensal, auc_val_mensal, seg_mensal = _treinar(
                modelo, pd.concat(bases[s] for s in safras_de_treino(d, todas, janela)))
            for modo, m, auc_val, seg in [("congelado", congelado, auc_val_cong, seg_cong),
                                          ("mensal", mensal, auc_val_mensal, seg_mensal)]:
                p = score_cong if modo == "congelado" else m.predict_proba(teste[X])[:, 1]
                linhas.append({"safra_teste": d, "modelo": nome, "modo": modo,
                               **metricas(teste[ALVO].to_numpy(), p),
                               "auc_validacao": auc_val, **monitor, "segundos_treino": seg})
    return pd.DataFrame(linhas, columns=COLUNAS)
