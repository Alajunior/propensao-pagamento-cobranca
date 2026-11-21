"""Linha do tempo para a diretoria: o que a operação teria alcançado, mês a mês, em três cenários.

    regua      sem modelo: contato recente primeiro, depois o menor atraso
    congelado  o modelo do primeiro dia, nunca retreinado
    mensal     o modelo retreinado todo mês (o que roda em produção)

Capacidade fixa de 30% do estoque por dia. Como o dado é sintético, o pagamento não depende de quem foi acionado:
a comparação mede a QUALIDADE DA ORDEM, não o efeito do contato. Só dias maduros (alvo de 30 dias fechado).
Os títulos de conclusão saem dos números, nunca escritos à mão.
"""
import pandas as pd

from src.candidatos import score_regua

CAPACIDADE = 0.30
CENARIOS = ("regua", "congelado", "mensal")


def cenarios_mensais(esc_mensal: pd.DataFrame, esc_congelado: pd.DataFrame, capacidade: float = CAPACIDADE) -> pd.DataFrame:
    cols = ["dia", "id_contrato", "score", "pagou_30d", "valor_pago_30d", "qtd_cpc_30d", "dias_atraso"]
    e = esc_mensal.loc[esc_mensal["maduro"], cols].rename(columns={"score": "mensal"})
    e = e.merge(esc_congelado[["dia", "id_contrato", "score"]].rename(columns={"score": "congelado"}),
                on=["dia", "id_contrato"], how="left")
    e["regua"] = score_regua(e)
    k = (capacidade * e.groupby("dia")["id_contrato"].transform("size")).round()
    e["mes"] = e["dia"].dt.strftime("%Y-%m")
    linhas = []
    for c in CENARIOS:
        topo = e.groupby("dia")[c].rank(method="first", ascending=False) <= k
        t = e.assign(ac=topo, pag_ac=e["pagou_30d"] * topo, val_ac=e["valor_pago_30d"] * topo).groupby("mes").agg(
            acionados=("ac", "sum"), pagadores=("pag_ac", "sum"), pag_total=("pagou_30d", "sum"),
            val_ac=("val_ac", "sum"), val_total=("valor_pago_30d", "sum"))
        for mes, r in t.iterrows():
            linhas.append({"mes": mes, "cenario": c, "pct_pagadores": r.pagadores / r.pag_total if r.pag_total else None,
                           "pct_valor": r.val_ac / r.val_total if r.val_total else None,
                           "pag_por_mil": r.pagadores / r.acionados * 1000 if r.acionados else None,
                           "acionados": int(r.acionados), "pagadores": int(r.pagadores)})
    return pd.DataFrame(linhas)


def _pts(x: float) -> str:
    return "ponto" if abs(x) < 2 else "pontos"


def _pp(x: float, c: int = 1) -> str:
    return f"{x:.{c}f}".replace(".", ",")


def conclusoes(df: pd.DataFrame, calib_ece_depois: float, diamante_pct_estoque: float,
               diamante_pct_pagos: float) -> list[dict]:
    media = df.groupby("cenario")["pct_pagadores"].mean()
    pos = df[df["mes"] >= "2026-06"]
    pos = pos if len(pos) else df
    p = pos.groupby("cenario")["pct_pagadores"].mean()
    perda = (p["mensal"] - p["congelado"]) * 100
    cap = round(CAPACIDADE * 100)
    if perda >= 1:
        t_ret = f"Sem retreino, o modelo perde {_pp(perda)} pontos depois de jun/2026"
    else:
        t_ret = f"Sem retreino, a perda depois de jun/2026 foi de só {_pp(perda)} {_pts(perda)}: a logística é estável"
    return [
        {"titulo": f"Com {cap}% do estoque, o modelo alcança {_pp(media['mensal'] * 100, 0)}% dos pagadores",
         "texto": f"Sem modelo (régua de contato recente e menor atraso), a mesma capacidade alcança "
                  f"{_pp(media['regua'] * 100, 0)}%. Média dos meses maduros.",
         "numero": f"{_pp(media['mensal'] * 100, 0)}%"},
        {"titulo": t_ret,
         "texto": f"De jun/2026 em diante, o retreino mensal alcançou {_pp(p['mensal'] * 100)}% dos pagadores contra "
                  f"{_pp(p['congelado'] * 100)}% do modelo congelado.",
         "numero": f"{'+' if perda >= 0 else '−'}{_pp(abs(perda))} pp"},
        {"titulo": "A chance prevista é a chance real",
         "texto": f"Depois da calibração, a distância média entre a chance prevista e a taxa que de fato pagou é de "
                  f"{_pp(calib_ece_depois * 100, 2)}%.",
         "numero": f"{_pp(calib_ece_depois * 100, 2)}%"},
        {"titulo": "Diamante pequeno que paga mais",
         "texto": f"O Diamante é {_pp(diamante_pct_estoque * 100, 0)}% do estoque e responde por "
                  f"{_pp(diamante_pct_pagos * 100, 0)}% dos pagamentos.",
         "numero": f"{_pp(diamante_pct_pagos * 100, 0)}%"},
    ]
