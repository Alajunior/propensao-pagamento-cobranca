"""Do score à operação: faixas em pirâmide dentro de cada faixa de atraso e simulação de capacidade.

Por que pirâmide POR faixa de atraso? Contratos com pouco atraso pagam muito mais que os antigos. Um corte
único mandaria quase todo o "Diamante" para o atraso curto, e as equipes das faixas antigas
ficariam sem prioridade. Cortando dentro de cada faixa, toda equipe recebe a sua fila ordenada.
"""
import numpy as np
import pandas as pd

FAIXAS_ATRASO = {"0001-0360": (0, 360), "0361-0720": (360, 720), "0721-1080": (720, 1080), "1081+": (1080, np.inf)}
PIRAMIDE = {"Diamante": .10, "Ouro": .20, "Prata": .30, "Bronze": .40}


def faixa_atraso(dias: pd.Series) -> pd.Series:
    cortes = [lim[0] for lim in FAIXAS_ATRASO.values()] + [np.inf]
    return pd.cut(dias, cortes, labels=list(FAIXAS_ATRASO)).astype(str)


def atribuir_faixas(df: pd.DataFrame, col_score: str = "score", piramide: dict = PIRAMIDE) -> pd.Series:
    """Diamante = 10% de maior score dentro da faixa de atraso, depois Ouro 20%, Prata 30%, Bronze 40%."""
    grupo = faixa_atraso(df["dias_atraso"])
    posicao = df.groupby(grupo)[col_score].rank(method="first", ascending=False)   # empate: ordem estável
    posicao = posicao / grupo.map(grupo.value_counts())                            # 0 < posicao <= 1
    limites = np.cumsum(list(piramide.values()))
    idx = np.searchsorted(limites, posicao.to_numpy() - 1e-9)
    return pd.Series(np.array(list(piramide))[np.minimum(idx, len(limites) - 1)], index=df.index)


def tabela_faixas(df: pd.DataFrame, col_score: str = "score", alvo: str = "pagou_30d") -> pd.DataFrame:
    """Taxa E participação por faixa: uma faixa grande pode somar mais pagadores mesmo com taxa menor."""
    d = df.assign(faixa_atraso=faixa_atraso(df["dias_atraso"]), faixa=atribuir_faixas(df, col_score))
    t = d.groupby(["faixa_atraso", "faixa"]).agg(contratos=(alvo, "size"), pagadores=(alvo, "sum"))
    t["taxa_pagamento"] = t["pagadores"] / t["contratos"]
    t["pct_carteira"] = t["contratos"] / t["contratos"].sum()
    t["pct_pagadores"] = t["pagadores"] / t["pagadores"].sum()
    t = t.reset_index()
    t["faixa_atraso"] = pd.Categorical(t["faixa_atraso"], list(FAIXAS_ATRASO), ordered=True)
    t["faixa"] = pd.Categorical(t["faixa"], list(PIRAMIDE), ordered=True)
    return t.sort_values(["faixa_atraso", "faixa"]).astype({"faixa_atraso": str, "faixa": str}) \
        .reset_index(drop=True)


def comparar_estrategias(df: pd.DataFrame, capacidade: float = 0.30, col_score: str = "score",
                         alvo: str = "pagou_30d", valor: str = "valor_pago_30d") -> pd.DataFrame:
    """Com capacidade para contatar `capacidade` da carteira: ranking global pelo score vs. Diamante→Bronze."""
    k = round(capacidade * len(df))
    d = df.assign(_ordem_faixa=atribuir_faixas(df, col_score).map({f: i for i, f in enumerate(PIRAMIDE)}))
    filas = {"ranking_global": d.sort_values(col_score, ascending=False),
             "por_faixa": d.sort_values(["_ordem_faixa", col_score], ascending=[True, False])}
    return pd.DataFrame([{"estrategia": nome, "capacidade": capacidade, "contratos_contatados": k,
                          "pct_pagadores": fila[alvo].head(k).sum() / d[alvo].sum(),
                          "pct_valor": fila[valor].head(k).sum() / d[valor].sum()}
                         for nome, fila in filas.items()])
