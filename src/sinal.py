"""Explicabilidade para a operação: medir o sinal antes de modelar, os motivos do score e o perfil de cada faixa.

- medir_sinal: cada variável SOZINHA, por fase — quantos pagadores ela acha nos 40% do topo contra o acaso (40%)
  e contra o modelo. Mostra de onde vem o poder do modelo antes de qualquer conta complexa.
- contribuicoes: na logística a explicação é exata, sem SHAP: log-odds = intercepto + Σ coef × z. Medido contra o
  contrato "médio" do treino, cada variável contribui coef × (z − média de z). Somando as colunas transformadas de
  uma mesma variável original (o valor, a flag de nulo, as categorias), sai a contribuição da variável.
- perfil_tipico: as medianas de cada faixa, para a operação saber QUEM está no Diamante.
"""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

from src.backtest import X, safras_de_treino
from src.base_analitica import ALVO, NUM, montar_base, safras
from src.faixas import faixa_atraso
from src.modelos import criar_modelos

TODAS = "__TODAS__"
NOMES = {"dias_atraso": "dias de atraso", "valor_divida": "valor da dívida", "idade": "idade",
         "renda_estimada": "renda estimada", "qtd_acion_total": "acionamentos no total",
         "qtd_acion_30d": "acionamentos em 30 dias", "qtd_cpc_total": "contatos certos no total",
         "qtd_cpc_30d": "contato certo em 30 dias", "qtd_promessa_total": "promessas de pagamento",
         "dias_desde_ult_cpc": "dias desde o último contato certo", "tem_acordo_antes": "já tem acordo",
         "qtd_contratos_cliente": "contratos do cliente", "produto": "produto", "uf": "UF"}
PERFIL = ("dias_atraso", "valor_divida", "idade", "renda_estimada", "qtd_acion_30d", "qtd_cpc_30d",
          "dias_desde_ult_cpc", "tem_acordo_antes")


def _original(col: str) -> str:
    nome = col.split("__", 1)[-1].replace("missingindicator_", "")
    for cat in ("produto", "uf"):
        if nome.startswith(cat + "_"):
            return cat
    return nome


def _z(modelo, X_df: pd.DataFrame) -> pd.DataFrame:
    return modelo[:-1].transform(X_df)


def referencia(modelo, X_treino: pd.DataFrame) -> pd.Series:
    """O contrato "médio" do treino, no espaço transformado que a logística enxerga."""
    return _z(modelo, X_treino).mean()


def contribuicoes(modelo, X_df: pd.DataFrame, ref: pd.Series) -> pd.DataFrame:
    z = _z(modelo, X_df)
    coef = pd.Series(modelo[-1].coef_[0], index=z.columns)
    c = (z - ref[z.columns]) * coef
    return c.T.groupby(c.columns.map(_original)).sum().T


def motivos(contrib: pd.DataFrame, k: int = 2) -> list[list[str]]:
    """As k variáveis que MAIS empurraram a chance para cima (só contribuições positivas)."""
    out = []
    for _, linha in contrib.iterrows():
        pos = linha[linha > 0].sort_values(ascending=False).head(k)
        out.append([NOMES.get(v, v) for v in pos.index])
    return out


def importancia(contrib: pd.DataFrame) -> pd.DataFrame:
    m = contrib.abs().mean().sort_values(ascending=False)
    return pd.DataFrame({"variavel": m.index, "nome": [NOMES.get(v, v) for v in m.index], "media_abs": m.to_numpy()})


def _captura(y: np.ndarray, valor: np.ndarray, pct: float = 0.40) -> float:
    k = max(1, round(pct * len(valor)))
    topo = np.argsort(-valor, kind="stable")[:k]
    return float(y[topo].sum() / y.sum())


def medir_sinal(tabelas: dict, inicio="2025-12-31", fim="2026-10-31") -> pd.DataFrame:
    todas = safras()
    testes = [d for d in todas if pd.Timestamp(inicio) <= d <= pd.Timestamp(fim)]
    precisa = sorted(set(testes) | {s for d in testes for s in safras_de_treino(d, todas, 3)})
    bases = {d: montar_base(d, tabelas) for d in precisa}
    partes = []
    for d in testes:
        treino = pd.concat(bases[s] for s in safras_de_treino(d, todas, 3))
        m = clone(criar_modelos()["logistica"]).fit(treino[X], treino[ALVO])
        t = bases[d]
        partes.append(t[NUM + [ALVO]].assign(fase=faixa_atraso(t["dias_atraso"]), score=m.predict_proba(t[X])[:, 1]))
    pilha = pd.concat(partes, ignore_index=True)
    linhas = []
    for fase, g in [*pilha.groupby("fase"), (TODAS, pilha)]:
        y = g[ALVO].to_numpy()
        if not 0 < y.sum() < len(y):
            continue
        cap_modelo = _captura(y, g["score"].to_numpy())
        for v in NUM:
            x = g[v].fillna(g[v].median()).to_numpy(dtype=float)
            auc = roc_auc_score(y, x)
            sentido = 1 if auc >= 0.5 else -1
            linhas.append({"fase": str(fase), "variavel": v, "nome": NOMES[v], "auc": max(auc, 1 - auc),
                           "direcao": "+" if sentido > 0 else "−", "captura_top40": _captura(y, sentido * x),
                           "captura_modelo": cap_modelo})
    return pd.DataFrame(linhas).sort_values(["fase", "auc"], ascending=[True, False]).reset_index(drop=True)


def perfil_tipico(base_com_faixa: pd.DataFrame, cols=PERFIL) -> dict:
    b = base_com_faixa.assign(fase=faixa_atraso(base_com_faixa["dias_atraso"]))
    out = {}
    for fase, g in [*b.groupby("fase"), (TODAS, b)]:
        d = {}
        for fx, gf in g.groupby("faixa"):
            vals = {c: gf[c].mean() if c == "tem_acordo_antes" else gf[c].median() for c in cols}
            d[str(fx)] = {c: None if pd.isna(v) else round(float(v), 4) for c, v in vals.items()}   # NaN quebra o JS
        out[str(fase)] = d
    return out
