"""Roda o projeto inteiro e grava em relatorios/ tudo o que o README mostra.

    python -m src.treinar                     # usa dados/carteira_exemplo/
    mlflow ui                                 # (opcional) compara as execuções registradas

Se o MLflow estiver instalado, cada modelo × modo é registrado no experimento "propensao-pagamento".
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src.avaliar import captura_por_decil  # noqa: E402
from src.backtest import X, _treinar, rodar_backtest, safras_de_treino  # noqa: E402
from src.base_analitica import ALVO, PASTA, carregar, montar_base, safras  # noqa: E402
from src.faixas import comparar_estrategias, tabela_faixas  # noqa: E402
from src.modelos import criar_modelos  # noqa: E402

ARQUIVOS = ["backtest.csv", "comparacao_modelos.csv", "faixas.csv", "estrategias.csv",
            "auc_no_tempo.png", "psi_no_tempo.png", "captura_decil.png"]
ETAPA_1, ETAPA_2 = pd.Timestamp("2026-02-01"), pd.Timestamp("2026-06-01")
MODELO_PRODUCAO = "logistica"   # a decisão do README; os gráficos de drift usam o LightGBM, onde o efeito é maior

# paleta categórica validada (ordem fixa) + tinta de texto e referências neutras
AZUL, LARANJA, VERDE_AGUA = "#2a78d6", "#eb6834", "#1baf7a"
TINTA, TINTA_2, GRADE = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"axes.edgecolor": GRADE, "axes.labelcolor": TINTA_2, "xtick.color": TINTA_2,
                     "ytick.color": TINTA_2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRADE, "grid.linewidth": 0.6, "font.size": 10,
                     "axes.titlesize": 12, "axes.titleweight": "bold", "figure.facecolor": "#fcfcfb",
                     "axes.facecolor": "#fcfcfb", "legend.frameon": False})


def _marcar_etapas(ax):
    for data, texto in [(ETAPA_1, "renda dos novos\nclientes cai"), (ETAPA_2, "relação com o\npagamento muda")]:
        ax.axvline(data, color=TINTA_2, lw=1, ls=":")
        ax.annotate(texto, (data, 1), xycoords=("data", "axes fraction"), xytext=(4, -4),
                    textcoords="offset points", va="top", fontsize=8, color=TINTA_2)


def _linha(ax, x, y, cor, rotulo, **kw):
    ax.plot(x, y, color=cor, lw=2, marker="o", ms=4, label=rotulo, **kw)
    ax.annotate(rotulo, (x.iloc[-1], y.iloc[-1]), xytext=(6, 0), textcoords="offset points",
                va="center", fontsize=8, color=TINTA)


def _graficos(bt: pd.DataFrame, captura: pd.DataFrame, saida: Path, modelo="lightgbm"):
    b = bt[bt.modelo == modelo].sort_values("safra_teste")
    cong, mens = b[b.modo == "congelado"], b[b.modo == "mensal"]

    fig, ax = plt.subplots(figsize=(9, 4.2))
    _linha(ax, cong.safra_teste, cong.AUC, LARANJA, "congelado (OOT)")
    _linha(ax, mens.safra_teste, mens.AUC, AZUL, "retreino mensal (OOT)")
    _linha(ax, cong.safra_teste, cong.auc_validacao, TINTA_2, "validação do congelado", ls="--")
    _marcar_etapas(ax)
    ax.set_title(f"AUC fora do tempo, safra a safra ({modelo})")
    ax.set_ylabel("AUC")
    ax.legend(loc="lower left", fontsize=8)
    ax.margins(x=0.12)
    fig.tight_layout()
    fig.savefig(saida / "auc_no_tempo.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.2))
    _linha(ax, cong.safra_teste, cong.psi_renda, LARANJA, "renda estimada")
    _linha(ax, cong.safra_teste, cong.psi_score, AZUL, "score do modelo")
    _linha(ax, cong.safra_teste, cong.psi_dias_atraso, VERDE_AGUA, "dias de atraso")
    ax.axhline(0.25, color=TINTA_2, lw=1)
    ax.annotate("0,25 = população mudou", (cong.safra_teste.iloc[0], 0.25), xytext=(0, 4),
                textcoords="offset points", fontsize=8, color=TINTA_2)
    _marcar_etapas(ax)
    ax.set_title("PSI contra a base de treino do modelo congelado")
    ax.set_ylabel("PSI")
    ax.legend(loc="upper left", bbox_to_anchor=(0, 0.85), fontsize=8)
    ax.margins(x=0.12)
    fig.tight_layout()
    fig.savefig(saida / "psi_no_tempo.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.2))
    x = pd.concat([pd.Series([0.0]), captura.pct_carteira]) * 100
    ax.plot(x, pd.concat([pd.Series([0.0]), captura.pct_pagadores_acum]) * 100, color=AZUL, lw=2,
            marker="o", ms=4, label="pagadores alcançados")
    ax.plot(x, pd.concat([pd.Series([0.0]), captura.pct_valor_acum]) * 100, color=LARANJA, lw=2,
            marker="o", ms=4, label="valor recuperado alcançado")
    ax.plot([0, 100], [0, 100], color=TINTA_2, lw=1, ls="--", label="sem modelo (aleatório)")
    top20 = captura.loc[captura.decil == 2, "pct_pagadores_acum"].iloc[0] * 100
    ax.annotate(f"top 20% → {top20:.0f}% dos pagadores", (20, top20), xytext=(-12, 10),
                textcoords="offset points", fontsize=8, color=TINTA)
    ax.set_title("Quanto se alcança contatando os X% de maior score")
    ax.set_xlabel("% da carteira contatada (do maior score para o menor)")
    ax.set_ylabel("% acumulado")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(saida / "captura_decil.png", dpi=130)
    plt.close(fig)


def _mlflow(bt: pd.DataFrame):
    try:
        import mlflow
    except ImportError:
        print("aviso: mlflow não instalado; pulando o registro dos experimentos (pip install mlflow)")
        return
    mlflow.set_experiment("propensao-pagamento")
    for (modelo, modo), g in bt.groupby(["modelo", "modo"]):
        with mlflow.start_run(run_name=f"{modelo}-{modo}"):
            mlflow.log_params({"modelo": modelo, "modo": modo, "janela_safras": 3})
            for i, linha in enumerate(g.sort_values("safra_teste").itertuples()):
                mlflow.log_metrics({"auc_oot": linha.AUC, "ks_oot": linha.KS, "psi_renda": linha.psi_renda}, step=i)
            mlflow.log_metrics(g[["AUC", "KS", "PR-AUC", "captura_top20"]].mean().add_suffix("_medio").to_dict())


def main(pasta_dados: Path = PASTA, pasta_saida: Path = Path("relatorios")) -> None:
    saida = Path(pasta_saida)
    saida.mkdir(parents=True, exist_ok=True)
    tabelas = carregar(pasta_dados)

    bt = rodar_backtest(tabelas)
    bt.to_csv(saida / "backtest.csv", index=False)

    # comparação justa: retreino mensal, antes da mudança na relação com o pagamento
    estavel = bt[(bt.modo == "mensal") & (bt.safra_teste < ETAPA_2)]
    comp = estavel.groupby("modelo")[["AUC", "KS", "PR-AUC", "captura_top20", "segundos_treino"]].mean()
    comp.sort_values("AUC", ascending=False).round(4).to_csv(saida / "comparacao_modelos.csv")

    # operação: última safra escorada pelo modelo escolhido para produção, retreinado
    todas = safras()
    ultima = todas[-1]
    treino = pd.concat(montar_base(d, tabelas) for d in safras_de_treino(ultima, todas))
    modelo, _, _ = _treinar(criar_modelos()[MODELO_PRODUCAO], treino)
    base = montar_base(ultima, tabelas)
    base["score"] = modelo.predict_proba(base[X])[:, 1]
    tabela_faixas(base).round(4).to_csv(saida / "faixas.csv", index=False)
    pd.concat([comparar_estrategias(base, c) for c in (0.2, 0.3, 0.5)]).round(4) \
        .to_csv(saida / "estrategias.csv", index=False)

    _graficos(bt, captura_por_decil(base[ALVO], base["score"], base["valor_pago_30d"]), saida)
    _mlflow(bt)
    print(comp.round(3).to_string())
    print(f"relatórios gravados em {saida.resolve()}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else PASTA)
