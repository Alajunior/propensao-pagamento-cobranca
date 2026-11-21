"""Base analítica: 1 linha por contrato da carteira ativa na data de referência D (uma "safra").

Pergunta de negócio: "na data D, quais contratos vão pagar algo nos próximos 30 dias?"
    - população: contratos que estão na carteira em D (entraram há menos de 180 dias) e já vencidos
    - features: só usam o que aconteceu ANTES de D (nada do futuro -> sem leakage)
    - alvo (pagou_30d): 1 se houve pagamento > 0 em (D, D + 30 dias]; valor_pago_30d guarda quanto

Uso:
    from src.base_analitica import carregar, montar_base, safras
    tabelas = carregar()                                   # lê os CSVs uma vez
    base = pd.concat(montar_base(d, tabelas) for d in safras())
"""
from pathlib import Path

import pandas as pd

PASTA = Path(__file__).resolve().parent.parent / "dados" / "carteira_exemplo"
JANELA_ALVO = 30         # dias
DIAS_NA_CARTEIRA = 180

NUM = ["dias_atraso", "valor_divida", "idade", "renda_estimada", "qtd_acion_total",
       "qtd_acion_30d", "qtd_cpc_total", "qtd_cpc_30d", "qtd_promessa_total",
       "dias_desde_ult_cpc", "tem_acordo_antes", "qtd_contratos_cliente"]
CAT = ["produto", "uf"]
ALVO = "pagou_30d"


def safras(inicio="2025-04-30", fim="2026-11-30") -> list[pd.Timestamp]:
    """Fins de mês entre inicio e fim (a última precisa de 30 dias de futuro dentro dos dados)."""
    return list(pd.date_range(inicio, fim, freq="ME"))


def carregar(pasta: Path = PASTA) -> dict[str, pd.DataFrame]:
    """Lê as 5 tabelas e corrige as sujeiras conhecidas."""
    pasta = Path(pasta)
    clientes = pd.read_csv(pasta / "clientes.csv")
    clientes["uf"] = clientes["uf"].str.strip().str.upper()
    contratos = pd.read_csv(pasta / "contratos.csv", parse_dates=["data_vencimento", "data_entrada"])
    acion = pd.read_csv(pasta / "acionamentos.csv", parse_dates=["data_hora"]).drop_duplicates()
    acordos = pd.read_csv(pasta / "acordos.csv", parse_dates=["data_acordo"])
    pag = pd.read_csv(pasta / "pagamentos.csv")
    pag["data_pagamento"] = pd.to_datetime(pag["data_pagamento"], format="%d/%m/%Y")
    pag = pag[pag["valor_pago"] > 0].merge(acordos[["id_acordo", "id_contrato"]], on="id_acordo")
    return {"clientes": clientes, "contratos": contratos, "acionamentos": acion,
            "acordos": acordos, "pagamentos": pag}


def montar_base(data_ref, tabelas: dict | None = None) -> pd.DataFrame:
    D = pd.Timestamp(data_ref)
    t = tabelas if tabelas is not None else carregar()
    ct = t["contratos"]

    # população: carteira ativa em D e já vencida
    ativa = (ct["data_entrada"] <= D) & (D < ct["data_entrada"] + pd.Timedelta(days=DIAS_NA_CARTEIRA))
    base = ct[ativa & (ct["data_vencimento"] < D)].copy()
    base["dias_atraso"] = (D - base["data_vencimento"]).dt.days
    base["qtd_contratos_cliente"] = base.groupby("id_cliente")["id_contrato"].transform("count")
    base = base.merge(t["clientes"][["id_cliente", "uf", "idade", "renda_estimada"]], on="id_cliente", how="left")

    # histórico de contato: SÓ antes de D
    a = t["acionamentos"]
    a = a[(a["data_hora"] < D) & a["id_contrato"].isin(base["id_contrato"])]
    a = a.assign(cpc=a["resultado"].isin(["cpc", "promessa"]),
                 recente=a["data_hora"] >= D - pd.Timedelta(days=30),
                 promessa=a["resultado"] == "promessa")
    a["cpc_30d"] = a["cpc"] & a["recente"]
    hist = a.groupby("id_contrato").agg(
        qtd_acion_total=("id_acionamento", "count"),
        qtd_acion_30d=("recente", "sum"),
        qtd_cpc_total=("cpc", "sum"),
        qtd_cpc_30d=("cpc_30d", "sum"),
        qtd_promessa_total=("promessa", "sum"),
    )
    hist["dias_desde_ult_cpc"] = (D - a[a["cpc"]].groupby("id_contrato")["data_hora"].max()).dt.days
    base = base.merge(hist, on="id_contrato", how="left")
    cont = ["qtd_acion_total", "qtd_acion_30d", "qtd_cpc_total", "qtd_cpc_30d", "qtd_promessa_total"]
    base[cont] = base[cont].fillna(0).astype(int)

    # acordo feito ANTES de D (o status final do acordo é do futuro: não usar!)
    ac = t["acordos"]
    base["tem_acordo_antes"] = base["id_contrato"].isin(ac.loc[ac["data_acordo"] < D, "id_contrato"]).astype(int)

    # alvo: pagou algo em (D, D+30]
    pag = t["pagamentos"]
    janela = pag[(pag["data_pagamento"] > D) & (pag["data_pagamento"] <= D + pd.Timedelta(days=JANELA_ALVO))]
    valor = janela.groupby("id_contrato")["valor_pago"].sum()
    base["valor_pago_30d"] = base["id_contrato"].map(valor).fillna(0.0)
    base[ALVO] = (base["valor_pago_30d"] > 0).astype(int)

    base.insert(0, "data_ref", D)
    return base.reset_index(drop=True)


if __name__ == "__main__":
    tabelas = carregar()
    for d in safras():
        df = montar_base(d, tabelas)
        print(f"{d:%Y-%m-%d}: {len(df):>6,} contratos | {ALVO} = {df[ALVO].mean():.2%}")
