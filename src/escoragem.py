"""Escoragem diária como em produção: cada dia útil D é escorado com o modelo VIGENTE em D.

Modelo vigente em D (retreino mensal):
    - elegíveis = safras (fins de mês) cujo alvo (d, d+30] já fechou em D;
    - calibração = a elegível mais recente; treino = as 3 anteriores a ela (sem 4 elegíveis, D não é escorado);
    - logística no treino inteiro + isotônica por fase ajustada na safra de calibração + ticket por fase do treino.
Modo "congelado": o vigente do primeiro dia, nunca retreinado (o contrafactual "e se não retreinássemos?").

A foto de D só sabe o que aconteceu até D. O alvo (pagou em D+1..D+30) só é conhecido quando D+30 <= HOJE:
antes disso o dia não está "maduro" e o alvo fica vazio (NaN), nunca zero.
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone

from src.backtest import X
from src.base_analitica import ALVO, JANELA_ALVO, montar_base, safras
from src.calibracao import aplicar_calibracao, ajustar_calibracao
from src.faixas import atribuir_faixas, faixa_atraso
from src.modelos import criar_modelos
from src.valor import aplicar_ticket, ticket_por_fase

HOJE = pd.Timestamp("2026-11-30")
INICIO = pd.Timestamp("2025-12-01")
COLUNAS = ["dia", "id_contrato", "id_cliente", "fase", "produto", "dias_atraso", "valor_divida", "qtd_cpc_30d",
           "score", "p_cal", "ticket", "valor_esperado", "faixa", "maduro", "pagou_30d", "valor_pago_30d"]


@dataclass
class Vigente:
    modelo: object
    cal: dict
    razao_ticket: dict
    safra_calib: pd.Timestamp
    safras_treino: list = field(default_factory=list)


def _elegiveis(D, datas) -> list[pd.Timestamp]:
    return [d for d in sorted(datas) if d + pd.Timedelta(days=JANELA_ALVO) <= pd.Timestamp(D)]


def treinar_vigente(D, bases: dict) -> Vigente | None:
    el = _elegiveis(D, bases)
    if len(el) < 4:
        return None
    calib, treino_ds = el[-1], el[-4:-1]
    treino = pd.concat(bases[d] for d in treino_ds)
    modelo = clone(criar_modelos()["logistica"]).fit(treino[X], treino[ALVO])
    cb = bases[calib]
    cal = ajustar_calibracao(modelo.predict_proba(cb[X])[:, 1], faixa_atraso(cb["dias_atraso"]), cb[ALVO])
    return Vigente(modelo, cal, ticket_por_fase(treino), calib, list(treino_ds))


def escorar_dia(D, tabelas: dict, vig: Vigente, hoje=HOJE) -> pd.DataFrame:
    D = pd.Timestamp(D)
    e = montar_base(D, tabelas)
    e["dia"] = D
    e["fase"] = faixa_atraso(e["dias_atraso"])
    e["score"] = vig.modelo.predict_proba(e[X])[:, 1]
    e["p_cal"] = aplicar_calibracao(vig.cal, e["score"], e["fase"])
    e["ticket"] = aplicar_ticket(e, vig.razao_ticket).to_numpy()
    e["valor_esperado"] = e["p_cal"] * e["ticket"]
    e["faixa"] = atribuir_faixas(e, "score")
    maduro = D + pd.Timedelta(days=JANELA_ALVO) <= pd.Timestamp(hoje)
    e["maduro"] = maduro
    if not maduro:
        e[ALVO] = np.nan
        e["valor_pago_30d"] = np.nan
    return e[COLUNAS].reset_index(drop=True)


def _cache_nome(cache: Path, modo, inicio, fim, tabelas) -> Path:
    # shortcut: a "impressão digital" da base é só o nº de contratos; apague dados/escoragem/ ao trocar a base
    return Path(cache) / f"escoragem_{modo}_{inicio:%Y%m%d}_{fim:%Y%m%d}_{len(tabelas['contratos'])}.pkl"


def escorar_periodo(tabelas: dict, modo: str = "mensal", inicio=INICIO, fim=HOJE, cache: Path | None = None,
                    hoje=HOJE) -> pd.DataFrame:
    inicio, fim = pd.Timestamp(inicio), pd.Timestamp(fim)
    if cache is not None:
        arq = _cache_nome(cache, modo, inicio, fim, tabelas)
        if arq.exists():
            return pd.read_pickle(arq)
    bases = {d: montar_base(d, tabelas) for d in safras()}
    vigentes: dict[pd.Timestamp, Vigente] = {}
    fixo = None
    partes = []
    for D in pd.bdate_range(inicio, fim):
        if modo == "congelado" and fixo is not None:
            vig = fixo
        else:
            el = _elegiveis(D, bases)
            if len(el) < 4:
                continue
            if el[-1] not in vigentes:
                vigentes[el[-1]] = treinar_vigente(D, bases)
            vig = vigentes[el[-1]]
            fixo = fixo or vig
        e = escorar_dia(D, tabelas, vig, hoje)
        e["versao"] = f"{modo}-{vig.safra_calib:%Y-%m}"
        partes.append(e)
    esc = pd.concat(partes, ignore_index=True)
    esc = esc.astype({c: "category" for c in ("fase", "produto", "faixa", "versao")})   # ~9 mi linhas: memória
    if cache is not None:
        arq.parent.mkdir(parents=True, exist_ok=True)
        esc.to_pickle(arq)
    return esc


def boletos(tabelas: dict, esc: pd.DataFrame, hoje=HOJE) -> pd.DataFrame:
    """Um boleto por PARCELA emitida até hoje, com a fase e a faixa da última foto do contrato até a emissão.

    A parcela j (1, 2, ...) é emitida em data_acordo + 30 × (j − 1) e vence 5 dias depois (como no gerador).
    O acordo para de emitir depois da primeira parcela não paga (quebra). Pagamento depois de hoje = em aberto.
    Contar só a 1ª parcela de acordos novos esconderia quem já tem acordo e paga em dia: justamente o Diamante.
    """
    hoje = pd.Timestamp(hoje)
    ac = tabelas["acordos"]
    pag = tabelas["pagamentos"].sort_values(["id_acordo", "data_pagamento"])
    pag = pag.assign(parcela=pag.groupby("id_acordo").cumcount() + 1)
    n_pago = pag.groupby("id_acordo")["parcela"].max()
    b = ac.loc[ac.index.repeat(ac["qtd_parcelas"])].copy()
    b["parcela"] = b.groupby("id_acordo").cumcount() + 1
    b["dia_emit"] = b["data_acordo"] + pd.to_timedelta(30 * (b["parcela"] - 1), unit="D")
    limite = b["id_acordo"].map(n_pago).fillna(0) + 1      # as pagas + a que quebrou (ou a em aberto)
    b = b[(b["parcela"] <= limite) & (b["dia_emit"] <= hoje)]
    b["rs_emit"] = b["valor_acordo"] / b["qtd_parcelas"]
    fotos = esc[["dia", "id_contrato", "fase", "faixa"]].sort_values("dia")
    b = pd.merge_asof(b.sort_values("dia_emit"), fotos, left_on="dia_emit", right_on="dia", by="id_contrato",
                      direction="backward").dropna(subset=["dia"])
    p = pag[pag["data_pagamento"] <= hoje][["id_acordo", "parcela", "data_pagamento", "valor_pago"]]
    b = b.merge(p, on=["id_acordo", "parcela"], how="left")
    b = b.rename(columns={"dia": "dia_foto", "data_pagamento": "dia_pago", "valor_pago": "rs_pago"})
    b["rs_pago"] = b["rs_pago"].fillna(0.0)
    return b[["id_acordo", "parcela", "id_contrato", "dia_emit", "dia_foto", "fase", "faixa", "rs_emit", "dia_pago",
              "rs_pago"]].reset_index(drop=True)
