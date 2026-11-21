"""Gera a "Carteira Exemplo": base FICTÍCIA de uma assessoria de cobrança (jan/2025 a dez/2026).

Uso:
    python dados/gerar_carteira.py                 # 100 mil clientes, seed 42
    python dados/gerar_carteira.py 3000            # base menor (usada na CI)
    python dados/gerar_carteira.py 3000 --saida /tmp/carteira

Saída (padrão em dados/carteira_exemplo/):
    clientes.csv, contratos.csv, acionamentos.csv, acordos.csv, pagamentos.csv
    carteira.db  -> as mesmas 5 tabelas em SQLite (para a versão SQL da base)

Como a carteira funciona:
    - cada contrato ENTRA na assessoria numa data (data_entrada) e é trabalhado por 180 dias;
    - a propensão real a pagar é latente (não é salva) e depende de atraso, valor, renda, idade e produto.

Mudança de regime em duas etapas (para o monitoramento ter o que detectar):
    1. a partir de 01/02/2026 entram clientes com renda menor  -> muda a POPULAÇÃO (o PSI dispara)
    2. a partir de 01/06/2026 a relação renda/atraso -> pagamento muda -> o modelo antigo perde ordenação

Sujeiras de propósito (qualidade de dados): renda vazia, UF com caixa/espaços errados,
acionamentos duplicados, pagamentos com valor <= 0 e data de pagamento em texto dd/mm/aaaa.

Tudo é sintético: nenhum dado real de cliente ou de empresa.
"""
import argparse
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

INICIO, FIM = pd.Timestamp("2025-01-01"), pd.Timestamp("2026-12-31")
ETAPA_1 = pd.Timestamp("2026-02-01")   # entra população de renda menor
ETAPA_2 = pd.Timestamp("2026-06-01")   # muda a relação feature -> pagamento
DIAS_NA_CARTEIRA = 180
SAIDA_PADRAO = Path(__file__).parent / "carteira_exemplo"


def _sig(x):
    return 1 / (1 + np.exp(-x))


def gerar(n_clientes: int, saida: Path, seed: int = 42) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    n = n_clientes

    # ---------- clientes (a data de entrada na assessoria é do cliente; todos os contratos dele entram juntos) ----------
    entrada_cli = INICIO + pd.to_timedelta(rng.integers(0, (FIM - INICIO).days + 1, n), "D")
    novos = entrada_cli >= ETAPA_1
    ufs = np.array(["SP", "MG", "RJ", "PR", "BA", "RS", "PE", "GO"])
    clientes = pd.DataFrame({
        "id_cliente": np.arange(1, n + 1),
        "uf": rng.choice(ufs, n, p=[.35, .14, .12, .09, .09, .08, .07, .06]),
        "idade": rng.integers(18, 81, n),
        "renda_estimada": np.round(rng.lognormal(np.where(novos, 7.45, 7.9), 0.6), 2),
        "data_cadastro": pd.Timestamp("2018-01-01") + pd.to_timedelta(rng.integers(0, 2500, n), "D"),
    })

    # ---------- contratos (1 a 3 por cliente: cliente != contrato) ----------
    qtd = rng.choice([1, 2, 3], n, p=[.7, .22, .08])
    ci = np.repeat(np.arange(n), qtd)
    m = len(ci)
    ct = pd.DataFrame({"id_contrato": np.arange(100001, 100001 + m), "id_cliente": clientes.id_cliente.to_numpy()[ci]})
    ct["produto"] = rng.choice(["cartao", "pessoal", "veiculo"], m, p=[.55, .30, .15])
    base_valor = ct.produto.map({"cartao": 7.2, "pessoal": 8.3, "veiculo": 9.6}).to_numpy()
    ct["valor_divida"] = np.round(rng.lognormal(base_valor, 0.7), 2)
    ct["data_entrada"] = entrada_cli[ci]
    ct["data_vencimento"] = ct.data_entrada - pd.to_timedelta(rng.integers(30, 1201, m), "D")
    ct["carteira"] = np.where(ct.produto == "veiculo", "VEICULOS", "VAREJO")

    # Propensão latente (o "segredo" que o modelo tenta descobrir). Não é salva.
    atraso = (ct.data_entrada - ct.data_vencimento).dt.days.clip(upper=900).to_numpy()
    renda = clientes.renda_estimada.to_numpy()[ci]
    idade = clientes.idade.to_numpy()[ci]
    comum = (0.9 - 0.25 * np.log(ct.valor_divida.to_numpy() / 1000)
             + 0.012 * (idade - 40)
             + ct.produto.map({"cartao": 0, "pessoal": 0.1, "veiculo": 0.5}).to_numpy()
             + rng.normal(0, 0.8, m))
    p_v1 = _sig(comum - 0.0035 * atraso + 0.35 * np.log(renda / 2700))
    # etapa 2: o efeito da renda se inverte e o atraso pesa o dobro
    p_v2 = _sig(comum - 0.0070 * atraso - 0.35 * np.log(renda / 2700) + 0.9)   # +0.9 mantém a taxa média parecida
    tel_bom = rng.random(m) < 0.6                                            # 40% dos telefones são ruins

    def p_na_data(idx, datas):
        return np.where(datas >= ETAPA_2, p_v2[idx], p_v1[idx])

    # ---------- acionamentos (só enquanto o contrato está na carteira) ----------
    k = rng.poisson(9, m)
    idx = np.repeat(np.arange(m), k)
    na = len(idx)
    ini = ct.data_entrada.to_numpy()[idx]
    fim_janela = np.minimum(ini + np.timedelta64(DIAS_NA_CARTEIRA, "D"), np.datetime64(FIM + pd.Timedelta(days=1)))
    dur_s = ((fim_janela - ini) / np.timedelta64(1, "s")).astype(np.int64)
    data_hora = pd.to_datetime(ini) + pd.to_timedelta((rng.random(na) * dur_s).astype(np.int64), "s")
    hora = data_hora.hour.to_numpy()
    a = pd.DataFrame({"id_contrato": ct.id_contrato.to_numpy()[idx], "data_hora": data_hora})
    a["canal"] = rng.choice(["discador", "whatsapp", "sms", "email"], na, p=[.6, .2, .12, .08])
    manha = (hora < 12).astype(float)
    p_cpc = np.where(tel_bom[idx], 0.22 + 0.05 * manha, 0.03)
    p_cpc = np.where(a.canal.isin(["sms", "email"]), 0.0, p_cpc)          # sem conversa nesses canais
    u = rng.random(na)
    cpc = u < p_cpc
    alo = (~cpc) & (u < p_cpc + 0.12)
    promessa = cpc & (rng.random(na) < 0.15 + 0.6 * p_na_data(idx, a.data_hora.to_numpy()))
    a["resultado"] = np.select([promessa, cpc, alo], ["promessa", "cpc", "alo"], "nao_atende")
    a["_idx"] = idx
    a = a.sort_values(["id_contrato", "data_hora"]).reset_index(drop=True)
    a.insert(0, "id_acionamento", np.arange(1, len(a) + 1))

    # ---------- acordos (nascem de promessas; no máximo 1 por contrato) ----------
    pr = a[a.resultado == "promessa"]
    pr = pr[rng.random(len(pr)) < 0.7].drop_duplicates("id_contrato", keep="first")
    pidx = pr._idx.to_numpy()
    acordos = pd.DataFrame({
        "id_acordo": np.arange(1, len(pr) + 1),
        "id_contrato": pr.id_contrato.to_numpy(),
        "data_acordo": pr.data_hora.dt.normalize().to_numpy(),
        "valor_acordo": np.round(ct.valor_divida.to_numpy()[pidx] * rng.uniform(0.35, 0.9, len(pr)), 2),
        "qtd_parcelas": rng.choice([1, 2, 3, 6, 10, 12], len(pr), p=[.35, .15, .15, .15, .1, .1]),
    })
    a = a.drop(columns="_idx")

    # ---------- pagamentos (parcela a parcela; parar de pagar = quebra) ----------
    linhas, status = [], []
    for ac, i in zip(acordos.itertuples(index=False), pidx):
        parcela = ac.valor_acordo / ac.qtd_parcelas
        st = "quitado"
        for j in range(ac.qtd_parcelas):
            venc = ac.data_acordo + pd.DateOffset(days=5 + 30 * j)
            if venc > FIM:
                st = "ativo"
                break
            p = p_v2[i] if venc >= ETAPA_2 else p_v1[i]
            if rng.random() < (min(0.95, 0.35 + 0.7 * p) if j == 0 else 0.88):
                pago = venc + pd.Timedelta(days=int(rng.integers(-4, 4)))
                linhas.append((ac.id_acordo, min(pago, FIM), round(parcela, 2)))
            else:
                st = "quebrado"
                break
        status.append(st)
    acordos["status"] = status
    pag = pd.DataFrame(linhas, columns=["id_acordo", "data_pagamento", "valor_pago"])
    pag.insert(0, "id_pagamento", np.arange(1, len(pag) + 1))

    # autoverificação: as relações entre tabelas fecham e o alvo tem sinal (antes das sujeiras)
    assert ct.id_cliente.isin(clientes.id_cliente).all()
    assert acordos.id_contrato.isin(ct.id_contrato).all()
    assert pag.id_acordo.isin(acordos.id_acordo).all()
    taxa = acordos.id_acordo.isin(pag.id_acordo).mean()
    assert 0.3 < taxa < 0.95, taxa

    # ---------- sujeiras de propósito ----------
    limpos = {"clientes": clientes.copy(), "contratos": ct.copy(), "acionamentos": a.copy(),
              "acordos": acordos.copy(), "pagamentos": pag.copy()}
    clientes.loc[rng.random(n) < 0.05, "renda_estimada"] = np.nan
    suj = rng.random(n) < 0.08
    clientes.loc[suj, "uf"] = [rng.choice([f" {x.lower()}", f"{x} ", x.title()]) for x in clientes.uf[suj]]
    a = pd.concat([a, a.sample(frac=0.01, random_state=seed)]).sort_values("id_acionamento")
    pag.loc[pag.sample(frac=0.003, random_state=seed).index, "valor_pago"] = 0.0
    pag["data_pagamento"] = pag.data_pagamento.dt.strftime("%d/%m/%Y")

    # ---------- salvar ----------
    saida = Path(saida)
    saida.mkdir(parents=True, exist_ok=True)
    tabelas = {"clientes": clientes, "contratos": ct, "acionamentos": a, "acordos": acordos, "pagamentos": pag}
    with sqlite3.connect(saida / "carteira.db") as con:
        for nome, df in tabelas.items():
            df.to_csv(saida / f"{nome}.csv", index=False)
            df.to_sql(nome, con, if_exists="replace", index=False)
    print(f"OK - {taxa:.0%} dos acordos tiveram ao menos 1 pagamento | "
          + " | ".join(f"{k} {len(v):,}" for k, v in tabelas.items()))
    return limpos


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("n_clientes", nargs="?", type=int, default=100_000)
    ap.add_argument("--saida", type=Path, default=SAIDA_PADRAO)
    args = ap.parse_args()
    gerar(args.n_clientes, args.saida)
