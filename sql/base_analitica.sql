-- Base analítica de uma safra: 1 linha por contrato da carteira ativa na data :data_ref.
-- Mesma regra de src/base_analitica.py (um teste confere que as duas saídas batem).
--
-- Escrita para SQLite (roda sem instalar nada), no estilo PostgreSQL. Equivalentes no Postgres:
--   datetime(:d, '-30 days')         -> :d::date - INTERVAL '30 days'
--   SUM(CASE WHEN x THEN 1 ELSE 0 END) -> COUNT(*) FILTER (WHERE x)
--   texto dd/mm/aaaa com substr       -> TO_DATE(data_pagamento, 'DD/MM/YYYY')
--   (strftime('%s', a) - strftime('%s', b)) / 86400 -> EXTRACT(DAY FROM a - b)

WITH params AS (
    SELECT datetime(:data_ref) AS d
),

-- população: contratos na carteira em D (entraram há menos de 180 dias) e já vencidos
populacao AS (
    SELECT c.id_contrato,
           c.id_cliente,
           c.data_vencimento,
           (strftime('%s', p.d) - strftime('%s', c.data_vencimento)) / 86400 AS dias_atraso
    FROM contratos c
    CROSS JOIN params p
    WHERE c.data_entrada <= p.d
      AND p.d < datetime(c.data_entrada, '+180 days')
      AND c.data_vencimento < p.d
),

-- a extração tem linhas repetidas: ROW_NUMBER mantém uma por id_acionamento
acion_dedup AS (
    SELECT *
    FROM (
        SELECT a.*,
               ROW_NUMBER() OVER (PARTITION BY a.id_acionamento ORDER BY a.data_hora) AS rn
        FROM acionamentos a
        JOIN populacao pop ON pop.id_contrato = a.id_contrato
    )
    WHERE rn = 1
),

-- histórico de contato: SÓ antes de D
historico AS (
    SELECT a.id_contrato,
           COUNT(*) AS qtd_acion_total,
           SUM(CASE WHEN a.data_hora >= datetime(p.d, '-30 days') THEN 1 ELSE 0 END) AS qtd_acion_30d,
           SUM(CASE WHEN a.resultado IN ('cpc', 'promessa') THEN 1 ELSE 0 END) AS qtd_cpc_total,
           SUM(CASE WHEN a.resultado IN ('cpc', 'promessa')
                     AND a.data_hora >= datetime(p.d, '-30 days') THEN 1 ELSE 0 END) AS qtd_cpc_30d,
           SUM(CASE WHEN a.resultado = 'promessa' THEN 1 ELSE 0 END) AS qtd_promessa_total,
           (strftime('%s', p.d)
            - strftime('%s', MAX(CASE WHEN a.resultado IN ('cpc', 'promessa') THEN a.data_hora END))) / 86400
               AS dias_desde_ult_cpc
    FROM acion_dedup a
    CROSS JOIN params p
    WHERE a.data_hora < p.d
    GROUP BY a.id_contrato
),

-- acordo feito ANTES de D (o status final é do futuro: não entra)
acordo_antes AS (
    SELECT DISTINCT ac.id_contrato
    FROM acordos ac
    CROSS JOIN params p
    WHERE ac.data_acordo < p.d
),

-- alvo: pagamentos > 0 em (D, D+30]; a data vem em texto dd/mm/aaaa do sistema de origem
pagamentos_iso AS (
    SELECT ac.id_contrato,
           pg.valor_pago,
           substr(pg.data_pagamento, 7, 4) || '-' || substr(pg.data_pagamento, 4, 2)
               || '-' || substr(pg.data_pagamento, 1, 2) AS data_pag
    FROM pagamentos pg
    JOIN acordos ac ON ac.id_acordo = pg.id_acordo
    WHERE pg.valor_pago > 0
),
alvo AS (
    SELECT pi.id_contrato, SUM(pi.valor_pago) AS valor_pago_30d
    FROM pagamentos_iso pi
    CROSS JOIN params p
    WHERE pi.data_pag > date(p.d)
      AND pi.data_pag <= date(p.d, '+30 days')
    GROUP BY pi.id_contrato
)

SELECT pop.id_contrato,
       pop.id_cliente,
       pop.dias_atraso,
       COUNT(*) OVER (PARTITION BY pop.id_cliente) AS qtd_contratos_cliente,
       COALESCE(h.qtd_acion_total, 0)    AS qtd_acion_total,
       COALESCE(h.qtd_acion_30d, 0)      AS qtd_acion_30d,
       COALESCE(h.qtd_cpc_total, 0)      AS qtd_cpc_total,
       COALESCE(h.qtd_cpc_30d, 0)        AS qtd_cpc_30d,
       COALESCE(h.qtd_promessa_total, 0) AS qtd_promessa_total,
       h.dias_desde_ult_cpc,
       CASE WHEN aa.id_contrato IS NOT NULL THEN 1 ELSE 0 END AS tem_acordo_antes,
       COALESCE(al.valor_pago_30d, 0)    AS valor_pago_30d,
       CASE WHEN al.id_contrato IS NOT NULL THEN 1 ELSE 0 END AS pagou_30d
FROM populacao pop
LEFT JOIN historico h     ON h.id_contrato = pop.id_contrato
LEFT JOIN acordo_antes aa ON aa.id_contrato = pop.id_contrato
LEFT JOIN alvo al         ON al.id_contrato = pop.id_contrato
ORDER BY pop.id_contrato;
