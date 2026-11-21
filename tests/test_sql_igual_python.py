import sqlite3
from pathlib import Path

import pandas as pd

from src.base_analitica import carregar, montar_base

SQL = Path(__file__).resolve().parent.parent / "sql" / "base_analitica.sql"
COLUNAS = ["id_contrato", "dias_atraso", "qtd_acion_total", "qtd_acion_30d", "qtd_cpc_total",
           "qtd_cpc_30d", "qtd_promessa_total", "dias_desde_ult_cpc", "tem_acordo_antes", "pagou_30d"]


def test_sql_igual_python(pasta_pequena):
    D = "2026-04-30"
    with sqlite3.connect(pasta_pequena / "carteira.db") as con:
        via_sql = pd.read_sql(SQL.read_text(encoding="utf-8"), con, params={"data_ref": D})
    via_py = montar_base(D, carregar(pasta_pequena))
    a = via_sql[COLUNAS].sort_values("id_contrato").reset_index(drop=True)
    b = via_py[COLUNAS].sort_values("id_contrato").reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b, check_dtype=False)
