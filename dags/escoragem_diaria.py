"""Esboço de DAG do Airflow para a escoragem diária (não roda neste repositório; mostra o desenho).

Fluxo, todo dia útil às 5h (antes de a operação começar):
    extrair  -> copia do banco de origem os eventos até ontem (carga incremental, idempotente por data)
    montar   -> montar_base(D) para a carteira ativa de hoje
    escorar  -> aplica o modelo vigente e as faixas em pirâmide
    gravar   -> grava score + faixa numa tabela de saída (delete/insert por data: rodar 2x não duplica)
O monitoramento (PSI do score e das features) roda no fim do `escorar` e alerta se passar de 0,25.
"""
from datetime import datetime, timedelta

try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
except ImportError:   # airflow não é dependência deste repositório
    DAG = None


def extrair(ds, **_):
    """Carga incremental dos eventos de `ds` (data de execução) para a área de staging."""


def montar(ds, **_):
    """from src.base_analitica import montar_base; montar_base(ds) -> parquet da safra do dia."""


def escorar(ds, **_):
    """Carrega o modelo vigente, gera score e faixa, calcula PSI contra a base de treino."""


def gravar(ds, **_):
    """DELETE da data + INSERT dos scores: tarefa idempotente."""


if DAG is not None:
    with DAG(
        dag_id="escoragem_diaria_propensao",
        schedule="0 5 * * 1-5",
        start_date=datetime(2026, 1, 1),
        catchup=False,
        default_args={"retries": 2, "retry_delay": timedelta(minutes=10)},
        tags=["cobranca", "propensao"],
    ) as dag:
        tarefas = [PythonOperator(task_id=f.__name__, python_callable=f) for f in (extrair, montar, escorar, gravar)]
        for antes, depois in zip(tarefas, tarefas[1:]):
            antes >> depois
