"""DAG do Airflow para a escoragem diária (o Airflow não é dependência deste repositório; as funções rodam sem ele).

Fluxo, todo dia útil às 5h (antes de a operação começar):
    retreinar -> no 1º dia útil do mês, treina o modelo vigente e grava modelo.pkl + manifest.json (sha256)
    escorar   -> src.producao.escorar_dia_guardado: travas (dia passado, hash, foto ausente, volume, pirâmide)
                 e publicação atômica de escoragem_AAAAMMDD.csv
Código diferente de 0 vira falha da tarefa: o Airflow tenta de novo (retries) e, se continuar, alerta.
A foto ausente (código 3) costuma se resolver sozinha quando a carga do estoque chega: daí os retries de 30 min.
"""
from datetime import datetime, timedelta
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
PASTA_MODELO = RAIZ / "modelos" / "vigente"
PASTA_SAIDA = RAIZ / "dados" / "publicado"

try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
except ImportError:   # airflow não é dependência deste repositório
    DAG = None


def retreinar(ds, **_):
    """No 1º dia útil do mês: treina o vigente (3 safras fechadas + calibração) e grava com manifest."""
    import pandas as pd

    from src.base_analitica import carregar, montar_base, safras
    from src.escoragem import treinar_vigente
    from src.producao import salvar_modelo
    D = pd.Timestamp(ds)
    if D != pd.bdate_range(D.replace(day=1), periods=1)[0] and (PASTA_MODELO / "manifest.json").exists():
        return "modelo do mês já publicado"
    t = carregar()
    return salvar_modelo(treinar_vigente(D, {d: montar_base(d, t) for d in safras()}), PASTA_MODELO)


def escorar(ds, **_):
    from src.base_analitica import carregar
    from src.producao import OK, escorar_dia_guardado
    r = escorar_dia_guardado(ds, carregar(), PASTA_MODELO, PASTA_SAIDA)
    if r["codigo"] != OK:
        raise RuntimeError(f"escoragem de {ds} saiu com código {r['codigo']}: {r['motivo']}")
    return r


if DAG is not None:
    with DAG(
        dag_id="escoragem_diaria_propensao",
        schedule="0 5 * * 1-5",
        start_date=datetime(2026, 1, 1),
        catchup=False,
        default_args={"retries": 3, "retry_delay": timedelta(minutes=30)},
        tags=["cobranca", "propensao"],
    ) as dag:
        PythonOperator(task_id="retreinar", python_callable=retreinar) >> PythonOperator(task_id="escorar", python_callable=escorar)
