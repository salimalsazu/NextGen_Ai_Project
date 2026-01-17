from datetime import datetime
from airflow import DAG
from airflow.operators.bash import BashOperator

DEFAULT_ARGS = {"owner": "nextgen", "retries": 0}

with DAG(
    dag_id="train_and_register_models",
    default_args=DEFAULT_ARGS,
    start_date=datetime(2025, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=["nextgen", "mlflow", "versioning"],
) as dag:

    reco = BashOperator(
        task_id="train_recommender",
        bash_command="python /opt/airflow/training/train_recommender.py",
    )
    pricing = BashOperator(
        task_id="train_pricing",
        bash_command="python /opt/airflow/training/train_pricing.py",
    )
    inventory = BashOperator(
        task_id="train_inventory",
        bash_command="python /opt/airflow/training/train_inventory.py",
    )

    reco >> pricing >> inventory
