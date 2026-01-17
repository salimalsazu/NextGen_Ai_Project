from datetime import datetime
from airflow import DAG
from airflow.operators.bash import BashOperator

DEFAULT_ARGS = {"owner": "nextgen", "retries": 0}

CSV_PATH = "/opt/airflow/training/data/retail_demo.csv"
MODELS_DIR = "/opt/airflow/models"

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
        bash_command=(
            "cd /opt/airflow && "
            f"RETAIL_CSV={CSV_PATH} MODELS_DIR={MODELS_DIR} "
            "python -m training.train_recommender"
        ),
    )
    pricing = BashOperator(
        task_id="train_pricing",
        bash_command=(
            "cd /opt/airflow && "
            f"RETAIL_CSV={CSV_PATH} MODELS_DIR={MODELS_DIR} "
            "python -m training.train_pricing"
        ),
    )
    inventory = BashOperator(
        task_id="train_inventory",
        bash_command=(
            "cd /opt/airflow && "
            f"RETAIL_CSV={CSV_PATH} MODELS_DIR={MODELS_DIR} "
            "python -m training.train_inventory"
        ),
    )

    reco >> pricing >> inventory
