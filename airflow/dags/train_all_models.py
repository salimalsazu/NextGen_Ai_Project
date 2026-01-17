from datetime import datetime
from airflow import DAG
from airflow.operators.bash import BashOperator

CSV_PATH = "/opt/airflow/training/data/retail_demo.csv"
MODELS_DIR = "/opt/airflow/models"

with DAG(
    dag_id="train_all_models",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["mlops", "training"],
) as dag:

    train_reco = BashOperator(
        task_id="train_recommender",
        bash_command=(
            "cd /opt/airflow && "
            f"RETAIL_CSV={CSV_PATH} MODELS_DIR={MODELS_DIR} "
            "python -m training.train_recommender"
        ),
    )

    train_pricing = BashOperator(
        task_id="train_pricing",
        bash_command=(
            "cd /opt/airflow && "
            f"RETAIL_CSV={CSV_PATH} MODELS_DIR={MODELS_DIR} "
            "python -m training.train_pricing"
        ),
    )

    train_inventory = BashOperator(
        task_id="train_inventory",
        bash_command=(
            "cd /opt/airflow && "
            f"RETAIL_CSV={CSV_PATH} MODELS_DIR={MODELS_DIR} "
            "python -m training.train_inventory"
        ),
    )

    train_reco >> train_pricing >> train_inventory
