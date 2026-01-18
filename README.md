# NextGen Retail AI/ML Project (FastAPI + MLflow + Airflow + Postgres + Alembic + Local Model Versioning)

This repository is an end-to-end **AI/ML project scaffold** for the **NextGen Retail** case study.
It includes:

- **Training pipelines** for 3 models (Recommender, Pricing, Inventory)
- **Model tracking + registry** with **MLflow**
- **Orchestration** with **Airflow**
- **Serving** with **FastAPI** (Swagger docs)
- **DB versioning** with **Alembic** (Postgres)
- **Local model saving + version control** so saved models can be reused later
- Docker Compose to run everything

---

## 1) Folder Structure

```
NextGen_Retail_AI_Project/
├── api/                       # FastAPI service (inference + DB logging)
├── airflow/                   # Airflow DAGs (train -> register)
├── mlflow_server/                    # MLflow server Dockerfile
├── training/                  # Training scripts (log to MLflow + save local versioned models)
├── models/                    # Local model versioning output (auto created at train time)
├── docs/                      # Written documents (expand for submission)
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## 2) Quick Start (Docker)

### Step A: Create `.env`

```bash
cp .env.example .env
```

### Step B: Start services

```bash
docker compose up -d --build
```

Services:

- Postgres: `localhost:5432`
- MLflow: `http://localhost:5000`
- FastAPI: `http://localhost:8000` (Swagger: `/docs`)
- Airflow: `http://localhost:18080` (see credentials section below)

### Airflow login (Standalone)

This project runs Airflow with:

```yaml
command: ["airflow", "standalone"]
```

In **standalone** mode, Airflow creates a default admin user:

- **username:** `admin`
- **password:** auto-generated on first start (printed in container logs and saved to a file in the container)

#### Get the generated password

```bash
docker exec -it nextgen_ai_project-airflow-1 bash -lc "cat /opt/airflow/standalone_admin_password.txt"
```

Or check logs:

```bash
docker logs nextgen_ai_project-airflow-1 --tail 200
```

You will see a line like:

`Login with username: admin  password: <PASSWORD>`

#### Reset password to `admin` (recommended for demos)

```bash
docker exec -it nextgen_ai_project-airflow-1 bash -lc "airflow users reset-password --username admin --password admin"
```

> **Windows PowerShell note:** if you run into quoting issues, keep the command on a single line exactly as above.

After that, log in using **admin/admin**.

---

## 3) Database Versioning (Alembic)

Inside the **api** container:

```bash
alembic -c app/db/alembic.ini upgrade head
```

This creates `inference_logs` table where all API requests/responses are stored.

### If `inference_logs` table is missing (quick fix)

If you see Postgres errors like `relation "inference_logs" does not exist`, you can create the table manually:

```bash
docker exec -it nextgen_ai_project-postgres-1 psql -U nextgen -d nextgen_db -c "
CREATE TABLE IF NOT EXISTS inference_logs (
  id SERIAL PRIMARY KEY,
  endpoint VARCHAR(50) NOT NULL,
  request_json JSON NOT NULL,
  response_json JSON NOT NULL,
  created_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_inference_logs_endpoint ON inference_logs(endpoint);
"
```

---

## 4) Training + Model Versioning (IMPORTANT)

### What happens during training?

Each training script does **two things**:

1. Logs run + artifacts to **MLflow** and registers the model in Model Registry.
2. Saves a **local versioned model** to disk for version control and future reuse.

Local saved structure:

```
models/
  recommender/
    vYYYYMMDD_HHMMSS/
      model.pkl
      meta.json
    current.json
  pricing/
    vYYYYMMDD_HHMMSS/
      model.pkl
      meta.json
    current.json
  inventory/
    vYYYYMMDD_HHMMSS/
      model.pkl
      meta.json
    current.json
```

`current.json` points to the latest version directory (like a simple local registry).

### Train locally (inside container or your Python environment)

```bash
python training/train_recommender.py
python training/train_pricing.py
python training/train_inventory.py
```

### Train via Airflow

Open Airflow UI → trigger DAG `train_and_register_models`.

---

## 5) Serving: Solve Problems Using Saved Models

FastAPI supports **two loading modes**:

### Mode A: Load from MLflow Registry (default)

Set in `.env`:

```
MODEL_SOURCE=mlflow
MODEL_STAGE=Production
```

### Mode B: Load from Local Saved Versioned Models

Set in `.env`:

```
MODEL_SOURCE=local
```

Then the API will read `models/<model_name>/current.json` and load `model.pkl` from that folder.

---

## 6) Test API (examples)

Health:

```bash
curl http://localhost:8000/health
```

Recommend:

```bash
curl -X POST http://localhost:8000/recommend -H "Content-Type: application/json" -d '{"user_id":10,"k":5}'
```

Pricing:

```bash
curl -X POST http://localhost:8000/pricing -H "Content-Type: application/json" -d '{"base_price":50,"demand":120,"stock":180}'
```

Inventory:

```bash
curl -X POST http://localhost:8000/inventory -H "Content-Type: application/json" -d '{"t":100}'
```

Chatbot:

```bash
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d '{"message":"Where is my delivery?"}'
```

---

Thanks
