#!/usr/bin/env bash
set -e

echo "==> Creating databases if they don't exist..."

until pg_isready -U "$POSTGRES_USER" -d "$POSTGRES_DB" >/dev/null 2>&1; do
  sleep 1
done

create_db() {
  local db="$1"
  echo "==> Ensuring database: $db"
  if psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -tAc "SELECT 1 FROM pg_database WHERE datname='${db}'" | grep -q 1; then
    echo "==> $db already exists"
  else
    echo "==> Creating $db"
    createdb -U "$POSTGRES_USER" "$db"
  fi
}

create_db "airflow_db"
create_db "mlflow_db"

echo "==> DB init done."