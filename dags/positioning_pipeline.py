from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.trigger_rule import TriggerRule

#PATHS
SCRIPTS_DIR   = "/opt/airflow/scripts"
ARTIFACTS_DIR = "/opt/airflow/artifacts"

RAW_DATA_DIR  = "/opt/airflow/data/collecting_NEW"

default_args = {
    "owner": "ble_pipeline",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
    "email_on_failure": False,
}

#define DAG
with DAG(
    dag_id="ble_positioning_pipeline",
    description="ETL pipeline: BLE telemetry → position estimates → KPI report",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,          # trigger manually; set e.g. "@daily" for automation
    catchup=False,
    tags=["ble", "positioning", "dt212g"],
    doc_md="""
## BLE Indoor Positioning Pipeline

End-to-end ETL DAG for the DT212G Project C assignment.

| Stage | Script | Output |
|---|---|---|
| Collect | Runner.py | raw_telemetry.csv |
| Transform | Transform.py | processed_positioning.csv |
| KPIs | Analytics.py | kpi_results.json |
| Report | Report.py | final_kpi_report.html |

Artifacts are written to `/opt/airflow/artifacts/<run_id>/`.
    """,
) as dag:

    start = EmptyOperator(task_id="start")

    #create run artifact director
    make_artifact_dir = BashOperator(
        task_id="make_artifact_dir",
        bash_command=(
            'mkdir -p {{ params.artifacts_dir }}/{{ run_id }} && '
            'echo "Artifact dir: {{ params.artifacts_dir }}/{{ run_id }}"'
        ),
        params={"artifacts_dir": ARTIFACTS_DIR},
    )

    #extract/collect data
    collect_telemetry = BashOperator(
        task_id="collect_telemetry",
        bash_command=(
            "python {{ params.scripts_dir }}/Runner.py "
            "  --base-dir  {{ params.raw_data_dir }} "
            "  --output-dir {{ params.artifacts_dir }}/{{ run_id }}"
        ),
        params={
            "scripts_dir":   SCRIPTS_DIR,
            "raw_data_dir":  RAW_DATA_DIR,
            "artifacts_dir": ARTIFACTS_DIR,
        },
    )

    #data transform
    transform = BashOperator(
        task_id="transform",
        bash_command=(
            "python {{ params.scripts_dir }}/Transform.py "
            "  --artifacts-dir {{ params.artifacts_dir }}/{{ run_id }}"
        ),
        params={
            "scripts_dir":   SCRIPTS_DIR,
            "artifacts_dir": ARTIFACTS_DIR,
        },
    )

    #compute kpis
    compute_kpis = BashOperator(
        task_id="compute_kpis",
        bash_command=(
            "python {{ params.scripts_dir }}/Analytics.py "
            "  --artifacts-dir {{ params.artifacts_dir }}/{{ run_id }}"
        ),
        params={
            "scripts_dir":   SCRIPTS_DIR,
            "artifacts_dir": ARTIFACTS_DIR,
        },
    )

    #report
    generate_report = BashOperator(
        task_id="generate_report",
        bash_command=(
            "python {{ params.scripts_dir }}/Report.py "
            "  --artifacts-dir {{ params.artifacts_dir }}/{{ run_id }} "
            "&& echo '✅ Report ready at {{ params.artifacts_dir }}/{{ run_id }}/final_kpi_report.html'"
        ),
        params={
            "scripts_dir":   SCRIPTS_DIR,
            "artifacts_dir": ARTIFACTS_DIR,
        },
    )

    #end
    end = EmptyOperator(
        task_id="end",
        trigger_rule=TriggerRule.ALL_DONE,
    )

    # dependencies
    (
        start
        >> make_artifact_dir
        >> collect_telemetry
        >> transform
        >> compute_kpis
        >> generate_report
        >> end
    )
