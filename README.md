**DT212G — Applied Computer Engineering**  
# BLE Indoor Positioning — ETL Pipeline

> Bluetooth Low Energy (BLE) Angle of Arrival positioning system using an automated Apache Airflow pipeline — comparing LOS vs NLOS accuracy.

---

## Overview

This project collects raw BLE radio data from a 2×2 m lab room (Room L211), processes it through a fully automated 5-stage ETL pipeline, and produces positioning accuracy KPIs comparing **Line of Sight (LOS)** vs **Non-Line of Sight (NLOS)** signal conditions.

The pipeline is orchestrated by **Apache Airflow** running inside **Docker**, so the entire stack can be reproduced with a single command on any machine.

---

## Pipeline Architecture

```
01 COLLECT → 02 RUNNER → 03 TRANSFORM → 04 ANALYTICS → 05 REPORT
```

| Stage | Script | Input | Output |
|---|---|---|---|
| Collect | *(physical)* | BLE hardware (u-blox ANNA-B4) | `.pkl` files |
| Runner | `Runner.py` | `collecting_NEW/` folder | `raw_telemetry.csv` |
| Transform | `Transform.py` | `raw_telemetry.csv` | `processed_positioning.csv` |
| Analytics | `Analytics.py` | `processed_positioning.csv` | `kpi_results.json` |
| Report | `Report.py` | `kpi_results.json` + `processed_positioning.csv` | `final_kpi_report.html` |

All output artifacts are written to `artifacts/<run_id>/` — each pipeline run gets its own folder so results are never overwritten.

---

## Project Structure

```
ble_pipeline/
├── dags/
│   └── positioning_pipeline.py   # Airflow DAG definition
├── scripts/
│   ├── Runner.py                 # Stage 2 — Extract .pkl → raw_telemetry.csv
│   ├── Transform.py              # Stage 3 — AoA Least Squares position estimation
│   ├── Analytics.py              # Stage 4 — RMSE, P90, NLOS degradation KPIs
│   └── Report.py                 # Stage 5 — HTML dashboard with embedded charts
├── data/
│   └── collecting_NEW/           # Raw .pkl files from lab collection (not in repo)
├── artifacts/                    # Pipeline outputs per run_id (auto-created)
└── docker-compose.yaml           # Full stack: Postgres + Airflow scheduler + webserver
```

---

## Running the Project

### Prerequisites

- [Docker](https://docs.docker.com) and Docker Compose installed
- Raw `.pkl` data placed under `data/collecting_NEW/`

### Run

```bash
# 1. Clone the repo
git clone <repo-url>

and change into the folder with the project

# 2. Start all services (Postgres + Airflow)
docker compose up -d

# 3. Open the Airflow UI
#    http://localhost:8080
#    username/password: admin

# 4. Trigger the DAG manually:
#    UI → DAGs → ble_positioning_pipeline → Trigger DAG
```

Airflow will run all 5 stages sequentially. The final report is saved to:
```
artifacts/<run_id>/final_kpi_report.html
```

### Seeing the result
```
# 1.  python -m http.server 9000 --directory artifacts

Results are now hosted now displayed at
  http://localhost:9000
```

### Shut down
```
# To shut everything down
  docker compose down
---



