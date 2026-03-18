import os
import json
import argparse
import numpy as np
import pandas as pd
from typing import Dict, Any



#  KPI calculation helpers


def compute_rmse(errors: np.ndarray) -> float:
    """Root Mean Square Error: sqrt(mean(error^2))"""
    if len(errors) == 0:
        return float("nan")
    return float(np.sqrt(np.mean(errors ** 2)))


def compute_percentile(errors: np.ndarray, pct: float) -> float:
    """p-th percentile of error values."""
    if len(errors) == 0:
        return float("nan")
    return float(np.percentile(errors, pct))



#  Main KPI function


def compute_kpis(df: pd.DataFrame) -> Dict[str, Any]:
  
    errors = df["error_m"].values
    kpis: Dict[str, Any] = {}

    # ── Overall ──
    kpis["overall"] = {
        "rmse":         round(compute_rmse(errors),           4),
        "p90_accuracy": round(compute_percentile(errors, 90), 4),
        "p50_accuracy": round(compute_percentile(errors, 50), 4),
        "mean_error":   round(float(np.mean(errors)),         4),
        "median_error": round(float(np.median(errors)),       4),
        "max_error":    round(float(np.max(errors)),          4),
        "min_error":    round(float(np.min(errors)),          4),
        "std_error":    round(float(np.std(errors)),          4),
        "n_estimates":  int(len(errors)),
    }

    # By setup type (LOS / NLOS) 
    kpis["by_setup_type"] = {}
    for setup_type, group in df.groupby("setup_type"):
        e = group["error_m"].values
        kpis["by_setup_type"][setup_type] = {
            "rmse":         round(compute_rmse(e),           4),
            "p90_accuracy": round(compute_percentile(e, 90), 4),
            "p50_accuracy": round(compute_percentile(e, 50), 4),
            "mean_error":   round(float(np.mean(e)),         4),
            "median_error": round(float(np.median(e)),       4),
            "max_error":    round(float(np.max(e)),          4),
            "std_error":    round(float(np.std(e)),          4),
            "n_estimates":  int(len(e)),
        }

    #  By position
    kpis["by_position"] = {}
    for (x, y), group in df.groupby(["x_true", "y_true"]):
        pos_key = f"({int(x)},{int(y)})"
        kpis["by_position"][pos_key] = {}
        for setup_type, sub in group.groupby("setup_type"):
            e = sub["error_m"].values
            kpis["by_position"][pos_key][setup_type] = {
                "rmse":       round(compute_rmse(e),           4),
                "p90":        round(compute_percentile(e, 90), 4),
                "mean_error": round(float(np.mean(e)),         4),
                "x_est":      round(float(sub["x_est"].mean()), 4),
                "y_est":      round(float(sub["y_est"].mean()), 4),
                "n_estimates": int(len(e)),
            }

    # NLOS impact analysis
    los_errors  = df[df["setup_type"] == "LOS"]["error_m"].values
    nlos_errors = df[df["setup_type"] == "NLOS"]["error_m"].values

    if len(los_errors) > 0 and len(nlos_errors) > 0:
        los_rmse  = compute_rmse(los_errors)
        nlos_rmse = compute_rmse(nlos_errors)
        degradation     = nlos_rmse - los_rmse
        degradation_pct = (degradation / los_rmse * 100) if los_rmse > 0 else 0.0
        kpis["nlos_impact"] = {
            "los_rmse":              round(los_rmse,       4),
            "nlos_rmse":             round(nlos_rmse,      4),
            "rmse_degradation_m":    round(degradation,    4),
            "rmse_degradation_pct":  round(degradation_pct, 2),
            "los_p90":               round(compute_percentile(los_errors,  90), 4),
            "nlos_p90":              round(compute_percentile(nlos_errors, 90), 4),
            "los_mean":              round(float(np.mean(los_errors)),  4),
            "nlos_mean":             round(float(np.mean(nlos_errors)), 4),
            "n_los":                 int(len(los_errors)),
            "n_nlos":                int(len(nlos_errors)),
        }
    else:
        kpis["nlos_impact"] = {
            "note": "Insufficient LOS or NLOS data for comparison"
        }

    return kpis



#  Console summary printer


def print_kpi_summary(kpis: Dict[str, Any]):
    """Pretty-print KPI results to stdout."""
    print("\n" + "=" * 60)
    print("  KPI REPORT SUMMARY")
    print("=" * 60)

    ov = kpis["overall"]
    print(f"\n  Overall RMSE:            {ov['rmse']:.4f} m")
    print(f"  90th Percentile (P90):   {ov['p90_accuracy']:.4f} m")
    print(f"  Mean Error:              {ov['mean_error']:.4f} m")
    print(f"  Median Error:            {ov['median_error']:.4f} m")
    print(f"  Max Error:               {ov['max_error']:.4f} m")
    print(f"  Min Error:               {ov['min_error']:.4f} m")
    print(f"  Std Dev:                 {ov['std_error']:.4f} m")
    print(f"  Total Estimates:         {ov['n_estimates']}")

    print(f"\n  --- By Setup Type ---")
    for st, v in kpis["by_setup_type"].items():
        print(f"  {st:>6s}: RMSE={v['rmse']:.4f}m  "
              f"P90={v['p90_accuracy']:.4f}m  "
              f"Mean={v['mean_error']:.4f}m  "
              f"N={v['n_estimates']}")

    print(f"\n  --- By Position ---")
    for pos_key, setups in kpis["by_position"].items():
        for st, v in setups.items():
            print(f"  {pos_key} {st:>6s}: RMSE={v['rmse']:.4f}m  "
                  f"est=({v['x_est']:.4f}, {v['y_est']:.4f})")

    if "nlos_impact" in kpis and "los_rmse" in kpis["nlos_impact"]:
        ni = kpis["nlos_impact"]
        sign = "+" if ni["rmse_degradation_m"] >= 0 else ""
        print(f"\n  --- NLOS Impact ---")
        print(f"  LOS  RMSE:   {ni['los_rmse']:.4f} m  (N={ni['n_los']})")
        print(f"  NLOS RMSE:   {ni['nlos_rmse']:.4f} m  (N={ni['n_nlos']})")
        print(f"  Degradation: {sign}{ni['rmse_degradation_m']:.4f} m "
              f"({sign}{ni['rmse_degradation_pct']:.1f}%)")
        print(f"  LOS  P90:    {ni['los_p90']:.4f} m")
        print(f"  NLOS P90:    {ni['nlos_p90']:.4f} m")

    print("\n" + "=" * 60)



#  Entry point


def main():
    script_dir        = os.path.dirname(os.path.abspath(__file__))
    default_artifacts  = os.path.join(script_dir, "artifacts")

    parser = argparse.ArgumentParser(description="Compute BLE positioning KPIs")
    parser.add_argument("--artifacts-dir", default=default_artifacts,
                        help=f"Artifacts directory containing processed_positioning.csv "
                             f"(default: {default_artifacts})")
    args = parser.parse_args()

    # Load processed positioning data
    proc_path = os.path.join(args.artifacts_dir, "processed_positioning.csv")
    df = pd.read_csv(proc_path)
    print(f"[analytics] Loaded {len(df)} position estimates from {proc_path}")
    print(f"[analytics] Setup types: {df['setup_type'].value_counts().to_dict()}")
    print(f"[analytics] Positions:   {sorted(df[['x_true','y_true']].drop_duplicates().apply(tuple,axis=1).tolist())}")

    # Compute KPIs
    kpis = compute_kpis(df)

    # Print to console
    print_kpi_summary(kpis)

    # Save to JSON
    out_path = os.path.join(args.artifacts_dir, "kpi_results.json")
    with open(out_path, "w") as f:
        json.dump(kpis, f, indent=2)
    print(f"\n[analytics] Saved KPI results → {out_path}")


if __name__ == "__main__":
    main()