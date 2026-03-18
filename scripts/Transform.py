import os
import json
import argparse
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict


#  Default anchor coordinates (meters)

DEFAULT_ANCHOR_COORDS = {
    "AP1": (0.0, 0.0),
    "AP2": (0.0, 2.0),
    "AP3": (2.0, 2.0),
    "AP4": (2.0, 0.0),
}



#    positioning algorithm


def estimate_position_aoa(
    anchors: list,
    azimuth_angles: list,
) -> Optional[np.ndarray]:
    """
    Estimate tag (x, y) using Angle of Arrival (AoA) via Linear Least Squares.

    For each anchor i at known coordinates (x_i, y_i) with measured azimuth theta_i:
        tan(theta_i) = (y - y_i) / (x - x_i)

    Linearised form:
        x*sin(theta_i) - y*cos(theta_i) = x_i*sin(theta_i) - y_i*cos(theta_i)

    With n >= 2 anchors this becomes an overdetermined system A @ u = B,
    solved via least squares: u = (A^T A)^{-1} A^T B

    Parameters
    ----------
    anchors        : list of (x, y) tuples — known anchor coordinates
    azimuth_angles : list of floats       — measured azimuth angles in degrees

    Returns
    -------
    np.ndarray([x_est, y_est]) or None if system is degenerate
    """
    if len(anchors) < 2 or len(azimuth_angles) < 2:
        return None

    A, B = [], []
    for i in range(len(anchors)):
        x_i, y_i  = anchors[i]
        theta_rad  = np.radians(azimuth_angles[i])
        sin_t      = np.sin(theta_rad)
        cos_t      = np.cos(theta_rad)
        A.append([sin_t, -cos_t])
        B.append(x_i * sin_t - y_i * cos_t)

    A = np.array(A)
    B = np.array(B)

    try:
        pos, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
        return pos
    except np.linalg.LinAlgError:
        return None


def compute_error(x_true: float, y_true: float,
                  x_est: float,  y_est: float) -> float:
    """Euclidean distance error in meters."""
    return float(np.sqrt((x_true - x_est) ** 2 + (y_true - y_est) ** 2))



#  Transform pipeline



def transform_telemetry(
    raw_df: pd.DataFrame,
    anchor_coords: Dict[str, Tuple[float, float]],
) -> pd.DataFrame:
    """
    Transform raw telemetry into one position estimate per (position, setup_type).

    Strategy
    --------
    For each unique (x_true, y_true, setup_type) group:
      1. Compute the mean azimuth across all packets for each anchor.
      2. Run AoA Least Squares on those mean angles.
      3. Record one definitive position estimate and its error.

    Using the mean azimuth (averaged over ~1000 packets) gives the most
    stable and accurate estimate — it suppresses per-packet noise while
    preserving the true signal geometry.

    Parameters
    ----------
    raw_df        : DataFrame from Runner.py (raw_telemetry.csv)
    anchor_coords : dict mapping anchor ID → (x, y)

    Returns
    -------
    pd.DataFrame with one row per (position, setup_type).
    """
    df = raw_df.dropna(subset=["azimuth"]).copy()

    if df.empty:
        print("[transform] WARNING: No packets with valid azimuth found!")
        return pd.DataFrame()

    results = []

    for (x_true, y_true, setup_type), group in df.groupby(["x_true", "y_true", "setup_type"]):

        # here we Collect mean azimuth and signal stats per anchor
        anchor_data = {}
        for anchor_id, adf in group.groupby("anchor_id"):
            if anchor_id not in anchor_coords:
                continue
            anchor_data[anchor_id] = {
                "mean_azimuth": float(adf["azimuth"].mean()),
                "std_azimuth":  float(adf["azimuth"].std()),
                "mean_rssi":    float(adf["rssi"].mean()) if "rssi" in adf.columns else None,
                "n_packets":    len(adf),
            }

        if len(anchor_data) < 2:
            print(f"[transform] SKIP ({x_true},{y_true}) {setup_type} — only {len(anchor_data)} anchor(s)")
            continue

        anchor_ids_used = sorted(anchor_data.keys())
        anchors_list    = [anchor_coords[aid]               for aid in anchor_ids_used]
        angles_list     = [anchor_data[aid]["mean_azimuth"] for aid in anchor_ids_used]

        # Estimate position
        pos = estimate_position_aoa(anchors_list, angles_list)
        if pos is None:
            print(f"[transform] SKIP ({x_true},{y_true}) {setup_type} — solver returned None")
            continue

        x_est, y_est = pos[0], pos[1]
        error_m      = compute_error(x_true, y_true, x_est, y_est)

        rssi_values = [anchor_data[aid]["mean_rssi"] for aid in anchor_ids_used
                       if anchor_data[aid]["mean_rssi"] is not None]

        results.append({
            "x_true":        x_true,
            "y_true":        y_true,
            "x_est":         round(x_est,   4),
            "y_est":         round(y_est,   4),
            "error_m":       round(error_m, 4),
            "setup_type":    setup_type,
            "n_anchors_used": len(anchor_data),
            "anchors_used":  ",".join(anchor_ids_used),
            "mean_rssi":     round(float(np.mean(rssi_values)), 4) if rssi_values else None,
            "total_packets": sum(anchor_data[aid]["n_packets"] for aid in anchor_ids_used),
        })

    out_df = pd.DataFrame(results)
    print(f"[transform] Produced {len(out_df)} position estimates "
          f"(1 per position per setup type)")
    if not out_df.empty:
        print(f"[transform] Setup types: {out_df['setup_type'].value_counts().to_dict()}")
        for _, row in out_df.iterrows():
            print(f"[transform]   ({row['x_true']},{row['y_true']}) {row['setup_type']:4s} "
                  f"→ est=({row['x_est']:.4f},{row['y_est']:.4f})  "
                  f"error={row['error_m']:.4f}m  anchors={row['anchors_used']}")
    return out_df



#  Entry point



def main():
    script_dir       = os.path.dirname(os.path.abspath(__file__))
    default_artifacts = os.path.join(script_dir, "artifacts")

    parser = argparse.ArgumentParser(
        description="Transform raw BLE telemetry into AoA position estimates")
    parser.add_argument("--artifacts-dir", default=default_artifacts,
                        help=f"Artifacts directory containing raw_telemetry.csv "
                             f"(default: {default_artifacts})")
    args = parser.parse_args()

    config_path = os.path.join(args.artifacts_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
        anchor_coords = {k: tuple(v) for k, v in config["anchor_coords"].items()}
        print(f"[transform] Loaded anchor config from {config_path}")
    else:
        anchor_coords = DEFAULT_ANCHOR_COORDS
        print("[transform] config.json not found — using default anchor coordinates")

    # Load raw telemetry
    raw_path = os.path.join(args.artifacts_dir, "raw_telemetry.csv")
    raw_df   = pd.read_csv(raw_path)
    print(f"[transform] Loaded {len(raw_df)} raw packets from {raw_path}")
    print(f"[transform] Positions: {sorted(raw_df[['x_true','y_true']].drop_duplicates().apply(tuple,axis=1).tolist())}")
    print(f"[transform] Setup types: {sorted(raw_df['setup_type'].unique())}")

    # Transform
    result_df = transform_telemetry(raw_df, anchor_coords)

    if result_df.empty:
        print("[transform] ERROR: No estimates produced. Check raw_telemetry.csv.")
        return

    # Save
    out_path = os.path.join(args.artifacts_dir, "processed_positioning.csv")
    result_df.to_csv(out_path, index=False)
    print(f"[transform] Saved → {out_path} ({len(result_df)} rows)")


if __name__ == "__main__":
    main()