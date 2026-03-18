import pickle
import re
import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
import pandas as pd



#  Anchor configuration



# Maps IP address (and short integer variants) to anchor ID
ANCHOR_IP_MAP = {
    "192.168.1.103": "AP1",
    "192.168.1.104": "AP2",
    "192.168.1.106": "AP3",
    "192.168.1.102": "AP4",
    "3": "AP1",   
    "4": "AP2",
    "6": "AP3",
    "2": "AP4",
    3: "AP1",
    4: "AP2",
    6: "AP3",
    2: "AP4",
}


# AP1 (192.168.1.103) → (0.0, 0.0)  bottom-left
# AP2 (192.168.1.104) → (0.0, 2.0)  top-left
# AP3 (192.168.1.106) → (2.0, 2.0)  top-right
# AP4 (192.168.1.102) → (2.0, 0.0)  bottom-right
ANCHOR_COORDS = {
    "AP1": (0.0, 0.0),
    "AP2": (0.0, 2.0),
    "AP3": (2.0, 2.0),
    "AP4": (2.0, 0.0),
}

CFG_PAT = re.compile(r"^(-?\d+)_(-?\d+)_ant(\d+)$")



#  Packet field extractors


def _normalize_xy(xy) -> tuple:
    """Extract (x, y) from either tuple or dict format."""
    if isinstance(xy, tuple):
        return xy
    if isinstance(xy, dict):
        return (xy.get("x", 0), xy.get("y", 0))
    return (0, 0)


def _extract_azimuth(pkt: dict) -> Optional[float]:
    """
    Extract azimuth angle from either packet format.
    Format A: direct 'azimuth' field.
    Format B: 'est' list where est[0] = azimuth, est[1] = elevation.
    """
    az = pkt.get("azimuth", None)
    if az is not None:
        return float(az)
    est = pkt.get("est", None)
    if est is not None and len(est) >= 1:
        return float(est[0])
    return None


def _extract_elevation(pkt: dict) -> Optional[float]:
    """Extract elevation angle from either packet format."""
    el = pkt.get("elevation", None)
    if el is not None:
        return float(el)
    est = pkt.get("est", None)
    if est is not None and len(est) >= 2:
        return float(est[1])
    return None


def _resolve_anchor_id(pkt: dict, ip_dir_name: str) -> str:
    """
    Resolve anchor ID from packet fields or directory name.
    Priority: packet ap_ip field → directory IP string.
    """
    ap_ip = pkt.get("ap_ip", None)
    if ap_ip in ANCHOR_IP_MAP:
        return ANCHOR_IP_MAP[ap_ip]
    if str(ap_ip) in ANCHOR_IP_MAP:
        return ANCHOR_IP_MAP[str(ap_ip)]
    for key in ANCHOR_IP_MAP:
        if isinstance(key, str) and key in ip_dir_name:
            return ANCHOR_IP_MAP[key]
    return "UNKNOWN"


def _extract_iq_power(pkt: dict, start: int = 8, stop: int = 80) -> Optional[float]:
    """Compute mean absolute IQ power from iq_samples[start:stop]."""
    iq = pkt.get("iq_samples", None)
    if iq is None:
        return None
    a = np.asarray(iq).ravel()
    if a.size <= start:
        return None
    seg = np.abs(a[start:min(stop, a.size)])
    return float(seg.mean()) if seg.size > 0 else None



#  Main extraction function



def read_all_packets(base_dir: str) -> pd.DataFrame:
    """
    Walk the collecting directory tree and extract all BLE packets into a DataFrame.

    Expected directory structure:
        base_dir/
          <ip_address>/
            <LOS|NLOS>/
              <x>_<y>_ant<n>/
                <orientation>.pkl

    Returns
    -------
    pd.DataFrame with one row per packet.
    """
    base = Path(base_dir).resolve()
    if not base.exists():
        raise FileNotFoundError(f"Base directory not found: {base}")

    rows: List[Dict] = []

    for ip_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        ip_name = ip_dir.name
        for setup_dir in sorted(p for p in ip_dir.iterdir() if p.is_dir()):
            setup_type = setup_dir.name  # LOS or NLOS
            for cfg_dir in sorted(p for p in setup_dir.iterdir() if p.is_dir()):
                m = CFG_PAT.match(cfg_dir.name)
                if not m:
                    continue
                dir_x = int(m.group(1))
                dir_y = int(m.group(2))
                dir_ant = int(m.group(3))

                for pkl_file in sorted(cfg_dir.glob("*.pkl")):
                    orientation = pkl_file.stem
                    try:
                        with open(pkl_file, "rb") as f:
                            pkt_idx = 0
                            while True:
                                try:
                                    pkt = pickle.load(f)
                                except EOFError:
                                    break
                                except Exception:
                                    break

                                if not isinstance(pkt, dict):
                                    continue

                                xy        = _normalize_xy(pkt.get("xy", (dir_x, dir_y)))
                                anchor_id = _resolve_anchor_id(pkt, ip_name)
                                azimuth   = _extract_azimuth(pkt)
                                elevation = _extract_elevation(pkt)
                                iq_power  = _extract_iq_power(pkt)

                                rows.append({
                                    "anchor_id":      anchor_id,
                                    "ip_dir":         ip_name,
                                    "setup_type":     pkt.get("setup_type", setup_type),
                                    "x_true":         xy[0],
                                    "y_true":         xy[1],
                                    "ant_placement":  pkt.get("ant_placement", dir_ant),
                                    "ant_orientation": pkt.get("ant_orientation",
                                                               int(orientation) if orientation.isdigit() else 0),
                                    "rssi":           pkt.get("rssi", None),
                                    "azimuth":        azimuth,
                                    "elevation":      elevation,
                                    "iq_power_mean":  iq_power,
                                    "timestamp":      pkt.get("timestamp", None),
                                    "channel":        pkt.get("channel", pkt.get("ch", None)),
                                    "pkt_index":      pkt_idx,
                                })
                                pkt_idx += 1
                    except Exception as e:
                        print(f"[WARN] Could not read {pkl_file}: {e}")

    df = pd.DataFrame(rows)
    print(f"[runner] Extracted {len(df)} packets from {base_dir}")
    print(f"[runner] Anchors found: {sorted(df['anchor_id'].unique())}")
    print(f"[runner] Setup types: {sorted(df['setup_type'].unique())}")
    print(f"[runner] Tag positions: {sorted(df[['x_true','y_true']].drop_duplicates().apply(tuple, axis=1).tolist())}")
    print(f"[runner] Packets with valid azimuth: {df['azimuth'].notna().sum()} / {len(df)}")
    return df



#  Entry point



def main():
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_base_dir   = os.path.join(script_dir, "collecting_NEW")
    default_output_dir = os.path.join(script_dir, "artifacts")

    parser = argparse.ArgumentParser(description="Extract raw BLE telemetry from pkl files")
    parser.add_argument("--base-dir",   default=default_base_dir,
                        help=f"Path to collecting directory (default: {default_base_dir})")
    parser.add_argument("--output-dir", default=default_output_dir,
                        help=f"Output artifacts directory (default: {default_output_dir})")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = read_all_packets(args.base_dir)

    # Save raw telemetry
    out_path = os.path.join(args.output_dir, "raw_telemetry.csv")
    df.to_csv(out_path, index=False)
    print(f"[runner] Saved raw telemetry → {out_path} ({len(df)} rows)")

    # Save anchor config for downstream stages
    config = {
        "anchor_coords": ANCHOR_COORDS,
        "anchor_ip_map": {k: v for k, v in ANCHOR_IP_MAP.items()
                          if isinstance(k, str) and "." in k},
    }
    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"[runner] Saved config → {config_path}")


if __name__ == "__main__":
    main()