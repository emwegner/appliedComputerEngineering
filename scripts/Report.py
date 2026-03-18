"""
Report.py  –  Generate an HTML KPI report from processed positioning data.

Usage:
    python Report.py --artifacts-dir artifacts/<run_id>
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (no display needed)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from io import BytesIO
import base64


# ── helpers ──────────────────────────────────────────────────────────────────

def _fig_to_b64(fig) -> str:
    """Encode a matplotlib figure as a base64 PNG string."""
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_estimated_vs_true(df: pd.DataFrame) -> str:
    """Scatter: ground-truth positions vs estimated positions, coloured by setup type."""
    fig, ax = plt.subplots(figsize=(6, 6))

    colors = {"LOS": "#2196F3", "NLOS": "#F44336"}
    for st, grp in df.groupby("setup_type"):
        c = colors.get(st, "#888")
        ax.scatter(grp["x_true"], grp["y_true"],
                   marker="x", s=80, color=c, alpha=0.7, label=f"{st} true")
        ax.scatter(grp["x_est"], grp["y_est"],
                   marker="o", s=40, color=c, alpha=0.4, label=f"{st} est")
        # Draw error lines
        for _, row in grp.iterrows():
            ax.plot([row["x_true"], row["x_est"]],
                    [row["y_true"], row["y_est"]],
                    color=c, alpha=0.25, linewidth=0.8)

    # Draw anchor positions
    anchor_coords = {"AP1": (0, 0), "AP2": (0, 2), "AP3": (2, 2), "AP4": (2, 0)}
    for aid, (ax_, ay_) in anchor_coords.items():
        ax.plot(ax_, ay_, "k^", markersize=10)
        ax.annotate(aid, (ax_, ay_), textcoords="offset points",
                    xytext=(6, 4), fontsize=8, fontweight="bold")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Estimated vs. Ground-Truth Positions")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.4)
    return _fig_to_b64(fig)


def plot_error_cdf(df: pd.DataFrame) -> str:
    """Cumulative Distribution Function of positioning error per setup type."""
    fig, ax = plt.subplots(figsize=(6, 4))

    colors = {"LOS": "#2196F3", "NLOS": "#F44336"}
    for st, grp in df.groupby("setup_type"):
        errors = np.sort(grp["error_m"].values)
        cdf = np.arange(1, len(errors) + 1) / len(errors)
        ax.plot(errors, cdf, color=colors.get(st, "#888"), linewidth=2, label=st)

    ax.axhline(0.90, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.text(ax.get_xlim()[0] if ax.get_xlim()[0] > 0 else 0,
            0.91, "P90", fontsize=8, color="gray")

    ax.set_xlabel("Positioning Error (m)")
    ax.set_ylabel("CDF")
    ax.set_title("CDF of Positioning Error")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    return _fig_to_b64(fig)


def plot_error_by_position(df: pd.DataFrame) -> str:
    """Heatmap-style bar chart: mean error per (x_true, y_true) and setup type."""
    grouped = (df.groupby(["x_true", "y_true", "setup_type"])["error_m"]
               .mean()
               .reset_index())
    grouped["pos_label"] = grouped.apply(
        lambda r: f"({int(r.x_true)},{int(r.y_true)})", axis=1)

    pos_labels = sorted(grouped["pos_label"].unique())
    setup_types = sorted(grouped["setup_type"].unique())
    x = np.arange(len(pos_labels))
    width = 0.35
    colors = {"LOS": "#2196F3", "NLOS": "#F44336"}

    fig, ax = plt.subplots(figsize=(max(6, len(pos_labels) * 1.2), 4))
    for i, st in enumerate(setup_types):
        sub = grouped[grouped["setup_type"] == st].set_index("pos_label")
        vals = [sub.loc[p, "error_m"] if p in sub.index else 0 for p in pos_labels]
        offset = (i - len(setup_types) / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width,
                      label=st, color=colors.get(st, "#888"), alpha=0.8)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(pos_labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Mean Error (m)")
    ax.set_title("Mean Positioning Error per Location")
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    return _fig_to_b64(fig)


# ── HTML builder ──────────────────────────────────────────────────────────────

def build_html(kpis: dict, df: pd.DataFrame, run_id: str) -> str:
    ov = kpis["overall"]
    ni = kpis.get("nlos_impact", {})

    img_scatter = plot_estimated_vs_true(df)
    img_cdf     = plot_error_cdf(df)
    img_bar     = plot_error_by_position(df)

    # Build per-setup-type table rows
    setup_rows = ""
    for st, v in kpis.get("by_setup_type", {}).items():
        setup_rows += f"""
        <tr>
            <td>{st}</td>
            <td>{v['rmse']:.4f}</td>
            <td>{v['p90_accuracy']:.4f}</td>
            <td>{v['mean_error']:.4f}</td>
            <td>{v['median_error']:.4f}</td>
            <td>{v['max_error']:.4f}</td>
            <td>{v['n_estimates']}</td>
        </tr>"""

    nlos_section = ""
    if "los_rmse" in ni:
        deg_class = "spike" if ni['rmse_degradation_m'] > 20 else "ok"
        nlos_section = f"""
        <section>
            <h2>⚠️ NLOS Impact Analysis (Signal Shadow)</h2>
            <div class="kpi-grid">
                <div class="kpi-card">
                    <div class="kpi-label">LOS RMSE</div>
                    <div class="kpi-value">{ni['los_rmse']:.4f} m</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">NLOS RMSE</div>
                    <div class="kpi-value">{ni['nlos_rmse']:.4f} m</div>
                </div>
                <div class="kpi-card {deg_class}">
                    <div class="kpi-label">RMSE Degradation</div>
                    <div class="kpi-value">+{ni['rmse_degradation_m']:.4f} m
                        ({ni['rmse_degradation_m']:.1f}%)</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">LOS P90</div>
                    <div class="kpi-value">{ni['los_p90']:.4f} m</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">NLOS P90</div>
                    <div class="kpi-value">{ni['nlos_p90']:.4f} m</div>
                </div>
            </div>
            <p class="insight">
                {'🚨 <strong>Significant signal shadow detected.</strong> NLOS conditions degrade positioning accuracy by more than 20%. This is consistent with multipath interference caused by environmental obstructions (e.g. metal shelving, walls). The estimated positions inside NLOS zones show systematic bias away from ground truth.' if ni['rmse_degradation_m'] > 20 else '✅ NLOS impact is within acceptable bounds (< 20% RMSE degradation).'}
            </p>
        </section>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>BLE Indoor Positioning – KPI Report</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', sans-serif; background: #f4f6f9; color: #333; }}
  header {{ background: #1565C0; color: white; padding: 24px 32px; }}
  header h1 {{ font-size: 1.6rem; }}
  header p  {{ opacity: 0.85; margin-top: 4px; font-size: 0.9rem; }}
  main {{ max-width: 1100px; margin: 32px auto; padding: 0 24px; }}
  section {{ background: white; border-radius: 8px; padding: 24px; margin-bottom: 24px;
             box-shadow: 0 1px 4px rgba(0,0,0,0.1); }}
  h2 {{ font-size: 1.15rem; margin-bottom: 16px; color: #1565C0; border-bottom: 2px solid #e3eaf5; padding-bottom: 8px; }}
  .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr)); gap: 16px; margin-bottom: 16px; }}
  .kpi-card {{ background: #f0f4ff; border-radius: 6px; padding: 14px 16px; text-align: center; }}
  .kpi-card.spike {{ background: #fff0f0; border: 1px solid #f44336; }}
  .kpi-card.ok    {{ background: #f0fff4; border: 1px solid #4caf50; }}
  .kpi-label {{ font-size: 0.75rem; color: #666; text-transform: uppercase; letter-spacing: 0.5px; }}
  .kpi-value {{ font-size: 1.3rem; font-weight: 700; margin-top: 4px; color: #1565C0; }}
  .kpi-card.spike .kpi-value {{ color: #d32f2f; }}
  .kpi-card.ok    .kpi-value {{ color: #2e7d32; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.88rem; }}
  th {{ background: #1565C0; color: white; padding: 8px 12px; text-align: left; }}
  td {{ padding: 8px 12px; border-bottom: 1px solid #eee; }}
  tr:hover td {{ background: #f5f8ff; }}
  .charts {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
  .charts img {{ width: 100%; border-radius: 6px; border: 1px solid #e0e0e0; }}
  .chart-full img {{ width: 100%; border-radius: 6px; border: 1px solid #e0e0e0; }}
  .insight {{ background: #fff8e1; border-left: 4px solid #ffc107; padding: 12px 16px;
              border-radius: 0 6px 6px 0; font-size: 0.9rem; margin-top: 12px; line-height: 1.5; }}
  footer {{ text-align: center; color: #999; font-size: 0.8rem; padding: 24px; }}
</style>
</head>
<body>
<header>
  <h1>📡 BLE Indoor Positioning – KPI Report</h1>
  <p>Run ID: <strong>{run_id}</strong> &nbsp;|&nbsp; {len(df)} position estimates &nbsp;|&nbsp;
     Setup types: {', '.join(sorted(df['setup_type'].unique()))}</p>
</header>
<main>

  <section>
    <h2>Overall Performance</h2>
    <div class="kpi-grid">
      <div class="kpi-card">
        <div class="kpi-label">RMSE</div>
        <div class="kpi-value">{ov['rmse']:.4f} m</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">P90 Accuracy</div>
        <div class="kpi-value">{ov['p90_accuracy']:.4f} m</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Mean Error</div>
        <div class="kpi-value">{ov['mean_error']:.4f} m</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Median Error</div>
        <div class="kpi-value">{ov['median_error']:.4f} m</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Max Error</div>
        <div class="kpi-value">{ov['max_error']:.4f} m</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Estimates</div>
        <div class="kpi-value">{ov['n_estimates']}</div>
      </div>
    </div>
  </section>

  <section>
    <h2>Performance by Setup Type</h2>
    <table>
      <thead>
        <tr>
          <th>Setup Type</th><th>RMSE (m)</th><th>P90 (m)</th>
          <th>Mean Error (m)</th><th>Median Error (m)</th>
          <th>Max Error (m)</th><th>N Estimates</th>
        </tr>
      </thead>
      <tbody>{setup_rows}</tbody>
    </table>
  </section>

  {nlos_section}

  <section>
    <h2>Visualisations</h2>
    <div class="charts">
      <div><img src="data:image/png;base64,{img_scatter}" alt="Estimated vs True positions"></div>
      <div><img src="data:image/png;base64,{img_cdf}" alt="Error CDF"></div>
    </div>
    <div class="chart-full" style="margin-top:20px">
      <img src="data:image/png;base64,{img_bar}" alt="Error by position">
    </div>
  </section>

</main>
<footer>Generated by BLE Positioning ETL Pipeline &nbsp;·&nbsp; DT212G Applied Computer Engineering</footer>
</body>
</html>"""
    return html


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate HTML KPI report")
    parser.add_argument("--artifacts-dir", required=True,
                        help="Artifacts directory with kpi_results.json and processed_positioning.csv")
    args = parser.parse_args()

    kpi_path  = os.path.join(args.artifacts_dir, "kpi_results.json")
    proc_path = os.path.join(args.artifacts_dir, "processed_positioning.csv")

    with open(kpi_path) as f:
        kpis = json.load(f)

    df = pd.read_csv(proc_path)
    run_id = os.path.basename(os.path.abspath(args.artifacts_dir))

    html = build_html(kpis, df, run_id)

    out_path = os.path.join(args.artifacts_dir, "final_kpi_report.html")
    with open(out_path, "w") as f:
        f.write(html)
    print(f"[report] Saved HTML report to {out_path}")


if __name__ == "__main__":
    main()
