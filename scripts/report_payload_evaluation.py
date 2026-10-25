"""Export compact tables, figures and a data archive from the two evaluations."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

from payload_metrics import read_csv, write_csv

MODELS = ("Asym_PPO", "DreamWaQ_CENet", "PayloadEstimator")
COLORS = ("#666666", "#377eb8", "#17944a")
LABELS = ("Asym PPO", "CENet", "PayloadEstimator")


def package(root):
    target = root.parent / (root.name + ".zip")
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix in (".csv", ".gz", ".json", ".md", ".png", ".pdf", ".log"):
                archive.write(path, path.relative_to(root),
                              compress_type=zipfile.ZIP_STORED if path.suffix == ".gz" else zipfile.ZIP_DEFLATED)
    print("ARCHIVE", target, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--package-only", action="store_true")
    args = parser.parse_args()
    root = args.results_root.resolve()
    if args.package_only:
        package(root)
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "savefig.dpi": 180})
    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    tables = {}
    for dataset in ("isaaclab_recomputed", "mujoco_fresh"):
        summary = read_csv(root / dataset / "overall_summary.csv")
        tables[dataset] = {r["model"]: r for r in summary}
        compact = []
        for r in summary:
            compact.append(dict(model=r["model"], trials=int(r["trials"]),
                                xy_rmse_m_s=float(r["linear_xy_vector_rmse_mean"]),
                                yaw_rmse_rad_s=float(r["wz_error_rmse_mean"]),
                                roll_std_deg=float(r["roll_std_deg_mean"]), pitch_std_deg=float(r["pitch_std_deg_mean"]),
                                survival_percent=100 * float(r["survival_rate"]),
                                legacy_completion_percent=100 * float(r["legacy_task_completion_rate"]) if r["legacy_task_completion_rate"] else "",
                                payload_mae_kg=r["mass_mae_steady_kg_mean"], payload_rmse_kg=r["mass_rmse_steady_kg_mean"]))
        write_csv(root / dataset / "paper_overall_table.csv", compact)
        conditions = read_csv(root / dataset / "condition_summary.csv")
        terrains = ("flat", "slope", "stairs") if dataset == "isaaclab_recomputed" else ("flat",)
        for plot_name, metrics, units in (
            ("tracking", ("linear_xy_vector_rmse", "wz_error_rmse"), ("XY velocity RMSE (m/s)", "Yaw-rate RMSE (rad/s)")),
            ("posture", ("roll_std_deg", "pitch_std_deg"), ("Roll fluctuation SD (deg)", "Pitch fluctuation SD (deg)"))):
            fig, axes = plt.subplots(2, len(terrains), squeeze=False, figsize=(4.1 * len(terrains), 6.4))
            for col, terrain in enumerate(terrains):
                for model, color, label in zip(MODELS, COLORS, LABELS):
                    selected = sorted((r for r in conditions if r["terrain"] == terrain and r["model"] == model and float(r["cmd_wz"]) == 0), key=lambda r: float(r["mass_kg"]))
                    for row, metric in enumerate(metrics):
                        axes[row, col].errorbar([float(r["mass_kg"]) for r in selected],
                            [float(r[metric + "_mean"]) for r in selected],
                            yerr=[float(r[metric + "_sd"]) for r in selected],
                            color=color, marker="o", capsize=3, label=label)
                axes[0, col].set_title(terrain.capitalize())
                for row in range(2):
                    axes[row, col].set(xlabel="Added payload (kg)", ylabel=units[row], xticks=[0, 4, 8])
                    axes[row, col].grid(alpha=0.2)
            axes[0, 0].legend(fontsize=8)
            engine = "Isaac Lab" if dataset == "isaaclab_recomputed" else "MuJoCo"
            fig.suptitle(f"{engine}: straight command [0.6, 0, 0]; bars = trial SD", fontsize=11)
            fig.tight_layout()
            for suffix in ("png", "pdf"):
                fig.savefig(figures / f"{dataset}_{plot_name}.{suffix}")
            plt.close(fig)
    curves = read_csv(root / "isaaclab_recomputed/mass_timecourse.csv")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for axis, terrain in zip(axes, ("flat", "slope", "stairs")):
        for mass, color in zip((0, 4, 8), ("#377eb8", "#ff7f00", "#984ea3")):
            selected = [r for r in curves if r["terrain"] == terrain and float(r["mass_kg"]) == mass and float(r["cmd_wz"]) == 0]
            x = [float(r["time_bin_start_s"]) for r in selected]
            y = [float(r["predicted_mass_kg_mean"]) for r in selected]
            sd = [float(r["predicted_mass_kg_sd"]) for r in selected]
            axis.plot(x, y, color=color, label=f"Estimate, {mass} kg")
            axis.fill_between(x, [a-b for a,b in zip(y,sd)], [a+b for a,b in zip(y,sd)], color=color, alpha=0.12)
            axis.axhline(mass, color=color, linestyle="--", linewidth=0.9)
        axis.axvline(2, color="grey", linestyle=":", linewidth=1)
        axis.set(title=terrain.capitalize(), xlabel="Policy time (s)", ylabel="Added payload estimate (kg)")
        axis.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("Isaac Lab mass estimates: dashed = truth; shaded = sample SD in 0.1 s bins", fontsize=11)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(figures / f"isaaclab_mass_estimates.{suffix}")
    plt.close(fig)

    definitions = [
        ("linear_xy_vector_rmse", "平面速度向量误差", "m/s", "sqrt(mean((vx-cmd_vx)^2+(vy-cmd_vy)^2)); body frame"),
        ("wz_error_rmse", "偏航角速度误差", "rad/s", "sqrt(mean((wz-cmd_wz)^2))"),
        ("roll_std_deg / pitch_std_deg", "姿态波动", "degree", "Population SD within each scored trial; mean attitude reported separately"),
        ("roll_peak_abs_deg / pitch_peak_abs_deg", "峰值倾角", "degree", "Maximum absolute angle in scored samples"),
        ("survival_rate", "存活率", "ratio", "No failure and observed full protocol duration"),
        ("legacy_task_completion_rate", "原协议通过率", "ratio", "Isaac Lab only: survival AND max world-x progress >=3.2m; turning interpretation limited"),
        ("mass_mae_steady_kg", "稳态质量估计 MAE", "kg", "mean(abs(predicted added mass - actual added mass)); scored window"),
        ("mass_rmse_steady_kg", "稳态质量估计 RMSE", "kg", "sqrt(mean((predicted added mass - actual added mass)^2)); scored window"),
        ("mass_bias_steady_kg", "质量估计偏差", "kg", "mean(prediction-truth); positive means overestimate"),
        ("reference_position_rmse_m", "平面参考轨迹位置误差", "m", "Integrate commands from initial yaw, ignoring terrain pitch; supplementary only"),
        ("*_sd", "试验间标准差", "same as metric", "Sample SD of trial metrics, not training-seed uncertainty"),
    ]
    write_csv(root / "metric_definitions.csv", [dict(field=k, chinese_name=n, unit=u, definition=d) for k,n,u,d in definitions])
    lines = ["# 三个最终模型评测数据", "",
             "本包包含两套独立数据：既有 Isaac Lab 540 次试验的原始轨迹复核，以及本次新运行的 MuJoCo 平地 90 次试验。两套数据不可混合平均。", "",
             "三个方法均只有一个训练 seed（42）。每个工况内的初态配对，统计单位是一次完整试验。", "",
             "## 汇总", ""]
    for dataset, table in tables.items():
        lines += [f"### {dataset}", "", "|方法|次数|XY RMSE m/s|Yaw RMSE rad/s|Roll SD °|Pitch SD °|存活率|原判据通过率|质量 MAE kg|",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for model in MODELS:
            r = table[model]
            completion = f"{100*float(r['legacy_task_completion_rate']):.1f}%" if r["legacy_task_completion_rate"] else "N/A"
            mass = f"{float(r['mass_mae_steady_kg_mean']):.3f}" if r["mass_mae_steady_kg_mean"] else "N/A"
            lines.append(f"|{model}|{r['trials']}|{float(r['linear_xy_vector_rmse_mean']):.5f}|{float(r['wz_error_rmse_mean']):.5f}|{float(r['roll_std_deg_mean']):.3f}|{float(r['pitch_std_deg_mean']):.3f}|{100*float(r['survival_rate']):.1f}%|{completion}|{mass}|")
        lines.append("")
    lines += ["## 测试范围与解释", "",
              "- Isaac Lab：平地、10°斜坡、6级10cm楼梯；0/4/8kg；速度指令[0.6,0,0]和[0.6,0,0.15]；每工况10组配对初态；每条12s，前2s不计入跟踪和姿态评分。共540条、324000行，所有原指标均从原始状态与指令重算。",
              "- 新 MuJoCo：同一 scene_flat.xml，0/4/8kg，同两种指令，每工况5组配对初态；先1s默认关节姿态PD稳定，再12s策略运行（前2s不评分）；0.002s物理步长、0.02s策略周期。增加躯干质量并按比例调整惯量，COM保持不变。",
              "- 新 MuJoCo 使用原 checkpoint 的 PyTorch Actor、同步PD，无DDS/键盘/实机；这是独立迁移验证，不代表完整C++异步链路的性能。",
              "- 质量指标只适用于PayloadEstimator；基线单元格留空，不能记为0。",
              "- 原协议通过率要求世界x方向最大进度>=3.2m；转弯会改变x进度。它作为历史协议指标保留，不等同于经轨迹定义校准的新任务成功率。新MuJoCo没有设置通过阈值，只报告存活及轨迹误差。",
              "- 姿态波动较低可能与低速/停滞有关，需要结合速度误差、进度和完成情况解释。平均姿态、峰值与RMS均在完整CSV中，不能仅凭波动SD声称更稳定。",
              "- 总表是工况平衡的每试验指标均值。总体SD还包含地形/负载/指令差异；工况表SD才表示同工况初态变化。没有多训练seed置信区间。",
              "- reference_* 是忽略地形俯仰的平面参考轨迹误差，仅作补充，不能替代任务通过判据。", "",
              "## 文件", "",
              "- 每套数据的 paper_overall_table.csv：适合直接查看的主表。",
              "- episode_metrics.csv：每次试验的完整指标；condition_summary.csv：各地形/负载/指令下的均值与SD。",
              "- terrain_summary.csv / load_summary.csv：按地形或负载分组；overall_summary.csv：全工况汇总。",
              "- *.csv.gz：原始时序数据，解压后为CSV。",
              "- manifest.json：协议、checkpoint哈希及来源文件哈希；recalculation_checks.csv：逐条原指标核对。",
              "- figures/：PNG与矢量PDF；误差棒为同工况试验SD，质量曲线阴影为0.1s分箱样本SD。",
              "- delivery_manifest.json：当前代码提交和仓库状态。", ""]
    (root / "README.md").write_text("\n".join(lines))
    print("REPORT_COMPLETE", root, flush=True)


if __name__ == "__main__":
    main()
