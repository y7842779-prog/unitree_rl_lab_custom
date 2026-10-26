"""Audit every rolling-payload trial and export mean/SD curves and loss counts."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from rolling_payload_support import (MODELS, METRICS, COURSES, CargoMonitor, aggregate,
    course_metrics, metrics, observation, sha, terrain_state, write_csv, write_json)

LABELS = dict(zip(MODELS,("Asym PPO","DreamWaQ-CENet","Ours")))


def read(path):
    with path.open(encoding="utf-8-sig") as handle:
        return [{k:v if k=="model" or v=="" else float(v) for k,v in r.items()} for r in csv.DictReader(handle)]


def display(value, digits=6):
    return f"{float(value):.{digits}f}" if value!="" else "NA"


def main(args):
    run = args.run_root
    manifest = json.loads((run/"manifest.json").read_text())
    protocol = json.loads((run/"protocol.json").read_text())
    terrain = protocol["args"].get("terrain","flat")
    course = COURSES[terrain]
    if terrain != "flat":
        assert protocol["course"] == course
    assert manifest["status"] == "complete"
    if not args.allow_smoke:
        assert manifest["trajectories"]==120 and manifest["trials_per_condition"]==10
        assert sorted(manifest["masses"])==[2.,4.,6.,8.]
        assert protocol["args"]["duration"]==12. and protocol["args"]["warmup"]==2.
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    episodes,conditions,inputs,anomalies,groups = [],[],[],[],{}
    largest_metric_error = largest_gravity_error = largest_roll_error = 0.
    observation_checks = 0
    for mass in sorted(manifest["masses"]):
        reference = None
        for label in MODELS:
            case = run/f"mass_{mass:g}kg"/label
            meta = json.loads((case/"manifest.json").read_text())
            assert meta["status"]=="complete" and meta["exact_integration_state_pairing"]
            assert meta["physics"]["base_mass_kg"]==6.921
            assert meta["physics"]["sphere_mass_kg"]==mass
            np.testing.assert_allclose(meta["physics"]["sphere_inertia"],.4*mass*.04**2,rtol=0,atol=1e-12)
            if reference is None:
                reference = meta["initial_state_sha256"]
            assert reference == meta["initial_state_sha256"]
            group = []
            for trial in range(manifest["trials_per_condition"]):
                path = case/f"trial_{trial:02d}.csv"
                rows = read(path)
                assert rows and [r["step"] for r in rows]==list(range(len(rows)))
                assert all(r["true_mass_kg"]==mass and r["actual_base_mass_kg"]==6.921 for r in rows)
                np.testing.assert_allclose([[r["cmd_vx"],r["cmd_vy"],r["cmd_wz"]] for r in rows],
                                          np.tile([.6,0,0],(len(rows),1)),rtol=0,atol=1e-6)
                q = np.array([[r[k] for k in ("qw","qx","qy","qz")] for r in rows])
                w,x,y,z = q.T
                g = np.column_stack((2*(w*y-x*z),-2*(y*z+w*x),-(1-2*(x*x+y*y))))
                largest_gravity_error = max(largest_gravity_error,float(np.max(np.abs(g-
                    [[r[k] for k in ("gx","gy","gz")] for r in rows]))))
                independent_roll = np.arctan2(2*(w*x+y*z),1-2*(x*x+y*y))
                largest_roll_error = max(largest_roll_error,float(np.max(np.abs(independent_roll-[r["roll"] for r in rows]))))
                monitor = CargoMonitor()
                for row in rows:
                    assert row["cargo_lost"]==monitor.update(row,row["time_after_s"])
                    assert row["scored"]==int(row["time_before_s"]>=protocol["args"]["warmup"]-1e-9)
                    if terrain != "flat":
                        local = terrain_state(row["base_x"],row["base_y"],row["height"],course)
                        for key,value in local.items():
                            assert abs(row[key]-value)<1e-10,(path,key,row[key],value)
                        failed = (local["base_terrain_clearance_m"]<.15 or
                                  row["tilt_cosine"]<math.cos(1.))
                        assert row["terminated"]==int(failed)
                calculated = metrics(rows,protocol["args"]["duration"],protocol["args"]["warmup"])
                if terrain != "flat":
                    calculated.update(course_metrics(rows,course,protocol["args"]["warmup"],
                                                     calculated["transport_completed"]))
                saved = json.loads((case/f"trial_{trial:02d}_metrics.json").read_text())
                for key,value in calculated.items():
                    if value=="":
                        assert saved[key]==""
                    else:
                        difference = abs(saved[key]-value)
                        largest_metric_error = max(largest_metric_error,difference)
                        assert difference < 1e-10,(path,key,difference)
                # Independently recompute the three formulas without shared metric code.
                selected = [r for r in rows if r["time_before_s"]>=protocol["args"]["warmup"]-1e-9 and not r["cargo_lost"]]
                if selected and not calculated["invalid_samples"]:
                    independent = (float(np.sqrt(np.mean([r["roll"]**2 for r in selected]))),
                        float(np.mean([r["gx"]**2+r["gy"]**2 for r in selected])),
                        float(np.sqrt(np.mean([(r["vx"]-r["cmd_vx"])**2+(r["vy"]-r["cmd_vy"])**2 for r in selected]))))
                    np.testing.assert_allclose([calculated[k] for k in METRICS],independent,rtol=0,atol=1e-12)
                obs_path = case/f"trial_{trial:02d}_initial_observation.npz"
                if obs_path.exists():
                    with np.load(obs_path) as obs:
                        length = {"Asym_PPO":0,"DreamWaQ_CENet":5,"PayloadEstimator":25}[label]
                        expected,_ = observation(obs["current"],[],length)
                        np.testing.assert_array_equal(expected,obs["observation"])
                        if label!="Asym_PPO":
                            with np.load(run/f"mass_{mass:g}kg"/"Asym_PPO"/obs_path.name) as reference_obs:
                                np.testing.assert_array_equal(obs["current"],reference_obs["current"])
                        observation_checks += 1
                else:
                    assert args.allow_smoke
                episode = dict(model=label,mass_kg=mass,trial=trial,evaluation_seed=rows[0]["evaluation_seed"],
                               training_seed=42,**calculated)
                group.append(episode)
                episodes.append(episode)
                if (not calculated["transport_completed"] or calculated["stall_flag"] or
                    calculated["invalid_samples"] or
                    (terrain != "flat" and not calculated["course_completed"])):
                    anomalies.append(episode)
                inputs.append(dict(path=str(path.resolve()),sha256=sha(path),rows=len(rows)))
            groups[mass,label] = group
            conditions.append(dict(mass_kg=mass,model=label,**aggregate(group)))
    assert len(episodes)==manifest["trajectories"] and largest_gravity_error<1e-10 and largest_roll_error<1e-10
    write_csv(args.output_dir/"episode_metrics.csv",episodes)
    write_csv(args.output_dir/"condition_summary.csv",conditions)
    write_csv(args.output_dir/"raw_inputs.csv",inputs)
    if anomalies:
        write_csv(args.output_dir/"anomalies.csv",anomalies)
    else:
        (args.output_dir/"anomalies.csv").write_text("model,mass_kg,trial,failed,cargo_lost,stall_flag,invalid_samples\n")
    lookup = {(r["mass_kg"],r["model"]):r for r in conditions}
    improvements = []
    for mass in sorted(manifest["masses"]):
        for baseline in MODELS[:2]:
            ours,other = lookup[mass,"PayloadEstimator"],lookup[mass,baseline]
            for metric in METRICS:
                a,b = ours[metric+"_mean"],other[metric+"_mean"]
                pairs = [(o,r) for o,r in zip(groups[mass,"PayloadEstimator"],groups[mass,baseline])
                         if ((o["course_completed"] and r["course_completed"])
                             if terrain != "flat" else
                             (o["transport_completed"] and r["transport_completed"]))]
                improvements.append(dict(mass_kg=mass,baseline=baseline,metric=metric,ours_mean=a,baseline_mean=b,
                    relative_improvement_percent=100*(b-a)/b if a!="" and b!="" and b else "",
                    ours_transport_completed=ours["transport_completed"],baseline_transport_completed=other["transport_completed"],
                    ours_course_completed=ours.get("course_completed",""),
                    baseline_course_completed=other.get("course_completed",""),
                    complete_pairs=len(pairs),paired_win_count=sum(o[metric]<r[metric] for o,r in pairs)))
    write_csv(args.output_dir/"relative_improvements.csv",improvements)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,
        "axes.spines.right":False,"pdf.fonttype":42,"ps.fonttype":42})
    names = ("Base Roll Deviation (rad)","Base Flatness Penalty (dimensionless)","Velocity Tracking Error (m/s)")
    for metric,ylabel in zip(METRICS,names):
        fig,ax = plt.subplots(figsize=(5.4,3.8),layout="constrained")
        for label,color,marker in zip(MODELS,("#0072B2","#E69F00","#009E73"),("o","s","^")):
            selected = [r for r in conditions if r["model"]==label]
            ax.errorbar([r["mass_kg"] for r in selected],
                [r[metric+"_mean"] if r[metric+"_mean"]!="" else np.nan for r in selected],
                yerr=[r[metric+"_sd"] if r[metric+"_sd"]!="" else 0. for r in selected],
                label=LABELS[label],color=color,marker=marker,linewidth=1.5,capsize=3)
        ax.set(xlabel="Payload Mass (kg)",ylabel=ylabel,xticks=sorted(manifest["masses"]),
               title=("Rolling sphere: mean ± trial SD" if terrain=="flat" else
                      ("Rolling sphere: 5° slope" if terrain=="slope" else "Rolling sphere: 4 × 4 cm stairs")))
        ax.grid(axis="y",alpha=.22)
        ax.legend(frameon=False,fontsize=9)
        for extension in ("png","pdf"):
            fig.savefig(args.output_dir/f"{metric}.{extension}",dpi=300)
        plt.close(fig)
    report = [f"# MuJoCo 滚动小球负载：{terrain} 三模型对比\n",
        f"本次新轨迹 {len(episodes)} 条，评估种子 {protocol['args']['seed']}，训练种子均为42。",
        "只在球上施加2/4/6/8 kg质量，base保持6.921 kg；球半径0.04m，惯量由编译器根据球质量计算。挡板未额外计重，保留原XML碰撞和摩擦。",
        "每负载/初态先共同站立PD预稳定2s，再恢复同一完整积分状态运行各策略；指令[0.6,0,0]，控制50Hz，物理步长0.002s。",
        f"每条{protocol['args']['duration']:g}s，前{protocol['args']['warmup']:g}s过渡；Roll RMS、单位重力水平分量平方均值、机体平面速度向量RMSE，均先逐轨迹计算，再取均值±样本SD。",
        "发生掉球时主要指标仅统计掉球前的满载窗口，并标记截断。原始数据和 full_window_* 辅助指标保留掉球后状态；不得把截短或失载后的低误差排名为更优。",
        "本实验独立于Isaac Lab固定base质量数据；不能将不同引擎的数值差异归因于负载形式。未重新训练、未消融、未修改checkpoint。",
        "\n|负载kg|模型|Roll RMS rad|Flatness|速度RMSE m/s|前进速度m/s|运输完成|掉球|摔倒|停滞|有效样本范围|",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|" ]
    for r in conditions:
        vals = [display(r[m+"_mean"])+" ± "+display(r[m+"_sd"]) for m in METRICS]
        report.append(f"|{r['mass_kg']:g}|{LABELS[r['model']]}|"+"|".join(vals)+
            f"|{display(r['mean_vx_mean'],4)}|{r['transport_completed']}/{r['trials']}|{r['cargo_losses']}|{r['robot_failures']}|{r['stalls']}|{r['scored_samples_min']}–{r['scored_samples_max']}|")
    if terrain != "flat":
        report += ["\n地形从世界x=1.5m开始；斜坡5°、水平长3m、过顶目标x=4.9m；楼梯4级×4cm、踏面40cm、过顶目标x=3.1m。末端接连续平台。",
            f"地形完成=机器人和球完整{protocol['args']['duration']:g}s且到达目标且不出侧边界；停在障碍前不算通过。地形暴露时间只统计过渡后仍承载且机身在地形上方的时段。",
            "三项主指标依然统计完整承载窗口；若未通过地形，低姿态误差不能解释为地形控制优势。斜坡的Flatness按世界重力计算，包含顺坡俯仰，不减5°。",
            "\n|负载kg|模型|通过/轨迹数|到达地形/轨迹数|到达目标/轨迹数|地形承载暴露秒/轨迹|最大世界x均值m|",
            "|---:|---|---:|---:|---:|---:|---:|"]
        for r in conditions:
            report.append(f"|{r['mass_kg']:g}|{LABELS[r['model']]}|{r['course_completed']}/{r['trials']}|{r['terrain_reached']}/{r['trials']}|{r['goal_reached']}/{r['trials']}|{display(r['terrain_scored_s_mean'],2)}|{display(r['max_world_x_m_mean'],3)}|")
    report += ["\n正数为Ours改善，负数为更差；100×(基线均值−Ours均值)/基线均值。发生失败时先比较完成数，截断误差不能直接排名。",
        "\n|负载kg|相对基线|Roll改善%|Flatness改善%|速度改善%|","|---:|---|---:|---:|---:|"]
    for mass in sorted(manifest["masses"]):
        for baseline in MODELS[:2]:
            selected = [r for r in improvements if r["mass_kg"]==mass and r["baseline"]==baseline]
            report.append(f"|{mass:g}|{LABELS[baseline]}|"+"|".join(display(r["relative_improvement_percent"],2) for r in selected)+"|")
    report += [f"\n异常轨迹 {len(anomalies)} 条，全部保留于episode_metrics.csv及anomalies.csv。",
        "10条重复仅表示初始状态变化，三个模型仍各只有一个训练种子。指标采用本实验定义，不声称与任何论文同名指标数值直接等价。",
        "原始CSV、初始积分状态、预稳定轨迹、模型配置和场景副本位于相邻运行目录。球真值只记录，不输入策略。",
        "需要从姿态、速度、运输完成及球运动一起判断；本对比不能单独将优势归因于显式质量估计。"]
    (args.output_dir/"README.md").write_text("\n".join(report)+"\n")
    write_json(args.output_dir/"audit.json",dict(status="passed",trajectories=len(episodes),anomalies=len(anomalies),
        max_metric_recalculation_error=largest_metric_error,max_quaternion_gravity_error=largest_gravity_error,
        max_quaternion_roll_error=largest_roll_error,independent_numpy_formulas_verified=True,
        common_integration_state_pairing_verified=True,initial_observations_verified=observation_checks==len(episodes),
        input_protocol_sha256=sha(run/"protocol.json")))
    print("REPORT_COMPLETE",args.output_dir,"trajectories",len(episodes),"anomalies",len(anomalies))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--allow-smoke",action="store_true")
    main(parser.parse_args())
