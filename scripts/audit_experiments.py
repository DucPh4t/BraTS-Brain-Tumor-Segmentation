"""
Mô tả: Duyệt và thu thập các thông số cấu hình cùng kết quả Dice/HD95 kiểm thử của toàn bộ 53 thí nghiệm. Hỗ trợ tạo bảng đối chiếu nhanh cho ablation study.
Đầu vào:
    Đọc tự động từ các file configs/exp*.yaml và outputs/exp*/results.json + history.json.
Đầu ra:
    In ra màn hình dạng bảng (hoặc log) tổng hợp tất cả các thí nghiệm.
"""

import argparse
import glob
import json
import os
from collections import defaultdict

import yaml


def _load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _metric_mean(metric_dict):
    if not metric_dict:
        return None
    if "Mean" in metric_dict:
        return metric_dict["Mean"]
    return round(sum(metric_dict.get(k, 0.0) for k in ("WT", "TC", "ET")) / 3, 2)


def collect_rows():
    rows = []
    for config_path in sorted(glob.glob("configs/exp*.yaml")):
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        exp_name = config["exp_name"]
        result = _load_json(os.path.join("outputs", exp_name, "results.json")) or {}
        history = _load_json(os.path.join("outputs", exp_name, "history.json")) or {}
        metrics = result.get("metrics", {})
        dice = metrics.get("DICE", {})
        hd95 = metrics.get("HD95", {})
        eval_cfg = config.get("evaluation", {})

        rows.append({
            "exp": exp_name,
            "config_path": config_path,
            "arch": config.get("model", {}).get("architecture", "unet2d"),
            "in_channels": config.get("model", {}).get("in_channels", 4),
            "context_slices": config.get("data", {}).get("context_slices", 0),
            "normalization": config.get("data", {}).get("normalization", ""),
            "loss": config.get("training", {}).get("loss"),
            "optimizer": config.get("training", {}).get("optimizer"),
            "scheduler": config.get("training", {}).get("scheduler", "none"),
            "source_exp": eval_cfg.get("source_exp", ""),
            "checkpoint_path": eval_cfg.get("checkpoint_path", ""),
            "thresholds_from_results_path": eval_cfg.get("thresholds_from_results_path", ""),
            "mean_dice": _metric_mean(dice),
            "wt_dice": dice.get("WT"),
            "tc_dice": dice.get("TC"),
            "et_dice": dice.get("ET"),
            "mean_hd95": _metric_mean(hd95),
            "et_hd95": hd95.get("ET"),
            "has_result": bool(result),
            "has_history": bool(history),
            "has_per_subject": bool(result.get("per_subject")),
            "max_val_dice": max(history.get("val_dice", [0])) if history else None,
            "epochs_recorded": len(history.get("val_dice", [])) if history else 0,
        })
    return rows


def print_table(rows):
    header = [
        "exp", "arch", "ch", "ctx", "norm", "loss", "opt", "sched", "src",
        "mean_dice", "WT", "TC", "ET", "mean_hd95", "ET_HD95",
        "per_subject", "max_val", "epochs",
    ]
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join(["---"] * len(header)) + "|")
    for row in rows:
        values = [
            row["exp"],
            row["arch"],
            row["in_channels"],
            row["context_slices"],
            row["normalization"],
            row["loss"],
            row["optimizer"],
            row["scheduler"],
            row["source_exp"],
            row["mean_dice"],
            row["wt_dice"],
            row["tc_dice"],
            row["et_dice"],
            row["mean_hd95"],
            row["et_hd95"],
            "yes" if row["has_per_subject"] else "no",
            row["max_val_dice"],
            row["epochs_recorded"],
        ]
        print("| " + " | ".join("" if v is None else str(v) for v in values) + " |")


def print_warnings(rows):
    warnings = []
    metric_groups = defaultdict(list)
    for row in rows:
        if row["mean_dice"] is not None:
            key = (row["wt_dice"], row["tc_dice"], row["et_dice"], row["mean_hd95"], row["et_hd95"])
            metric_groups[key].append(row["exp"])

        if row["source_exp"] and row["checkpoint_path"] and not os.path.exists(row["checkpoint_path"]):
            warnings.append(f"{row['exp']}: missing source checkpoint {row['checkpoint_path']}")
        if row["thresholds_from_results_path"] and not os.path.exists(row["thresholds_from_results_path"]):
            warnings.append(f"{row['exp']}: missing threshold source {row['thresholds_from_results_path']}")
        if row["has_result"] and not row["has_per_subject"] and row["exp"] >= "exp023":
            warnings.append(f"{row['exp']}: result exists but has no per_subject details")
        if row["arch"].endswith("2_5d") and row["in_channels"] != 4 * int(row["context_slices"]):
            warnings.append(f"{row['exp']}: 2.5D in_channels does not match 4 * context_slices")

    for metrics, exps in metric_groups.items():
        if len(exps) > 1:
            warnings.append(f"duplicate result metrics: {', '.join(exps)}")

    print("\n## Warnings")
    if not warnings:
        print("- none")
        return
    for warning in warnings:
        print(f"- {warning}")


def print_rankings(rows):
    completed = [row for row in rows if row["mean_dice"] is not None]
    print("\n## Top Mean Dice")
    for row in sorted(completed, key=lambda item: item["mean_dice"], reverse=True)[:8]:
        print(f"- {row['exp']}: Mean Dice={row['mean_dice']} | ET={row['et_dice']} | HD95 Mean={row['mean_hd95']} | ET-HD95={row['et_hd95']}")

    print("\n## Top HD95 Mean")
    for row in sorted(completed, key=lambda item: item["mean_hd95"])[:8]:
        print(f"- {row['exp']}: HD95 Mean={row['mean_hd95']} | Mean Dice={row['mean_dice']} | ET-HD95={row['et_hd95']}")


def main():
    parser = argparse.ArgumentParser(description="Audit experiment configs and result files.")
    parser.add_argument("--output", type=str, default=None, help="Optional markdown path to write the audit report.")
    args = parser.parse_args()

    rows = collect_rows()
    if args.output:
        import contextlib
        with open(args.output, "w", encoding="utf-8") as f, contextlib.redirect_stdout(f):
            print_table(rows)
            print_rankings(rows)
            print_warnings(rows)
        print(f"Audit report saved to {args.output}")
    else:
        print_table(rows)
        print_rankings(rows)
        print_warnings(rows)


if __name__ == "__main__":
    main()
