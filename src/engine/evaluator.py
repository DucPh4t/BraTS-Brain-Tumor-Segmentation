"""
Mô tả: Lớp Evaluator thực hiện chạy suy diễn (inference) mô hình trên toàn bộ tập dữ liệu kiểm thử (test set) theo dạng 3D khối (volume-based). Tính toán các chỉ số 3D Dice Score, 95% Hausdorff Distance (HD95) và lưu trữ kết quả.
Đầu vào:
    model (nn.Module): Kiến trúc mô hình cần đánh giá.
    config (dict): Từ điển cấu hình thí nghiệm (YAML).
    device (torch.device): Thiết bị phần cứng để suy diễn (CUDA/MPS/CPU).
Đầu ra:
    Xuất kết quả chi tiết từng ca bệnh và ghi lại kết quả trung bình vào file kết quả (results.json).
"""

import itertools
import json
import os

import nibabel as nib
import numpy as np
import torch
from tqdm import tqdm

from src.data.processors import get_preprocessor
from src.data.dataset import normalize_context_slices
from src.utils.metrics import calc_dice_3d, calc_hd95_3d
from src.utils.postprocessing import REGION_NAMES, postprocess_regions
from src.utils.provenance import build_provenance


class Evaluator:
    def __init__(self, model, config, device, test_subjects, val_subjects=None, train_subjects=None):
        self.model = model
        self.config = config
        self.device = device
        self.test_subjects = test_subjects
        self.val_subjects = val_subjects or []
        self.train_subjects = train_subjects or []

        self.exp_name = config["exp_name"]
        self.out_dir = os.path.join("outputs", self.exp_name)
        os.makedirs(self.out_dir, exist_ok=True)

        self.eval_cfg = config.get("evaluation", {})
        self.best_model_path = self.eval_cfg.get(
            "checkpoint_path",
            os.path.join(self.out_dir, "best_model.pth"),
        )

        self.batch_size = int(self.eval_cfg.get("batch_size", 16))
        self.preprocessor = get_preprocessor(self.config["data"].get("normalization", "zscore_volume"))

    def run(self):
        print(f"=== 3D EVALUATING {self.exp_name} ===")
        print(f"[EVAL] Checkpoint: {self.best_model_path}")
        self.model.load_state_dict(torch.load(self.best_model_path, map_location=self.device, weights_only=True))
        self.model.eval()
        self.model.to(self.device)

        thresholds = self._load_or_default_thresholds()
        postprocess_cfg = self._base_postprocess_config()
        et_rescue_cfg = self._base_et_rescue_config()

        if self.eval_cfg.get("threshold_tuning", {}).get("enabled", False):
            thresholds, threshold_tuning = self._tune_thresholds(thresholds, postprocess_cfg)
        else:
            threshold_tuning = None

        if self.eval_cfg.get("et_rescue", {}).get("tune", {}).get("enabled", False):
            et_rescue_cfg, et_rescue_tuning = self._tune_et_rescue(thresholds, postprocess_cfg, et_rescue_cfg)
        else:
            et_rescue_tuning = None

        if self.eval_cfg.get("postprocess", {}).get("tune", {}).get("enabled", False):
            postprocess_cfg, postprocess_tuning = self._tune_postprocess(thresholds, postprocess_cfg, et_rescue_cfg)
        else:
            postprocess_tuning = None

        split_name, subjects = self._resolve_eval_split(self.eval_cfg.get("split", "test"))
        subjects = self._limit_subjects(subjects, "max_subjects")
        final_metrics = self._evaluate_subjects(
            subjects,
            thresholds,
            postprocess_cfg,
            et_rescue_cfg,
            desc=f"{split_name} subjects",
        )

        self._print_report(final_metrics, split_name)
        self._save_results(
            final_metrics,
            split_name,
            thresholds,
            postprocess_cfg,
            et_rescue_cfg,
            threshold_tuning,
            et_rescue_tuning,
            postprocess_tuning,
        )

    def evaluate_current_model(
        self,
        subjects,
        thresholds=None,
        postprocess_cfg=None,
        et_rescue_cfg=None,
        desc="3D validation subjects",
        show_progress=False,
    ):
        """Đánh giá mô hình đã được tải sẵn vào bộ nhớ mà không cần tải lại checkpoint từ đĩa."""
        self.model.eval()
        self.model.to(self.device)
        thresholds = thresholds or self._load_or_default_thresholds()
        postprocess_cfg = postprocess_cfg or self._base_postprocess_config()
        et_rescue_cfg = et_rescue_cfg or self._base_et_rescue_config()
        return self._evaluate_subjects(
            subjects,
            thresholds,
            postprocess_cfg,
            et_rescue_cfg,
            desc=desc,
            show_progress=show_progress,
        )

    def _resolve_eval_split(self, split_name):
        if split_name == "val":
            return "val", self.val_subjects
        if split_name == "test":
            return "test", self.test_subjects
        raise ValueError(f"Unknown evaluation split: {split_name}")

    def _load_or_default_thresholds(self):
        thresholds = {"WT": 0.5, "TC": 0.5, "ET": 0.5}
        thresholds.update(self.eval_cfg.get("thresholds", {}))

        results_path = self.eval_cfg.get("thresholds_from_results_path")
        if not results_path:
            return thresholds

        if not os.path.exists(results_path):
            print(f"[WARN] Threshold source not found, using config/default thresholds: {results_path}")
            return thresholds

        with open(results_path, "r", encoding="utf-8") as f:
            source_results = json.load(f)

        source_thresholds = source_results.get("evaluation", {}).get("thresholds")
        if source_thresholds:
            thresholds.update(source_thresholds)
            print(f"[EVAL] Loaded thresholds from {results_path}: {thresholds}")
        else:
            print(f"[WARN] No evaluation.thresholds in {results_path}, using config/default thresholds")

        return thresholds

    def _base_postprocess_config(self):
        post_cfg = self.eval_cfg.get("postprocess", {}).copy()
        post_cfg.pop("tune", None)
        post_cfg.setdefault("enabled", False)
        post_cfg.setdefault("enforce_hierarchy", True)
        return post_cfg

    def _base_et_rescue_config(self):
        rescue_cfg = self.eval_cfg.get("et_rescue", {}).copy()
        rescue_cfg.pop("tune", None)
        rescue_cfg.setdefault("enabled", False)
        rescue_cfg.setdefault("restrict_to_tc", True)
        return rescue_cfg

    def _limit_subjects(self, subjects, key):
        limit = self.eval_cfg.get(key)
        if limit is None:
            return subjects
        return subjects[:int(limit)]

    def _tune_thresholds(self, fallback_thresholds, postprocess_cfg):
        tuning_cfg = self.eval_cfg["threshold_tuning"]
        _, subjects = self._resolve_eval_split(tuning_cfg.get("split", "val"))
        subjects = self._limit_subjects(subjects, "max_tuning_subjects")
        grids = tuning_cfg.get("grids", {})
        candidates = [
            dict(zip(REGION_NAMES, values))
            for values in itertools.product(*(grids.get(region, [fallback_thresholds[region]]) for region in REGION_NAMES))
        ]

        best_thresholds = fallback_thresholds
        best_metrics = None
        print(f"[TUNE] Threshold candidates: {len(candidates)} on {len(subjects)} validation subjects")

        dice_only_metrics = self._evaluate_threshold_candidates(
            candidates,
            subjects,
            postprocess_cfg,
            self._base_et_rescue_config(),
            include_hd95=False,
            desc="Threshold tuning dice pass",
        )
        top_k = min(int(tuning_cfg.get("top_k_hd95", len(candidates))), len(candidates))
        ranked = sorted(
            zip(candidates, dice_only_metrics),
            key=lambda item: item[1]["summary"]["DICE"]["Mean"],
            reverse=True,
        )
        top_candidates = [candidate for candidate, _ in ranked[:top_k]]
        print(f"[TUNE] Computing HD95 for top-{top_k} threshold candidates")

        full_metrics = self._evaluate_threshold_candidates(
            top_candidates,
            subjects,
            postprocess_cfg,
            self._base_et_rescue_config(),
            include_hd95=True,
            desc="Threshold tuning hd95 pass",
        )

        for candidate, metrics in zip(top_candidates, full_metrics):
            if self._is_better(metrics, best_metrics):
                best_thresholds = candidate
                best_metrics = metrics

        print(f"[TUNE] Best thresholds: {best_thresholds}")
        return best_thresholds, {
            "enabled": True,
            "split": tuning_cfg.get("split", "val"),
            "num_candidates": len(candidates),
            "top_k_hd95": top_k,
            "best_metrics": best_metrics["summary"],
        }

    def _tune_postprocess(self, thresholds, fallback_postprocess_cfg, et_rescue_cfg):
        tuning_cfg = self.eval_cfg["postprocess"]["tune"]
        _, subjects = self._resolve_eval_split(tuning_cfg.get("split", "val"))
        subjects = self._limit_subjects(subjects, "max_tuning_subjects")
        grids = tuning_cfg.get("candidate_min_component_voxels", {})
        candidates = [
            dict(zip(REGION_NAMES, values))
            for values in itertools.product(*(grids.get(region, [0]) for region in REGION_NAMES))
        ]

        best_cfg = fallback_postprocess_cfg
        best_metrics = None
        print(f"[TUNE] Postprocess candidates: {len(candidates)} on {len(subjects)} validation subjects")

        candidate_cfgs = []
        for min_voxels in candidates:
            candidate_cfg = fallback_postprocess_cfg.copy()
            candidate_cfg["enabled"] = True
            candidate_cfg["min_component_voxels"] = min_voxels
            candidate_cfgs.append(candidate_cfg)

        candidate_subject_scores = [[] for _ in candidate_cfgs]
        with torch.no_grad():
            for subject_id in tqdm(subjects, desc="Postprocess tuning subjects"):
                stack_4d, gt_volume, spacing_zyx = self._load_subject(subject_id)
                prob_volume = self._predict_prob_volume(stack_4d)
                binary_volume = self._apply_thresholds(prob_volume, thresholds)
                for idx, candidate_cfg in enumerate(candidate_cfgs):
                    pred_volume = self._apply_et_rescue(binary_volume, prob_volume, candidate_cfg, et_rescue_cfg)
                    pred_volume = postprocess_regions(pred_volume, candidate_cfg)
                    candidate_subject_scores[idx].append(
                        self._score_subject(subject_id, pred_volume, gt_volume, spacing_zyx)
                    )

        for candidate_cfg, per_subject in zip(candidate_cfgs, candidate_subject_scores):
            metrics = {
                "summary": self._summarize(per_subject),
                "per_subject": per_subject,
                "num_subjects": len(subjects),
            }
            if self._is_better(metrics, best_metrics):
                best_cfg = candidate_cfg
                best_metrics = metrics

        print(f"[TUNE] Best postprocess min_component_voxels: {best_cfg.get('min_component_voxels', {})}")
        return best_cfg, {
            "enabled": True,
            "split": tuning_cfg.get("split", "val"),
            "num_candidates": len(candidates),
            "best_metrics": best_metrics["summary"],
        }

    def _tune_et_rescue(self, thresholds, postprocess_cfg, fallback_rescue_cfg):
        tuning_cfg = self.eval_cfg["et_rescue"]["tune"]
        _, subjects = self._resolve_eval_split(tuning_cfg.get("split", "val"))
        subjects = self._limit_subjects(subjects, "max_tuning_subjects")
        thresholds_grid = tuning_cfg.get("candidate_thresholds", [fallback_rescue_cfg.get("prob_threshold", 0.5)])
        min_voxels_grid = tuning_cfg.get("candidate_min_voxels", [fallback_rescue_cfg.get("min_voxels", 10)])
        restrict_grid = tuning_cfg.get("candidate_restrict_to_tc", [fallback_rescue_cfg.get("restrict_to_tc", True)])
        candidates = [
            {
                "enabled": True,
                "prob_threshold": float(prob_threshold),
                "min_voxels": int(min_voxels),
                "restrict_to_tc": bool(restrict_to_tc),
            }
            for prob_threshold, min_voxels, restrict_to_tc in itertools.product(
                thresholds_grid,
                min_voxels_grid,
                restrict_grid,
            )
        ]

        best_cfg = fallback_rescue_cfg
        best_metrics = None
        print(f"[TUNE] ET rescue candidates: {len(candidates)} on {len(subjects)} validation subjects")

        candidate_subject_scores = [[] for _ in candidates]
        with torch.no_grad():
            for subject_id in tqdm(subjects, desc="ET rescue tuning subjects"):
                stack_4d, gt_volume, spacing_zyx = self._load_subject(subject_id)
                prob_volume = self._predict_prob_volume(stack_4d)
                binary_volume = self._apply_thresholds(prob_volume, thresholds)
                for idx, candidate_cfg in enumerate(candidates):
                    pred_volume = self._apply_et_rescue(binary_volume, prob_volume, postprocess_cfg, candidate_cfg)
                    if postprocess_cfg.get("enabled", False):
                        pred_volume = postprocess_regions(pred_volume, postprocess_cfg)
                    candidate_subject_scores[idx].append(
                        self._score_subject(subject_id, pred_volume, gt_volume, spacing_zyx)
                    )

        for candidate_cfg, per_subject in zip(candidates, candidate_subject_scores):
            metrics = {
                "summary": self._summarize(per_subject),
                "per_subject": per_subject,
                "num_subjects": len(subjects),
            }
            if self._is_better(metrics, best_metrics):
                best_cfg = candidate_cfg
                best_metrics = metrics

        print(f"[TUNE] Best ET rescue config: {best_cfg}")
        return best_cfg, {
            "enabled": True,
            "split": tuning_cfg.get("split", "val"),
            "num_candidates": len(candidates),
            "best_metrics": best_metrics["summary"],
        }

    def _evaluate_threshold_candidates(
        self,
        candidates,
        subjects,
        postprocess_cfg,
        et_rescue_cfg,
        include_hd95,
        desc,
    ):
        candidate_subject_scores = [[] for _ in candidates]
        with torch.no_grad():
            for subject_id in tqdm(subjects, desc=desc):
                stack_4d, gt_volume, spacing_zyx = self._load_subject(subject_id)
                prob_volume = self._predict_prob_volume(stack_4d)
                for idx, candidate in enumerate(candidates):
                    pred_volume = self._apply_thresholds(prob_volume, candidate)
                    pred_volume = self._apply_et_rescue(pred_volume, prob_volume, postprocess_cfg, et_rescue_cfg)
                    if postprocess_cfg.get("enabled", False):
                        pred_volume = postprocess_regions(pred_volume, postprocess_cfg)
                    candidate_subject_scores[idx].append(
                        self._score_subject(
                            subject_id,
                            pred_volume,
                            gt_volume,
                            spacing_zyx,
                            include_hd95=include_hd95,
                        )
                    )

        return [
            {
                "summary": self._summarize(per_subject),
                "per_subject": per_subject,
                "num_subjects": len(subjects),
            }
            for per_subject in candidate_subject_scores
        ]

    def _is_better(self, metrics, best_metrics):
        if best_metrics is None:
            return True
        current = metrics["summary"]
        best = best_metrics["summary"]
        if current["DICE"]["Mean"] != best["DICE"]["Mean"]:
            return current["DICE"]["Mean"] > best["DICE"]["Mean"]
        return current["HD95"]["ET"] < best["HD95"]["ET"]

    def _load_subject(self, subject_id):
        data_dir = os.path.join(self.config["data"]["root_dir"], subject_id)
        volumes = []
        for mod in ["flair", "t1", "t1ce", "t2"]:
            path = self._find_subject_nifti(data_dir, f"{subject_id}_{mod}")
            vol_3d = nib.load(path).get_fdata()
            volumes.append(self.preprocessor(vol_3d))

        stack_4d = np.stack(volumes, axis=0)
        stack_4d = np.transpose(stack_4d, (3, 0, 1, 2)).astype(np.float32)
        stack_4d = self._add_context_slices(stack_4d)

        try:
            mask_path = self._find_subject_nifti(data_dir, f"{subject_id}_seg")
        except FileNotFoundError:
            candidates = [
                os.path.join(data_dir, filename)
                for filename in os.listdir(data_dir)
                if "seg" in filename.lower() and filename.endswith((".nii", ".nii.gz"))
            ]
            if not candidates:
                raise
            mask_path = candidates[0]

        mask_img = nib.load(mask_path)
        mask_3d = np.transpose(mask_img.get_fdata(), (2, 0, 1))
        spacing = mask_img.header.get_zooms()[:3]
        spacing_zyx = (spacing[2], spacing[0], spacing[1])

        gt_volume = np.stack(
            [
                (mask_3d > 0).astype(np.float32),
                np.logical_or(mask_3d == 1, mask_3d == 4).astype(np.float32),
                (mask_3d == 4).astype(np.float32),
            ],
            axis=0,
        )

        return stack_4d, gt_volume, spacing_zyx

    def _find_subject_nifti(self, data_dir, stem):
        for extension in (".nii", ".nii.gz"):
            path = os.path.join(data_dir, stem + extension)
            if os.path.exists(path):
                return path
        raise FileNotFoundError(f"Missing NIfTI file {stem}.nii[.gz] in {data_dir}")

    def _add_context_slices(self, stack_4d):
        context_slices = normalize_context_slices(
            self.config["data"].get("context_slices", 1)
        )
        if context_slices <= 1:
            return stack_4d

        radius = (context_slices - 1) // 2
        expanded = []
        for z_idx in range(stack_4d.shape[0]):
            channels = []
            for modality_idx in range(stack_4d.shape[1]):
                for offset in range(-radius, radius + 1):
                    ctx_idx = int(np.clip(z_idx + offset, 0, stack_4d.shape[0] - 1))
                    channels.append(stack_4d[ctx_idx, modality_idx])
            expanded.append(np.stack(channels, axis=0))
        return np.stack(expanded, axis=0).astype(np.float32)

    def _predict_prob_volume(self, stack_4d):
        all_probs = []
        for i in range(0, stack_4d.shape[0], self.batch_size):
            batch_imgs = stack_4d[i:i + self.batch_size]
            tensor_imgs = torch.from_numpy(batch_imgs).to(self.device)
            probs = self._predict_batch_probs(tensor_imgs).cpu().numpy()
            all_probs.append(probs)
        return np.concatenate(all_probs, axis=0)

    def _predict_batch_probs(self, tensor_imgs):
        tta_cfg = self.eval_cfg.get("tta", {})
        if not tta_cfg.get("enabled", False):
            return torch.sigmoid(self._forward_logits(tensor_imgs))

        probs = []
        for flip_name in tta_cfg.get("flips", ["none", "h", "w", "hw"]):
            dims = self._flip_dims(flip_name)
            aug_imgs = torch.flip(tensor_imgs, dims=dims) if dims else tensor_imgs
            aug_probs = torch.sigmoid(self._forward_logits(aug_imgs))
            if dims:
                aug_probs = torch.flip(aug_probs, dims=dims)
            probs.append(aug_probs)
        return torch.stack(probs, dim=0).mean(dim=0)

    def _forward_logits(self, tensor_imgs):
        output = self.model(tensor_imgs)
        return output[0] if isinstance(output, tuple) else output

    def _flip_dims(self, flip_name):
        if flip_name in ("none", None):
            return ()
        if flip_name == "h":
            return (2,)
        if flip_name == "w":
            return (3,)
        if flip_name == "hw":
            return (2, 3)
        raise ValueError(f"Unknown TTA flip: {flip_name}")

    def _apply_thresholds(self, prob_volume, thresholds):
        threshold_array = np.array([thresholds[region] for region in REGION_NAMES], dtype=np.float32)
        threshold_array = threshold_array.reshape(1, 3, 1, 1)
        return (prob_volume > threshold_array).astype(np.float32)

    def _apply_et_rescue(self, pred_volume, prob_volume, postprocess_cfg, et_rescue_cfg):
        if not et_rescue_cfg.get("enabled", False):
            return pred_volume

        rescued = pred_volume.astype(np.float32).copy()
        if rescued[:, 2].sum() > 0:
            return rescued

        et_prob = prob_volume[:, 2]
        prob_threshold = float(et_rescue_cfg.get("prob_threshold", 0.5))
        candidate = et_prob >= prob_threshold

        if et_rescue_cfg.get("restrict_to_tc", True):
            candidate = np.logical_and(candidate, rescued[:, 1] > 0)

        if candidate.sum() < int(et_rescue_cfg.get("min_voxels", 1)):
            return rescued

        rescued[:, 2] = candidate.astype(np.float32)
        rescued[:, 1] = np.logical_or(rescued[:, 1] > 0, rescued[:, 2] > 0)
        rescued[:, 0] = np.logical_or(rescued[:, 0] > 0, rescued[:, 1] > 0)
        return rescued.astype(np.float32)

    def _evaluate_subjects(self, subjects, thresholds, postprocess_cfg, et_rescue_cfg, desc, show_progress=True):
        per_subject = []
        iterator = tqdm(subjects, desc=desc, leave=False) if show_progress and desc else subjects

        with torch.no_grad():
            for subject_id in iterator:
                stack_4d, gt_volume, spacing_zyx = self._load_subject(subject_id)
                prob_volume = self._predict_prob_volume(stack_4d)
                pred_volume = self._apply_thresholds(prob_volume, thresholds)
                pred_volume = self._apply_et_rescue(pred_volume, prob_volume, postprocess_cfg, et_rescue_cfg)

                if postprocess_cfg.get("enabled", False):
                    pred_volume = postprocess_regions(pred_volume, postprocess_cfg)

                per_subject.append(self._score_subject(subject_id, pred_volume, gt_volume, spacing_zyx))

        return {
            "summary": self._summarize(per_subject),
            "per_subject": per_subject,
            "num_subjects": len(subjects),
        }

    def _score_subject(self, subject_id, pred_volume, gt_volume, spacing_zyx, include_hd95=True):
        scores = {
            "subject_id": subject_id,
            "DICE": {},
            "HD95": {},
            "PRECISION": {},
            "SENSITIVITY": {},
        }
        for channel, region in enumerate(REGION_NAMES):
            pred = pred_volume[:, channel]
            gt = gt_volume[channel]
            scores["DICE"][region] = float(calc_dice_3d(pred, gt))
            if include_hd95:
                scores["HD95"][region] = float(calc_hd95_3d(pred, gt, voxelspacing=spacing_zyx))

            pred_positive = pred > 0
            gt_positive = gt > 0
            true_positive = int(np.logical_and(pred_positive, gt_positive).sum())
            false_positive = int(np.logical_and(pred_positive, ~gt_positive).sum())
            false_negative = int(np.logical_and(~pred_positive, gt_positive).sum())
            precision_denom = true_positive + false_positive
            sensitivity_denom = true_positive + false_negative
            scores["PRECISION"][region] = (
                float(true_positive / precision_denom)
                if precision_denom > 0
                else (1.0 if sensitivity_denom == 0 else 0.0)
            )
            scores["SENSITIVITY"][region] = (
                float(true_positive / sensitivity_denom) if sensitivity_denom > 0 else None
            )

        pred_wt = pred_volume[:, 0] > 0
        pred_tc = pred_volume[:, 1] > 0
        pred_et = pred_volume[:, 2] > 0
        et_outside_tc = int(np.logical_and(pred_et, ~pred_tc).sum())
        tc_outside_wt = int(np.logical_and(pred_tc, ~pred_wt).sum())
        et_voxels = int(pred_et.sum())
        tc_voxels = int(pred_tc.sum())
        scores["STRUCTURE"] = {
            "pred_wt_voxels": int(pred_wt.sum()),
            "pred_tc_voxels": tc_voxels,
            "pred_et_voxels": et_voxels,
            "et_outside_tc_voxels": et_outside_tc,
            "tc_outside_wt_voxels": tc_outside_wt,
            "et_inside_tc_rate": float(1.0 - et_outside_tc / et_voxels) if et_voxels else 1.0,
            "tc_inside_wt_rate": float(1.0 - tc_outside_wt / tc_voxels) if tc_voxels else 1.0,
            "fully_compliant": et_outside_tc == 0 and tc_outside_wt == 0,
        }

        gt_et_voxels = int((gt_volume[2] > 0).sum())
        size_cfg = self.eval_cfg.get("et_size_groups", {})
        tiny_max = int(size_cfg.get("tiny_max_voxels", 100))
        small_max = int(size_cfg.get("small_max_voxels", 1000))
        if gt_et_voxels == 0:
            et_group = "empty"
        elif gt_et_voxels <= tiny_max:
            et_group = "tiny"
        elif gt_et_voxels <= small_max:
            et_group = "small"
        else:
            et_group = "normal_large"
        scores["GT_VOXELS"] = {
            region: int((gt_volume[channel] > 0).sum())
            for channel, region in enumerate(REGION_NAMES)
        }
        scores["ET_GROUP"] = et_group
        return scores

    def _summarize(self, per_subject):
        summary = {"DICE": {}, "HD95": {}, "PRECISION": {}, "SENSITIVITY": {}}
        for metric in ("DICE", "HD95", "PRECISION", "SENSITIVITY"):
            for region in REGION_NAMES:
                values = np.array(
                    [
                        item[metric][region]
                        for item in per_subject
                        if region in item.get(metric, {}) and item[metric][region] is not None
                    ],
                    dtype=np.float32,
                )
                if values.size == 0:
                    continue
                multiplier = 100.0 if metric in ("DICE", "PRECISION", "SENSITIVITY") else 1.0
                summary[metric][region] = round(float(values.mean() * multiplier), 2)
                summary[metric][f"{region}_std"] = round(float(values.std() * multiplier), 2)
                summary[metric][f"{region}_median"] = round(float(np.median(values) * multiplier), 2)
                q1, q3 = np.percentile(values, [25, 75])
                summary[metric][f"{region}_q1"] = round(float(q1 * multiplier), 2)
                summary[metric][f"{region}_q3"] = round(float(q3 * multiplier), 2)
                summary[metric][f"{region}_iqr"] = round(float((q3 - q1) * multiplier), 2)

        summary["DICE"]["Mean"] = round(float(np.mean([summary["DICE"][region] for region in REGION_NAMES])), 2)
        if all(region in summary["HD95"] for region in REGION_NAMES):
            summary["HD95"]["Mean"] = round(float(np.mean([summary["HD95"][region] for region in REGION_NAMES])), 2)

        structure_items = [item["STRUCTURE"] for item in per_subject if "STRUCTURE" in item]
        total_et = sum(item["pred_et_voxels"] for item in structure_items)
        total_tc = sum(item["pred_tc_voxels"] for item in structure_items)
        total_et_outside = sum(item["et_outside_tc_voxels"] for item in structure_items)
        total_tc_outside = sum(item["tc_outside_wt_voxels"] for item in structure_items)
        summary["STRUCTURE"] = {
            "et_outside_tc_voxels": total_et_outside,
            "tc_outside_wt_voxels": total_tc_outside,
            "et_inside_tc_rate": float(1.0 - total_et_outside / total_et) if total_et else 1.0,
            "tc_inside_wt_rate": float(1.0 - total_tc_outside / total_tc) if total_tc else 1.0,
            "subject_compliance_rate": (
                float(np.mean([item["fully_compliant"] for item in structure_items]))
                if structure_items
                else 1.0
            ),
        }
        summary["ET_GROUP_COUNTS"] = {
            group: sum(item.get("ET_GROUP") == group for item in per_subject)
            for group in ("empty", "tiny", "small", "normal_large")
        }
        return summary

    def _print_report(self, metrics, split_name):
        summary = metrics["summary"]
        print("\n=============================================")
        print(f" REPORT — {self.exp_name} ({split_name.upper()} 3D VOLUME EVAL)")
        print("=============================================")
        print("          [ 3D DICE SCORE (Cao tốt) ]        ")
        print(f"Whole Tumor  (WT): {summary['DICE']['WT']:.2f}%")
        print(f"Tumor Core   (TC): {summary['DICE']['TC']:.2f}%")
        print(f"Enhancing    (ET): {summary['DICE']['ET']:.2f}%")
        print(f"Mean Dice        : {summary['DICE']['Mean']:.2f}%")
        print("---------------------------------------------")
        print("      [ 3D HAUSDORFF 95 (Nhỏ tốt) mm ]       ")
        print(f"Whole Tumor  (WT): {summary['HD95']['WT']:.2f}")
        print(f"Tumor Core   (TC): {summary['HD95']['TC']:.2f}")
        print(f"Enhancing    (ET): {summary['HD95']['ET']:.2f}")
        print(f"Mean HD95        : {summary['HD95']['Mean']:.2f}")
        print("=============================================")

    def _save_results(
        self,
        metrics,
        split_name,
        thresholds,
        postprocess_cfg,
        et_rescue_cfg,
        threshold_tuning,
        et_rescue_tuning,
        postprocess_tuning,
    ):
        checkpoint_metadata = self._load_checkpoint_metadata()
        split_subjects = {
            "train": self.train_subjects,
            "validation": self.val_subjects,
            "test": self.test_subjects,
        }
        results = {
            "experiment_name": self.exp_name,
            "description": self.config.get("description", "")
            + " evaluated with true 3D Dice and HD95 metrics over all 155 slices.",
            "source_experiment": self.eval_cfg.get("source_exp", self.exp_name),
            "checkpoint_path": self.best_model_path,
            "split": split_name,
            "metrics": metrics["summary"],
            "evaluation": {
                "thresholds": thresholds,
                "tta": self.eval_cfg.get("tta", {"enabled": False}),
                "postprocess": postprocess_cfg,
                "et_rescue": et_rescue_cfg,
                "threshold_tuning": threshold_tuning,
                "et_rescue_tuning": et_rescue_tuning,
                "postprocess_tuning": postprocess_tuning,
                "num_subjects": metrics["num_subjects"],
                "evaluated_subject_ids": [item["subject_id"] for item in metrics["per_subject"]],
                "external_overlap_audit": self.eval_cfg.get("external_overlap_audit"),
                "metric_empty_mask_policy": {
                    "dice": "both empty=1; exactly one empty=0",
                    "hd95": "both empty=0; exactly one empty=physical volume diagonal",
                    "precision": "both empty=1; positive GT with empty prediction=0",
                    "sensitivity": "undefined for empty GT and excluded from aggregate",
                },
            },
            "provenance": build_provenance(
                self.config,
                split_subjects,
                checkpoint_metadata=checkpoint_metadata,
            ),
            "per_subject": metrics["per_subject"],
            "status": "done",
        }

        res_path = os.path.join(self.out_dir, "results.json")
        with open(res_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        print(f"Results saved to: {res_path}")

    def _load_checkpoint_metadata(self):
        checkpoint_dir = os.path.dirname(self.best_model_path)
        checkpoint_name = os.path.basename(self.best_model_path)
        if checkpoint_name == "best_model_3d.pth":
            metadata_name = "best_model_3d_meta.json"
        elif checkpoint_name == "best_model.pth":
            metadata_name = "best_model_meta.json"
        else:
            metadata_name = f"{os.path.splitext(checkpoint_name)[0]}_meta.json"
        metadata_path = os.path.join(checkpoint_dir, metadata_name)
        if not os.path.exists(metadata_path):
            return {"path": self.best_model_path, "metadata_available": False}
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        metadata["path"] = self.best_model_path
        metadata["metadata_available"] = True
        return metadata
