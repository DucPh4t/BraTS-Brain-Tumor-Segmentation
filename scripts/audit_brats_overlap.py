"""
Mô tả: Kiểm tra và phát hiện các ca bệnh trùng lặp hoặc có độ tương quan cấu trúc cao giữa hai tập dữ liệu BraTS 2020 và BraTS 2023 GLI. Sử dụng mã hóa SHA256 và các đặc trưng hình học robust.
Đầu vào:
    --brats2020-root: Đường dẫn đến thư mục dữ liệu BraTS 2020.
    --brats2023-root: Đường dẫn đến thư mục dữ liệu BraTS 2023.
    --output: Đường dẫn xuất file JSON báo cáo audit trùng lặp.
Đầu ra:
    File JSON chứa danh sách mã băm trùng khớp, hệ số tương quan Pearson và thống kê chồng lấn.
"""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np


MODALITIES_2020 = {
    "flair": "_flair",
    "t1": "_t1",
    "t1ce": "_t1ce",
    "t2": "_t2",
}
MODALITIES_2023 = {
    "flair": "-t2f",
    "t1": "-t1n",
    "t1ce": "-t1c",
    "t2": "-t2w",
}


def robust_descriptor(array, grid_size=12):
    data = np.asarray(array, dtype=np.float32)
    if data.ndim != 3:
        raise ValueError(f"Expected a 3D MRI volume, got shape {data.shape}")
    indices = [
        np.rint(np.linspace(0, size - 1, grid_size)).astype(int)
        for size in data.shape
    ]
    thumbnail = data[np.ix_(*indices)]
    mask = thumbnail != 0
    normalized = np.zeros_like(thumbnail, dtype=np.float32)
    if mask.any():
        values = thumbnail[mask]
        normalized[mask] = (values - values.mean()) / (values.std() + 1e-8)
    vector = np.clip(normalized, -8.0, 8.0).reshape(-1)
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector


def find_nifti(subject_dir, subject_id, suffix):
    for extension in (".nii", ".nii.gz"):
        candidate = subject_dir / f"{subject_id}{suffix}{extension}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing {subject_id}{suffix}.nii[.gz] in {subject_dir}")


def fingerprint_array(array):
    data = np.asarray(array, dtype=np.float32)
    shape_bytes = np.asarray(data.shape, dtype=np.int32).tobytes()
    exact = hashlib.sha256(shape_bytes + np.ascontiguousarray(data).tobytes()).hexdigest()

    thumbnail = data[::8, ::8, ::5]
    nonzero = thumbnail[thumbnail != 0]
    normalized = np.zeros_like(thumbnail, dtype=np.float32)
    if nonzero.size:
        normalized[thumbnail != 0] = (nonzero - nonzero.mean()) / (nonzero.std() + 1e-8)
    quantized = np.round(np.clip(normalized, -8.0, 8.0) * 128.0).astype(np.int16)
    robust = hashlib.sha256(shape_bytes + np.ascontiguousarray(quantized).tobytes()).hexdigest()
    return {"exact": exact, "robust": robust, "descriptor": robust_descriptor(data)}


def fingerprint_nifti(path):
    image = nib.load(str(path))
    return fingerprint_array(np.asanyarray(image.dataobj, dtype=np.float32))


def subject_fingerprints(root, subject_id, suffixes):
    subject_dir = root / subject_id
    return {
        modality: fingerprint_nifti(find_nifti(subject_dir, subject_id, suffix))
        for modality, suffix in suffixes.items()
    }


def subject_ids(root, prefix):
    return sorted(path.name for path in root.iterdir() if path.is_dir() and path.name.startswith(prefix))


def build_hash_index(root, ids, suffixes):
    index = {
        "exact": defaultdict(list),
        "robust": defaultdict(list),
        "descriptors": {modality: {"subject_ids": [], "vectors": []} for modality in suffixes},
    }
    records = {}
    for position, subject_id in enumerate(ids, start=1):
        print(f"[{position}/{len(ids)}] fingerprint reference {subject_id}", flush=True)
        records[subject_id] = subject_fingerprints(root, subject_id, suffixes)
        for modality, fingerprints in records[subject_id].items():
            for kind in ("exact", "robust"):
                index[kind][(modality, fingerprints[kind])].append(subject_id)
            index["descriptors"][modality]["subject_ids"].append(subject_id)
            index["descriptors"][modality]["vectors"].append(fingerprints["descriptor"])
    for descriptor_index in index["descriptors"].values():
        descriptor_index["vectors"] = np.stack(descriptor_index["vectors"], axis=0)
    return records, index


def match_subject(fingerprints, reference_index, correlation_threshold=0.995):
    exact_counts, robust_counts, correlation_counts = Counter(), Counter(), Counter()
    correlation_scores = defaultdict(list)
    evidence = []
    for modality, values in fingerprints.items():
        for candidate in reference_index["exact"].get((modality, values["exact"]), []):
            exact_counts[candidate] += 1
            evidence.append({"candidate": candidate, "modality": modality, "kind": "exact"})
        for candidate in reference_index["robust"].get((modality, values["robust"]), []):
            robust_counts[candidate] += 1
            evidence.append({"candidate": candidate, "modality": modality, "kind": "robust"})

        descriptor_index = reference_index.get("descriptors", {}).get(modality)
        if descriptor_index and len(descriptor_index.get("subject_ids", [])):
            scores = descriptor_index["vectors"] @ values["descriptor"]
            best_index = int(np.argmax(scores))
            best_score = float(scores[best_index])
            if best_score >= correlation_threshold:
                candidate = descriptor_index["subject_ids"][best_index]
                correlation_counts[candidate] += 1
                correlation_scores[candidate].append(best_score)
                evidence.append({
                    "candidate": candidate,
                    "modality": modality,
                    "kind": "descriptor_correlation",
                    "score": best_score,
                })

    candidates = sorted(set(exact_counts) | set(robust_counts) | set(correlation_counts))
    ranked = sorted(
        candidates,
        key=lambda candidate: (
            exact_counts[candidate],
            robust_counts[candidate],
            correlation_counts[candidate],
            np.mean(correlation_scores[candidate]) if correlation_scores[candidate] else -1.0,
        ),
        reverse=True,
    )
    best = ranked[0] if ranked else None
    exact_matches = exact_counts[best] if best else 0
    robust_matches = robust_counts[best] if best else 0
    correlation_matches = correlation_counts[best] if best else 0
    if exact_matches >= 2 or robust_matches >= 2:
        status = "confirmed_duplicate_candidate"
    elif exact_matches >= 1 or robust_matches >= 1 or correlation_matches >= 2:
        status = "uncertain_overlap"
    else:
        status = "no_fingerprint_match"
    return {
        "status": status,
        "best_brats2020_candidate": best,
        "exact_modalities_matched": exact_matches,
        "robust_modalities_matched": robust_matches,
        "correlated_modalities_matched": correlation_matches,
        "correlation_scores": correlation_scores[best] if best else [],
        "evidence": evidence,
    }


def subject_list_hash(ids):
    return hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Audit possible BraTS2020/BraTS2023 subject overlap.")
    parser.add_argument("--brats2020-root", type=Path, required=True)
    parser.add_argument("--brats2023-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-2023-subjects", type=int)
    parser.add_argument("--correlation-threshold", type=float, default=0.995)
    parser.add_argument("--brats2020-version", default="BraTS2020 TrainingData")
    parser.add_argument("--brats2023-version", default="BraTS2023 GLI TrainingData")
    parser.add_argument("--brats2023-source", default="TODO: record official download source/version")
    parser.add_argument(
        "--official-mapping-reviewed",
        action="store_true",
        help="Assert that official provenance/mapping has also been reviewed outside this fingerprint audit.",
    )
    args = parser.parse_args()

    ids_2020 = subject_ids(args.brats2020_root, "BraTS20_Training_")
    all_ids_2023 = subject_ids(args.brats2023_root, "BraTS-GLI-")
    ids_2023 = list(all_ids_2023)
    if args.max_2023_subjects:
        ids_2023 = ids_2023[: args.max_2023_subjects]
    if not ids_2020 or not ids_2023:
        raise SystemExit("Both roots must contain recognizable BraTS subject directories.")

    _, reference_index = build_hash_index(args.brats2020_root, ids_2020, MODALITIES_2020)
    matches = {}
    for position, subject_id in enumerate(ids_2023, start=1):
        print(f"[{position}/{len(ids_2023)}] audit target {subject_id}", flush=True)
        fingerprints = subject_fingerprints(args.brats2023_root, subject_id, MODALITIES_2023)
        matches[subject_id] = match_subject(
            fingerprints,
            reference_index,
            correlation_threshold=args.correlation_threshold,
        )

    confirmed = sorted(
        subject_id for subject_id, item in matches.items()
        if item["status"] == "confirmed_duplicate_candidate"
    )
    uncertain = sorted(
        subject_id for subject_id, item in matches.items()
        if item["status"] == "uncertain_overlap"
    )
    nonmatched = sorted(
        subject_id for subject_id, item in matches.items()
        if item["status"] == "no_fingerprint_match"
    )
    audit_complete = len(ids_2023) == len(all_ids_2023)
    source_recorded = bool(args.brats2023_source.strip()) and not args.brats2023_source.startswith("TODO:")
    manuscript_eligible = (
        bool(args.official_mapping_reviewed)
        and audit_complete
        and not uncertain
        and source_recorded
    )
    report = {
        "method": {
            "exact": "SHA256 of canonical float32 voxel arrays",
            "robust": "SHA256 of z-scored, downsampled, quantized voxel thumbnails",
            "duplicate_rule": "at least two exact or two robust modality matches to one BraTS2020 subject",
            "uncertain_rule": "one exact/robust hash match or at least two high-correlation modality descriptors",
            "descriptor_correlation_threshold": args.correlation_threshold,
        },
        "reference": {
            "dataset": "BraTS2020",
            "version": args.brats2020_version,
            "num_subjects": len(ids_2020),
            "subject_list_sha256": subject_list_hash(ids_2020),
        },
        "target": {
            "dataset": "BraTS2023 GLI",
            "version": args.brats2023_version,
            "source": args.brats2023_source,
            "num_subjects_available": len(all_ids_2023),
            "num_subjects_audited": len(ids_2023),
            "subject_list_sha256": subject_list_hash(all_ids_2023),
            "audited_subject_ids": ids_2023,
        },
        "audit_complete": audit_complete,
        "source_recorded": source_recorded,
        "official_mapping_reviewed": bool(args.official_mapping_reviewed),
        "independence_status": (
            "eligible_after_fingerprint_and_official_mapping_review"
            if manuscript_eligible
            else "candidate_nonoverlap_only_not_officially_proven"
        ),
        "manuscript_eligible": manuscript_eligible,
        "confirmed_duplicate_subjects": confirmed,
        "uncertain_overlap_subjects": uncertain,
        "manual_review_required_subjects": uncertain,
        "candidate_nonoverlap_subjects": nonmatched,
        "excluded_subjects": sorted(set(confirmed) | set(uncertain)),
        "matches": matches,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "confirmed_duplicates": len(confirmed),
        "uncertain": len(uncertain),
        "candidate_nonoverlap": len(nonmatched),
        "audit_complete": audit_complete,
        "source_recorded": source_recorded,
        "manuscript_eligible": manuscript_eligible,
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
