"""Prepare private patient folds without decoding images or running a model.

Run on Rangpur: python -B prepare_cv_folds.py --config PATH/config.json
Repeat with --verify to compare saved manifests against the current source data.
Only the existing loader enumerates Test filenames to preserve patient exclusion;
no Test prediction file, image contents, or performance metric is read.
"""
import argparse
import hashlib
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path


SEED = 42
FOLDS = 5


class FoldError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise FoldError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True).encode()).hexdigest()


def split_fingerprint(samples):
    result = hashlib.sha256()
    for path, label in samples:
        result.update((str(path) + "\t" + str(label) + "\n").encode("utf-8"))
    return result.hexdigest()


def build_manifest(records, provenance, expected=None):
    """Pure partitioning: records are in the original ImageFolder index order."""
    require(bool(records), "Empty development set")
    paths, scans, patients = set(), {}, defaultdict(list)
    for row in records:
        require(set(row) == {"path", "label", "scan", "patient"}, "Invalid record schema")
        require(all(isinstance(row[k], str) and row[k] for k in ("path", "scan", "patient")),
                "Missing identity or path")
        require(type(row["label"]) is int and row["label"] in (0, 1), "Invalid class")
        require(row["path"] not in paths, "Duplicate development path")
        paths.add(row["path"])
        link = (row["patient"], row["label"])
        require(row["scan"] not in scans or scans[row["scan"]] == link,
                "Conflicting scan identity or labels")
        scans[row["scan"]] = link
        patients[row["patient"]].append(row)
    labels = {}
    for patient, rows in patients.items():
        values = {r["label"] for r in rows}
        require(len(values) == 1, "Mixed-label patient detected; no majority voting allowed")
        labels[patient] = next(iter(values))
    counts = Counter(labels.values())
    require(set(counts) == {0, 1} and min(counts.values()) >= FOLDS,
            "Each class needs at least five patients")
    if expected is not None:
        require((len(patients), len(scans), len(records), counts[0], counts[1]) == expected,
                "Development counts differ from approved audit")
    rng = random.Random(SEED)
    assignment = {}
    for label in (0, 1):
        members = sorted(p for p in patients if labels[p] == label)
        rng.shuffle(members)
        assignment.update((p, index % FOLDS) for index, p in enumerate(members))
    all_patients = set(patients)
    all_indices = set(range(len(records)))
    validation_seen = set()
    folds = []
    for fold_id in range(FOLDS):
        validation = {p for p in patients if assignment[p] == fold_id}
        training = all_patients - validation
        require(not validation_seen & validation, "Validation folds overlap")
        validation_seen.update(validation)
        train_indices = [i for i, r in enumerate(records) if r["patient"] in training]
        val_indices = [i for i, r in enumerate(records) if r["patient"] in validation]
        require(not set(train_indices) & set(val_indices), "Slice overlap")
        require(set(train_indices) | set(val_indices) == all_indices, "Missing slices")
        train_scans = {records[i]["scan"] for i in train_indices}
        val_scans = {records[i]["scan"] for i in val_indices}
        require(not train_scans & val_scans, "Scan overlap")
        require({labels[p] for p in validation} == {0, 1} and
                {labels[p] for p in training} == {0, 1}, "Fold missing class")
        body = dict(fold_id=fold_id, train_patients=sorted(training),
                    validation_patients=sorted(validation), train_indices=train_indices,
                    validation_indices=val_indices,
                    validation_patient_class_counts={str(c): sum(labels[p] == c for p in validation)
                                                     for c in (0, 1)})
        body["fingerprint"] = digest(body)
        folds.append(body)
    require(validation_seen == all_patients, "Validation coverage incomplete")
    require(sorted(i for f in folds for i in f["validation_indices"]) == list(range(len(records))),
            "Validation slices missing or repeated")
    body = dict(version=1, seed=SEED, n_folds=FOLDS, class_to_idx={"AD": 0, "NC": 1},
                algorithm="sorted patient IDs; Random(42); shuffle AD then NC; index modulo 5",
                provenance=provenance, samples=records, patient_to_fold=assignment,
                development_fingerprint=digest(records), folds=folds)
    body["fingerprint"] = digest(body)
    return body


def load_source(config_path):
    # Lazy imports keep synthetic tests independent of torch/torchvision.
    import contextlib
    import io
    from torch.utils.data import Subset
    from dataset import load_datasets, load_patient_map, get_image_id, DATA_ROOT

    config = json.loads(config_path.read_text(encoding="utf-8"))
    require(config.get("model") == "convnext" and not config.get("smoke_test", True),
            "Expected historical full-data ConvNeXt config")
    require(config.get("class_to_idx") == {"AD": 0, "NC": 1}, "Historical class map mismatch")
    # No __getitem__, image decoder, DataLoader, or model invocation.
    with contextlib.redirect_stdout(io.StringIO()):
        train, validation, test = load_datasets()
    mapping = load_patient_map()
    require(isinstance(train, Subset) and isinstance(validation, Subset) and
            train.dataset is validation.dataset, "Unexpected historical loader structure")
    source = train.dataset
    require(source.class_to_idx == {"AD": 0, "NC": 1}, "Source class map mismatch")
    fingerprints = {}
    for name, ds in (("train", train), ("validation", validation)):
        samples = [source.samples[int(i)] for i in ds.indices]
        fingerprints[name] = split_fingerprint(samples)
        require(fingerprints[name] == config.get("split_fingerprint", {}).get(name),
                "Historical split fingerprint mismatch: " + name)
    require(not set(train.indices) & set(validation.indices), "Historical slice overlap")
    def patient(path):
        value = mapping.get(get_image_id(path))
        require(value is not None, "Missing scan-to-patient mapping")
        return value
    train_patients = {patient(source.samples[int(i)][0]) for i in train.indices}
    val_patients = {patient(source.samples[int(i)][0]) for i in validation.indices}
    require(not train_patients & val_patients, "Historical patient overlap")
    # Labels are deliberately discarded: Test identities only certify exclusion.
    test_patients = {patient(path) for path, _ in test.samples}
    require(not (train_patients | val_patients) & test_patients, "Development/Test patient overlap")
    indices = sorted(list(train.indices) + list(validation.indices))
    records = []
    root = DATA_ROOT / "train"
    for i in indices:
        path, label = source.samples[int(i)]
        require(Path(path).is_file(), "Development file missing")
        records.append(dict(path=Path(path).relative_to(root).as_posix(), label=int(label),
                            scan=get_image_id(path), patient=patient(path)))
    return build_manifest(records, dict(historical_split_fingerprints=fingerprints,
                          historical_config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
                          sample_path_base="DATA_ROOT/train", test_identity_exclusion_verified=True),
                          expected=(349, 635, 12700, 90, 259))


def artifacts(manifest):
    files = {"manifest.json": manifest}
    for fold in manifest["folds"]:
        files["fold_{}.json".format(fold["fold_id"])] = dict(
            manifest_fingerprint=manifest["fingerprint"], **fold)
    files["fingerprints.json"] = {name: digest(value) for name, value in files.items()}
    return files


def write_or_verify(output, manifest, verify=False):
    files = artifacts(manifest)
    if verify:
        require(output.is_dir() and not output.is_symlink(), "Missing or invalid output directory")
        for name, expected in files.items():
            path = output / name
            require(path.is_file() and not path.is_symlink(), "Missing or invalid manifest artifact")
            require(json.loads(path.read_text(encoding="utf-8")) == expected,
                    "Saved artifacts differ from current validated source")
            if os.name == "posix":
                require(path.stat().st_mode & 0o077 == 0, "Private file permissions too broad")
        if os.name == "posix":
            require(output.stat().st_mode & 0o077 == 0, "Private directory permissions too broad")
        return
    require(not output.exists(), "Output exists; use --verify or choose a new directory")
    # Rangpur/Linux permissions are mandatory for writing patient-bearing outputs.
    require(os.name == "posix", "Create private manifests on Rangpur/Linux, not a shared Windows directory")
    output.mkdir(mode=0o700, parents=True)
    os.chmod(output, 0o700)
    for name, value in files.items():
        with os.fdopen(os.open(str(output / name), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600),
                       "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
    with os.fdopen(os.open(str(output / ".gitignore"), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600),
                   "w", encoding="utf-8") as handle:
        handle.write("*\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("diagnostics/cv_folds_seed42"))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true", help="Validate and partition in memory; write nothing")
    mode.add_argument("--verify", action="store_true", help="Revalidate source and compare saved artifacts; write nothing")
    args = parser.parse_args()
    manifest = load_source(args.config)
    if not args.check_only:
        write_or_verify(args.output_dir, manifest, args.verify)
    print("PASS: 349 patients, 635 scans, 12700 slices; five disjoint validation folds.")
    for fold in manifest["folds"]:
        print("Fold {}: validation AD={}, NC={}; train slices={}, validation slices={}".format(
            fold["fold_id"], fold["validation_patient_class_counts"]["0"],
            fold["validation_patient_class_counts"]["1"], len(fold["train_indices"]),
            len(fold["validation_indices"])))
    print("Manifest fingerprint:", manifest["fingerprint"])


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # Third-party exceptions can expose restricted paths/identifiers.
        print("CV preparation failed: " + (str(error) if isinstance(error, FoldError)
                                            else type(error).__name__), file=sys.stderr)
        sys.exit(1)
