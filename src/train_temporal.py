"""Evaluate a temporal head against the single-window CNN and Random Forest."""

import argparse
from copy import deepcopy
import json

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.ml_data import resolve_path, select_model_features, write_json
from src.ml_metrics import metrics_by_bis_range, regression_metrics
from src.phase2 import run_phase2
from src.predict import make_estimator
from src.temporal import TemporalReadout, case_inputs, encode_cases, history_indices
from src.train_cnn import KEYS, fit_single_window, load_training_cases, run_epoch
from src.vitaldb_loader import load_config, run_phase1


def fit_temporal(cases, encoded, train, validation, count, hop, cfg, device):
    torch.manual_seed(cfg["seed"])
    train_inputs, train_targets = zip(*(case_inputs(case, cases, encoded, count, hop) for case in train))
    data = TensorDataset(torch.cat(train_inputs), torch.cat(train_targets))
    loader = DataLoader(data, batch_size=cfg["batch_size"], shuffle=True)
    val_inputs, val_targets = case_inputs(validation, cases, encoded, count, hop)
    model = TemporalReadout().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"])
    loss_fn = nn.SmoothL1Loss(beta=0.1)
    best = (float("inf"), None, 0)
    for epoch in range(1, cfg["temporal_epochs"] + 1):
        train_loss = run_epoch(model, loader, loss_fn, device, optimizer)
        model.eval()
        with torch.no_grad():
            pred = torch.cat([model(batch.to(device)).cpu() for batch in val_inputs.split(256)])
        val_mae = float((pred.clamp(0, 1) - val_targets).abs().mean() * 100)
        print(f"Temporal {count} windows | epoch {epoch:02d} | loss {train_loss:.4f} | validation MAE {val_mae:.3f}", flush=True)
        if val_mae < best[0]:
            best = (val_mae, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, epoch)
        if epoch - best[2] >= cfg["patience"]:
            break
    return best


def transition_metrics(frame, predicted, hop, threshold=60, tolerance=30):
    """Match each crossing to the closest unused same-direction crossing."""
    time = frame.window_end_sec.to_numpy()
    consecutive = history_indices(frame.window_index.to_numpy(), time, 2, hop)[:, 0] >= 0
    def crossings(values):
        direction = np.diff((values >= threshold).astype(np.int8))
        return {sign: time[1:][consecutive[1:] & (direction == sign)] for sign in (-1, 1)}
    actual = crossings(frame.BIS_target.to_numpy())
    estimates = crossings(frame[predicted].to_numpy())
    delays = []
    for sign in actual:
        unused = list(estimates[sign])
        for point in actual[sign]:
            if not unused:
                break
            nearest = int(np.argmin(np.abs(np.asarray(unused) - point)))
            if abs(unused[nearest] - point) <= tolerance:
                delays.append(float(unused.pop(nearest) - point))
    return {
        "reference_crossings": sum(map(len, actual.values())),
        "predicted_crossings": sum(map(len, estimates.values())),
        "matched_crossings": len(delays),
        "median_signed_lag_sec": float(np.median(delays)) if delays else None,
        "median_absolute_lag_sec": float(np.median(np.abs(delays))) if delays else None,
    }


def cohort_config(config, prepare):
    config = deepcopy(config)
    settings = config["temporal"]
    root = resolve_path(settings["cohort_dir"])
    config["paths"].update(metadata_dir=str(root / "metadata"), figures_dir=str(root / "acquisition_figures"))
    config["download"].update(max_cases=settings["max_cases"], max_attempts=24, crosscheck_vitalfile=False)
    config["phase2"].update(
        features_csv=str(root / "eeg_features.csv"), rejections_csv=str(root / "window_rejections.csv"),
        summary_json=str(root / "phase2_summary.json"), windows_dir=str(root / "windows"),
        figures_dir=str(root / "processing_figures"),
    )
    manifest = root / "metadata" / "selected_cases.json"
    if prepare and not manifest.exists():
        run_phase1(config)
    config["phase2"]["case_ids"] = json.loads(manifest.read_text(encoding="utf-8"))["accepted_caseids"]
    if prepare:
        run_phase2(config)
    return config


def run_experiment(config, device):
    settings = config["temporal"]
    cfg = {**config["phase4"], "epochs": settings["epochs"], "temporal_epochs": settings["epochs"]}
    config["phase4"] = cfg
    torch.set_num_threads(cfg["cpu_threads"])
    torch.use_deterministic_algorithms(True, warn_only=True)
    cases = load_training_cases(config)
    # The split uses IDs and a fixed seed. No case is chosen from its BIS values.
    ids = np.random.default_rng(cfg["seed"]).permutation(sorted(cases)).tolist()
    held_out = settings["test_cases"]
    if held_out < 1 or len(ids) < held_out + 2:
        raise ValueError("The cohort needs distinct training, validation, and test cases.")
    test, validation, train = sorted(ids[:held_out]), ids[held_out], sorted(ids[held_out + 1:])
    print(f"Training {train}, validation {validation}, test {test}", flush=True)
    models = resolve_path(settings["models_dir"])
    reports = resolve_path(settings["reports_dir"])
    models.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    baseline, checkpoint, loss = fit_single_window(
        cases, train, validation, config, device, models / "single_window.pt", cfg["seed"],
    )
    baseline.load_state_dict(checkpoint["weights"])
    dev_cases = {case: cases[case] for case in [*train, validation]}
    encoded = encode_cases(baseline, checkpoint, dev_cases)
    hop = config["windows"]["length_sec"] * (1 - config["windows"]["overlap_fraction"])
    candidates = {count: fit_temporal(cases, encoded, train, validation, count, hop, cfg, device)
                  for count in settings["history_windows"]}
    count = min(candidates, key=lambda candidate: candidates[candidate][0])
    temporal = TemporalReadout().to(device)
    temporal.load_state_dict(candidates[count][1])
    temporal.eval()
    temporary = models / "temporal.pt.tmp"
    torch.save({"version": 1, "encoder": checkpoint, "head": candidates[count][1],
                "history_windows": count, "hop_sec": hop, "test_cases": test}, temporary)
    temporary.replace(models / "temporal.pt")

    train_frame = pd.concat([cases[case][0] for case in train], ignore_index=True)
    features = select_model_features(train_frame.columns)
    forest = make_estimator("random_forest", config["phase3"])
    forest.set_params(model__n_jobs=cfg["cpu_threads"])
    with threadpool_limits(limits=cfg["cpu_threads"]):
        forest.fit(train_frame[features].to_numpy(), train_frame.BIS_target)
    encoded = encode_cases(baseline, checkpoint, {case: cases[case] for case in test})
    rows = []
    for case in test:
        frame = cases[case][0]
        result = frame[KEYS].copy()
        result["single_window"] = np.clip(encoded[case][1] * 100, 0, 100)
        inputs, _ = case_inputs(case, cases, encoded, count, hop)
        with torch.inference_mode():
            result["temporal"] = torch.cat([temporal(batch.to(device)).cpu() for batch in inputs.split(256)]).mul(100).clamp(0, 100).numpy()
        result["random_forest"] = np.clip(forest.predict(frame[features].to_numpy()), 0, 100)
        rows.append(result)
    predictions = pd.concat(rows, ignore_index=True)
    predictions.to_csv(reports / "test_predictions.csv", index=False)
    pd.DataFrame(loss).to_csv(reports / "single_window_loss.csv", index=False)
    metrics, ranges, by_case, transitions = {}, [], [], []
    for name in ("single_window", "temporal", "random_forest"):
        metrics[name] = regression_metrics(predictions.BIS_target, predictions[name])
        case_scores = []
        for case, frame in predictions.groupby("case_id"):
            score = regression_metrics(frame.BIS_target, frame[name])
            case_scores.append(score["mae"])
            by_case.append(dict(case_id=int(case), model=name, **score))
            transitions.append(dict(case_id=int(case), model=name, **transition_metrics(frame, name, hop)))
        metrics[name]["case_averaged_mae"] = float(np.mean(case_scores))
        ranges.extend(dict(model=name, **row) for row in metrics_by_bis_range(
            predictions.BIS_target, predictions[name], predictions.case_id))
        print(f"{name}: pooled MAE {metrics[name]['mae']:.3f}, case-averaged MAE {metrics[name]['case_averaged_mae']:.3f}", flush=True)
    pd.DataFrame(ranges).to_csv(reports / "metrics_by_bis_range.csv", index=False)
    pd.DataFrame(by_case).to_csv(reports / "metrics_by_case.csv", index=False)
    pd.DataFrame(transitions).to_csv(reports / "transitions.csv", index=False)
    summary = dict(
        training_cases=train, validation_case=validation, test_cases=test, history_windows=count,
        history_seconds=config["windows"]["length_sec"] + (count - 1) * hop,
        validation_mae={length: score[0] for length, score in candidates.items()},
        validation_epoch={length: score[2] for length, score in candidates.items()},
        n_test_windows=len(predictions), models=metrics,
        transition_threshold_bis=60, transition_matching_tolerance_sec=30,
        seed=cfg["seed"], device=str(device), training_configuration=cfg,
    )
    write_json(reports / "metrics.json", summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true", help="Download and process the expanded cohort.")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    run_experiment(cohort_config(load_config(), args.prepare), torch.device(args.device))


if __name__ == "__main__":
    main()
