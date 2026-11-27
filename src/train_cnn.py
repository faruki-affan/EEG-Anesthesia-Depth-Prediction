"""Train and compare waveform CNN and feature regressors on held-out cases."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.cnn import BISConvNet, normalize, predict_windows
from src.ml_data import MODEL_KEYS, assert_case_separation, load_feature_table
from src.ml_data import resolve_path, select_model_features, write_json
from src.ml_metrics import metrics_by_bis_range, regression_metrics
from src.phase2 import load_processed_windows
from src.predict import make_estimator
from src.vitaldb_loader import load_config

KEYS = ["case_id", "window_index", "window_start_sec", "window_end_sec", "BIS_target"]


def case_folds(case_ids, seed):
    ordered = sorted(set(case_ids))
    if len(ordered) < 3:
        raise ValueError("Training, validation and testing require at least three cases.")
    rotation = np.random.default_rng(seed).permutation(ordered).tolist()
    validation_cases = dict(zip(rotation, rotation[1:] + rotation[:1]))
    for test in ordered:
        candidates = [case for case in ordered if case != test]
        validation = validation_cases[test]
        training = [case for case in candidates if case != validation]
        assert_case_separation(training, [validation, test])
        yield training, validation, test


def load_training_cases(config):
    features = load_feature_table(config["phase2"]["features_csv"])
    window_config = config["windows"]
    directory = resolve_path(config["phase2"]["windows_dir"])
    cases = {}
    for case_id, group in features.groupby("case_id"):
        archive = load_processed_windows(directory / f"case_{case_id:05d}.npz")
        order = np.argsort(archive["window_index"])
        table = group.sort_values("window_index").reset_index(drop=True)
        eeg = archive["eeg"][order]
        matches = (
            archive["case_id"] == case_id
            and eeg.shape == (len(table), window_config["expected_window_samples"])
            and np.isfinite(eeg).all()
            and not table["window_index"].duplicated().any()
            and np.isclose(archive["sampling_rate_hz"], window_config["expected_sampling_rate_hz"], rtol=0.02)
        )
        if not matches or not all(
            np.allclose(table[column], archive[key][order], rtol=0, atol=1e-7)
            for column, key in [
                ("window_index", "window_index"),
                ("window_start_sec", "window_start_sec"),
                ("window_end_sec", "window_end_sec"),
                ("BIS_target", "bis_target"),
            ]
        ):
            raise ValueError(f"Case {case_id} has inconsistent feature and EEG window archives.")
        for edge in ("low", "high"):
            key = f"bandpass_{edge}_hz"
            if archive[key] != config["preprocessing"][key]:
                raise ValueError(f"Case {case_id} archive {key} differs from config.yaml.")
        cases[int(case_id)] = (table, eeg, archive["sampling_rate_hz"])
    return cases


def batches(eeg, bis, mean, scale, batch_size, seed=None):
    inputs = normalize(eeg, mean, scale)
    targets = torch.tensor(bis / 100, dtype=torch.float32)
    generator = None if seed is None else torch.Generator().manual_seed(seed)
    return DataLoader(TensorDataset(inputs, targets), batch_size=batch_size,
                      shuffle=seed is not None, generator=generator)


def run_epoch(model, loader, loss_fn, device, optimizer=None):
    model.train(optimizer is not None)
    total = 0.0
    with torch.set_grad_enabled(optimizer is not None):
        for eeg, target in loader:
            loss = loss_fn(model(eeg.to(device)), target.to(device))
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item() * len(target)
    return total / len(loader.dataset)


def fit_fold(cases, training, validation, test, config, device, path):
    cfg = config["phase4"]
    seed = cfg["seed"] + test
    torch.manual_seed(seed)
    train_frame = pd.concat([cases[case][0] for case in training], ignore_index=True)
    train_eeg = np.concatenate([cases[case][1] for case in training])
    val_frame, val_eeg, _ = cases[validation]
    mean = float(train_eeg.mean(dtype=np.float64))
    scale = max(float(train_eeg.std(dtype=np.float64)), 1e-6)
    batch_size = cfg["batch_size"]
    train_batches = batches(train_eeg, train_frame.BIS_target.to_numpy(), mean, scale, batch_size, seed)
    val_batches = batches(val_eeg, val_frame.BIS_target.to_numpy(), mean, scale, batch_size)
    model = BISConvNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=cfg["lr_patience"])
    loss_fn = nn.SmoothL1Loss(beta=0.1)
    history = []
    best_loss = float("inf")
    best = None
    for epoch in range(1, cfg["epochs"] + 1):
        train_loss = run_epoch(model, train_batches, loss_fn, device, optimizer)
        val_loss = run_epoch(model, val_batches, loss_fn, device)
        history.append(dict(test_case=test, epoch=epoch, train_loss=train_loss, validation_loss=val_loss))
        print(f"Test case {test} | epoch {epoch:02d} | train {train_loss:.4f} | validation {val_loss:.4f}", flush=True)
        scheduler.step(val_loss)
        if val_loss < best_loss:
            best_loss = val_loss
            best = dict(
                version=1, weights={key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
                mean=mean, scale=scale, samples=train_eeg.shape[1], batch_size=batch_size,
                sampling_rate_hz=float(config["windows"]["expected_sampling_rate_hz"]),
                train_cases=training, validation_case=validation, test_case=test,
                epoch=epoch, validation_loss=val_loss, seed=seed,
                signal_config={key: config[key] for key in ("windows", "preprocessing", "quality")},
            )
        if best is not None and epoch - best["epoch"] >= cfg["patience"]:
            break
    if best is None:
        raise ValueError(f"Validation loss never became finite for test case {test}.")
    temporary = path.with_suffix(".tmp")
    torch.save(best, temporary)
    temporary.replace(path)

    test_frame, test_eeg, fs = cases[test]
    results = test_frame[KEYS].copy()
    model.load_state_dict(best["weights"])
    results["pred_cnn"] = predict_windows((model.cpu().eval(), best), test_eeg, fs)
    feature_names = select_model_features(train_frame.columns)
    with threadpool_limits(limits=cfg["cpu_threads"]):
        for name in MODEL_KEYS:
            estimator = make_estimator(name, config["phase3"])
            if name == "random_forest":
                estimator.set_params(model__n_jobs=cfg["cpu_threads"])
            estimator.fit(train_frame[feature_names].to_numpy(), train_frame.BIS_target)
            values = estimator.predict(test_frame[feature_names].to_numpy())
            results[f"pred_{name}"] = np.clip(values, 0, 100)
    return results, history, {key: best[key] for key in ("train_cases", "validation_case", "test_case", "epoch")}


def run_phase4(config, device_name="auto"):
    from src.evaluate import save_evaluation_figures

    cfg = config["phase4"]
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if device_name == "auto" else device_name)
    torch.set_num_threads(cfg["cpu_threads"])
    torch.use_deterministic_algorithms(True, warn_only=True)
    cases = load_training_cases(config)
    models = resolve_path(cfg["models_dir"])
    reports = resolve_path(cfg["reports_dir"])
    models.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    predictions, history, folds = [], [], []
    for training, validation, test in case_folds(cases, cfg["seed"]):
        print(f"Training {training}, validation {validation}, test {test} on {device}", flush=True)
        result, losses, fold = fit_fold(
            cases, training, validation, test, config, device, models / f"test_case_{test:05d}.pt",
        )
        predictions.append(result)
        history.extend(losses)
        folds.append(fold)
    predictions = pd.concat(predictions, ignore_index=True)
    history = pd.DataFrame(history)
    predictions.to_csv(reports / "oof_predictions.csv", index=False)
    history.to_csv(reports / "loss.csv", index=False)
    metrics = {}
    ranges = []
    by_case = []
    for name in (*MODEL_KEYS, "cnn"):
        column = f"pred_{name}"
        metrics[name] = regression_metrics(predictions.BIS_target, predictions[column])
        for case, frame in predictions.groupby("case_id"):
            by_case.append(dict(model=name, case_id=int(case), **regression_metrics(frame.BIS_target, frame[column])))
        for row in metrics_by_bis_range(predictions.BIS_target, predictions[column], predictions.case_id):
            ranges.append(dict(model=name, **row))

    # Count the union of accepted time intervals, since neighboring windows overlap.
    duration = 0.0
    for _, frame in predictions.groupby("case_id"):
        end = -np.inf
        for start, stop in frame.sort_values("window_start_sec")[["window_start_sec", "window_end_sec"]].itertuples(index=False):
            duration += max(0.0, stop - max(start, end))
            end = max(end, stop)
    rejections = resolve_path(config["phase2"]["rejections_csv"])
    rejected = pd.read_csv(rejections).case_id.isin(cases).sum() if rejections.exists() else None
    summary = dict(models=metrics, by_case=by_case, folds=folds, test_cases=sorted(cases),
                   n_cases=len(cases), n_windows=len(predictions), usable_eeg_seconds=duration,
                   rejected_windows=rejected, device=str(device), configuration=cfg,
                   baseline_configuration=config["phase3"], torch_version=str(torch.__version__))
    write_json(reports / "metrics.json", summary)
    pd.DataFrame(ranges).to_csv(reports / "metrics_by_bis_range.csv", index=False)
    save_evaluation_figures(predictions, history, resolve_path(cfg["figures_dir"]))
    for name, result in metrics.items():
        print(f"{name}: MAE {result['mae']:.2f}, RMSE {result['rmse']:.2f}", flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    run_phase4(load_config(args.config), args.device)

