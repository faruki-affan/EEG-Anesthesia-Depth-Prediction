"""Dashboard case discovery and educational BIS labels."""

from src.ml_data import resolve_path


def available_cases(config):
    raw = resolve_path(config["paths"]["raw_dir"])
    processed = resolve_path(config["phase2"]["windows_dir"])
    paths = [*raw.glob("case_*/eeg_primary.npz"), *processed.glob("case_*.npz")]
    names = [path.parent.name if path.name == "eeg_primary.npz" else path.stem for path in paths]
    return sorted({int(name[5:]) for name in names if name[5:].isdigit()})


def educational_range(value):
    if value <= 20:
        return "Very low BIS, associated with marked EEG suppression"
    if value <= 40:
        return "Low BIS, associated with a deeper hypnotic state"
    if value <= 60:
        return "Range commonly associated with general anesthesia"
    if value <= 80:
        return "Range associated with lighter hypnosis"
    return "High BIS, associated with wakefulness or lighter sedation"

