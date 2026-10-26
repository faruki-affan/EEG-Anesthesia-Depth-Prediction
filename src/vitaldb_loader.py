"""Download VitalDB metadata and EEG/BIS tracks for a small case subset.

Track names are confirmed against the live ``/trks`` index. Signals are read
from the documented gzip-CSV track endpoint ``https://api.vitaldb.net/{tid}``.
The official library warns that those CSV timestamps can differ from the packed
``.vital`` timeline by a per-case constant, so one accepted case can be
cross-checked with ``vitaldb.VitalFile``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import yaml
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CLINICAL_COLUMNS = (
    "caseid",
    "subjectid",
    "caseend",
    "anestart",
    "aneend",
    "opstart",
    "opend",
    "ane_type",
    "age",
    "sex",
    "department",
    "optype",
    "emop",
)


class VitalDBError(Exception):
    """Base error for dataset access."""


class DownloadError(VitalDBError):
    """A remote file could not be retrieved intact."""


class TrackFormatError(VitalDBError):
    """A downloaded track did not match the documented CSV layout."""


@dataclass
class WaveSummary:
    track_name: str
    tid: str
    unit: str
    n_samples: int
    n_missing: int
    missing_fraction: float
    t_start_sec: float | None
    sample_interval_sec: float | None
    t_end_header_sec: float | None
    sampling_rate_hz: float | None
    duration_sec: float | None
    finite_time_header: list[float]
    value_min: float | None
    value_max: float | None
    value_std: float | None
    flatline: bool
    sha256: str


@dataclass
class NumericSummary:
    track_name: str
    tid: str
    unit: str
    n_samples: int
    t_start_sec: float | None
    t_end_sec: float | None
    duration_sec: float | None
    median_interval_sec: float | None
    max_gap_sec: float | None
    n_gaps_over_5s: int
    value_min: float | None
    value_max: float | None
    value_median: float | None
    fraction_in_0_100: float | None
    sha256: str


@dataclass
class CaseRecord:
    caseid: int
    subjectid: int | None
    accepted: bool
    reason: str
    tracks_present: list[str]
    clinical: dict[str, Any] = field(default_factory=dict)
    eeg: WaveSummary | None = None
    bis: NumericSummary | None = None
    optional: list[NumericSummary] = field(default_factory=list)
    overlap_sec: float | None = None
    sampling_rate_matches_catalog: bool | None = None
    vitalfile_crosscheck: dict[str, Any] | None = None


def load_config(path: Path | None = None) -> dict[str, Any]:
    config_path = path or (PROJECT_ROOT / "config.yaml")
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise VitalDBError(f"Configuration at {config_path} is empty.")
    return config


def project_path(config: dict[str, Any], key: str) -> Path:
    relative = Path(config["paths"][key])
    path = relative if relative.is_absolute() else PROJECT_ROOT / relative
    path.mkdir(parents=True, exist_ok=True)
    return path


def _session(config: dict[str, Any]) -> requests.Session:
    retries = int(config["vitaldb"]["max_retries"])
    retry = Retry(
        total=retries,
        connect=retries,
        read=retries,
        status=retries,
        backoff_factor=float(config["vitaldb"]["retry_backoff_sec"]),
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False,
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers["User-Agent"] = "eeg-anesthesia-depth-phase1/0.1 (educational)"
    return session


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sniff(path: Path) -> str:
    with path.open("rb") as handle:
        head = handle.read(256)
    if head.startswith(b"\x1f\x8b"):
        return "gzip"
    lowered = head.lower()
    if lowered.startswith(b"<") or lowered.startswith(b"<!doctype") or b"<html" in lowered:
        raise DownloadError(f"{path.name} is an HTML error page, not a CSV track.")
    if not head:
        raise DownloadError(f"{path.name} is empty.")
    return "csv"


def _stored_table(directory: Path, stem: str) -> Path:
    """Return the on-disk table, whether the response was left gzip-compressed."""
    for name in (f"{stem}.csv", f"{stem}.csv.gz"):
        candidate = directory / name
        if candidate.exists():
            return candidate
    return directory / f"{stem}.csv.gz"


def _match_suffix_to_contents(path: Path) -> Path:
    """Requests may already decode Content-Encoding, so a .gz name can be plain CSV."""
    kind = _sniff(path)
    if kind == "csv" and path.name.endswith(".gz"):
        target = path.with_name(path.name[:-3])
        target.unlink(missing_ok=True)
        path.replace(target)
        return target
    if kind == "gzip" and not path.name.endswith(".gz"):
        target = path.with_name(path.name + ".gz")
        target.unlink(missing_ok=True)
        path.replace(target)
        return target
    return path


def download_file(
    session: requests.Session,
    url: str,
    dest: Path,
    timeout: float,
    refresh: bool = False,
) -> Path:
    """Stream a URL to disk. A partial file is removed if the transfer fails."""
    if dest.exists() and not refresh:
        try:
            dest = _match_suffix_to_contents(dest)
            logger.info("  using cached %s", dest.name)
            return dest
        except DownloadError:
            dest.unlink(missing_ok=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary = dest.with_suffix(dest.suffix + ".part")
    try:
        with session.get(url, stream=True, timeout=timeout) as response:
            if response.status_code != 200:
                raise DownloadError(f"{url} returned HTTP {response.status_code}.")
            written = 0
            next_report = 5 * 1024 * 1024
            with temporary.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1 << 16):
                    if not chunk:
                        continue
                    handle.write(chunk)
                    written += len(chunk)
                    if written >= next_report:
                        logger.info("  downloaded %.1f MB from %s", written / 1e6, url)
                        next_report += 5 * 1024 * 1024
        if written == 0:
            raise DownloadError(f"{url} returned an empty body.")
        temporary.replace(dest)
        dest = _match_suffix_to_contents(dest)
    except requests.RequestException as exc:
        temporary.unlink(missing_ok=True)
        raise DownloadError(f"Network failure while downloading {url}: {exc}") from exc
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    _sniff(dest)
    return dest


def _read_csv(path: Path, dtype: dict[str, Any] | None = None) -> pd.DataFrame:
    kind = _sniff(path)
    kwargs = {
        "compression": "gzip" if kind == "gzip" else None,
        "na_values": ["-nan(ind)"],
        "dtype": dtype,
        "low_memory": False,
    }
    frame = pd.read_csv(path, **kwargs)
    first = str(frame.columns[0]).strip().lstrip("\ufeff")
    if first.lower() not in {"time", "dt", "t"} and _is_number(first):
        frame = pd.read_csv(path, header=None, **kwargs)
    return frame


def _is_number(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return None if np.isnan(number) else number
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, Path):
        return str(value)
    return value


def _finite_stats(values: np.ndarray) -> tuple[float | None, float | None, float | None, int]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None, None, None, 0
    return float(np.min(finite)), float(np.max(finite)), float(np.std(finite)), int(finite.size)


def parse_waveform(frame: pd.DataFrame, track_name: str) -> dict[str, Any]:
    """Reconstruct a waveform from the documented three-value time header.

    VitalDB waveform CSVs leave the time blank on sample rows. The finite time
    entries are the start, the sample interval, and the end time.
    """
    if frame.shape[1] < 2:
        raise TrackFormatError(f"{track_name} does not have Time and Value columns.")
    time = pd.to_numeric(frame.iloc[:, 0], errors="coerce").to_numpy(dtype=np.float64)
    values = pd.to_numeric(frame.iloc[:, 1], errors="coerce").to_numpy(dtype=np.float32)
    finite_mask = np.isfinite(time)
    finite_time = time[finite_mask]
    if finite_time.size == 0:
        raise TrackFormatError(f"{track_name} has no finite time values.")
    if finite_time.size > 10 and finite_mask.mean() > 0.5:
        return _parse_timestamped_waveform(time, values, finite_time)
    if finite_time.size < 2:
        raise TrackFormatError(
            f"{track_name} waveform header has {finite_time.size} finite time value; "
            "the documentation describes start, interval, and end."
        )

    t_start = float(finite_time[0])
    interval = float(finite_time[1])
    t_end = float(finite_time[-1]) if finite_time.size >= 3 else None
    if not np.isfinite(interval) or interval <= 0:
        raise TrackFormatError(f"{track_name} sample interval is {interval}.")

    duration = None if t_end is None else float(t_end - t_start)
    sampling_rate = float(1.0 / interval)
    n_missing = int(np.isnan(values).sum())
    vmin, vmax, vstd, n_finite = _finite_stats(values)
    return {
        "time_header": [float(item) for item in finite_time[:8]],
        "values": values,
        "t_start_sec": t_start,
        "sample_interval_sec": interval,
        "t_end_header_sec": t_end,
        "sampling_rate_hz": sampling_rate,
        "duration_sec": duration,
        "n_samples": int(values.size),
        "n_missing": n_missing,
        "missing_fraction": float(n_missing / values.size) if values.size else 1.0,
        "value_min": vmin,
        "value_max": vmax,
        "value_std": vstd,
        "n_finite": n_finite,
        "flatline": bool(n_finite > 100 and vstd is not None and vstd < 1e-3),
    }


def parse_numeric(frame: pd.DataFrame, track_name: str) -> dict[str, Any]:
    """Read a numeric track. Missing rows are already omitted by VitalDB."""
    if frame.shape[1] < 2:
        raise TrackFormatError(f"{track_name} does not have Time and Value columns.")
    time = pd.to_numeric(frame.iloc[:, 0], errors="coerce").to_numpy(dtype=np.float64)
    values = pd.to_numeric(frame.iloc[:, 1], errors="coerce").to_numpy(dtype=np.float64)
    valid = np.isfinite(time) & np.isfinite(values)
    time = time[valid]
    values = values[valid]
    if time.size == 0:
        raise TrackFormatError(f"{track_name} has no finite samples.")
    order = np.argsort(time, kind="mergesort")
    time = time[order]
    values = values[order]
    gaps = np.diff(time) if time.size > 1 else np.empty(0)
    positive_gaps = gaps[gaps > 0]
    median_interval = float(np.median(positive_gaps)) if positive_gaps.size else None
    return {
        "time": time,
        "values": values,
        "n_samples": int(time.size),
        "t_start_sec": float(time[0]),
        "t_end_sec": float(time[-1]),
        "duration_sec": float(time[-1] - time[0]),
        "median_interval_sec": median_interval,
        "max_gap_sec": float(np.max(gaps)) if gaps.size else 0.0,
        "n_gaps_over_5s": int(np.sum(gaps > 5.0)) if gaps.size else 0,
        "value_min": float(np.min(values)),
        "value_max": float(np.max(values)),
        "value_median": float(np.median(values)),
        "fraction_in_0_100": float(np.mean((values >= 0.0) & (values <= 100.0))),
    }


def _save_waveform(path: Path, parsed: dict[str, Any]) -> None:
    np.savez_compressed(
        path,
        values=parsed["values"],
        t_start_sec=np.float64(parsed["t_start_sec"]),
        sample_interval_sec=np.float64(parsed["sample_interval_sec"]),
        t_end_header_sec=np.float64(
            np.nan if parsed["t_end_header_sec"] is None else parsed["t_end_header_sec"]
        ),
    )


def _save_numeric(path: Path, parsed: dict[str, Any]) -> None:
    np.savez_compressed(path, time=parsed["time"], values=parsed["values"])


def load_waveform(path: Path) -> dict[str, Any]:
    with np.load(path) as archive:
        t_end = float(archive["t_end_header_sec"])
        return {
            "values": archive["values"].astype(np.float32, copy=False),
            "t_start_sec": float(archive["t_start_sec"]),
            "sample_interval_sec": float(archive["sample_interval_sec"]),
            "t_end_header_sec": None if np.isnan(t_end) else t_end,
        }


def load_numeric(path: Path) -> dict[str, Any]:
    with np.load(path) as archive:
        return {"time": archive["time"], "values": archive["values"]}


class VitalDBLoader:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.api_base = str(config["vitaldb"]["api_base"]).rstrip("/")
        self.timeout = float(config["vitaldb"]["request_timeout_sec"])
        self.metadata_dir = project_path(config, "metadata_dir")
        self.raw_dir = project_path(config, "raw_dir")
        self.session = _session(config)

    def ensure_index(self, refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
        trks_path = _stored_table(self.metadata_dir, "trks")
        cases_path = _stored_table(self.metadata_dir, "cases")
        if refresh or not trks_path.exists():
            logger.info("Downloading track index from %s/trks", self.api_base)
            download_file(self.session, f"{self.api_base}/trks", trks_path, self.timeout)
        if refresh or not cases_path.exists():
            logger.info("Downloading clinical table from %s/cases", self.api_base)
            download_file(self.session, f"{self.api_base}/cases", cases_path, self.timeout)
        trks = _read_csv(trks_path, dtype={"caseid": "int64", "tname": "string", "tid": "string"})
        cases = _read_csv(cases_path)
        self._validate_index(trks, cases)
        return trks, cases

    @staticmethod
    def _validate_index(trks: pd.DataFrame, cases: pd.DataFrame) -> None:
        missing = {"caseid", "tname", "tid"} - set(trks.columns)
        if missing:
            raise TrackFormatError(f"Track index is missing columns: {sorted(missing)}")
        if "caseid" not in cases.columns:
            raise TrackFormatError("Clinical table is missing caseid.")
        if trks["tid"].isna().any() or trks["tname"].isna().any():
            raise TrackFormatError("Track index contains blank track names or identifiers.")

    def discover(self, trks: pd.DataFrame, cases: pd.DataFrame) -> dict[str, Any]:
        names = trks["tname"].astype(str)
        bis_related = sorted(names[names.str.contains("BIS|EEG", case=False, regex=True)].unique())
        counts = (
            trks[trks["tname"].isin(bis_related)]
            .groupby("tname")["caseid"]
            .nunique()
            .sort_values(ascending=False)
        )
        primary = self.config["tracks"]["eeg_primary"]
        secondary = self.config["tracks"]["eeg_secondary"]
        bis_name = self.config["tracks"]["bis"]
        missing_names = [name for name in (primary, secondary, bis_name) if name not in set(counts.index)]
        primary_ids = set(trks.loc[trks["tname"] == primary, "caseid"].astype(int))
        secondary_ids = set(trks.loc[trks["tname"] == secondary, "caseid"].astype(int))
        bis_ids = set(trks.loc[trks["tname"] == bis_name, "caseid"].astype(int))
        both = sorted(primary_ids & bis_ids)
        clinical = cases.copy()
        clinical["caseid"] = clinical["caseid"].astype(int)
        both_clinical = clinical[clinical["caseid"].isin(both)]
        subject_counts = None
        if "subjectid" in both_clinical.columns:
            subject_counts = both_clinical.groupby("subjectid")["caseid"].nunique()
        ane_counts = None
        if "ane_type" in both_clinical.columns:
            ane_counts = both_clinical["ane_type"].fillna("missing").value_counts().to_dict()
        report = {
            "api_base": self.api_base,
            "n_tracks": int(len(trks)),
            "n_cases_in_clinical_table": int(clinical["caseid"].nunique()),
            "bis_or_eeg_track_names": bis_related,
            "cases_per_track": {str(name): int(count) for name, count in counts.items()},
            "configured_names_missing_from_index": missing_names,
            "n_cases_with_primary_eeg": len(primary_ids),
            "n_cases_with_secondary_eeg": len(secondary_ids),
            "n_cases_with_bis": len(bis_ids),
            "n_cases_with_primary_eeg_and_bis": len(both),
            "n_cases_with_both_eeg_channels_and_bis": len(primary_ids & secondary_ids & bis_ids),
            "n_primary_eeg_without_bis": len(primary_ids - bis_ids),
            "n_bis_without_primary_eeg": len(bis_ids - primary_ids),
            "n_unique_subjects_with_eeg_and_bis": (
                None if subject_counts is None else int(subject_counts.shape[0])
            ),
            "n_subjects_with_multiple_eeg_bis_cases": (
                None if subject_counts is None else int((subject_counts > 1).sum())
            ),
            "anesthesia_type_counts_for_eeg_bis_cases": ane_counts,
            "candidate_caseids": both,
        }
        destination = self.metadata_dir / "discovery.json"
        destination.write_text(json.dumps(_jsonable(report), indent=2), encoding="utf-8")
        logger.info(
            "Index has %s tracks. %s cases contain %s and %s.",
            report["n_tracks"],
            report["n_cases_with_primary_eeg_and_bis"],
            primary,
            bis_name,
        )
        if missing_names:
            logger.warning("Configured track names absent from the live index: %s", missing_names)
        return report

    def _clinical_row(self, cases: pd.DataFrame, caseid: int) -> dict[str, Any]:
        rows = cases[cases["caseid"].astype(int) == caseid]
        if rows.empty:
            return {}
        row = rows.iloc[0]
        available = [column for column in CLINICAL_COLUMNS if column in rows.columns]
        payload = {}
        for column in available:
            value = row[column]
            if pd.isna(value):
                payload[column] = None
            elif column in {"caseid", "subjectid"}:
                payload[column] = int(value)
            else:
                payload[column] = value.item() if hasattr(value, "item") else value
        return payload

    def _tid(self, trks: pd.DataFrame, caseid: int, track_name: str) -> str | None:
        rows = trks[(trks["caseid"].astype(int) == caseid) & (trks["tname"] == track_name)]
        if rows.empty:
            return None
        if len(rows) > 1:
            logger.warning("Case %s has %s rows for %s; using the first.", caseid, len(rows), track_name)
        return str(rows.iloc[0]["tid"])

    def _download_track(self, case_dir: Path, track_name: str, tid: str) -> Path:
        slug = track_name.replace("/", "__")
        destination = case_dir / f"{slug}.csv.gz"
        url = f"{self.api_base}/{tid}"
        logger.info("  GET %s (%s)", track_name, tid)
        return download_file(self.session, url, destination, self.timeout)

    def _catalog(self, track_name: str) -> dict[str, Any]:
        return dict(self.config["tracks"]["catalog"].get(track_name, {}))

    def download_cases(self, trks: pd.DataFrame, cases: pd.DataFrame, discovery: dict[str, Any]) -> list[CaseRecord]:
        if discovery["configured_names_missing_from_index"]:
            raise VitalDBError(
                "Refusing to download because these configured tracks are not in the live index: "
                + ", ".join(discovery["configured_names_missing_from_index"])
            )
        settings = self.config["download"]
        wanted = int(settings["max_cases"])
        attempts = int(settings["max_attempts"])
        records: list[CaseRecord] = []
        accepted = 0
        for caseid in discovery["candidate_caseids"]:
            if accepted >= wanted or len(records) >= attempts:
                break
            logger.info("Case %s (%s accepted, attempt %s/%s)", caseid, accepted, len(records) + 1, attempts)
            record = self._download_one(trks, cases, int(caseid))
            records.append(record)
            if record.accepted:
                accepted += 1
            else:
                logger.info("  excluded case %s: %s", caseid, record.reason)
        if accepted and settings.get("crosscheck_vitalfile", False):
            first = next(record for record in records if record.accepted)
            first.vitalfile_crosscheck = self._crosscheck_vitalfile(first.caseid)
        self._write_case_report(records)
        return records

    def _download_one(self, trks: pd.DataFrame, cases: pd.DataFrame, caseid: int) -> CaseRecord:
        primary = self.config["tracks"]["eeg_primary"]
        bis_name = self.config["tracks"]["bis"]
        secondary = self.config["tracks"]["eeg_secondary"]
        optional_names = list(self.config["tracks"].get("optional", []))
        present = []
        tids = {}
        for name in [primary, secondary, bis_name, *optional_names]:
            tid = self._tid(trks, caseid, name)
            if tid is not None:
                present.append(name)
                tids[name] = tid
        clinical = self._clinical_row(cases, caseid)
        subjectid = clinical.get("subjectid")
        case_dir = self.raw_dir / f"case_{caseid:05d}"
        if primary not in tids or bis_name not in tids:
            return CaseRecord(
                caseid=caseid,
                subjectid=subjectid,
                accepted=False,
                reason="Primary EEG or BIS track id is missing from the index.",
                tracks_present=present,
                clinical=clinical,
            )
        try:
            eeg_path = self._download_track(case_dir, primary, tids[primary])
            bis_path = self._download_track(case_dir, bis_name, tids[bis_name])
            eeg_frame = _read_csv(eeg_path)
            bis_frame = _read_csv(bis_path)
            eeg = parse_waveform(eeg_frame, primary)
            bis = parse_numeric(bis_frame, bis_name)
        except (DownloadError, TrackFormatError, pd.errors.ParserError, OSError, EOFError) as exc:
            return CaseRecord(
                caseid=caseid,
                subjectid=subjectid,
                accepted=False,
                reason=f"{type(exc).__name__}: {exc}",
                tracks_present=present,
                clinical=clinical,
            )

        _save_waveform(case_dir / "eeg_primary.npz", eeg)
        _save_numeric(case_dir / "bis.npz", bis)
        self._download_secondary(case_dir, secondary, tids.get(secondary))
        eeg_summary = WaveSummary(
            track_name=primary,
            tid=tids[primary],
            unit=str(self._catalog(primary).get("unit", "")),
            n_samples=eeg["n_samples"],
            n_missing=eeg["n_missing"],
            missing_fraction=eeg["missing_fraction"],
            t_start_sec=eeg["t_start_sec"],
            sample_interval_sec=eeg["sample_interval_sec"],
            t_end_header_sec=eeg["t_end_header_sec"],
            sampling_rate_hz=eeg["sampling_rate_hz"],
            duration_sec=eeg["duration_sec"],
            finite_time_header=eeg["time_header"],
            value_min=eeg["value_min"],
            value_max=eeg["value_max"],
            value_std=eeg["value_std"],
            flatline=eeg["flatline"],
            sha256=_sha256(eeg_path),
        )
        bis_summary = NumericSummary(
            track_name=bis_name,
            tid=tids[bis_name],
            unit=str(self._catalog(bis_name).get("unit", "")),
            n_samples=bis["n_samples"],
            t_start_sec=bis["t_start_sec"],
            t_end_sec=bis["t_end_sec"],
            duration_sec=bis["duration_sec"],
            median_interval_sec=bis["median_interval_sec"],
            max_gap_sec=bis["max_gap_sec"],
            n_gaps_over_5s=bis["n_gaps_over_5s"],
            value_min=bis["value_min"],
            value_max=bis["value_max"],
            value_median=bis["value_median"],
            fraction_in_0_100=bis["fraction_in_0_100"],
            sha256=_sha256(bis_path),
        )
        optional_summaries = self._download_optional(case_dir, optional_names, tids)
        reason = self._quality_reason(eeg_summary, bis_summary)
        overlap = _overlap_seconds(eeg_summary, bis_summary)
        if reason is None and (overlap is None or overlap < float(self.config["download"]["min_overlap_sec"])):
            reason = f"EEG and BIS overlap is {overlap} seconds."
        expected_rate = float(self._catalog(primary).get("sampling_rate_hz", 128))
        tolerance = float(self.config["download"]["sampling_rate_tolerance"])
        rate_ok = (
            eeg_summary.sampling_rate_hz is not None
            and abs(eeg_summary.sampling_rate_hz - expected_rate) / expected_rate <= tolerance
        )
        if reason is None and not rate_ok:
            reason = (
                f"Measured EEG sampling rate {eeg_summary.sampling_rate_hz} Hz "
                f"is outside {tolerance:.0%} of the catalog rate {expected_rate} Hz."
            )
        return CaseRecord(
            caseid=caseid,
            subjectid=subjectid,
            accepted=reason is None,
            reason="accepted" if reason is None else reason,
            tracks_present=present,
            clinical=clinical,
            eeg=eeg_summary,
            bis=bis_summary,
            optional=optional_summaries,
            overlap_sec=overlap,
            sampling_rate_matches_catalog=rate_ok,
        )

    def _download_secondary(self, case_dir: Path, track_name: str, tid: str | None) -> None:
        if tid is None:
            return
        try:
            path = self._download_track(case_dir, track_name, tid)
            parsed = parse_waveform(_read_csv(path), track_name)
            _save_waveform(case_dir / "eeg_secondary.npz", parsed)
        except (DownloadError, TrackFormatError, pd.errors.ParserError, OSError, EOFError) as exc:
            logger.warning("  secondary EEG %s skipped: %s", track_name, exc)

    def _download_optional(
        self,
        case_dir: Path,
        names: list[str],
        tids: dict[str, str],
    ) -> list[NumericSummary]:
        summaries = []
        for name in names:
            tid = tids.get(name)
            if tid is None:
                continue
            try:
                path = self._download_track(case_dir, name, tid)
                parsed = parse_numeric(_read_csv(path), name)
            except (DownloadError, TrackFormatError, pd.errors.ParserError, OSError, EOFError) as exc:
                logger.warning("  optional track %s skipped: %s", name, exc)
                continue
            slug = name.replace("/", "__")
            _save_numeric(case_dir / f"{slug}.npz", parsed)
            summaries.append(
                NumericSummary(
                    track_name=name,
                    tid=tid,
                    unit=str(self._catalog(name).get("unit", "")),
                    n_samples=parsed["n_samples"],
                    t_start_sec=parsed["t_start_sec"],
                    t_end_sec=parsed["t_end_sec"],
                    duration_sec=parsed["duration_sec"],
                    median_interval_sec=parsed["median_interval_sec"],
                    max_gap_sec=parsed["max_gap_sec"],
                    n_gaps_over_5s=parsed["n_gaps_over_5s"],
                    value_min=parsed["value_min"],
                    value_max=parsed["value_max"],
                    value_median=parsed["value_median"],
                    fraction_in_0_100=parsed["fraction_in_0_100"],
                    sha256=_sha256(path),
                )
            )
        return summaries

    def _quality_reason(self, eeg: WaveSummary, bis: NumericSummary) -> str | None:
        settings = self.config["download"]
        if eeg.n_samples == 0 or eeg.value_std is None:
            return "EEG track has no finite samples."
        if eeg.flatline:
            return "EEG track is a flatline."
        if eeg.missing_fraction > float(settings["max_eeg_missing_fraction"]):
            return f"EEG missing fraction is {eeg.missing_fraction:.2%}."
        if eeg.duration_sec is None or eeg.duration_sec < float(settings["min_overlap_sec"]):
            return f"EEG duration is {eeg.duration_sec} seconds."
        if bis.n_samples < 60:
            return f"BIS track has only {bis.n_samples} samples."
        if bis.duration_sec is None or bis.duration_sec < float(settings["min_overlap_sec"]):
            return f"BIS duration is {bis.duration_sec} seconds."
        if bis.fraction_in_0_100 is not None and bis.fraction_in_0_100 < 0.95:
            return f"Only {bis.fraction_in_0_100:.1%} of BIS values lie in 0–100."
        gap_limit = float(settings["max_bis_missing_gap_sec"])
        if bis.max_gap_sec is not None and bis.max_gap_sec > gap_limit:
            logger.warning(
                "  BIS has a %.1f s gap (threshold %.1f s). The case is kept and the gap is recorded.",
                bis.max_gap_sec,
                gap_limit,
            )
        return None

    def _crosscheck_vitalfile(self, caseid: int) -> dict[str, Any]:
        """Compare CSV length with the packed vital file for one case.

        A mismatch is recorded and does not delete the CSV download. Phase 2
        should use the timeline that this check shows is reliable.
        """
        primary = self.config["tracks"]["eeg_primary"]
        bis_name = self.config["tracks"]["bis"]
        version = self.config["vitaldb"]["dataset_version"]
        url = f"{self.api_base}/{version}/{caseid}.vital"
        try:
            import vitaldb
        except ImportError as exc:
            return {"status": "skipped", "reason": f"vitaldb import failed: {exc}"}
        try:
            logger.info("Cross-checking case %s against %s", caseid, url)
            vital = vitaldb.VitalFile(int(caseid), [primary, bis_name])
            track_names = list(vital.get_track_names())
            eeg_track = vital.trks.get(primary)
            eeg_rate = None if eeg_track is None else eeg_track.srate
            array = vital.to_numpy([primary, bis_name], 1.0 / 128.0)
        except Exception as exc:
            return {"status": "failed", "url": url, "reason": f"{type(exc).__name__}: {exc}"}
        if array is None or len(np.shape(array)) != 2 or array.shape[1] < 2:
            return {"status": "failed", "url": url, "reason": "VitalFile.to_numpy returned no EEG/BIS array."}
        eeg = array[:, 0]
        bis = array[:, 1]
        local = load_waveform(self.raw_dir / f"case_{caseid:05d}" / "eeg_primary.npz")
        return {
            "status": "ok",
            "url": url,
            "vital_track_names": track_names,
            "vital_eeg_srate": None if eeg_rate is None else float(eeg_rate),
            "vital_samples_at_128hz": int(array.shape[0]),
            "vital_eeg_finite": int(np.isfinite(eeg).sum()),
            "vital_bis_finite": int(np.isfinite(bis).sum()),
            "csv_eeg_samples": int(local["values"].size),
            "sample_count_ratio_csv_over_vital": (
                float(local["values"].size / array.shape[0]) if array.shape[0] else None
            ),
        }

    def _write_case_report(self, records: list[CaseRecord]) -> None:
        payload = {
            "accepted_caseids": [record.caseid for record in records if record.accepted],
            "excluded": [
                {"caseid": record.caseid, "reason": record.reason}
                for record in records
                if not record.accepted
            ],
            "cases": [_jsonable(asdict(record)) for record in records],
        }
        path = self.metadata_dir / "selected_cases.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("Wrote %s", path)


def _parse_timestamped_waveform(
    time: np.ndarray,
    values: np.ndarray,
    finite_time: np.ndarray,
) -> dict[str, Any]:
    """Fallback for a waveform whose time column is filled on every row."""
    valid = np.isfinite(time)
    time = time[valid]
    values = values[valid]
    finite_time = np.sort(finite_time)
    gaps = np.diff(finite_time)
    positive = gaps[gaps > 0]
    if positive.size == 0:
        raise TrackFormatError("Timestamped waveform has no increasing time values.")
    interval = float(np.median(positive))
    if np.max(positive) > interval * 1.05 or np.min(positive) < interval * 0.95:
        raise TrackFormatError(
            "Timestamped waveform is not uniformly sampled "
            f"(median interval {interval}, range {float(np.min(positive))}–{float(np.max(positive))})."
        )
    order = np.argsort(time, kind="mergesort")
    values = values[order]
    n_missing = int(np.isnan(values).sum())
    vmin, vmax, vstd, n_finite = _finite_stats(values)
    t_start = float(finite_time[0] if finite_time[0] <= finite_time[-1] else finite_time[-1])
    t_end = float(max(finite_time[0], finite_time[-1]))
    return {
        "time_header": [t_start, interval, t_end],
        "values": values.astype(np.float32, copy=False),
        "t_start_sec": t_start,
        "sample_interval_sec": interval,
        "t_end_header_sec": t_end,
        "sampling_rate_hz": float(1.0 / interval),
        "duration_sec": float(t_end - t_start),
        "n_samples": int(values.size),
        "n_missing": n_missing,
        "missing_fraction": float(n_missing / values.size) if values.size else 1.0,
        "value_min": vmin,
        "value_max": vmax,
        "value_std": vstd,
        "n_finite": n_finite,
        "flatline": bool(n_finite > 100 and vstd is not None and vstd < 1e-3),
    }


def _overlap_seconds(eeg: WaveSummary, bis: NumericSummary) -> float | None:
    if eeg.t_start_sec is None or eeg.t_end_header_sec is None:
        return None
    if bis.t_start_sec is None or bis.t_end_sec is None:
        return None
    start = max(eeg.t_start_sec, bis.t_start_sec)
    end = min(eeg.t_end_header_sec, bis.t_end_sec)
    return float(end - start)


def run_phase1(config: dict[str, Any] | None = None, refresh_index: bool = False) -> dict[str, Any]:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    resolved = config or load_config()
    loader = VitalDBLoader(resolved)
    trks, cases = loader.ensure_index(refresh=refresh_index)
    discovery = loader.discover(trks, cases)
    records = loader.download_cases(trks, cases, discovery)
    return {"discovery": discovery, "records": records}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Discover VitalDB EEG/BIS tracks and download a few cases.")
    parser.add_argument("--refresh-index", action="store_true", help="Download the track and clinical tables again.")
    args = parser.parse_args()
    result = run_phase1(refresh_index=args.refresh_index)
    accepted = [record.caseid for record in result["records"] if record.accepted]
    print(f"Accepted cases: {accepted}")


if __name__ == "__main__":
    main()
