"""NASA C-MAPSS turbofan degradation: the cross-domain test with real ground truth.

Public F1 telemetry contains no measured tyre wear, so there is no way to check a
degradation estimate against reality in motorsport. C-MAPSS does have that: 709
simulated engines run to failure, with published remaining-useful-life labels for
the held-out set and a large body of comparable results.

So the cross-domain claim gets tested rather than asserted. The *same*
state-space estimator, with no tyre-specific code, is pointed at engine
degradation and scored against the published labels.

Dataset: NASA Prognostics Center of Excellence, Saxena & Goebel (2008).
https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/

    FD001  100 engines, 1 operating condition,  1 fault mode
    FD002  260 engines, 6 operating conditions, 1 fault mode
    FD003  100 engines, 1 operating condition,  2 fault modes
    FD004  249 engines, 6 operating conditions, 2 fault modes

Each row is one engine at one flight cycle: unit id, cycle, 3 operating settings,
21 sensors. Training units run to failure. Test units stop partway, and the RUL
file gives how many cycles they had left.
"""

from __future__ import annotations

import io
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("data/external/cmapss")
DOWNLOAD_URL = (
    "https://phm-datasets.s3.amazonaws.com/NASA/"
    "6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
)

COLUMNS = (
    ["unit", "cycle", "setting_1", "setting_2", "setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)

#: Sensors that carry a usable degradation trend in FD001.
#:
#: Of the 21 channels, several are constant for the whole dataset and several
#: more are pure noise. Selecting on *observed* behaviour rather than on domain
#: knowledge keeps this honest -- `select_trending_sensors` derives the list from
#: the data, and this constant is only the cached answer for the common case.
DEFAULT_TREND_SENSORS = (
    "sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_8", "sensor_9",
    "sensor_11", "sensor_12", "sensor_13", "sensor_14", "sensor_15",
    "sensor_17", "sensor_20", "sensor_21",
)


def download(target: Path | None = None) -> Path:
    """Fetch and unpack C-MAPSS if it is not already on disk.

    Args:
        target: Directory to unpack into.

    Returns:
        The directory containing the .txt files.

    Raises:
        RuntimeError: If the download fails and the data is not already present.
    """
    target = target or DATA_DIR
    if (target / "train_FD001.txt").exists():
        return target

    target.mkdir(parents=True, exist_ok=True)
    try:
        request = urllib.request.Request(DOWNLOAD_URL, headers={"User-Agent": "Mozilla/5.0"})
        outer = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(request, timeout=600).read()))
        # The archive nests a second zip inside a descriptive folder.
        inner_name = next(n for n in outer.namelist() if n.endswith("CMAPSSData.zip"))
        zipfile.ZipFile(io.BytesIO(outer.read(inner_name))).extractall(target)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"could not download C-MAPSS to {target} ({type(exc).__name__}: {exc}). "
            f"Download {DOWNLOAD_URL} manually and unpack it there."
        ) from exc

    return target


@dataclass
class CmapssSubset:
    """One C-MAPSS sub-dataset.

    Attributes:
        name: FD001 through FD004.
        train: Run-to-failure units.
        test: Units truncated before failure.
        test_rul: True remaining cycles for each test unit, indexed by unit id.
        n_conditions: Distinct operating conditions.
    """

    name: str
    train: pd.DataFrame
    test: pd.DataFrame
    test_rul: pd.Series
    n_conditions: int


def load_subset(name: str = "FD001", directory: Path | None = None) -> CmapssSubset:
    """Load one C-MAPSS sub-dataset.

    Args:
        name: FD001 through FD004.
        directory: Where the .txt files live. Downloaded if absent.

    Returns:
        A CmapssSubset.

    Raises:
        FileNotFoundError: If the files are missing and cannot be fetched.
    """
    directory = download(directory)

    def read(prefix: str) -> pd.DataFrame:
        path = directory / f"{prefix}_{name}.txt"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; run tyremind.assets.cmapss.download()")
        frame = pd.read_csv(path, sep=r"\s+", header=None)
        # Trailing whitespace in the source files produces phantom empty columns.
        frame = frame.dropna(axis=1, how="all")
        frame.columns = COLUMNS[: frame.shape[1]]
        return frame

    train, test = read("train"), read("test")
    rul = pd.read_csv(directory / f"RUL_{name}.txt", sep=r"\s+", header=None).iloc[:, 0]
    rul.index = sorted(test["unit"].unique())

    n_conditions = int(
        train[["setting_1", "setting_2", "setting_3"]].round(0).drop_duplicates().shape[0]
    )

    return CmapssSubset(
        name=name, train=train, test=test, test_rul=rul, n_conditions=n_conditions
    )


def select_trending_sensors(train: pd.DataFrame, min_abs_correlation: float = 0.4) -> list[str]:
    """Pick sensors whose reading trends with engine age.

    Several of the 21 channels are constant across the whole dataset and several
    more are noise. Including them adds variance and no signal.

    Selection is on the *training* units only and uses correlation with cycle
    number within each unit, then averages across units. Doing it on pooled data
    instead would let differences between engines masquerade as a trend.

    Args:
        train: Run-to-failure units.
        min_abs_correlation: Minimum mean absolute correlation to keep a sensor.

    Returns:
        Sensor column names, strongest trend first.
    """
    sensors = [c for c in train.columns if c.startswith("sensor_")]
    scores: dict[str, float] = {}

    for sensor in sensors:
        per_unit = []
        for _, unit in train.groupby("unit"):
            values = unit[sensor].to_numpy(dtype=float)
            if np.std(values) < 1e-9:
                continue  # constant channel: no information
            per_unit.append(np.corrcoef(unit["cycle"].to_numpy(dtype=float), values)[0, 1])
        if per_unit:
            scores[sensor] = float(np.abs(np.nanmean(per_unit)))

    kept = {s: v for s, v in scores.items() if v >= min_abs_correlation}
    return sorted(kept, key=lambda s: -kept[s])


def build_health_index(
    frame: pd.DataFrame, sensors: list[str], reference: pd.DataFrame | None = None
) -> pd.Series:
    """Fuse selected sensors into a single scalar health signal in [0, 1].

    Each sensor is standardised against a healthy reference, sign-aligned so that
    increasing means *more degraded*, then averaged. The result is mapped so that
    1.0 is as-new and 0.0 is failed.

    Averaging rather than PCA is deliberate. PCA's leading component would be
    slightly tidier, but its loadings are fitted per dataset and would change
    between train and test, which is exactly the kind of quiet inconsistency that
    made the model ladder benchmark wrong earlier. A fixed, sign-aligned mean is
    reproducible and good enough -- and the state-space model is what extracts
    the trend anyway.

    Args:
        frame: Rows to score.
        sensors: Sensor columns to fuse.
        reference: Frame used to fit the standardisation. Defaults to `frame`.
            Always pass the TRAINING frame when scoring test data.

    Returns:
        Health index per row, aligned to `frame.index`.
    """
    reference = frame if reference is None else reference

    early = reference.groupby("unit").head(20)
    late = reference.groupby("unit").tail(20)

    values = np.zeros((len(frame), len(sensors)))
    for j, sensor in enumerate(sensors):
        mean = float(early[sensor].mean())
        sd = float(reference[sensor].std())
        if sd < 1e-9:
            continue
        z = (frame[sensor].to_numpy(dtype=float) - mean) / sd
        # Align the sign so that positive always means degraded, using the
        # healthy-to-worn direction observed in the reference set.
        direction = np.sign(float(late[sensor].mean()) - float(early[sensor].mean()))
        values[:, j] = z * (direction if direction != 0 else 1.0)

    damage = values.mean(axis=1)
    # Map to [0, 1] using the reference spread, so train and test share a scale.
    scale = float(np.percentile(np.abs(damage), 98)) or 1.0
    return pd.Series(np.clip(1.0 - damage / (2.0 * scale), 0.0, 1.0), index=frame.index)


def to_observations(
    frame: pd.DataFrame, health: pd.Series, *, max_units: int | None = None
) -> list:
    """Translate C-MAPSS rows into generic `DegradationObservation`s.

    Each engine is one asset and one run-to-replacement unit. Operating settings
    are discretised into a mode label, which the estimator treats exactly as it
    treats a tyre compound -- pooling a degradation baseline per mode.

    Args:
        frame: C-MAPSS rows.
        health: Health index aligned to `frame`.
        max_units: Cap on engines, for speed.

    Returns:
        A list of DegradationObservation.
    """
    from tyremind.assets.profile import DegradationObservation

    units = sorted(frame["unit"].unique())
    if max_units:
        units = units[:max_units]

    observations = []
    for unit in units:
        rows = frame[frame["unit"] == unit].sort_values("cycle")
        for i, (index, row) in enumerate(rows.iterrows()):
            # Six discrete operating conditions in FD002/FD004; one otherwise.
            mode = f"cond_{int(round(float(row['setting_3'])))}_{int(round(float(row['setting_1'])))}"
            observations.append(
                DegradationObservation(
                    asset_id=f"engine_{int(unit):03d}",
                    unit_id=int(unit),
                    sequence=int(row["cycle"]),
                    age=float(row["cycle"]),
                    performance=float(health.loc[index]),
                    mode=mode,
                    elapsed_in_run=i,
                )
            )
    return observations
