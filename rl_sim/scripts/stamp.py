"""
stamp.py — The settings stamp carried by every figure and CSV.

Every generated artefact records the settings it was made with, so a figure or
a table found on its own can still be traced back to a model version. The
values are read from :mod:`fanet_sim.config` and from the loss curve — never
hardcoded — so a stamp cannot silently disagree with the run that produced it.

Used as a one-line footer on figures, and as a leading ``#`` comment line in
every CSV.
"""

from __future__ import annotations

import csv
import os
import subprocess
import textwrap
from typing import Any, Dict, List, Optional, Sequence

from fanet_sim import config
from fanet_sim.envs import channel


def git_commit() -> str:
    """Return the current short commit hash, marked if the tree is dirty.

    Returns:
        Something like ``"bd7cb3e"``, ``"bd7cb3e-dirty"``, or ``"unknown"``
        when git cannot answer (not a repository, or git unavailable).
    """
    repo = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    try:
        commit = subprocess.run(
            ["git", "-C", repo, "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", repo, "status", "--porcelain"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
        return f"{commit}-dirty" if dirty else commit
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def settings_stamp(
    loads_ms: Optional[Sequence[int]] = None,
    placements: Optional[int] = None,
    extra: Optional[str] = None,
) -> str:
    """Return the one-line stamp describing the current model settings.

    Args:
        loads_ms:   Traffic load(s) the artefact covers, in milliseconds.
                    Defaults to the configured working load.
        placements: How many placement seeds the artefact covers, if any.
        extra:      Anything else worth recording, appended at the end.

    Returns:
        A single line, safe to use as a figure footer or a CSV comment.
    """
    if loads_ms is None:
        loads_ms = [int(round(config.PACKET_INTERVAL_STEPS
                              * config.TIMESTEP * 1000))]
    load_text = "/".join(str(int(load)) for load in loads_ms) + " ms"

    parts = [
        f"D50={config.LOSS_50_DISTANCE_M:.0f} m",
        f"slope={config.LOSS_SLOPE_PER_M}/m",
        f"LINK_MAX_LOSS={config.LINK_MAX_LOSS}",
        f"reach={channel.max_link_distance():.0f} m",
        f"area={config.WIDTH:.0f}x{config.HEIGHT:.0f} m",
        f"{config.NUM_M_DRONES}M+{config.NUM_C_DRONES}C",
        f"Q={config.QUEUE_CAPACITY}/link",
        f"N={config.MAX_TX_PER_STEP}/link/step",
        f"load={load_text}",
    ]
    if placements is not None:
        parts.append(f"placements={placements}")
    parts.append(
        f"window=[{config.measurement_window()[0]},"
        f"{config.measurement_window()[1]}) of {config.MAX_STEPS}"
    )
    parts.append(f"commit={git_commit()}")
    if extra:
        parts.append(extra)
    return "  |  ".join(parts)


def stamp_lines(width: int = 118, **kwargs: object) -> str:
    """Return the stamp wrapped to *width*, for use as a figure footer.

    A figure is a fixed width; an unwrapped stamp simply runs off the edge and
    loses its tail, which is where the commit hash lives.

    Args:
        width:    Maximum characters per line.
        **kwargs: Passed through to :func:`settings_stamp`.

    Returns:
        The stamp, newline-separated if it does not fit on one line.
    """
    return "\n".join(textwrap.wrap(
        settings_stamp(**kwargs),  # type: ignore[arg-type]
        width=width, break_long_words=False, break_on_hyphens=False,
    ))


def csv_comment(**kwargs: object) -> str:
    """Return the stamp as a CSV comment line, including the trailing newline.

    Args:
        **kwargs: Passed through to :func:`settings_stamp`.

    Returns:
        ``"# <stamp>\\n"``.
    """
    return f"# {settings_stamp(**kwargs)}\n"  # type: ignore[arg-type]


def read_stamped_csv(path: str) -> List[Dict[str, Any]]:
    """Read a CSV written with a leading stamp comment.

    ``csv.DictReader`` would otherwise take the ``#`` line for the header, so
    every reader of these files must go through here.

    Args:
        path: Path to the CSV.

    Returns:
        The rows, as dicts keyed by column name.
    """
    with open(path, newline="") as handle:
        lines = [line for line in handle if not line.startswith("#")]
    return list(csv.DictReader(lines))


def read_stamp(path: str) -> Optional[str]:
    """Return the stamp line a CSV was written with, if it has one.

    Args:
        path: Path to the CSV.

    Returns:
        The stamp text without the leading ``"# "``, or None.
    """
    with open(path, newline="") as handle:
        first = handle.readline()
    return first[1:].strip() if first.startswith("#") else None
