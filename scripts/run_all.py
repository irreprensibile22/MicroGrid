"""Run the whole pipeline: download, cleaning, training, simulation, charts.

Usage:

    python scripts/run_all.py                  # everything, skipping downloads already there
    python scripts/run_all.py --force-download # download the raw data again
    python scripts/run_all.py --no-notebooks   # do not re-run the notebooks

Each step matches a `python -m microgrid.<module>` command and can also be run
on its own. The parameters live in `config/default.yaml`.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from collections.abc import Callable

from microgrid import data, dataset, evaluate, forecast, optimize, simulate
from microgrid.config import PROJECT_ROOT, load_config

NOTEBOOKS = ("01_exploration.ipynb", "02_forecasting.ipynb", "03_results.ipynb")


def run_notebooks() -> None:
    """Re-run the notebooks in place, so charts and tables reflect the new results."""
    for name in NOTEBOOKS:
        path = PROJECT_ROOT / "notebooks" / name
        subprocess.run(
            [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace", str(path)],
            check=True,
        )


def main() -> None:
    """Run every step of the pipeline in order."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force-download", action="store_true", help="download the raw data even if present")
    parser.add_argument("--no-notebooks", action="store_true", help="do not re-run the notebooks")
    args = parser.parse_args()

    # The module `main` functions parse sys.argv themselves: hide our own flags from them.
    sys.argv = sys.argv[:1]

    steps: list[tuple[str, Callable[[], object]]] = [
        ("Data download", lambda: data.download_all(load_config(), force=args.force_download)),
        ("Cleaning and dataset", dataset.main),
        ("Forecast", forecast.main),
        ("Plan for an example day", optimize.main),
        ("Rolling-horizon simulation", simulate.main),
        ("Strategy comparison", evaluate.main),
    ]
    if not args.no_notebooks:
        steps.append(("Notebooks", run_notebooks))

    start = time.perf_counter()
    for number, (title, step) in enumerate(steps, start=1):
        print(f"\n=== {number}/{len(steps)}  {title} ===", flush=True)
        step_start = time.perf_counter()
        step()
        print(f"--- {title}: {time.perf_counter() - step_start:.0f} s", flush=True)
    print(f"\nPipeline completed in {(time.perf_counter() - start) / 60:.1f} minutes.")


if __name__ == "__main__":
    main()
