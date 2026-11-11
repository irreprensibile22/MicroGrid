"""Esegue l'intero pipeline: download, pulizia, addestramento, simulazione, grafici.

Uso:

    python scripts/run_all.py                  # tutto, saltando i download già presenti
    python scripts/run_all.py --force-download # riscarica anche i dati grezzi
    python scripts/run_all.py --no-notebooks   # non riesegue i notebook

Ogni passo corrisponde a un comando `python -m microgrid.<modulo>` e si può
eseguire anche da solo. I parametri sono in `config/default.yaml`.
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
    """Riesegue i notebook sul posto, così grafici e tabelle riflettono i nuovi risultati."""
    for name in NOTEBOOKS:
        path = PROJECT_ROOT / "notebooks" / name
        subprocess.run(
            [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace", str(path)],
            check=True,
        )


def main() -> None:
    """Esegue in ordine tutti i passi del pipeline."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force-download", action="store_true", help="riscarica anche i dati già presenti")
    parser.add_argument("--no-notebooks", action="store_true", help="non riesegue i notebook")
    args = parser.parse_args()

    steps: list[tuple[str, Callable[[], object]]] = [
        ("Download dei dati", lambda: data.download_all(load_config(), force=args.force_download)),
        ("Pulizia e dataset", dataset.main),
        ("Previsione", forecast.main),
        ("Piano di un giorno di esempio", optimize.main),
        ("Simulazione a orizzonte mobile", simulate.main),
        ("Confronto fra strategie", evaluate.main),
    ]
    if not args.no_notebooks:
        steps.append(("Notebook", run_notebooks))

    start = time.perf_counter()
    for number, (title, step) in enumerate(steps, start=1):
        print(f"\n=== {number}/{len(steps)}  {title} ===", flush=True)
        step_start = time.perf_counter()
        step()
        print(f"--- {title}: {time.perf_counter() - step_start:.0f} s", flush=True)
    print(f"\nPipeline completato in {(time.perf_counter() - start) / 60:.1f} minuti.")


if __name__ == "__main__":
    main()
