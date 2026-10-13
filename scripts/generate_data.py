"""Write the seeded dataset to data/ (or --out). ``make data`` runs this."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from genai_recommender.datagen import Scale, generate


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--customers", type=int, default=Scale.customers)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    t0 = time.perf_counter()
    counts = generate(args.out, Scale(customers=args.customers), seed=args.seed)
    print(" ".join(f"{k}={v:,}" for k, v in counts.items()), f"({time.perf_counter() - t0:.0f}s)")


if __name__ == "__main__":
    main()
