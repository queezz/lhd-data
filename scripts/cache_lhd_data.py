"""Bulk-download LHD diagnostics into the local cache.

Examples:
    python scripts/cache_lhd_data.py
    python scripts/cache_lhd_data.py --figures
    python scripts/cache_lhd_data.py --diagnostics nbpwr_tot_temporal fircall thomson
"""

from __future__ import annotations

import argparse

from lhd_data.io.cache import CORE_DIAGNOSTICS, FIGURE_DIAGNOSTICS, cache_shots

DEFAULT_FIRST_SHOT = 193772
DEFAULT_LAST_SHOT = 193819


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-shot", type=int, default=DEFAULT_FIRST_SHOT)
    parser.add_argument("--last-shot", type=int, default=DEFAULT_LAST_SHOT)
    parser.add_argument("--subshot", type=int, default=1)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument(
        "--figures",
        action="store_true",
        help="Download all diagnostics needed by Figure1/Figure2.",
    )
    parser.add_argument(
        "--diagnostics",
        nargs="+",
        default=None,
        help="Override diagnostics to download.",
    )
    args = parser.parse_args()

    diagnostics = args.diagnostics
    if diagnostics is None:
        diagnostics = FIGURE_DIAGNOSTICS if args.figures else CORE_DIAGNOSTICS

    shots = range(args.first_shot, args.last_shot + 1)
    results = cache_shots(
        shots,
        diagnostics,
        subshot=args.subshot,
        cache_dir=args.cache_dir,
        refresh=args.refresh,
    )

    failures = 0
    for (shot, diag_name), result in results.items():
        if isinstance(result, Exception):
            failures += 1
            print(f"FAILED {shot} {diag_name}: {result}")
        else:
            print(f"OK     {shot} {diag_name}: {result}")

    print(f"Finished {len(results) - failures}/{len(results)} downloads")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
