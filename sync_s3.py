"""
Sync EgoVerse data from S3/R2 to a local directory.

Example:
    # All episodes, no named filters
    python egomimic/scripts/data_download/sync_s3.py --local-dir /tmp/egoverse
    python egomimic/scripts/data_download/sync_s3.py --local-dir /tmp/egoverse --filters aria-fold-clothes

How to use differnt filters:
    - There a few filters already defined in the DATA_FILTERS dictionary, you can use them by name with --filters
    - to make a new filter, define a new DatasetFilter object in the DATA_FILTERS dictionary
    - the filters are lambda functions that are applied to the rows of the DB table, you can use any column name from the sql table to make the filter
    - use the filter name with --filters
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from egomimic.rldb.filters import DatasetFilter
from egomimic.rldb.zarr.zarr_dataset_multi import S3EpisodeResolver
from egomimic.utils.aws.aws_data_utils import load_env

logging.basicConfig(level=logging.INFO, format="%(message)s")


# Named presets for --filters. Omit --filters to sync with no predicates (all DB episodes).
DATA_FILTERS = {
    "aria-fold-clothes": DatasetFilter(
        filter_lambdas=[
            "lambda row: row.get('embodiment') == 'aria'",
            "lambda row: row.get('task') == 'fold_clothes'",
        ]
    ),
    "aria-all": DatasetFilter(
        filter_lambdas=[
            "lambda row: row.get('embodiment') == 'aria'",
        ]
    ),
    "eva-all": DatasetFilter(
        filter_lambdas=[
            "lambda row: row.get('embodiment') == 'eva'",
        ]
    ),
    "mecka-fold-clothes": DatasetFilter(
        filter_lambdas=[
            "lambda row: row.get('embodiment') == 'mecka'",
            "lambda row: row.get('task') == 'fold_clothes'",
        ]
    ),
    "target_aria_episodes": DatasetFilter(
        filter_lambdas=[
            "lambda row: row['zarr_mp4_path'] in {"
            "    's3://rldb/processed_v3/aria/2025-12-12-21-57-03-498000.mp4',"
            "    's3://rldb/processed_v3/aria/2026-01-16-21-59-57-172000.mp4',"
            "    's3://rldb/processed_v3/aria/2025-11-27-23-13-48-498000.mp4',"
            "    's3://rldb/processed_v3/aria/2026-01-19-21-47-50-000000.mp4',"
            "    's3://rldb/processed_v3/aria/2026-01-16-19-18-08-251000.mp4',"
            "    's3://rldb/processed_v3/aria/2025-11-30-22-27-52-168000.mp4',"
            "    's3://rldb/processed_v3/aria/2025-11-27-23-45-40-750000.mp4',"
            "    's3://rldb/processed_v3/aria/2026-01-14-23-22-51-248000.mp4',"
            "    's3://rldb/processed_v3/aria/2026-01-24-05-29-04-698000.mp4',"
            "    's3://rldb/processed_v3/aria/2025-11-30-16-16-40-282000.mp4',"
            "    's3://rldb/processed_v3/aria/2025-11-27-23-12-34-299000.mp4',"
            "    's3://rldb/processed_v3/aria/2025-11-24-23-59-28-546000.mp4',"
            "    's3://rldb/processed_v3/aria/2026-01-20-01-38-24-213000.mp4',"
            "    's3://rldb/processed_v3/aria/2025-11-27-23-18-17-895000.mp4',"
            "    's3://rldb/processed_v3/aria/2026-01-16-21-54-10-092000.mp4',"
            "    's3://rldb/processed_v3/aria/2025-11-27-23-15-59-065000.mp4',"
            "    's3://rldb/processed_v3/aria/2025-11-27-23-45-29-883000.mp4',"
            "    's3://rldb/processed_v3/aria/2025-11-27-23-16-45-029000.mp4',"
            "    's3://rldb/processed_v3/aria/2025-11-25-23-53-35-467000.mp4',"
            "    's3://rldb/processed_v3/aria/2026-01-24-05-29-04-696000.mp4'"
            "}"
        ]
    )
}


def parse_dataset_filter_key(filter_key: str) -> DatasetFilter:
    try:
        return DATA_FILTERS[filter_key]
    except KeyError as exc:
        raise ValueError(
            f"Unknown filter key {filter_key!r}. "
            f"Available filter keys: {sorted(DATA_FILTERS)}"
        ) from exc


def main():
    parser = argparse.ArgumentParser(
        description="Sync EgoVerse data from S3/R2 to a local directory."
    )
    parser.add_argument(
        "--local-dir",
        type=str,
        required=True,
        help="Local directory to sync into.",
    )
    parser.add_argument(
        "--workers", type=int, default=128, help="s5cmd parallel workers."
    )
    parser.add_argument(
        "--filters",
        type=str,
        default=None,
        help=(
            "Optional named filter preset. "
            "If omitted, no filter predicates are applied (sync every episode in the DB). "
            f"Presets: {', '.join(sorted(DATA_FILTERS))}"
        ),
    )
    args = parser.parse_args()

    filters = parse_dataset_filter_key(args.filters) if args.filters else None

    load_env()
    S3EpisodeResolver.sync_from_filters(
        bucket_name="rldb",
        filters=filters,
        local_dir=Path(args.local_dir),
        numworkers=args.workers,
    )


if __name__ == "__main__":
    main()
