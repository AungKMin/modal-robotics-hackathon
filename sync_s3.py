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
    # The 50 episodes behind somundane/egoverse-cup50 (which only ships 1 fps frames).
    # Substring match on the episode id so the embodiment directory does not matter.
    "cup50_episodes": DatasetFilter(
        filter_lambdas=[
            "lambda row: any(e in (row.get('zarr_mp4_path') or '') for e in {'2025-11-14-16-19-50-305000','2025-11-14-16-39-40-009000','2025-11-24-19-42-18-324000','2025-11-30-16-25-06-265000','2025-12-24-19-36-56-086000','2025-12-25-20-00-08-755000','2025-12-26-00-56-41-044000','2025-12-26-18-05-07-214000','2025-12-26-18-31-01-838000','2026-01-11-18-13-58-430000','2026-01-11-23-11-22-998000','2026-01-20-19-49-03-357000','2026-01-24-05-29-04-636000','692e98927641010d04354574','692e9b6668228e362d908f0e','692e9fa092a31767e35da22c','692ea164e2322e3b092b5dd8','692ea2012c8fefa9948e8dd0','692ea3beffdc0ca6345c4246','692ea4da74b24813e759755d','692ea671dbc4294a49cc727e','692ea6ffc621d7f4aac3aafc','692ea773b77ab81b8b41ee87','692ea7b368228e362d90908e','692ea7ff4e7eab2cafd26992','692ea886e2322e3b092b5ee5','692ea8fb727c13b350cb7cb8','692ea92a99488ff84776f987','692ea97bea43f09edf26f5fb','692ea9bc95aad87d3e34466a','692ea9ffaec602a46af10605','692eaa05c7cb0e94dc84bbf5','692eaa59a0e165ab2e42e516','692eaa5fd3d807884d4dd7ae','692eaa76dfa4113987776a57','692eaac0c621d7f4aac3ab9d','692eab1cb77ab81b8b41eeec','692eac0739719ab57395b969','692eac2caec602a46af1065e','692eac7274b24813e759760d','692ead5d3019385fbbf5683e','692ead5e3019385fbbf5684a','692eae07424dbf22e75ed141','692eae54c621d7f4aac3abd6','692eae813019385fbbf56873','692eaeaa40338017b7f999ff','692eaeba9cebf0ec017ddd36','692eaee539719ab57395b989','692eafd7eee2f8cb10d0f8e0','692eafd9eee2f8cb10d0f8f7'})"
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
