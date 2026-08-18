#!/usr/bin/env python

import argparse

from src.pipeline.config import DEFAULT_CSV_PATH
from src.pipeline.training import (
    run_modality_comparison,
    run_modality_comparison_across_seeds,
    run_modality_comparison_leave_one_hospital_out,
)


def main():
    parser = argparse.ArgumentParser(
        description='Compare modality groups with nested cross-validation.',
    )
    parser.add_argument('-d', '--data_folder', required=True)
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--csv_path', default=DEFAULT_CSV_PATH)
    parser.add_argument('--seeds', type=int, nargs='+')
    parser.add_argument('--leave-one-hospital-out', action='store_true')
    args = parser.parse_args()
    if args.leave_one_hospital_out:
        run_modality_comparison_leave_one_hospital_out(
            args.data_folder,
            args.verbose,
            args.csv_path,
        )
        return
    if args.seeds:
        run_modality_comparison_across_seeds(
            args.data_folder,
            args.verbose,
            args.csv_path,
            args.seeds,
        )
        return
    run_modality_comparison(args.data_folder, args.verbose, args.csv_path)


if __name__ == '__main__':
    main()