import argparse
from datetime import date

from tornado_data import (
    get_dat_tornadoes,
    get_states,
    assign_states,
    create_counts,
)

OUTPUT_FILE = "tornado_counts.csv"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Count surveyed NWS tornadoes per state for a date range."
    )
    parser.add_argument(
        "--start",
        type=date.fromisoformat,
        default=date(date.today().year, 1, 1),
        help="Start date, YYYY-MM-DD (default: Jan 1 of the current year).",
    )
    parser.add_argument(
        "--end",
        type=date.fromisoformat,
        default=date.today(),
        help="End date, inclusive, YYYY-MM-DD (default: today).",
    )

    args = parser.parse_args()

    if args.start > args.end:
        parser.error("--start must not be after --end")

    return args


# ------------------------------------------------------------
# Main program
# ------------------------------------------------------------

def main():
    args = parse_args()

    tornadoes = get_dat_tornadoes(args.start, args.end)

    states = get_states()

    intersections = assign_states(
        tornadoes,
        states
    )

    result = create_counts(
        intersections,
        states
    )

    result.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print("=" * 60)
    print(f"TORNADO COUNTS BY STATE: {args.start} to {args.end}")
    print("=" * 60)

    if result["tornadoes"].sum() == 0:
        print("No surveyed tornadoes in this range.")
        print()

    print(
        result.to_string(index=False)
    )

    print()
    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
