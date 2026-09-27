"""Date range guards for verification module.

Enforces Hard Rule 4: Leaked train/calibrate data into verification invalidates skill credibility.
Verification queries must strictly evaluate ONLY against the held-out TEST period.
"""

from typing import Tuple

# Hard Rule 4: Held-out scoring period
TEST_START = "2024-07-01"
TEST_END = "2024-08-31"

ALLOWED_TEST_WINDOW: Tuple[str, str] = (TEST_START, TEST_END)


def assert_strictly_test_period(start_date: str, end_date: str) -> None:
    """Enforce that verification queries strictly touch ONLY the held-out TEST period.
    
    Raises AssertionError if start_date < TEST_START or end_date > TEST_END.
    """
    if start_date > end_date:
        raise ValueError(f"Invalid date range order: {start_date} > {end_date}")

    if start_date < TEST_START:
        raise AssertionError(
            f"VERIFICATION LEAKAGE VIOLATION: Query start_date ({start_date}) is earlier than "
            f"reserved TEST_START ({TEST_START}). Verification must not evaluate on TRAIN or CALIBRATE data!"
        )

    if end_date > TEST_END:
        raise AssertionError(
            f"VERIFICATION WINDOW VIOLATION: Query end_date ({end_date}) exceeds "
            f"reserved TEST_END ({TEST_END}). Allowed verification period is {TEST_START} to {TEST_END}."
        )
