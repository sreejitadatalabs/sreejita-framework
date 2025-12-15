"""
Shadow Diff Logger

Runs v1 and v2 domain application in parallel
and logs differences without affecting execution.
"""

from typing import Optional, Dict, Any
import logging

logger = logging.getLogger("sreejita.shadow_diff")


def compare_outputs(
    df_original,
    df_v1,
    df_v2,
    domain_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compare v1 and v2 outputs at a high level.
    """

    diff: Dict[str, Any] = {
        "domain": domain_name,
        "column_diff": None,
        "row_count_diff": None,
    }

    try:
        cols_v1 = set(df_v1.columns)
        cols_v2 = set(df_v2.columns)

        if cols_v1 != cols_v2:
            diff["column_diff"] = {
                "v1_only": sorted(cols_v1 - cols_v2),
                "v2_only": sorted(cols_v2 - cols_v1),
            }

        if len(df_v1) != len(df_v2):
            diff["row_count_diff"] = {
                "v1": len(df_v1),
                "v2": len(df_v2),
            }

    except Exception as e:
        diff["error"] = str(e)

    return diff


def log_shadow_diff(
    df,
    apply_v1,
    apply_v2,
    domain_name: Optional[str] = None,
) -> None:
    """
    Run v1 and v2 in shadow mode and log differences.
    """

    try:
        df_v1 = apply_v1(df, domain_name) if domain_name else apply_v1(df)
        df_v2 = apply_v2(df)

        diff = compare_outputs(df, df_v1, df_v2, domain_name)

        if diff.get("column_diff") or diff.get("row_count_diff"):
            logger.warning("Shadow diff detected: %s", diff)

    except Exception as e:
        # Shadow mode must NEVER affect execution
        logger.debug("Shadow diff skipped due to error: %s", e)
