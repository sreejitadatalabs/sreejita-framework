import os
from pathlib import Path

import pytest

from sreejita.reports.hybrid import run


# -------------------------------------------------
# Fixtures
# -------------------------------------------------

@pytest.fixture
def sample_retail_csv(tmp_path):
    """
    Minimal Retail CSV for regression tests.
    """
    csv_path = tmp_path / "retail.csv"
    csv_path.write_text(
        "order_id,sales,profit,discount,shipping_cost\n"
        "1,100,20,0.1,8\n"
        "2,200,40,0.2,18\n"
        "3,300,60,0.15,25\n"
    )
    return csv_path


@pytest.fixture
def base_config():
    """
    Minimal config required to generate report.
    """
    return {
        "dataset": {
            "sales": "sales",
            "profit": "profit",
        }
    }


# -------------------------------------------------
# Regression Tests — MUST NEVER BREAK
# -------------------------------------------------

def test_report_generates_successfully(sample_retail_csv, base_config, tmp_path):
    """
    Report must always generate without crashing.
    """
    output_path = tmp_path / "report.pdf"
    path = run(
        input_path=str(sample_retail_csv),
        config=base_config,
        output_path=output_path,
    )
    assert Path(path).exists()


def test_executive_brief_present(sample_retail_csv, base_config, tmp_path):
    """
    Executive Brief section must always exist.
    """
    output_path = tmp_path / "report.pdf"
    run(str(sample_retail_csv), base_config, output_path)

    content = output_path.read_bytes()
    assert b"EXECUTIVE BRIEF" in content


def test_available_quick_wins_present(sample_retail_csv, base_config, tmp_path):
    """
    Available Quick Wins must always be rendered
    (even if quantified later).
    """
    output_path = tmp_path / "report.pdf"
    run(str(sample_retail_csv), base_config, output_path)

    content = output_path.read_bytes()
    assert b"Available Quick Wins" in content


def test_quick_wins_not_duplicated(sample_retail_csv, base_config, tmp_path):
    """
    Prevent duplicate Quick Wins rendering.
    """
    output_path = tmp_path / "report.pdf"
    run(str(sample_retail_csv), base_config, output_path)

    content = output_path.read_bytes()
    assert content.count(b"Available Quick Wins") == 1


def test_data_quality_section_present(sample_retail_csv, base_config, tmp_path):
    """
    Data Quality Scorecard must always be present.
    """
    output_path = tmp_path / "report.pdf"
    run(str(sample_retail_csv), base_config, output_path)

    content = output_path.read_bytes()
    assert b"Data Quality" in content