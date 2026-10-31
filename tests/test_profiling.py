"""The dataset profile must surface the messiness the agents need to clean."""
from analyst.profiling import profile_files
from eval.reference import DATA_DIR


def test_profile_flags_messy_data(tmp_path):
    text = profile_files([DATA_DIR / "sales.csv", DATA_DIR / "hr_employees.xlsx"])
    assert "25 fully duplicated rows" in text
    assert "numbers stored as text" in text                       # revenue '$1,234.50', salary '85,000'
    assert "inconsistent casing/whitespace" in text               # region / department
    assert "extreme outliers" in text and "999" in text           # units = 999
    assert "zero values" in text                                  # age = 0
    assert "hr_employees.xlsx" in text


def test_unreadable_file_reported_not_raised(tmp_path):
    bad = tmp_path / "broken.xlsx"
    bad.write_text("not an excel file")
    assert "Could not be read" in profile_files([bad])
