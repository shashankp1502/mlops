"""Unit tests for src/data/download_data.py's parsing logic.

No network calls here on purpose — `fetch_raw_text` is the only function
that touches the internet, and it's a two-line requests.get() wrapper not
worth mocking extensively. `parse_cleveland` is where the actual logic
(missing-value markers, column naming, target binarization) lives, so
that's what gets tested, against a small hand-written sample in the
exact format UCI serves.
"""

from src.data.download_data import COLUMN_NAMES, parse_cleveland

# 3 real-looking rows, including one with a '?' missing value (as the
# genuine Cleveland file has for a handful of `ca`/`thal` entries) and a
# severity label > 1 to check binarization
SAMPLE_RAW_TEXT = (
    "63.0,1.0,1.0,145.0,233.0,1.0,2.0,150.0,0.0,2.3,3.0,0.0,6.0,0\n"
    "67.0,1.0,4.0,160.0,286.0,0.0,2.0,108.0,1.0,1.5,2.0,3.0,3.0,2\n"
    "41.0,0.0,2.0,130.0,204.0,0.0,2.0,172.0,0.0,1.4,1.0,?,3.0,0\n"
)


def test_parse_cleveland_column_names():
    df = parse_cleveland(SAMPLE_RAW_TEXT)
    expected_cols = [c for c in COLUMN_NAMES if c != "num"] + ["target"]
    assert list(df.columns) == expected_cols


def test_parse_cleveland_row_count():
    df = parse_cleveland(SAMPLE_RAW_TEXT)
    assert len(df) == 3


def test_parse_cleveland_handles_question_mark_as_nan():
    df = parse_cleveland(SAMPLE_RAW_TEXT)
    assert df.loc[2, "ca"] != df.loc[2, "ca"]  # NaN != NaN


def test_parse_cleveland_binarizes_target():
    df = parse_cleveland(SAMPLE_RAW_TEXT)
    # num=0 -> target=0, num=2 -> target=1, num=0 -> target=0
    assert df["target"].tolist() == [0, 1, 0]


def test_parse_cleveland_drops_original_num_column():
    df = parse_cleveland(SAMPLE_RAW_TEXT)
    assert "num" not in df.columns
