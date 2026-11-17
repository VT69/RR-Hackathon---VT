"""Tests for the Phase 7 labels and the report cross-check mechanics."""

from __future__ import annotations

from src.report.check_report import Claim, check, normalise, unchecked_numbers
from src.report.final_outputs import ALL_ROUND, PRESSURE_SPECIFIC, profile_label, record_note


def test_profile_label_uses_option_b_rank_threshold() -> None:
    """B rank beyond the threshold is "Pressure-specific"; at or inside it, "All-round"."""
    assert profile_label(88, 30) == PRESSURE_SPECIFIC
    assert profile_label(31, 30) == PRESSURE_SPECIFIC
    assert profile_label(30, 30) == ALL_ROUND
    assert profile_label(1, 30) == ALL_ROUND


def test_record_note_flags_only_short_records() -> None:
    """Fewer pressure balls than the cutoff gets the short-record note; otherwise the note is empty."""
    assert "Short record" in record_note(177, 200)
    assert record_note(200, 200) == ""
    assert record_note(626, 200) == ""


def test_normalise_unifies_minus_signs_and_whitespace() -> None:
    """A true minus, an en dash and a hyphen compare equal, and line breaks do not matter."""
    assert normalise("−0.07 and 1–4\n  names") == "-0.07 and 1-4 names"


def test_check_passes_mismatches_and_misses() -> None:
    """A rendered phrase found = ok; same words with another number = MISMATCH; words absent = MISSING."""
    report = "Pressure value reproduces (reliability 0.33 batters, 0.19 bowlers). Cap at 17.9 runs."
    claims = [
        Claim("a", "(reliability {:.2f} batters, {:.2f} bowlers)", (0.331, 0.192)),
        Claim("b", "Cap at {:.1f} runs", (21.3,)),
        Claim("c", "Uplift reliability is {:.2f}", (0.0,)),
    ]
    result, spans = check(report, claims)
    assert list(result["status"]) == ["ok", "MISMATCH", "MISSING"]
    assert result.loc[1, "report_says"] == "Cap at 17.9 runs"
    assert len(spans) == 1


def test_unchecked_numbers_skip_covered_and_label_numbers() -> None:
    """Numbers inside a passing claim and label-like numbers (sections, figures, top 10) are not listed."""
    report = "See Figure 3 and §4.2: the top 10 had 0.33, and 774 matches were used."
    result, spans = check(report, [Claim("a", "had {:.2f}", (0.33,))])
    loose = [token for _, token, _ in unchecked_numbers(report, spans)]
    assert loose == ["774"]


def test_tie_break_note_names_the_tied_players_and_b_ranks() -> None:
    """A lifted player is listed with everyone they passed, each with option A and option B ranks."""
    import pandas as pd

    from src.report.check_report import tie_break_note

    final = pd.DataFrame({
        "role": ["batter"] * 3, "rank": [1, 2, 3], "player_name": ["X", "Y", "Z"],
        "rank_option_a": [2, 1, 3], "rank_option_b": [4, 30, 9], "set_by_tie_break": [True, True, False],
    })
    note = tie_break_note(final, "batter", range(1, 4))
    assert note == "‡ Set by the tie-break: X (A 2, B 4) cannot be told apart from Y (A 1, B 30), so B orders them."
    assert tie_break_note(final.assign(set_by_tie_break=False), "batter", range(1, 4)) == ""


def test_pdf_tables_right_align_numbers_and_mark_ranking_tables() -> None:
    """Numeric columns get the num class; a table with a Rating column gets the compact ranking style."""
    from src.report.build_pdf import table_html

    html = table_html([["Player", "Rating", "Interval"], ["---", "---", "---"],
                       ["A", "2.51", "13.5 to 55.0"], ["B", "−1.20", "−2.2 to 40.8"]])
    assert '<table class="rank">' in html
    assert '<td class="num">2.51</td>' in html and '<td class="num">13.5 to 55.0</td>' in html
    assert "<td>A</td>" in html
