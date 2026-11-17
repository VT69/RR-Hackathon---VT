"""Cross-check the numbers in ``outputs/report/methodology_report.md`` against the generated output tables.

Run: ``python -m src.report.check_report`` (also run at the end of ``python -m src.pipeline``).

How it works:

* Every checked number is a **claim**: a short phrase from the report with the number(s) in it, written as a
  format template, plus the value(s) recomputed from the output tables (or ``config.yaml``).
* A claim **passes** when the rendered phrase appears in the report. If the phrase's words are there but the
  number differs, it is a **mismatch** (the report's number is shown). If the words are not found, it is
  **missing** (the report was reworded, so the check itself needs updating).
* The ranking tables in the report are checked row by row: each row is rendered from the output tables by
  ``ranking_rows`` (the same function used to write them), and must appear verbatim.
* Finally every number in the report that no claim covers is listed, so unchecked numbers are visible.

The result is written to ``outputs/report/crosscheck.md``; the exit code is 1 if anything mismatches or is missing.
"""

from __future__ import annotations

import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from src.config import load_config, resolve_path

logger = logging.getLogger(__name__)
NUMBER = r"[-+]?\d[\d,]*(?:\.\d+)?"
MINUS = "−"


@dataclass
class Claim:
    """One checked phrase: ``template.format(*values)`` must appear in the report."""

    section: str
    template: str
    values: tuple

    @property
    def text(self) -> str:
        return normalise(self.template.format(*self.values))

    def pattern(self) -> re.Pattern:
        """Regex for the phrase with any numbers in the placeholders (to report what the report says)."""
        parts = re.split(r"\{[^}]*\}", normalise(self.template))
        return re.compile(NUMBER.join(re.escape(p) for p in parts))


def normalise(text: str) -> str:
    """Unify minus signs and whitespace so formatting differences do not count as mismatches."""
    return re.sub(r"\s+", " ", text.replace(MINUS, "-").replace("–", "-"))


class Tables:
    """Lazy access to the output tables: ``T('phase2b/li_summary.csv')`` or ``T.root('final_rankings.csv')``."""

    def __init__(self, outputs: Path) -> None:
        self.outputs, self._cache = outputs, {}

    def __call__(self, name: str) -> pd.DataFrame:
        if name not in self._cache:
            path = self.outputs / name if name.startswith(("final_", "audit/")) else self.outputs / "tables" / name
            self._cache[name] = pd.read_csv(path)
        return self._cache[name]


# ---------------------------------------------------------------- ranking tables (written into the report by this code)
def fmt_signed(x: float, digits: int = 1) -> str:
    """Signed number with a true minus sign, e.g. +34.3 / −2.2."""
    s = f"{x:+.{digits}f}"
    return s.replace("-", MINUS)


def fmt_num(x: float, digits: int = 1) -> str:
    """Unsigned format with a true minus sign for negatives."""
    return f"{x:.{digits}f}".replace("-", MINUS)


RANKING_HEADER = ("| # | Player | ID | Team(s) | Pressure balls | Rating | Shrunk value | 90% interval | Consistency "
                  "| P(top 10) | Stability | Label |")


def ranking_rows(final: pd.DataFrame, role: str, ranks: range) -> list[str]:
    """Markdown rows of the report's ranking tables, rendered from ``final_rankings_top15.csv``.

    Rating is the composite that sets the order; † = short record; ‡ = position set by the tie-break.
    """
    rows = []
    d = final[(final["role"] == role) & final["rank"].isin(list(ranks))].sort_values("rank")
    for r in d.itertuples():
        stab = f"{r.stability} ({r.share_of_variants_in_top_10:.0%})"
        flag = (" †" if isinstance(r.note, str) and r.note else "") + (" ‡" if r.set_by_tie_break else "")
        rows.append(
            f"| {r.rank} | {r.player_name}{flag} | {r.player_id} | {r.teams} | {r.pressure_balls} | {fmt_num(r.rating, 2)} | "
            f"{fmt_num(r.pressure_value_shrunk)} | {fmt_num(r.pressure_value_p05)} to {fmt_num(r.pressure_value_p95)} | "
            f"{fmt_signed(r.consistency_pts_above_base)} | {r.p_top_10:.2f} | {stab} | {r.label} |"
        )
    return rows


def _join(items: list[str]) -> str:
    """'a', 'a and b', 'a, b and c'."""
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def tie_break_note(final: pd.DataFrame, role: str, ranks: range) -> str:
    """Footnote for the ‡ marks: each player the tie-break lifted, the players they could not be told apart from
    and passed, with option A and option B ranks (rendered from ``final_rankings_top15.csv``)."""
    d = final[(final["role"] == role) & final["rank"].isin(list(ranks))].sort_values("rank")
    lifted = d[d["set_by_tie_break"] & (d["rank"] < d["rank_option_a"])]
    sentences = []
    for p in lifted.itertuples():
        passed = d[(d["rank_option_a"] < p.rank_option_a) & (d["rank"] > p.rank)]
        names = _join([f"{q.player_name} (A {q.rank_option_a}, B {q.rank_option_b})" for q in passed.itertuples()])
        sentences.append(f"{p.player_name} (A {p.rank_option_a}, B {p.rank_option_b}) cannot be told apart from {names}, "
                         f"so B orders them.")
    return "‡ Set by the tie-break: " + " ".join(sentences) if sentences else ""


# ---------------------------------------------------------------- the claims
def claims(T: Tables, config: dict[str, Any]) -> list[Claim]:
    """Every checked phrase in the report, with its value recomputed from the tables (filled in report_claims)."""
    from src.report.report_claims import build

    return build(T, config)


# ---------------------------------------------------------------- running the check
def check(report: str, items: list[Claim]) -> tuple[pd.DataFrame, list[tuple[int, int]]]:
    """Status of each claim, and the character spans of the report covered by passing claims."""
    text = normalise(report)
    rows, spans = [], []
    for c in items:
        found = [m.span() for m in re.finditer(re.escape(c.text), text)]
        if found:
            spans.extend(found)
            rows.append({"section": c.section, "status": "ok", "expected": c.text, "report_says": ""})
            continue
        m = c.pattern().search(text)
        status = "MISMATCH" if m else "MISSING"
        rows.append({"section": c.section, "status": status, "expected": c.text, "report_says": m.group(0) if m else ""})
    return pd.DataFrame(rows), spans


# Numbers that are labels, not data: section and figure references, list numbering, "top 10", "T20", season years.
BENIGN = [r"Figure [A-Z]?\d+", r"§\d+(?:\.\d+)?", r"(?m)^\s*\d+\.\s", r"top[ -]?1[05]", r"T20I?", r"ranks? \d+-\d+",
          r"\b20(?:22|23|24|25|26)\b", r"\b2022-2026\b", r"\b2022-26\b", r"D\d+-D\d+", r"Season 2, Round 1",
          r"final_rankings_top15", r"(?m)^\| \d+ \|", r"#\d+"]


def unchecked_numbers(report: str, spans: list[tuple[int, int]]) -> list[tuple[int, str, str]]:
    """Numbers in the report body not inside any passing claim (headings, figure links, code and labels are skipped)."""
    text = normalise(report)
    covered = sorted(spans)
    out: list[tuple[int, str, str]] = []
    in_code = False
    # Map each report line onto the normalised text by searching forward from the previous line.
    norm_lines = [normalise(line) for line in report.split("\n")]
    cursor = 0
    for i, line in enumerate(norm_lines, start=1):
        start = text.find(line, cursor) if line.strip() else cursor
        if start < 0:
            start = cursor
        cursor = start + len(line)
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code = not in_code
            continue
        if in_code or stripped.startswith(("#", "![", "<!--")) or not stripped:
            continue
        benign = [b.span() for p in BENIGN for b in re.finditer(p, line)]
        for m in re.finditer(NUMBER, line):
            a, b = start + m.start(), start + m.end()
            if any(s <= a and b <= e for s, e in covered) or any(s <= m.start() and m.end() <= e for s, e in benign):
                continue
            token = m.group(0)
            context = line[max(0, m.start() - 30): m.end() + 30]
            out.append((i, token, context))
    return out


def run(outputs: Path | None = None, config: dict[str, Any] | None = None) -> int:
    """Check the report; write ``crosscheck.md``; return the number of failed claims."""
    config = config or load_config()
    outputs = outputs or resolve_path(config["paths"]["outputs_dir"])
    report_dir = outputs / config["report"]["dir"]
    report = (report_dir / "methodology_report.md").read_text(encoding="utf-8")
    T = Tables(outputs)
    items = claims(T, config)
    final = T("final_rankings_top15.csv")
    top_n, zone = config["rating"]["top_n"], config["robustness"]["tie_break_zone"]
    for role in ("batter", "bowler"):
        for row in ranking_rows(final, role, range(1, zone + 1)):
            items.append(Claim(f"ranking table ({role}s)", "{}", (row,)))
        for span in (range(1, top_n + 1), range(top_n + 1, zone + 1)):
            note = tie_break_note(final, role, span)
            if note:
                items.append(Claim(f"tie-break note ({role}s)", "{}", (note,)))
    result, spans = check(report, items)
    loose = unchecked_numbers(report, spans)
    failed = result[result["status"] != "ok"]
    lines = [
        "# Report cross-check",
        "",
        f"Checked `{config['report']['dir']}/methodology_report.md` against the output tables.",
        "",
        f"- **Claims checked:** {len(result)}",
        f"- **Passed:** {int((result['status'] == 'ok').sum())}",
        f"- **Mismatches:** {int((result['status'] == 'MISMATCH').sum())}",
        f"- **Missing (phrase not found; the check needs updating):** {int((result['status'] == 'MISSING').sum())}",
        f"- **Numbers in the report not covered by any check:** {len(loose)}",
        "",
    ]
    if len(failed):
        lines += ["## Failures", "", "| Section | Status | Expected (from tables) | Report says |", "|---|---|---|---|"]
        lines += [f"| {r.section} | {r.status} | {r.expected} | {r.report_says} |" for r in failed.itertuples()]
        lines.append("")
    lines += ["## Numbers not covered by an automatic check", "",
              "Mostly definitions, worked arithmetic and figures quoted from the data audit. Line number, number, context.", ""]
    lines += [f"- line {i}: **{tok}** — …{ctx}…" for i, tok, ctx in loose] or ["- none"]
    (report_dir / "crosscheck.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for r in failed.itertuples():
        logger.warning("Report cross-check %s [%s]: expected %r, report says %r", r.status, r.section, r.expected, r.report_says)
    logger.info("Report cross-check: %d claims, %d failed, %d numbers not covered (see %s)", len(result), len(failed),
                len(loose), report_dir / "crosscheck.md")
    return len(failed)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    sys.exit(1 if run() else 0)
