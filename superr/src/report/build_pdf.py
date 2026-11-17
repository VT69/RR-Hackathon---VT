"""Phase 9: build the submission PDF from the methodology report.

Run: ``python -m src.report.build_pdf`` (also the last step of ``python -m src.pipeline``).

Markdown -> styled HTML (title page, table of contents, embedded figures) -> PDF, printed by a headless
Chromium browser (Microsoft Edge or Chrome). Page numbers and the running header come from CSS page-margin
boxes; PDF bookmarks come from the browser's document outline. The converter handles exactly the Markdown the
report uses: headings, paragraphs, lists, pipe tables, images with a caption line, bold, italics, code and links.
"""

from __future__ import annotations

import base64
import html
import io
import logging
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

from src.config import load_config, resolve_path

logger = logging.getLogger(__name__)
NUMERIC_CELL = re.compile(r"^[−\-+]?[\d,]*\.?\d+%?( to [−\-+]?[\d,]*\.?\d+)?$|^\d+ → \d+$|^\*\*[−\-+]?[\d.]+\*\*$")


# ---------------------------------------------------------------- Markdown -> HTML
def slug(text: str) -> str:
    """Anchor id for a heading."""
    return re.sub(r"[^a-z0-9]+", "-", re.sub(r"<[^>]+>", "", text).lower()).strip("-")


def inline(text: str) -> str:
    """Inline Markdown: code, bold, italics, links."""
    parts = re.split(r"(`[^`]+`)", text)
    out = []
    for part in parts:
        if part.startswith("`") and part.endswith("`") and len(part) > 1:
            out.append(f"<code>{html.escape(part[1:-1])}</code>")
            continue
        t = html.escape(part, quote=False)
        t = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', t)
        t = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", t)
        t = re.sub(r"(?<![*\w])\*([^*\s][^*]*)\*(?![*\w])", r"<em>\1</em>", t)
        out.append(t)
    return "".join(out)


def embed_image(path: Path, width_in: float, dpi: int) -> str:
    """The figure as a PNG data URI, downsampled to ``dpi`` at the printed width."""
    im = Image.open(path).convert("RGB")
    target = int(round(width_in * dpi))
    if im.width > target:
        im = im.resize((target, int(round(im.height * target / im.width))), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def table_html(rows: list[list[str]]) -> str:
    """A pipe table: header row repeated on page breaks, numeric cells right-aligned."""
    head, body = rows[0], rows[2:]
    numeric_cols = [all(NUMERIC_CELL.match(r[j].strip()) or not r[j].strip() for r in body if j < len(r)) and
                    any(r[j].strip() for r in body if j < len(r)) for j in range(len(head))]
    cls = lambda j: ' class="num"' if numeric_cols[j] else ""  # noqa: E731
    table_cls = ' class="rank"' if "Rating" in head else ""
    out = [f"<table{table_cls}><thead><tr>" + "".join(f"<th{cls(j)}>{inline(c)}</th>" for j, c in enumerate(head)) + "</tr></thead><tbody>"]
    for r in body:
        if all(not c.strip() for c in r[1:]) and r[0].startswith("**"):
            out.append(f'<tr class="group"><td colspan="{len(head)}">{inline(r[0])}</td></tr>')
            continue
        out.append("<tr>" + "".join(f"<td{cls(j)}>{inline(c)}</td>" for j, c in enumerate(r)) + "</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def body_html(lines: list[str], report_dir: Path, width_in: float, dpi: int) -> tuple[str, list[tuple[int, str, str]]]:
    """Convert the report body; returns the HTML and the headings (level, id, text) for the contents page."""
    out, heads, i = [], [], 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            out.append(table_html(rows))
            continue
        m = re.match(r"^(#{2,4}) (.*)", line)
        if m:
            level, text = len(m.group(1)), m.group(2)
            hid = slug(text)
            heads.append((level, hid, text))
            extra = ' class="appendix-start"' if text.startswith("Appendix A") else ""
            out.append(f'<h{level} id="{hid}"{extra}>{inline(text)}</h{level}>')
        elif line.startswith("!["):
            src = re.search(r"\((.*?)\)", line).group(1)
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            caption = lines[j].strip() if j < len(lines) and lines[j].strip().startswith("*Figure") else ""
            data = embed_image(report_dir / src, width_in, dpi)
            cap = f"<figcaption>{inline(caption.strip('*'))}</figcaption>" if caption else ""
            out.append(f'<figure><img src="{data}" alt="">{cap}</figure>')
            i = j + 1 if caption else i + 1
            continue
        elif re.match(r"^\s*(-|\d+\.) ", line):
            ordered = bool(re.match(r"^\s*\d+\.", line))
            items = []
            while i < len(lines) and re.match(r"^\s*(-|\d+\.) ", lines[i]):
                items.append(re.sub(r"^\s*(-|\d+\.) ", "", lines[i]))
                i += 1
            tag = "ol" if ordered else "ul"
            out.append(f"<{tag}>" + "".join(f"<li>{inline(x)}</li>" for x in items) + f"</{tag}>")
            continue
        elif line.startswith(">"):
            out.append(f"<blockquote>{inline(line.lstrip('> '))}</blockquote>")
        elif line.strip() and line.strip() != "---":
            out.append(f"<p>{inline(line)}</p>")
        i += 1
    return "\n".join(out), heads


# ---------------------------------------------------------------- the document
CSS = """
@page { size: A4; margin: 20mm 19mm 18mm 19mm;
  @top-left { content: "%(running)s"; font: 7.5pt "Segoe UI", Arial, sans-serif; color: #8a8f98; }
  @bottom-right { content: "Page " counter(page) " of " counter(pages); font: 7.5pt "Segoe UI", Arial, sans-serif; color: #8a8f98; } }
@page :first { @top-left { content: none; } @bottom-right { content: none; } }
html { font-family: "Segoe UI", Inter, Arial, sans-serif; font-size: 10pt; line-height: 1.42; color: #16181d; }
body { margin: 0; }
h1, h2, h3, h4 { color: %(accent)s; line-height: 1.25; break-after: avoid; page-break-after: avoid; }
h2 { font-size: 14pt; margin: 18pt 0 6pt; padding-bottom: 2pt; border-bottom: 0.8pt solid #d5dbe5; }
h3 { font-size: 11pt; margin: 12pt 0 4pt; }
h2 + *, h3 + * { break-before: avoid; }
p, li { margin: 3pt 0; orphans: 3; widows: 3; }
ul, ol { margin: 3pt 0 4pt 15pt; padding: 0; }
a { color: %(accent)s; text-decoration: none; }
code { font-family: Consolas, "Cascadia Mono", monospace; font-size: 8.6pt; background: #f2f4f7; padding: 0 2pt; border-radius: 2pt; }
table { border-collapse: collapse; width: 100%%; font-size: 8pt; line-height: 1.3; margin: 6pt 0 8pt; }
thead { display: table-header-group; }
tr { break-inside: avoid; page-break-inside: avoid; }
th { text-align: left; font-weight: 600; border-bottom: 1pt solid #5d6573; padding: 3pt 4pt; vertical-align: bottom; }
td { padding: 2.5pt 4pt; border-bottom: 0.4pt solid #e1e5ec; vertical-align: top; }
tbody tr:nth-child(even) td { background: #f5f7fa; }
th.num { text-align: right; }
td.num { text-align: right; white-space: nowrap; font-variant-numeric: tabular-nums; }
table.rank { font-size: 7.3pt; }
table.rank th, table.rank td { padding: 2pt 3pt; }
table.rank td:nth-child(2) { min-width: 23mm; }
table.rank td:nth-child(4) { min-width: 34mm; }
table.rank td:nth-child(11) { min-width: 15mm; }
tr.group td { background: #e9eef6 !important; font-weight: 600; }
figure { margin: 8pt 0 10pt; break-inside: avoid; page-break-inside: avoid; }
figure img { width: 100%%; display: block; }
figcaption { font-size: 8.6pt; font-style: italic; color: #3b4150; margin-top: 3pt; }
blockquote { margin: 6pt 0; padding-left: 8pt; border-left: 2pt solid %(accent)s; }
.title-page { break-after: page; }
.title-page h1 { font-size: 23pt; margin: 18mm 0 4pt; }
.title-page .subtitle { font-size: 13pt; color: #3b4150; margin: 0 0 14pt; }
.title-page .author { font-size: 10.5pt; margin: 0; }
.title-page .date { font-size: 9.5pt; color: #5d6573; margin: 2pt 0 14pt; }
.title-page .definition { font-size: 10.5pt; border-left: 3pt solid %(accent)s; padding: 6pt 10pt; background: #f2f6fc; margin: 0 0 12pt; }
.title-page table { font-size: 9pt; }
.title-page .note { font-size: 8pt; color: #5d6573; margin-top: 10pt; }
.toc { break-after: page; }
.toc h2 { border: none; }
.toc ol { list-style: none; margin: 0; padding: 0; }
.toc li { margin: 2.5pt 0; }
.toc li.l3 { margin-left: 14pt; font-size: 9pt; }
.appendix-start { break-before: page; }
"""


def title_page(front: list[str], summary_def: str, final: pd.DataFrame, cfg: dict[str, Any], top_n: int) -> str:
    """Title, subtitle, author, date, the pressure definition and both top 10s."""
    title = front[0].lstrip("# ").strip()
    author = next((l for l in front if l.startswith("**Vaibhav")), "")
    note = next((l for l in front if l.startswith("Every number")), "")
    bat = final[(final["role"] == "batter") & (final["rank"] <= top_n)].sort_values("rank")
    bowl = final[(final["role"] == "bowler") & (final["rank"] <= top_n)].sort_values("rank")
    rows = "".join(f"<tr><td class='num'>{a.rank}</td><td>{html.escape(a.player_name)}</td>"
                   f"<td>{html.escape(b.player_name)}</td></tr>" for a, b in zip(bat.itertuples(), bowl.itertuples()))
    return (f'<section class="title-page"><h1>{inline(title)}</h1>'
            f'<p class="subtitle">{html.escape(cfg["subtitle"])}</p>'
            f'<p class="author">{inline(author)}</p><p class="date">{html.escape(cfg["date"])}</p>'
            f'<p class="definition"><strong>Pressure, in one sentence:</strong> {inline(summary_def)}</p>'
            f'<table><thead><tr><th class="num">#</th><th>Top 10 batters</th><th>Top 10 bowlers</th></tr></thead>'
            f'<tbody>{rows}</tbody></table><p class="note">{inline(note)}</p></section>')


def contents(heads: list[tuple[int, str, str]]) -> str:
    """Clickable table of contents (sections and subsections)."""
    items = "".join(f'<li class="l{lvl}"><a href="#{hid}">{inline(text)}</a></li>' for lvl, hid, text in heads if lvl <= 3)
    return f'<section class="toc"><h2 id="contents">Contents</h2><ol>{items}</ol></section>'


def build_html(config: dict[str, Any], outputs: Path) -> str:
    """The full HTML document."""
    cfg = config["report"]["pdf"]
    report_dir = outputs / config["report"]["dir"]
    lines = (report_dir / "methodology_report.md").read_text(encoding="utf-8").split("\n")
    start = next(i for i, l in enumerate(lines) if l.startswith("## 0."))
    front, body = [l for l in lines[:start] if l.strip() and l.strip() != "---"], lines[start:]
    definition = next(l for l in body if l.startswith("**Pressure, in one sentence:**")).split("**", 2)[2].strip()
    final = pd.read_csv(outputs / "final_rankings.csv")
    html_body, heads = body_html(body, report_dir, float(config["report"]["figure_width_in"]), int(cfg["image_dpi"]))
    title = front[0].lstrip("# ").strip()
    css = CSS % {"accent": cfg["accent"], "running": cfg["running_header"].replace('"', "'")}
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{html.escape(title)}</title>'
            f"<style>{css}</style></head><body>"
            + title_page(front, definition, final, cfg, int(config["rating"]["top_n"]))
            + contents(heads) + html_body + "</body></html>")


# ---------------------------------------------------------------- printing
def find_browser(candidates: list[str]) -> str:
    """First installed Chromium-based browser from the configured list."""
    for c in candidates:
        if Path(c).exists() or shutil.which(c):
            return c
    raise FileNotFoundError(f"No headless browser found; tried {candidates}")


def print_pdf(html_path: Path, pdf_path: Path, browser: str, timeout_s: int) -> None:
    """Print the HTML to PDF with a headless browser (fresh profile, document outline for bookmarks)."""
    pdf_path.unlink(missing_ok=True)
    with tempfile.TemporaryDirectory() as profile:
        cmd = [browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--generate-pdf-document-outline",
               "--allow-file-access-from-files", f"--user-data-dir={profile}", f"--print-to-pdf={pdf_path}",
               html_path.resolve().as_uri()]
        subprocess.run(cmd, check=False, timeout=timeout_s, capture_output=True)
        deadline = time.time() + timeout_s
        last = -1
        while time.time() < deadline:          # the launcher can return before the file is complete
            size = pdf_path.stat().st_size if pdf_path.exists() else -1
            if size > 0 and size == last:
                break
            last = size
            time.sleep(1)
    if not pdf_path.exists() or pdf_path.stat().st_size == 0:
        raise RuntimeError(f"The browser did not write {pdf_path}")


def fix_outline(pdf_path: Path, titles: list[str]) -> None:
    """Rewrite the bookmark titles with the clean heading texts.

    The browser builds the outline from the rendered headings; a heading split across a line or pushed to the
    next page can come out doubled or with a missing space. Pages and nesting are kept; only titles change.
    """
    from pypdf import PdfReader, PdfWriter

    # The browser's cross-reference table under-counts objects; pypdf notes it and repairs it. Not an error.
    logging.getLogger("pypdf").setLevel(logging.ERROR)
    reader = PdfReader(str(pdf_path))
    entries: list[tuple[int, str, int]] = []

    def walk(items: list, depth: int) -> None:
        for item in items:
            if isinstance(item, list):
                walk(item, depth + 1)
            else:
                entries.append((depth, item.title, reader.get_destination_page_number(item)))

    walk(reader.outline, 0)
    if len(entries) != len(titles):
        logger.warning("Outline has %d entries for %d headings; titles left as printed", len(entries), len(titles))
        return
    writer = PdfWriter(clone_from=reader)
    writer._root_object.pop("/Outlines", None)        # drop the printed outline, rebuild it with clean titles
    parents: dict[int, Any] = {}
    for (depth, _, page), title in zip(entries, titles):
        parents[depth] = writer.add_outline_item(title=title, page_number=page, parent=parents.get(depth - 1))
    writer.page_mode = "/UseOutlines"
    with open(pdf_path, "wb") as handle:
        writer.write(handle)


def plain(text: str) -> str:
    """Heading text without Markdown or HTML."""
    return html.unescape(re.sub(r"<[^>]+>", "", inline(text)))


def run(config: dict[str, Any] | None = None) -> Path:
    """Build the HTML and print the PDF; returns the PDF path."""
    config = config or load_config()
    outputs = resolve_path(config["paths"]["outputs_dir"])
    cfg = config["report"]["pdf"]
    work = outputs / "cache" / "report_build"
    work.mkdir(parents=True, exist_ok=True)
    html_path = work / "methodology_report.html"
    html_path.write_text(build_html(config, outputs), encoding="utf-8")
    pdf_path = outputs / config["report"]["dir"] / cfg["file_name"]
    print_pdf(html_path, pdf_path, find_browser(cfg["browsers"]), int(cfg["timeout_s"]))
    lines = (outputs / config["report"]["dir"] / "methodology_report.md").read_text(encoding="utf-8").split("\n")
    heads = [plain(m.group(2)) for l in lines if (m := re.match(r"^(#{2,3}) (.*)", l))]
    fix_outline(pdf_path, [plain(lines[0].lstrip("# ").strip()), "Contents", *heads])
    logger.info("PDF written: %s (%.1f MB)", pdf_path, pdf_path.stat().st_size / 1e6)
    return pdf_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run()
