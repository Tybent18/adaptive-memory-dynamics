"""Create a versioned PDF from the actual Stage Two Markdown report and figures."""
from __future__ import annotations

import re
from html import escape
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Image, LongTable, Paragraph, SimpleDocTemplate, Spacer, TableStyle

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "docs/STAGE_TWO_RESULTS.md"
OUTPUT = ROOT / "docs/STAGE_TWO_RESULTS.pdf"
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="ResearchTitle", parent=styles["Title"], fontSize=18, leading=23, textColor=colors.HexColor("#123349")))
styles.add(ParagraphStyle(name="ResearchHeader", parent=styles["Heading2"], fontSize=12, leading=16, spaceBefore=16, textColor=colors.HexColor("#123349")))
styles.add(ParagraphStyle(name="ResearchSub", parent=styles["Heading3"], fontSize=10, leading=13, spaceBefore=10))
styles.add(ParagraphStyle(name="ResearchBody", parent=styles["BodyText"], fontSize=9, leading=13.5, spaceAfter=8))
styles.add(ParagraphStyle(name="ResearchCell", parent=styles["BodyText"], fontSize=7.1, leading=10, wordWrap="CJK"))

def plain(value: str) -> str:
    value = escape(value).replace("—", " - ").replace("×", " x ").replace("±", "+/-")
    value = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", value)
    value = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", value)
    return value

def main() -> None:
    lines = REPORT.read_text(encoding="utf-8").splitlines()
    story = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line:
            continue
        if line.startswith("|"):
            raw = []
            while True:
                pieces = [cell.strip() for cell in line.strip("|").split("|")]
                if not all(re.fullmatch(r"[-: ]+", cell) for cell in pieces):
                    raw.append([Paragraph(plain(cell), styles["ResearchCell"]) for cell in pieces])
                if i >= len(lines) or not lines[i].strip().startswith("|"):
                    break
                line = lines[i].strip()
                i += 1
            if raw:
                tab = LongTable(raw, colWidths=[465 / len(raw[0])] * len(raw[0]), repeatRows=1)
                tab.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DCEAF1")),
                    ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#C7D9E0")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]))
                story.extend([tab, Spacer(1, 12)])
            continue
        if line.startswith("!["):
            match = re.match(r"!\[[^]]*]\(([^)]+)\)", line)
            if match:
                file = (REPORT.parent / match.group(1)).resolve()
                if file.exists():
                    graph = Image(str(file))
                    graph.drawWidth = 455
                    graph.drawHeight = 455 * graph.imageHeight / graph.imageWidth
                    story.extend([Spacer(1, 6), graph, Spacer(1, 12)])
            continue
        if line.startswith("# "):
            style, body = "ResearchTitle", line[2:]
        elif line.startswith("## "):
            style, body = "ResearchHeader", line[3:]
        elif line.startswith("### "):
            style, body = "ResearchSub", line[4:]
        else:
            style, body = "ResearchBody", line
        story.append(Paragraph(plain(body), styles[style]))
    doc = SimpleDocTemplate(str(OUTPUT), pagesize=(612, 792),
                            topMargin=63, bottomMargin=58, leftMargin=72, rightMargin=72,
                            title="Adaptive Memory Stage Two Research Results")
    def footer(canvas, document):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#C3DAE6"))
        canvas.line(72, 748, 540, 748)
        canvas.setFont("Helvetica", 7)
        canvas.drawString(72, 756, "ADAPTIVE MEMORY DYNAMICS  /  STAGE TWO")
        canvas.drawString(72, 37, "Exploratory simulated study - NOT peer reviewed")
        canvas.drawRightString(540, 37, str(document.page))
        canvas.restoreState()
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    assert OUTPUT.stat().st_size > 10000
    print("PDF generated:", OUTPUT, OUTPUT.stat().st_size)

if __name__ == "__main__":
    main()
