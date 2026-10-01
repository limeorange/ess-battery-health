"""Build the polished Day 1 submission PDF from the generated Markdown report.

The analysis itself lives in ``src/day1_analysis.py``.  This module only turns
the checked report and its figures into a deterministic A4 PDF.
"""

from __future__ import annotations

import argparse
import html
import re
from pathlib import Path

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    HRFlowable,
    Image,
    KeepTogether,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT / "reports" / "DAY1_분석_보고서.md"
DEFAULT_OUTPUT = ROOT / "output" / "pdf" / "DS-MINI-Design-Day1-배터리수명예측-제출본.pdf"

NAVY = colors.HexColor("#17324D")
TEAL = colors.HexColor("#0F8B8D")
ORANGE = colors.HexColor("#F29E4C")
INK = colors.HexColor("#17212B")
MUTED = colors.HexColor("#5F6B76")
PALE = colors.HexColor("#F3F7FA")
GRID = colors.HexColor("#D7E0E7")
WHITE = colors.white


def register_fonts() -> tuple[str, str]:
    """Register Korean fonts available on the target macOS workstation."""

    candidates = [
        (
            Path("/Users/lsh/Library/Fonts/NotoSansKR-Regular.otf"),
            Path("/Users/lsh/Library/Fonts/NotoSansKR-Bold.otf"),
        ),
        (
            Path("/Users/lsh/Library/Fonts/AppleSDGothicNeoR.ttf"),
            Path("/Users/lsh/Library/Fonts/AppleSDGothicNeoB.ttf"),
        ),
    ]
    for regular, bold in candidates:
        if not (regular.exists() and bold.exists()):
            continue
        try:
            pdfmetrics.registerFont(TTFont("ReportKR", str(regular)))
            pdfmetrics.registerFont(TTFont("ReportKRBold", str(bold)))
            return "ReportKR", "ReportKRBold"
        except Exception:
            continue
    raise RuntimeError("Korean font files could not be registered.")


FONT, FONT_BOLD = register_fonts()


def inline_markup(text: str) -> str:
    """Convert the small Markdown subset used by the report to ReportLab XML."""

    text = (
        text.replace("·", " / ")
        .replace("—", " - ")
        .replace("–", "-")
        .replace("‑", "-")
    )
    escaped = html.escape(text.strip())
    escaped = re.sub(r"`([^`]+)`", r'<font color="#0F6B78">\1</font>', escaped)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", escaped)
    return escaped


def build_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "body": ParagraphStyle(
            "BodyKR",
            parent=base["BodyText"],
            fontName=FONT,
            fontSize=9.4,
            leading=15.2,
            textColor=INK,
            spaceAfter=6,
            wordWrap="CJK",
        ),
        "h1": ParagraphStyle(
            "Heading1KR",
            parent=base["Heading1"],
            fontName=FONT_BOLD,
            fontSize=20,
            leading=26,
            textColor=NAVY,
            spaceBefore=3,
            spaceAfter=10,
            keepWithNext=True,
            wordWrap="CJK",
        ),
        "h2": ParagraphStyle(
            "Heading2KR",
            parent=base["Heading2"],
            fontName=FONT_BOLD,
            fontSize=16,
            leading=21,
            textColor=NAVY,
            spaceBefore=4,
            spaceAfter=9,
            keepWithNext=True,
            wordWrap="CJK",
        ),
        "h3": ParagraphStyle(
            "Heading3KR",
            parent=base["Heading3"],
            fontName=FONT_BOLD,
            fontSize=11.5,
            leading=16,
            textColor=TEAL,
            spaceBefore=9,
            spaceAfter=5,
            keepWithNext=True,
            wordWrap="CJK",
        ),
        "bullet": ParagraphStyle(
            "BulletKR",
            parent=base["BodyText"],
            fontName=FONT,
            fontSize=9.1,
            leading=14.2,
            leftIndent=12,
            firstLineIndent=-7,
            bulletIndent=2,
            textColor=INK,
            spaceAfter=3,
            wordWrap="CJK",
        ),
        "quote": ParagraphStyle(
            "QuoteKR",
            parent=base["BodyText"],
            fontName=FONT,
            fontSize=9.2,
            leading=14.5,
            leftIndent=10,
            rightIndent=10,
            borderColor=TEAL,
            borderWidth=0,
            borderPadding=8,
            backColor=colors.HexColor("#EAF5F5"),
            textColor=INK,
            wordWrap="CJK",
        ),
        "caption": ParagraphStyle(
            "CaptionKR",
            parent=base["BodyText"],
            fontName=FONT,
            fontSize=8,
            leading=11,
            textColor=MUTED,
            alignment=TA_CENTER,
            spaceBefore=4,
            spaceAfter=8,
            wordWrap="CJK",
        ),
        "table": ParagraphStyle(
            "TableKR",
            parent=base["BodyText"],
            fontName=FONT,
            fontSize=7.6,
            leading=10,
            textColor=INK,
            wordWrap="CJK",
        ),
        "table_small": ParagraphStyle(
            "TableSmallKR",
            parent=base["BodyText"],
            fontName=FONT,
            fontSize=6.1,
            leading=8.1,
            textColor=INK,
            wordWrap="CJK",
        ),
        "table_header": ParagraphStyle(
            "TableHeaderKR",
            parent=base["BodyText"],
            fontName=FONT_BOLD,
            fontSize=7.6,
            leading=10,
            textColor=WHITE,
            wordWrap="CJK",
        ),
        "table_header_small": ParagraphStyle(
            "TableHeaderSmallKR",
            parent=base["BodyText"],
            fontName=FONT_BOLD,
            fontSize=6.1,
            leading=8.1,
            textColor=WHITE,
            wordWrap="CJK",
        ),
        "toc_title": ParagraphStyle(
            "TOCTitleKR",
            parent=base["Heading1"],
            fontName=FONT_BOLD,
            fontSize=20,
            leading=25,
            textColor=NAVY,
            spaceAfter=14,
        ),
        "small": ParagraphStyle(
            "SmallKR",
            parent=base["BodyText"],
            fontName=FONT,
            fontSize=7.5,
            leading=10,
            textColor=MUTED,
            wordWrap="CJK",
        ),
    }


STYLES = build_styles()


class Day1DocTemplate(BaseDocTemplate):
    def __init__(self, filename: str):
        super().__init__(
            filename,
            pagesize=A4,
            rightMargin=18 * mm,
            leftMargin=18 * mm,
            topMargin=20 * mm,
            bottomMargin=18 * mm,
            title="Day 1 배터리 수명 예측 EDA 보고서",
            author="DS Mini Project Team",
            subject="초기 100사이클 기반 배터리 수명 예측 EDA 및 모델 전략",
        )

        pw, ph = A4
        lw, lh = landscape(A4)
        self.addPageTemplates(
            [
                PageTemplate(
                    id="cover",
                    pagesize=A4,
                    frames=[Frame(18 * mm, 18 * mm, pw - 36 * mm, ph - 36 * mm, id="cover_frame")],
                    onPage=self._cover_page,
                ),
                PageTemplate(
                    id="portrait",
                    pagesize=A4,
                    frames=[Frame(18 * mm, 18 * mm, pw - 36 * mm, ph - 38 * mm, id="portrait_frame")],
                    onPage=self._content_page,
                ),
                PageTemplate(
                    id="landscape",
                    pagesize=landscape(A4),
                    frames=[Frame(14 * mm, 16 * mm, lw - 28 * mm, lh - 34 * mm, id="landscape_frame")],
                    onPage=self._content_page,
                ),
            ]
        )

    @staticmethod
    def _cover_page(canvas, doc):
        canvas.saveState()
        width, height = doc.pagesize
        canvas.setFillColor(NAVY)
        canvas.rect(0, height - 16 * mm, width, 16 * mm, fill=1, stroke=0)
        canvas.setFillColor(TEAL)
        canvas.rect(0, 0, width, 7 * mm, fill=1, stroke=0)
        canvas.restoreState()

    @staticmethod
    def _content_page(canvas, doc):
        canvas.saveState()
        width, height = doc.pagesize
        canvas.setStrokeColor(GRID)
        canvas.setLineWidth(0.5)
        canvas.line(14 * mm, height - 12 * mm, width - 14 * mm, height - 12 * mm)
        canvas.setFont(FONT_BOLD, 7.5)
        canvas.setFillColor(NAVY)
        canvas.drawString(14 * mm, height - 9 * mm, "DS MINI PROJECT  |  DAY 1")
        canvas.setFont(FONT, 7.5)
        canvas.setFillColor(MUTED)
        canvas.drawRightString(width - 14 * mm, 9 * mm, f"{doc.page} / 분석 보고서")
        canvas.restoreState()

    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph):
            style_name = flowable.style.name
            if style_name == "Heading1KR":
                self.notify("TOCEntry", (0, flowable.getPlainText(), self.page))


def cover_story() -> list:
    title = Paragraph(
        "초기 100사이클 기반<br/>배터리 수명 예측",
        ParagraphStyle(
            "CoverTitle",
            fontName=FONT_BOLD,
            fontSize=27,
            leading=36,
            textColor=NAVY,
            alignment=TA_LEFT,
        ),
    )
    subtitle = Paragraph(
        "Day 1 탐색적 데이터 분석 및 Day 2 모델 전략",
        ParagraphStyle(
            "CoverSubtitle",
            fontName=FONT,
            fontSize=15,
            leading=22,
            textColor=TEAL,
            alignment=TA_LEFT,
        ),
    )
    eyebrow = Paragraph(
        "DS MINI PROJECT  /  REGRESSION",
        ParagraphStyle(
            "CoverEyebrow",
            fontName=FONT_BOLD,
            fontSize=9,
            leading=12,
            textColor=ORANGE,
            charSpace=0.8,
        ),
    )

    card_style = ParagraphStyle(
        "CoverCard",
        fontName=FONT,
        fontSize=9,
        leading=13,
        textColor=INK,
        alignment=TA_CENTER,
    )
    cards = Table(
        [
            [
                Paragraph("<b>139</b><br/>원본 배터리 셀", card_style),
                Paragraph("<b>129</b><br/>유효 수명 라벨", card_style),
                Paragraph("<b>12</b><br/>한국어 분석 그래프", card_style),
            ]
        ],
        colWidths=[52 * mm, 52 * mm, 52 * mm],
        rowHeights=[30 * mm],
    )
    cards.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), PALE),
                ("BOX", (0, 0), (-1, -1), 0.7, GRID),
                ("INNERGRID", (0, 0), (-1, -1), 0.7, WHITE),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )

    return [
        Spacer(1, 37 * mm),
        eyebrow,
        Spacer(1, 8 * mm),
        title,
        Spacer(1, 6 * mm),
        subtitle,
        Spacer(1, 17 * mm),
        HRFlowable(width="100%", thickness=2.2, color=TEAL, spaceAfter=15 * mm),
        cards,
        Spacer(1, 42 * mm),
        Paragraph(
            "Batch 1/2/3 통합 품질 감사  |  ΔQ(V) 조기 열화 신호  |  재현 가능한 feature 결정",
            ParagraphStyle(
                "CoverNote",
                fontName=FONT,
                fontSize=9.5,
                leading=15,
                textColor=MUTED,
                alignment=TA_LEFT,
            ),
        ),
        Spacer(1, 8 * mm),
        Paragraph("작성일  2026. 10. 01.", STYLES["small"]),
        NextPageTemplate("portrait"),
        PageBreak(),
    ]


def toc_story() -> list:
    toc = TableOfContents()
    toc.levelStyles = [
        ParagraphStyle(
            "TOCLevel1KR",
            fontName=FONT_BOLD,
            fontSize=9.5,
            leading=14,
            leftIndent=0,
            firstLineIndent=0,
            textColor=NAVY,
            spaceBefore=2,
        ),
        ParagraphStyle(
            "TOCLevel2KR",
            fontName=FONT,
            fontSize=8.5,
            leading=14,
            leftIndent=12,
            firstLineIndent=0,
            textColor=MUTED,
        ),
    ]
    return [
        Paragraph("목차", STYLES["toc_title"]),
        Paragraph(
            "그래프/통계에서 관찰을 도출하고, 원인 가설과 Day 2 feature/모델 전략으로 연결했습니다.",
            STYLES["body"],
        ),
        Spacer(1, 5 * mm),
        toc,
        PageBreak(),
    ]


def _table_widths(rows: list[list[str]], available: float) -> list[float]:
    ncols = len(rows[0])
    header = [c.strip() for c in rows[0]]
    if header and header[0] == "Feature" and ncols >= 10:
        widths = [60, 50, 53, 76, 37, 37, 37, 48, 42, 62, 146, 48]
        scale = available / sum(widths)
        return [w * scale for w in widths]

    weights = []
    for col in range(ncols):
        longest = max(len(re.sub(r"[`*]", "", row[col])) for row in rows[: min(len(rows), 16)])
        weights.append(min(max(longest, 5), 28))
    minimum = 34 if ncols >= 8 else 48
    raw = [max(minimum, w * 4.4) for w in weights]
    scale = available / sum(raw)
    return [w * scale for w in raw]


def make_table(rows: list[list[str]], wide: bool) -> Table:
    available = (landscape(A4)[0] - 28 * mm) if wide else (A4[0] - 36 * mm)
    style_key = "table_small" if wide else "table"
    header_style_key = "table_header_small" if wide else "table_header"
    cells = [
        [
            Paragraph(
                inline_markup(cell),
                STYLES[header_style_key] if row_idx == 0 else STYLES[style_key],
            )
            for cell in row
        ]
        for row_idx, row in enumerate(rows)
    ]
    table = Table(
        cells,
        colWidths=_table_widths(rows, available),
        repeatRows=1,
        hAlign="LEFT",
        splitByRow=1,
    )
    rules = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("FONTNAME", (0, 0), (-1, 0), FONT_BOLD),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.35, GRID),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    for row_idx in range(1, len(rows)):
        if row_idx % 2 == 0:
            rules.append(("BACKGROUND", (0, row_idx), (-1, row_idx), PALE))
    table.setStyle(TableStyle(rules))
    return table


def make_figure(path: Path, caption: str) -> KeepTogether:
    if not path.exists():
        raise FileNotFoundError(f"Figure not found: {path}")
    with PILImage.open(path) as source:
        width_px, height_px = source.size
    max_width = A4[0] - 38 * mm
    max_height = 145 * mm
    ratio = min(max_width / width_px, max_height / height_px)
    image = Image(str(path), width=width_px * ratio, height=height_px * ratio)
    image.hAlign = "CENTER"
    return KeepTogether(
        [
            Spacer(1, 3 * mm),
            image,
            Paragraph(f"그림. {inline_markup(caption)}", STYLES["caption"]),
        ]
    )


def parse_report(report_path: Path) -> list:
    lines = report_path.read_text(encoding="utf-8").splitlines()
    story: list = []
    paragraph_buffer: list[str] = []
    section_count = 0
    idx = 0

    def flush_paragraph() -> None:
        if paragraph_buffer:
            story.append(Paragraph(inline_markup(" ".join(paragraph_buffer)), STYLES["body"]))
            paragraph_buffer.clear()

    while idx < len(lines):
        line = lines[idx].rstrip()
        stripped = line.strip()

        if not stripped:
            flush_paragraph()
            idx += 1
            continue

        image_match = re.fullmatch(r"!\[([^]]*)\]\(([^)]+)\)", stripped)
        if image_match:
            flush_paragraph()
            caption, rel_path = image_match.groups()
            image_path = (report_path.parent / rel_path).resolve()
            story.append(make_figure(image_path, caption))
            idx += 1
            continue

        if stripped.startswith("|"):
            flush_paragraph()
            block: list[str] = []
            while idx < len(lines) and lines[idx].strip().startswith("|"):
                block.append(lines[idx].strip())
                idx += 1
            parsed = [[cell.strip() for cell in row.strip("|").split("|")] for row in block]
            if len(parsed) > 1 and all(re.fullmatch(r":?-{3,}:?", c.replace(" ", "")) for c in parsed[1]):
                parsed.pop(1)
            wide = len(parsed[0]) >= 8
            if wide:
                # Keep a section title with its opening wide table instead of
                # leaving an almost-empty portrait page before the table.
                heading_flowable = None
                if (
                    story
                    and isinstance(story[-1], Paragraph)
                    and story[-1].style.name == "Heading1KR"
                ):
                    heading_flowable = story.pop()
                story.extend([NextPageTemplate("landscape"), PageBreak()])
                if heading_flowable is not None:
                    story.append(heading_flowable)
                story.append(make_table(parsed, True))
                story.extend([NextPageTemplate("portrait"), PageBreak()])
            else:
                story.extend([Spacer(1, 2 * mm), make_table(parsed, False), Spacer(1, 4 * mm)])
            continue

        heading = re.match(r"^(#{1,3})\s+(.+)$", stripped)
        if heading:
            flush_paragraph()
            level, text = len(heading.group(1)), heading.group(2)
            if level == 1:
                # The cover already carries the report title.
                idx += 1
                continue
            if level == 2:
                story.append(Paragraph(inline_markup(text), STYLES["h1"]))
                section_count += 1
            else:
                story.append(Paragraph(inline_markup(text), STYLES["h2"]))
            idx += 1
            continue

        if stripped == "---":
            flush_paragraph()
            story.append(HRFlowable(width="100%", thickness=0.6, color=GRID, spaceBefore=4, spaceAfter=7))
            idx += 1
            continue

        if stripped.startswith("> "):
            flush_paragraph()
            story.append(Paragraph(inline_markup(stripped[2:]), STYLES["quote"]))
            idx += 1
            continue

        bullet_match = re.match(r"^[-*]\s+(.+)$", stripped)
        numbered_match = re.match(r"^(\d+)\.\s+(.+)$", stripped)
        if bullet_match:
            flush_paragraph()
            story.append(Paragraph(inline_markup(bullet_match.group(1)), STYLES["bullet"], bulletText="•"))
            idx += 1
            continue
        if numbered_match:
            flush_paragraph()
            story.append(
                Paragraph(
                    inline_markup(numbered_match.group(2)),
                    STYLES["bullet"],
                    bulletText=f"{numbered_match.group(1)}.",
                )
            )
            idx += 1
            continue

        paragraph_buffer.append(stripped)
        idx += 1

    flush_paragraph()
    return story


def build_pdf(report_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc = Day1DocTemplate(str(output_path))
    story = cover_story() + toc_story() + parse_report(report_path)
    doc.multiBuild(story)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    build_pdf(args.report.resolve(), args.output.resolve())
    print(args.output.resolve())


if __name__ == "__main__":
    main()
