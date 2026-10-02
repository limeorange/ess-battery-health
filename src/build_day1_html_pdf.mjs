#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";
import { pathToFileURL } from "node:url";

const projectRoot = path.resolve(import.meta.dirname, "..");
const reportPath = path.join(projectRoot, "reports", "DAY1_분석_보고서.md");
const htmlPath = path.join(projectRoot, "reports", "DAY1_분석_보고서.html");
const pdfPath = path.join(projectRoot, "reports", "DAY1_분석_보고서.pdf");
const designTemplatePath = path.join(
  projectRoot,
  "reports",
  "archive",
  "DAY1_분석_보고서_before_plain_language_expansion_2026-10-02.html",
);
const dependencyRoot = process.env.CODEX_NODE_MODULES;

if (!dependencyRoot) {
  throw new Error("CODEX_NODE_MODULES must point to the bundled node_modules directory.");
}

const require = createRequire(path.join(dependencyRoot, "package.json"));
const { marked } = require("marked");
const { chromium } = require("playwright");

function extractExistingDesign(htmlFile) {
  const source = fs.readFileSync(htmlFile, "utf8");
  const css = source.match(/<style>([\s\S]*?)<\/style>/i)?.[1];
  const cover = source.match(/<section class="page cover">[\s\S]*?<\/section>/i)?.[0];
  if (!css || !cover) {
    throw new Error("Existing HTML design template could not be extracted.");
  }
  return { css, cover };
}

function preprocessMarkdown(markdown) {
  const start = markdown.indexOf("## 0. Executive Summary");
  if (start < 0) {
    throw new Error("Executive Summary heading was not found in the Markdown report.");
  }
  const body = markdown.slice(start);
  let processed = body.replace(
    /!\[([^\]]+)\]\(([^)]+)\)\n\n\*Figure\s+([^*]+)\*/g,
    (_, alt, src, caption) =>
      `<figure class="chart"><img src="${src}" alt="${alt}"><figcaption>${caption}</figcaption></figure>`,
  );

  // 이 보고서에서 물결표는 모두 수치 범위 표기다. GFM의 취소선 문법으로
  // 해석되지 않도록 먼저 escape한다.
  processed = processed.replace(/~/g, "\\~");

  // 한국어 조사처럼 닫는 ** 뒤에 문자가 바로 이어지면 marked가 일부 강조를
  // 인식하지 못한다. 강조 내부의 inline 문법은 유지하면서 명시적 HTML로 바꾼다.
  processed = processed.replace(/\*\*([^*\n]+?)\*\*/g, (_, content) =>
    `<strong>${marked.parseInline(content, { gfm: true }).trim()}</strong>`,
  );

  return processed;
}

function renderSections(markdown) {
  const sections = markdown.split(/(?=^##\s+)/m).filter((part) => part.trim());
  return sections
    .map((section, index) => {
      const heading = section.match(/^##\s+(.+)$/m)?.[1] ?? `Section ${index + 1}`;
      const slug = heading
        .replace(/[^0-9A-Za-z가-힣]+/g, "-")
        .replace(/^-|-$/g, "")
        .toLowerCase();
      const sectionClass = heading.startsWith("3. EDA") ? "report-section eda-section" : "report-section";
      return `<section class="${sectionClass}" id="${slug}">${marked.parse(section, {
        gfm: true,
        breaks: false,
      })}</section>`;
    })
    .join("\n");
}

function tocPage() {
  return `
  <section class="page toc-page">
    <div class="page-header"><span class="section-name">Contents</span><span>Report Navigation</span></div>
    <p class="eyebrow">Report Map</p>
    <h1>초기 신호에서<br>Day 2 전략까지</h1>
    <p class="toc-intro">데이터를 이해하고, 다섯 가지 EDA 질문에서 근거를 쌓은 뒤, 재현 가능한 Regression 검증 전략으로 수렴한다.</p>
    <div class="toc-grid">
      <div class="toc-block" data-no="01">
        <span class="toc-kicker">CONTEXT &amp; DATA</span>
        <div class="toc-title"><strong>문제와 데이터</strong><span class="pages">03-07</span></div>
        <ul class="toc-list">
          <li><span>0. Executive Summary</span><span>03</span></li>
          <li><span>1. 문제 정의와 핵심 용어</span><span>04-05</span></li>
          <li><span>2. 데이터와 분석 원칙</span><span>06-07</span></li>
        </ul>
      </div>
      <div class="toc-block featured" data-no="02">
        <span class="toc-kicker">EXPLORATORY ANALYSIS</span>
        <div class="toc-title"><strong>다섯 개 EDA 질문</strong><span class="pages">08-20</span></div>
        <ul class="toc-list">
          <li><span>Q1. Cycle Life 분포와 저수명 군집</span><span>08-10</span></li>
          <li><span>Q2. Capacity 열화와 Knee</span><span>11-12</span></li>
          <li><span>Q3. ΔQ(V) 조기 열화 신호</span><span>13-15</span></li>
          <li><span>Q4-5. 충전 조건·상관·중복</span><span>15-20</span></li>
        </ul>
      </div>
      <div class="toc-block featured" data-no="03">
        <span class="toc-kicker">MODEL STRATEGY</span>
        <div class="toc-title"><strong>Feature와 모델 전략</strong><span class="pages">21-24</span></div>
        <ul class="toc-list">
          <li><span>4. 분석의 논리 사슬</span><span>21</span></li>
          <li><span>5. Feature Engineering</span><span>22</span></li>
          <li><span>6. Regression과 모델 후보</span><span>23</span></li>
          <li><span>7. Feature Ablation</span><span>24</span></li>
        </ul>
      </div>
      <div class="toc-block" data-no="04">
        <span class="toc-kicker">VALIDATION &amp; DELIVERY</span>
        <div class="toc-title"><strong>검증과 실행</strong><span class="pages">25-28</span></div>
        <ul class="toc-list">
          <li><span>8. Validation Strategy</span><span>25</span></li>
          <li><span>9. 평가요소 대응</span><span>26</span></li>
          <li><span>10. 한계와 다음 단계</span><span>27</span></li>
          <li><span>Appendix. 재현 산출물</span><span>28</span></li>
        </ul>
      </div>
    </div>
    <div class="reading-guide">
      <h3 class="no-margin">읽는 방법</h3>
      <div class="reading-guide-row">
        <div class="reading-step"><b>QUESTION</b><span>무엇을 확인하는가</span></div>
        <div class="reading-step"><b>EVIDENCE</b><span>어떤 수치와 그림인가</span></div>
        <div class="reading-step"><b>INTERPRET</b><span>어디까지 말할 수 있는가</span></div>
        <div class="reading-step"><b>DECIDE</b><span>Day 2에 무엇을 남기는가</span></div>
      </div>
    </div>
  </section>`;
}

const supplementalCss = `
  /* DAY1_AUTOGEN_OVERRIDES */
  .report-body { width: 210mm; margin: 0 auto; }
  .report-section {
    width: 210mm;
    min-height: 278mm;
    margin: 0 auto 10mm;
    padding: 15mm 16mm 16mm;
    overflow: visible;
    background: #fff;
    box-shadow: 0 8px 28px rgba(23, 50, 77, 0.14);
    break-before: page;
    page-break-before: always;
    -webkit-box-decoration-break: clone;
    box-decoration-break: clone;
  }

  .report-section > h2:first-child {
    margin: -3mm 0 6mm;
    padding: 0 0 3.2mm;
    border-bottom: 0.55mm solid var(--teal);
    font-size: 19pt;
  }

  .report-section h3 {
    margin-top: 8mm;
    padding-top: 2mm;
    border-top: 0.35mm solid var(--line);
    font-size: 14pt;
    break-after: avoid;
    page-break-after: avoid;
  }


  .report-section h4 {
    margin-top: 4mm;
    margin-bottom: 1.5mm;
    color: var(--teal);
    font-size: 10pt;
    break-after: avoid;
    page-break-after: avoid;
  }

  .report-section p,
  .report-section li { font-size: 9.35pt; line-height: 1.62; }
  .report-section p { orphans: 3; widows: 3; }
  .report-section ul, .report-section ol { margin-top: 1.5mm; margin-bottom: 3.2mm; }
  .report-section li { margin-bottom: 1.1mm; }
  .report-section code {
    padding: 0.15em 0.35em;
    border-radius: 0.8mm;
    background: #edf3f6;
    color: #0f6873;
    font-family: "SFMono-Regular", Consolas, monospace;
    font-size: 0.88em;
  }

  .report-section blockquote {
    position: relative;
    margin: 3.2mm 0 4mm;
    padding: 3.5mm 4.5mm 3.5mm 5.5mm;
    border: 0;
    border-radius: 1.8mm;
    background: var(--teal-soft);
    color: #244244;
    break-inside: avoid;
    page-break-inside: avoid;
  }
  .report-section blockquote::before {
    content: "";
    position: absolute;
    inset: 0 auto 0 0;
    width: 1.2mm;
    border-radius: 1.8mm 0 0 1.8mm;
    background: var(--teal);
  }
  .report-section blockquote p { margin: 0; }
  .report-section blockquote strong { color: var(--navy); }

  .report-section table {
    margin: 3mm 0 4mm;
    font-size: 7.1pt;
    line-height: 1.34;
    break-inside: avoid;
    page-break-inside: avoid;
  }
  .report-section table.wide { font-size: 6.35pt; }
  .report-section table.wide th,
  .report-section table.wide td { padding: 1.35mm 1.1mm; }
  .report-section th { background: var(--navy); color: #fff; }
  .report-section td:first-child { font-weight: 650; }

  .report-section figure.chart {
    margin: 4mm 0 5mm;
    padding: 2.4mm;
    border: 0.3mm solid var(--line);
    border-radius: 2mm;
    background: #fff;
    break-inside: avoid;
    page-break-inside: avoid;
  }
  .report-section figure.chart img {
    display: block;
    width: 100%;
    max-width: 100%;
    max-height: 112mm;
    margin: 0 auto;
    object-fit: contain;
  }
  .report-section figure.chart figcaption {
    margin-top: 1.5mm;
    color: var(--muted);
    font-size: 7.3pt;
    line-height: 1.45;
    text-align: center;
  }

  .toc-page .toc-title { margin-top: 2mm; }
  .toc-page .toc-list li { grid-template-columns: 1fr; }
  .toc-page .reading-guide { margin-top: 7mm; }

  @media print {
    @page { size: A4 portrait; margin: 0 0 8mm; }
    html, body { width: 210mm; margin: 0; padding: 0; background: #fff; }
    .page {
      width: 210mm;
      height: 289mm;
      margin: 0;
      box-shadow: none;
      break-after: page;
      page-break-after: always;
    }
    .cover { height: 289mm; }
    .report-body { width: 210mm; margin: 0; }
    .report-section {
      width: 210mm;
      min-height: 281mm;
      margin: 0;
      padding: 14mm 16mm 12mm;
      box-shadow: none;
      break-before: page;
      page-break-before: always;
    }
    .report-section h3 { break-before: auto; page-break-before: auto; }
    .report-section:first-child { break-before: page; page-break-before: always; }
  }
`;

async function build() {
  const design = extractExistingDesign(designTemplatePath);
  const markdown = fs.readFileSync(reportPath, "utf8");
  const sectionsHtml = renderSections(preprocessMarkdown(markdown));
  const cover = design.cover
    .replace("12</b><br/>한국어 분석 그래프", "13</b><br/>한국어 분석 그래프")
    .replace("작성일  2026. 10. 01.", "작성일  2026. 10. 01. / 설명 보강  2026. 10. 02.");

  const html = `<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="author" content="U094 이수현">
  <meta name="description" content="초기 100 cycle 기반 배터리 수명 예측 Day 1 최종 분석 보고서">
  <title>초기 100 Cycle 기반 배터리 수명 예측 - Day 1</title>
  <style>${design.css}\n${supplementalCss}</style>
</head>
<body>
${cover}
${tocPage()}
<main class="report-body">${sectionsHtml}</main>
<script>
  document.querySelectorAll("table").forEach((table) => {
    const columns = table.rows[0]?.cells.length ?? 0;
    if (columns >= 6) table.classList.add("wide");
  });
</script>
</body>
</html>`;

  fs.writeFileSync(htmlPath, html, "utf8");

  const chromePath = process.env.CHROME_BIN ?? "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
  const browser = await chromium.launch({ headless: true, executablePath: chromePath });
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 1000 }, deviceScaleFactor: 1 });
    await page.goto(pathToFileURL(htmlPath).href, { waitUntil: "networkidle" });
    await page.evaluate(async () => {
      await document.fonts.ready;
      await Promise.all(
        [...document.images].map((image) =>
          image.complete
            ? Promise.resolve()
            : new Promise((resolve, reject) => {
                image.addEventListener("load", resolve, { once: true });
                image.addEventListener("error", reject, { once: true });
              }),
        ),
      );
    });

    await page.pdf({
      path: pdfPath,
      format: "A4",
      printBackground: true,
      preferCSSPageSize: true,
      displayHeaderFooter: true,
      headerTemplate: "<div></div>",
      footerTemplate: `<div style="width:100%;padding:0 16mm;color:#71808c;font-family:Arial,sans-serif;font-size:7px;display:flex;justify-content:space-between;align-items:center;"><span>DS MINI PROJECT / DAY 1 / U094 이수현</span><span><span class="pageNumber"></span> / <span class="totalPages"></span></span></div>`,
      margin: { top: "0mm", right: "0mm", bottom: "8mm", left: "0mm" },
    });
  } finally {
    await browser.close();
  }

  console.log(htmlPath);
  console.log(pdfPath);
}

await build();
