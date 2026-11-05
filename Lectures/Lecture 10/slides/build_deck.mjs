/** Rebuild the lecture using the existing deck's slide templates.
 * Requires the Codex @oai/artifact-tool runtime for authoring only.
 * Students use the PPTX directly and do not need Node.js.
 */
import fs from "node:fs/promises";
import path from "node:path";
import crypto from "node:crypto";
import { fileURLToPath } from "node:url";

const { FileBlob, PresentationFile } = await import(process.env.ARTIFACT_TOOL_MODULE || "@oai/artifact-tool");
const here = path.dirname(fileURLToPath(import.meta.url));
const build = process.env.DECK_BUILD_DIR || path.join(here, ".build");
await fs.mkdir(build, { recursive: true });
const contents = JSON.parse(await fs.readFile(path.join(here, "content.json"), "utf8"));
const source = process.env.DECK_TEMPLATE || path.join(here, "UCB_MFE_Lecture_10_Stock_Research_Harness.pptx");
const deck = await PresentationFile.importPptx(await FileBlob.load(source));
const templates = [...deck.slides.items];
const layouts = await Promise.all(templates.map(async slide => JSON.parse(await (await slide.export({format:"layout"})).text())));
const C = { navy: "#002B49", gold: "#FDB515", ink: "#17364B", muted: "#61717D", green: "#147D64", blue: "#005A9C" };
const fontPolicy = { basis: "reference", families: ["Arial", "Courier New"], referencePath: source,
  referenceSha256: crypto.createHash("sha256").update(await fs.readFile(source)).digest("hex") };

function text(slide, value, x, y, w, h, size = 30, color = C.ink, bold = false, font = "Arial", existing = null) {
  const shape = existing || slide.shapes.add({ geometry: "textbox", position: { left: x, top: y, width: w, height: h },
    fill: "none", line: { fill: "none", width: 0 } });
  shape.position = {left:x, top:y, width:w, height:h};
  shape.text = value;
  shape.text.style = { typeface: font, fontSize: size, color, bold };
  return shape;
}

const tableOwners = [], kept=[];
for (const [i, item] of contents.entries()) {
  const dark = ["cover", "section"].includes(item.type);
  // Clone an existing composition, retaining its theme, layout, and slide frame.
  let index = item.type === "cover" ? 0 : layouts.findIndex(l => l.elements.some(e => e.kind === "textbox" && e.text === "Break"));
  if (!dark) index = layouts.findIndex(l => l.elements.some(e => e.kind === "table") === Boolean(item.table) && !l.elements.some(e => e.text === "Break" || e.text === "UC BERKELEY MFE"));
  if (index < 0) index = dark ? 0 : 1;
  const slide = templates[index].duplicate();
  kept.push(`sl/${slide.id.replace(/^sl\//, "")}`);
  const layout = JSON.parse(await (await slide.export({format:"layout"})).text());
  const textElements = layout.elements.filter(e => e.kind === "textbox");
  const heading = textElements.find(e => e.text !== "UC BERKELEY MFE" && !/^\d{2}$/.test(e.text));
  const footer = textElements.find(e => /^\d{2}$/.test(e.text));
  const retained = new Set([heading?.id,footer?.id].filter(Boolean));
  deck.delete(layout.elements.filter(e => !retained.has(e.id) && /^(sh|tb|im|ch)\//.test(e.id)).map(e => e.id));
  slide.background.fill = dark ? C.navy : "#FFFFFF";
  const headingShape = heading ? deck.resolve(heading.id) : null;
  if (item.type === "cover") {
    text(slide, "UC BERKELEY MFE", 72, 72, 900, 44, 26, C.gold);
    text(slide, item.title, 72, 190, 1100, 230, 72, "#FFFFFF", true, "Arial", headingShape);
    text(slide, item.subtitle, 72, 480, 1050, 160, 30, "#FFFFFF");
  } else if (item.type === "section") {
    text(slide, item.title, 72, 170, 1100, 110, 76, "#FFFFFF", true, "Arial", headingShape);
    text(slide, item.subtitle, 72, 360, 1100, 160, 36, C.gold);
  } else {
    text(slide, item.title, 72, 48, 1136, 112, 44, C.navy, true, "Arial", headingShape);
    let top = 185;
    if (item.table) {
      const rows = item.table.length, columns = item.table[0].length;
      const height = Math.min(385, rows * 66);
      const widths = item.widths || (columns === 2 ? [390, 746] : [340, 340, 456]);
      const table = slide.tables.add({ rows, columns, left: 72, top, width: 1136, height,
        columnWidths: widths, values: item.table });
      table.borders.assign({ fill: "#D9E2E8", width: 0.7 });
      for (let r = 0; r < rows; r++) {
        table.rows[r].height = height / rows;
        for (let c = 0; c < columns; c++) {
          const cell = table.getCell(r, c);
          cell.fill = r === 0 ? C.navy : (r % 2 ? "#F0F5F8" : "#FFFFFF");
          cell.text.style = { typeface: "Arial", fontSize: 25, color: r === 0 ? "#FFFFFF" : C.ink, bold: r === 0 };
        }
      }
      tableOwners.push(i + 1);
      top += height + 24;
    }
    if (item.code) {
      const lines = item.code.split("\n").length;
      const size = item.code.split("\n").some(l=>l.length>66) ? 24 : 27;
      const height = lines * (size * 1.3) + 18;
      text(slide, item.code, 84, top, 1110, height, size, C.blue, false, "Courier New");
      top += height + 20;
    }
    if (item.body) {
      const remaining = (item.callout ? 566 : 640) - top;
      const each = remaining / item.body.length;
      item.body.forEach((paragraph, j) => text(slide, paragraph, 72, top + each * j, 1120, each - 10,
        item.code || item.table ? 28 : 32, C.ink));
    }
    if (item.callout) text(slide, item.callout, 72, 598, 1110, 63, 25, C.green, true);
  }
  text(slide, String(i + 1).padStart(2, "0"), 1164, 675, 56, 28, 18, dark ? "#FFFFFF" : C.muted, false, "Arial", footer ? deck.resolve(footer.id) : null);
  slide.speakerNotes.text = item.notes || "";
}
deck.slides.keep(kept);
deck.slides.reorder(kept);
await (await PresentationFile.exportPptx(deck)).save(path.join(build, "candidate.pptx"));
for (const [i, slide] of deck.slides.items.entries()) {
  const image = await deck.export({ slide, format: "png", scale: 1 });
  await fs.writeFile(path.join(build, `slide-${String(i + 1).padStart(2, "0")}.png`), new Uint8Array(await image.arrayBuffer()));
  const layout = await slide.export({ format: "layout" });
  await fs.writeFile(path.join(build, `slide-${String(i + 1).padStart(2, "0")}.layout.json`), await layout.text());
}
await fs.writeFile(path.join(build, "requirements.json"), JSON.stringify({
  explicitTotalSlideCount: contents.length, requiredNativeTableOwnerSlides: tableOwners,
  requiredNativeChartOwnerSlides: [], fontPolicy,
}, null, 2));
console.log(`Exported ${contents.length} slides with speaker notes to ${build}`);
