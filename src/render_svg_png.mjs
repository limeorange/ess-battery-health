#!/usr/bin/env node
import path from "node:path";
import { createRequire } from "node:module";

const dependencyRoot = process.env.CODEX_NODE_MODULES;
if (!dependencyRoot) {
  throw new Error("CODEX_NODE_MODULES must point to the bundled node_modules directory.");
}
const require = createRequire(path.join(dependencyRoot, "package.json"));
const sharp = require("sharp");

if (process.argv.length < 3) {
  throw new Error("Usage: render_svg_png.mjs <input.svg> [output.png]");
}

const input = path.resolve(process.argv[2]);
const output = path.resolve(process.argv[3] ?? input.replace(/\.svg$/i, ".png"));

await sharp(input, { density: 180 }).png().toFile(output);
console.log(output);
