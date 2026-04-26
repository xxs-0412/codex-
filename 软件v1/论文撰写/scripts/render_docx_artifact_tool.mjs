#!/usr/bin/env node

import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";

function parseArgs(argv) {
  const args = {
    inputPath: undefined,
    outputDir: undefined,
    scale: 2,
    artifactToolPackage: process.env.ARTIFACT_TOOL_PACKAGE_DIR,
    metadataJson: undefined,
    verbose: false,
  };

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--output_dir") {
      args.outputDir = argv[index + 1];
      index += 1;
    } else if (arg === "--scale") {
      args.scale = Number(argv[index + 1]);
      index += 1;
    } else if (arg === "--artifact_tool_package") {
      args.artifactToolPackage = argv[index + 1];
      index += 1;
    } else if (arg === "--metadata_json") {
      args.metadataJson = argv[index + 1];
      index += 1;
    } else if (arg === "--verbose") {
      args.verbose = true;
    } else if (!args.inputPath) {
      args.inputPath = arg;
    } else {
      throw new Error(`Unexpected argument: ${arg}`);
    }
  }

  if (!args.inputPath) {
    throw new Error(
      "usage: render_docx_artifact_tool.mjs <input.docx> --output_dir <out>"
    );
  }
  if (!args.outputDir) {
    const parsed = path.parse(args.inputPath);
    args.outputDir = path.join(parsed.dir, parsed.name);
  }
  if (!args.artifactToolPackage) {
    throw new Error(
      "--artifact_tool_package is required unless ARTIFACT_TOOL_PACKAGE_DIR is set."
    );
  }
  if (!Number.isFinite(args.scale) || args.scale <= 0) {
    throw new Error("--scale must be a positive number.");
  }

  args.inputPath = path.resolve(args.inputPath);
  args.outputDir = path.resolve(args.outputDir);
  args.artifactToolPackage = path.resolve(args.artifactToolPackage);
  if (args.metadataJson) {
    args.metadataJson = path.resolve(args.metadataJson);
  }
  return args;
}

function moduleUrl(filePath) {
  return pathToFileURL(filePath).href;
}

async function importArtifactTool(artifactToolPackage) {
  return import(
    moduleUrl(path.join(artifactToolPackage, "dist", "artifact_tool.mjs"))
  );
}

function createCanvas(width, height) {
  if (typeof OffscreenCanvas === "undefined") {
    throw new Error(
      "OffscreenCanvas is unavailable. artifact-tool DOCX rendering needs skia-canvas."
    );
  }
  const canvas = new OffscreenCanvas(width, height);
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    throw new Error("Canvas 2D context is unavailable.");
  }
  return { canvas, ctx };
}

async function blobToBuffer(blob) {
  const arrayBuffer = await blob.arrayBuffer();
  return Buffer.from(arrayBuffer);
}

function pageFileName(pageIndex) {
  return `page-${pageIndex + 1}.png`;
}

async function renderDocumentPages({
  model,
  outputDir,
  scale,
  drawDocumentToCtx,
  preloadDocumentImageBitmaps,
}) {
  await fs.mkdir(outputDir, { recursive: true });

  const bootstrap = createCanvas(1, 1);
  const initialDraw = drawDocumentToCtx(model, bootstrap.ctx, { pageIndex: 0 });
  const pages = initialDraw.pages;
  const imageBitmaps = await preloadDocumentImageBitmaps(model, pages);

  if (pages.length === 0) {
    const placeholder = createCanvas(
      Math.max(1, Math.ceil(800 * scale)),
      Math.max(1, Math.ceil(1000 * scale))
    );
    const outputPath = path.join(outputDir, pageFileName(0));
    const buffer = await blobToBuffer(
      await placeholder.canvas.convertToBlob({ type: "image/png" })
    );
    await fs.writeFile(outputPath, buffer);
    return { pagePaths: [outputPath], pageCount: 0 };
  }

  const pagePaths = [];
  for (let pageIndex = 0; pageIndex < pages.length; pageIndex += 1) {
    const page = pages[pageIndex];
    const width = Math.max(1, Math.ceil(page.widthPx * scale));
    const height = Math.max(1, Math.ceil(page.heightPx * scale));
    const { canvas, ctx } = createCanvas(width, height);
    ctx.save();
    ctx.scale(scale, scale);
    drawDocumentToCtx(model, ctx, {
      pageIndex,
      clear: true,
      imageBitmaps,
    });
    ctx.restore();

    const outputPath = path.join(outputDir, pageFileName(pageIndex));
    const buffer = await blobToBuffer(
      await canvas.convertToBlob({ type: "image/png" })
    );
    await fs.writeFile(outputPath, buffer);
    pagePaths.push(outputPath);
  }

  return { pagePaths, pageCount: pages.length };
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const artifactTool = await importArtifactTool(args.artifactToolPackage);
  const model = await artifactTool.DocumentFile.importDocx(
    await artifactTool.FileBlob.load(args.inputPath)
  );
  const result = await renderDocumentPages({
    model,
    outputDir: args.outputDir,
    scale: args.scale,
    drawDocumentToCtx: artifactTool.drawDocumentToCtx,
    preloadDocumentImageBitmaps: artifactTool.preloadDocumentImageBitmaps,
  });

  if (args.metadataJson) {
    await fs.mkdir(path.dirname(args.metadataJson), { recursive: true });
    await fs.writeFile(
      args.metadataJson,
      JSON.stringify(
        {
          renderer: "artifact-tool",
          inputPath: args.inputPath,
          outputDir: args.outputDir,
          artifactToolPackage: args.artifactToolPackage,
          scale: args.scale,
          pageCount: result.pageCount,
          pagePaths: result.pagePaths,
        },
        null,
        2
      ),
      "utf8"
    );
  }

  if (args.verbose) {
    console.log(
      `artifact-tool rendered ${result.pagePaths.length} page image(s).`
    );
  }
  console.log(`Pages rendered to ${args.outputDir}`);
}

main().catch((error) => {
  console.error(
    error instanceof Error ? error.stack || error.message : String(error)
  );
  process.exit(1);
});
