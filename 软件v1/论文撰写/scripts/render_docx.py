from __future__ import annotations

import argparse
import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from os import makedirs, replace
from os.path import abspath, basename, dirname, exists, expanduser, join, splitext
from typing import Sequence, cast
from zipfile import ZipFile

TWIPS_PER_INCH: int = 1440
BASE_RENDER_DPI = 96
DEFAULT_RUNTIME_PACKAGE_GLOBS = [
    "~/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool",
    "~/.cache/codex-runtimes/codex-primary-runtime-*/dependencies/node/node_modules/@oai/artifact-tool",
    "~/.codex/environments/*/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool",
]


def find_soffice() -> str | None:
    """Return a LibreOffice binary path if one is already available."""

    env_bin = os.environ.get("LIBREOFFICE_BIN")
    if env_bin and exists(expanduser(env_bin)):
        return abspath(expanduser(env_bin))

    for name in ("soffice", "libreoffice"):
        resolved = shutil.which(name)
        if resolved:
            return resolved

    mac_candidates = [
        "/Applications/LibreOffice.app/Contents/MacOS/soffice",
        "/Applications/LibreOffice.app/Contents/MacOS/libreoffice",
    ]
    for candidate in mac_candidates:
        if exists(candidate):
            return candidate

    return None


def _node_executable_name() -> str:
    return "node.exe" if os.name == "nt" else "node"


def _is_artifact_tool_package_dir(path: str) -> bool:
    return exists(join(path, "package.json")) and exists(join(path, "dist", "artifact_tool.mjs"))


def _candidate_artifact_tool_dirs(start_dir: str) -> list[str]:
    current = abspath(start_dir)
    candidates: list[str] = []
    while True:
        candidates.append(join(current, "node_modules", "@oai", "artifact-tool"))
        parent = dirname(current)
        if parent == current:
            break
        current = parent
    return candidates


def find_artifact_tool_package_dir(input_path: str) -> str:
    candidates: list[str] = []

    env_package_dir = os.environ.get("ARTIFACT_TOOL_PACKAGE_DIR")
    if env_package_dir:
        candidates.append(env_package_dir)

    env_node_modules = os.environ.get("CODEX_NODE_MODULES") or os.environ.get("NODE_MODULES_PATH")
    if env_node_modules:
        candidates.append(join(env_node_modules, "@oai", "artifact-tool"))

    for pattern in DEFAULT_RUNTIME_PACKAGE_GLOBS:
        candidates.extend(glob.glob(expanduser(pattern)))

    candidates.extend(_candidate_artifact_tool_dirs(os.getcwd()))
    candidates.extend(_candidate_artifact_tool_dirs(dirname(input_path)))
    candidates.extend(_candidate_artifact_tool_dirs(dirname(abspath(__file__))))

    seen: set[str] = set()
    for candidate in candidates:
        resolved = abspath(expanduser(candidate))
        if resolved in seen:
            continue
        seen.add(resolved)
        if _is_artifact_tool_package_dir(resolved):
            return resolved

    raise RuntimeError(
        "Could not find the bundled @oai/artifact-tool package. "
        "Install or refresh the Codex primary runtime bundle, or set "
        "ARTIFACT_TOOL_PACKAGE_DIR to the package directory."
    )


def find_node_executable(artifact_tool_package_dir: str) -> str:
    env_node = os.environ.get("CODEX_NODE_PATH") or os.environ.get("NODE_BIN")
    if env_node and exists(expanduser(env_node)):
        return abspath(expanduser(env_node))

    package_scope_dir = dirname(artifact_tool_package_dir)
    node_modules_dir = dirname(package_scope_dir)
    runtime_node_root = dirname(node_modules_dir)
    derived_node = join(runtime_node_root, "bin", _node_executable_name())
    if exists(derived_node):
        return derived_node

    default_runtime_nodes = [
        join(
            expanduser("~"),
            ".cache",
            "codex-runtimes",
            "codex-primary-runtime",
            "dependencies",
            "node",
            "bin",
            _node_executable_name(),
        )
    ]
    default_runtime_nodes.extend(
        glob.glob(
            expanduser(
                join(
                    "~",
                    ".codex",
                    "environments",
                    "*",
                    "codex-primary-runtime",
                    "dependencies",
                    "node",
                    "bin",
                    _node_executable_name(),
                )
            )
        )
    )

    for candidate in default_runtime_nodes:
        resolved = abspath(expanduser(candidate))
        if exists(resolved):
            return resolved

    resolved = shutil.which("node")
    if resolved:
        return resolved

    raise RuntimeError(
        "Could not find a Node.js executable for the bundled @oai/artifact-tool "
        "package. Install or refresh the Codex primary runtime bundle, or set "
        "CODEX_NODE_PATH."
    )


def calc_dpi_via_ooxml_docx(input_path: str, max_w_px: int, max_h_px: int) -> int:
    """Calculate DPI from OOXML `word/document.xml` page size (w:pgSz in twips).

    DOCX stores page dimensions in section properties as twips (1/1440 inch).
    We read the first encountered section's page size and compute an isotropic DPI
    that fits within the target max pixel dimensions.
    """

    with ZipFile(input_path, "r") as zf:
        xml = zf.read("word/document.xml")
    root = ET.fromstring(xml)
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}

    # Common placements: w:body/w:sectPr or w:body/w:p/w:pPr/w:sectPr
    sect_pr = root.find(".//w:sectPr", ns)
    if sect_pr is None:
        raise RuntimeError("Section properties not found in document.xml")
    pg_sz = sect_pr.find("w:pgSz", ns)
    if pg_sz is None:
        raise RuntimeError("Page size not found in section properties")

    # Values are in twips
    w_twips_str = pg_sz.get(
        "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}w"
    ) or pg_sz.get("w")
    h_twips_str = pg_sz.get(
        "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}h"
    ) or pg_sz.get("h")

    if not w_twips_str or not h_twips_str:
        raise RuntimeError("Page size attributes missing in pgSz")

    width_in = int(w_twips_str) / TWIPS_PER_INCH
    height_in = int(h_twips_str) / TWIPS_PER_INCH
    if width_in <= 0 or height_in <= 0:
        raise RuntimeError("Invalid page size values in document.xml")
    # Best-effort warning for multi-section docs.
    try:
        dims = set()
        for s in root.findall(".//w:sectPr", ns):
            ps = s.find("w:pgSz", ns)
            if ps is None:
                continue
            wv = ps.get(f"{{{ns['w']}}}w") or ps.get("w")
            hv = ps.get(f"{{{ns['w']}}}h") or ps.get("h")
            if wv and hv:
                dims.add((wv, hv))
        if len(dims) > 1:
            print(
                "[render_docx] NOTE: multiple section page sizes/orientations detected; "
                "DPI is computed from the first section only. Consider --dpi to override."
            )
    except Exception:
        pass

    return round(min(max_w_px / width_in, max_h_px / height_in))


def _build_lo_env(user_profile: str) -> dict:
    """Build a container-safe env for LibreOffice.

    LibreOffice sometimes tries to write configuration/cache under a default,
    non-writable HOME or XDG dirs. We force a writable HOME under the
    per-run user_profile.
    """

    env = os.environ.copy()
    env["HOME"] = user_profile
    env.setdefault("XDG_CONFIG_HOME", join(user_profile, "xdg_config"))
    env.setdefault("XDG_CACHE_HOME", join(user_profile, "xdg_cache"))
    os.makedirs(env["XDG_CONFIG_HOME"], exist_ok=True)
    os.makedirs(env["XDG_CACHE_HOME"], exist_ok=True)
    return env


def _run_cmd(cmd: list[str], env: dict, verbose: bool) -> subprocess.CompletedProcess:
    """Run a command and capture output for debuggability."""

    proc = subprocess.run(
        cmd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )

    if verbose:
        print("[render_docx] $ " + " ".join(cmd))
        if proc.stdout:
            print(proc.stdout)
        if proc.stderr:
            print(proc.stderr)

    return proc


def convert_to_pdf(
    doc_path: str,
    user_profile: str,
    convert_tmp_dir: str,
    stem: str,
    verbose: bool,
    soffice_bin: str,
) -> tuple[str, str]:
    """Convert input into a PDF.

    Returns (pdf_path, debug_log). If conversion fails, pdf_path == "" and
    debug_log contains captured stdout/stderr for diagnosis.
    """

    env = _build_lo_env(user_profile)
    logs: list[str] = []

    def _log_result(label: str, cmd: list[str], proc: subprocess.CompletedProcess) -> None:
        logs.append(f"--- {label} ---")
        logs.append("CMD: " + " ".join(cmd))
        logs.append(f"EXIT: {proc.returncode}")
        if proc.stdout:
            logs.append("STDOUT:\n" + proc.stdout.strip())
        if proc.stderr:
            logs.append("STDERR:\n" + proc.stderr.strip())

    def _nonempty(path: str) -> bool:
        try:
            return exists(path) and os.path.getsize(path) > 0
        except Exception:
            return exists(path)

    pdf_path = join(convert_tmp_dir, f"{stem}.pdf")

    # Try direct DOC(X) -> PDF
    cmd_pdf = [
        soffice_bin,
        "-env:UserInstallation=file://" + user_profile,
        "--invisible",
        "--headless",
        "--norestore",
        "--convert-to",
        "pdf",
        "--outdir",
        convert_tmp_dir,
        doc_path,
    ]
    proc = _run_cmd(cmd_pdf, env=env, verbose=verbose)
    _log_result("DOCX->PDF", cmd_pdf, proc)

    # LibreOffice sometimes prints scary-looking stderr even on successful output.
    # Prefer file existence + non-empty size over stderr content.
    if _nonempty(pdf_path):
        return pdf_path, "\n".join(logs)

    # Sometimes LO names the output slightly differently; fall back to any PDF it emitted.
    pdf_glob = glob.glob(join(convert_tmp_dir, "*.pdf"))
    if pdf_glob:
        cand = min(pdf_glob)
        if _nonempty(cand):
            return cand, "\n".join(logs)

    # Fallback: DOCX -> ODT, then ODT -> PDF
    cmd_odt = [
        soffice_bin,
        "-env:UserInstallation=file://" + user_profile,
        "--invisible",
        "--headless",
        "--norestore",
        "--convert-to",
        "odt",
        "--outdir",
        convert_tmp_dir,
        doc_path,
    ]
    proc = _run_cmd(cmd_odt, env=env, verbose=verbose)
    _log_result("DOCX->ODT", cmd_odt, proc)

    odt_path = join(convert_tmp_dir, f"{stem}.odt")

    if exists(odt_path):
        cmd_odt_pdf = [
            soffice_bin,
            "-env:UserInstallation=file://" + user_profile,
            "--invisible",
            "--headless",
            "--norestore",
            "--convert-to",
            "pdf",
            "--outdir",
            convert_tmp_dir,
            odt_path,
        ]
        proc = _run_cmd(cmd_odt_pdf, env=env, verbose=verbose)
        _log_result("ODT->PDF", cmd_odt_pdf, proc)
        if _nonempty(pdf_path):
            return pdf_path, "\n".join(logs)
        pdf_glob = glob.glob(join(convert_tmp_dir, "*.pdf"))
        if pdf_glob:
            cand = min(pdf_glob)
            if _nonempty(cand):
                return cand, "\n".join(logs)

    return "", "\n".join(logs)


def calc_dpi_via_pdf(
    input_path: str,
    max_w_px: int,
    max_h_px: int,
    verbose: bool,
    soffice_bin: str,
) -> int:
    """Convert input to PDF and compute DPI from its page size."""

    from pdf2image import pdfinfo_from_path

    with tempfile.TemporaryDirectory(prefix="soffice_profile_") as user_profile:
        with tempfile.TemporaryDirectory(prefix="soffice_convert_") as convert_tmp_dir:
            stem = splitext(basename(input_path))[0]
            pdf_path, debug = convert_to_pdf(
                input_path,
                user_profile,
                convert_tmp_dir,
                stem,
                verbose=verbose,
                soffice_bin=soffice_bin,
            )
            if not (pdf_path and exists(pdf_path)):
                raise RuntimeError("Failed to convert input to PDF for DPI computation.\n" + debug)

            info = pdfinfo_from_path(pdf_path)
            size_val = info.get("Page size")
            if not size_val:
                for k, v in info.items():
                    if isinstance(v, str) and "size" in k.lower() and "pts" in v:
                        size_val = v
                        break
            if not isinstance(size_val, str):
                raise RuntimeError("Failed to read PDF page size for DPI computation.")

            # Example formats:
            # - "612 x 792 pts"
            # - "612.0 x 792.0 pts (letter)"
            m = re.search(r"(\d+(?:\.\d+)?)\s*x\s*(\d+(?:\.\d+)?)\s*pts", size_val)
            if not m:
                raise RuntimeError("Unrecognized PDF page size format.")
            width_pts = float(m.group(1))
            height_pts = float(m.group(2))
            width_in = width_pts / 72.0
            height_in = height_pts / 72.0
            if width_in <= 0 or height_in <= 0:
                raise RuntimeError("Invalid PDF page size values.")
            return round(min(max_w_px / width_in, max_h_px / height_in))


def rasterize(
    doc_path: str,
    out_dir: str,
    dpi: int,
    verbose: bool,
    emit_pdf: bool,
    soffice_bin: str,
) -> Sequence[str]:
    """Rasterize DOCX-like input to images placed in out_dir and return their paths.

    Images are named as page-<N>.png with pages starting at 1.
    """

    from pdf2image import convert_from_path

    makedirs(out_dir, exist_ok=True)
    doc_path = abspath(doc_path)
    stem = splitext(basename(doc_path))[0]

    with tempfile.TemporaryDirectory(prefix="soffice_profile_") as user_profile:
        with tempfile.TemporaryDirectory(prefix="soffice_convert_") as convert_tmp_dir:
            pdf_path, debug = convert_to_pdf(
                doc_path,
                user_profile,
                convert_tmp_dir,
                stem,
                verbose=verbose,
                soffice_bin=soffice_bin,
            )

            if not pdf_path or not exists(pdf_path):
                raise RuntimeError(
                    "Failed to produce PDF for rasterization (direct and ODT fallback).\n" + debug
                )

            if emit_pdf:
                # Optional: persist the intermediate PDF for debugging / archival.
                # This is OFF by default so agents don't confuse intermediates with deliverables.
                dst_pdf = join(out_dir, f"{stem}.pdf")
                tmp_pdf = dst_pdf + ".tmp"
                shutil.copy2(pdf_path, tmp_pdf)
                replace(tmp_pdf, dst_pdf)

            paths_raw = cast(
                list[str],
                convert_from_path(
                    pdf_path,
                    dpi=dpi,
                    fmt="png",
                    thread_count=8,
                    output_folder=out_dir,
                    paths_only=True,
                    output_file="page",
                ),
            )

    # Rename convert_from_path's output format f'page{thread_id:04d}-{page_num:02d}.png' to 'page-<num>.png'
    pages: list[tuple[int, str]] = []
    for src_path in paths_raw:
        base = splitext(basename(src_path))[0]
        page_num_str = base.split("-")[-1]
        page_num = int(page_num_str)
        dst_path = join(out_dir, f"page-{page_num}.png")
        replace(src_path, dst_path)
        pages.append((page_num, dst_path))
    pages.sort(key=lambda t: t[0])
    return [path for _, path in pages]


def calc_artifact_tool_scale(
    input_path: str, max_w_px: int, max_h_px: int, dpi: int | None
) -> float:
    """Calculate an artifact-tool raster scale from the requested max image size."""

    if dpi is not None:
        return max(dpi / BASE_RENDER_DPI, 0.1)

    try:
        resolved_dpi = calc_dpi_via_ooxml_docx(input_path, max_w_px, max_h_px)
        return max(resolved_dpi / BASE_RENDER_DPI, 0.1)
    except Exception:
        return 2.0


def run_artifact_tool_renderer(
    input_path: str,
    out_dir: str,
    scale: float,
    verbose: bool,
) -> Sequence[str]:
    """Render a DOCX to page PNGs with the bundled @oai/artifact-tool package."""

    if verbose:
        print(f"[render_docx] using artifact-tool renderer at scale {scale:.3f}")

    script_path = join(dirname(abspath(__file__)), "scripts", "render_docx_artifact_tool.mjs")
    if not exists(script_path):
        raise RuntimeError(f"artifact-tool renderer script not found: {script_path}")

    artifact_tool_package_dir = find_artifact_tool_package_dir(input_path)
    node_bin = find_node_executable(artifact_tool_package_dir)

    cmd = [
        node_bin,
        script_path,
        input_path,
        "--output_dir",
        out_dir,
        "--scale",
        f"{scale:.6f}",
        "--artifact_tool_package",
        artifact_tool_package_dir,
    ]
    if verbose:
        cmd.append("--verbose")
        print("[render_docx] $ " + " ".join(cmd))

    proc = subprocess.run(
        cmd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=os.environ.copy(),
    )

    if verbose:
        if proc.stdout:
            print(proc.stdout)
        if proc.stderr:
            print(proc.stderr, file=sys.stderr)

    if proc.returncode != 0:
        raise RuntimeError(
            "artifact-tool render failed.\n"
            + "STDOUT:\n"
            + (proc.stdout or "").strip()
            + "\nSTDERR:\n"
            + (proc.stderr or "").strip()
        )

    rendered = sorted(
        glob.glob(join(out_dir, "page-*.png")),
        key=lambda path: (
            int(re.search(r"page-(\d+)\.png$", basename(path)).group(1))
            if re.search(r"page-(\d+)\.png$", basename(path))
            else 0
        ),
    )
    if not rendered:
        raise RuntimeError(f"artifact-tool render did not produce page PNGs in {out_dir}")
    return rendered


def main() -> None:
    parser = argparse.ArgumentParser(description="Render DOCX-like file to PNG page images.")
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to the input DOCX file (or compatible).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=(
            "Output directory for the rendered images. "
            "Defaults to a folder next to the input named after the input file (without extension)."
        ),
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1600,
        help=(
            "Approximate maximum width in pixels after isotropic scaling (default 1600). "
            "The actual value may exceed slightly."
        ),
    )
    parser.add_argument(
        "--height",
        type=int,
        default=2000,
        help=(
            "Approximate maximum height in pixels after isotropic scaling (default 2000). "
            "The actual value may exceed slightly."
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=None,
        help="Override computed DPI. If provided, skips DOCX/PDF-based DPI calculation.",
    )
    parser.add_argument(
        "--emit_pdf",
        action="store_true",
        help=(
            "LibreOffice renderer only: also write an intermediate PDF to --output_dir as <input_stem>.pdf. "
            "Default is PNG-only to avoid confusing intermediates with deliverables."
        ),
    )
    parser.add_argument(
        "--renderer",
        choices=("artifact-tool", "granola", "libreoffice", "auto"),
        default="artifact-tool",
        help=(
            "Renderer backend. documents defaults to artifact-tool. "
            "`granola` is kept as a deprecated alias for backward compatibility. "
            "Use libreoffice only for an explicit LibreOffice cross-check."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print renderer commands and captured stdout/stderr (useful for debugging).",
    )

    args = parser.parse_args()

    input_path = abspath(expanduser(args.input_path))
    out_dir = abspath(expanduser(args.output_dir)) if args.output_dir else splitext(input_path)[0]
    renderer = args.renderer
    soffice_bin = find_soffice()
    if renderer == "auto":
        renderer = "libreoffice" if soffice_bin else "artifact-tool"
    elif renderer == "granola":
        renderer = "artifact-tool"

    if renderer == "artifact-tool":
        if args.emit_pdf:
            raise RuntimeError(
                "--emit_pdf is only supported with --renderer libreoffice; "
                "artifact-tool emits PNG page images only."
            )
        scale = calc_artifact_tool_scale(input_path, args.width, args.height, args.dpi)
        run_artifact_tool_renderer(input_path, out_dir, scale=scale, verbose=args.verbose)
        print("Pages rendered to " + out_dir)
        return

    if not soffice_bin:
        raise RuntimeError(
            "LibreOffice renderer requested but no soffice/libreoffice binary was found. "
            "Use --renderer artifact-tool or set LIBREOFFICE_BIN."
        )

    if args.dpi is not None:
        dpi = int(args.dpi)
    else:
        try:
            if input_path.lower().endswith((".docx", ".docm", ".dotx", ".dotm")):
                dpi = calc_dpi_via_ooxml_docx(input_path, args.width, args.height)
            else:
                raise RuntimeError("Skip OOXML DPI; not a DOCX container")
        except Exception:
            dpi = calc_dpi_via_pdf(
                input_path,
                args.width,
                args.height,
                verbose=args.verbose,
                soffice_bin=soffice_bin,
            )

    rasterize(
        input_path,
        out_dir,
        dpi,
        verbose=args.verbose,
        emit_pdf=args.emit_pdf,
        soffice_bin=soffice_bin,
    )
    print("Pages rendered to " + out_dir)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        raise SystemExit(f"ERROR: {exc}") from None
