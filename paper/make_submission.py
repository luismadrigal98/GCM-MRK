#!/usr/bin/env python3
"""Flatten the manuscript into a self-contained bundle for submission.

The working manuscript is split across several files and reads its numbers,
tables and figures from ``results/`` and ``figures/`` so that re-running an
analysis updates the paper.  A submission system compiles whatever files it is
given in a single flat directory, with no guarantee that subdirectories or a
multi-file ``\\input`` chain survive upload.

This script resolves the whole chain once and writes ``submission/``: one master
``manuscript.tex`` with every macro definition, generated value, table body and
bibliography entry inlined, the figures copied out with the names their figure
numbers imply, and the class and ``.bib`` alongside.  The result compiles on its
own in an empty directory.

Run it after the analyses and after a full LaTeX build (the ``.bbl`` is reused):

    python3 paper/make_submission.py

It fails rather than falling back if a generated file is missing, so the bundle
can never silently carry a placeholder value.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parent
OUT = PAPER / "submission"

WRAPPER = "gcmrk_paper_peerj.tex"
BBL = "gcmrk_paper_peerj.bbl"
MASTER = "manuscript.tex"


class MissingResult(RuntimeError):
    pass


def read(name: str) -> str:
    path = PAPER / name
    if not path.exists():
        raise MissingResult(
            f"{name} is missing. Run the analyses (and a full LaTeX build for "
            f"the .bbl) before building the submission bundle."
        )
    return path.read_text()


def inline_macro_files(text: str) -> tuple[str, int]:
    """Replace ``\\InputIfFileExists{results/X}{}{}`` with the contents of X."""
    pattern = re.compile(r"\\InputIfFileExists\{(results/[^}]+)\}\{\}\{\}")
    count = 0

    def sub(match: re.Match) -> str:
        nonlocal count
        count += 1
        name = match.group(1)
        return f"% --- inlined from {name} ---\n{read(name).rstrip()}"

    return pattern.sub(sub, text), count


def _matching_brace(text: str, open_index: int) -> int:
    """Index of the ``}`` closing the ``{`` at *open_index*, ignoring escapes."""
    depth = 0
    i = open_index
    while i < len(text):
        char = text[i]
        if char == "\\":
            i += 2
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    raise RuntimeError(f"unbalanced braces from offset {open_index}")


def inline_tables(text: str) -> tuple[str, int]:
    """Replace ``\\IfFileExists{results/T}{\\input{results/T}}{fallback}``.

    The fallback is a multi-line ``tabular``, so the closing brace is found by
    brace matching rather than by a regular expression.
    """
    pattern = re.compile(
        r"\\IfFileExists\{(results/[^}]+)\}\{\\input\{results/[^}]+\}\}\{"
    )
    count = 0
    while True:
        match = pattern.search(text)
        if match is None:
            return text, count
        end = _matching_brace(text, match.end() - 1)
        name = match.group(1)
        body = read(name).rstrip()
        text = (
            text[: match.start()]
            + f"% --- inlined from {name} ---\n{body}"
            + text[end + 1 :]
        )
        count += 1


def renumber_figures(text: str) -> list[tuple[str, str]]:
    """Rename figure files to ``figureN.pdf`` in order of first appearance."""
    order: list[str] = []
    for name in re.findall(r"\\previewfig\{([^}]+)\}", text):
        if name not in order:
            order.append(name)
    return [(src, f"figure{i}.pdf") for i, src in enumerate(order, start=1)]


def build() -> None:
    text = read(WRAPPER)

    # 1. Shared body files.
    for part in ("gcmrk_macros", "gcmrk_abstract", "gcmrk_body"):
        marker = "\\input{%s}" % part
        if marker not in text:
            raise RuntimeError(f"{marker} not found in {WRAPPER}")
        text = text.replace(marker, f"% --- inlined from {part}.tex ---\n"
                                    + read(part + ".tex").rstrip())

    # 2. Generated macro files and table bodies.
    text, n_macro_files = inline_macro_files(text)
    text, n_tables = inline_tables(text)

    # Ignore the provenance comments this script itself inserts.
    uncommented = "\n".join(re.sub(r"(?<!\\\\)%.*$", "", line)
                            for line in text.splitlines())
    leftover = re.findall(r"results/[^}\s]+", uncommented)
    if leftover:
        raise RuntimeError(f"unresolved results references: {sorted(set(leftover))}")

    # 3. Figures live beside the master file, under their figure numbers.
    text = text.replace("\\graphicspath{{figures/}}\n", "")
    mapping = renumber_figures(text)
    for src, dst in mapping:
        text = text.replace("\\previewfig{%s}" % src, "\\previewfig{%s}" % dst)

    # 4. Bibliography: the resolved .bbl, so the compile needs no BibTeX pass.
    bbl = read(BBL).rstrip()
    if "\\bibliography{references}" not in text:
        raise RuntimeError("\\bibliography{references} not found")
    text = text.replace(
        "\\bibliography{references}",
        f"% --- inlined from {BBL} (also submitted as references.bib) ---\n{bbl}",
    )

    # 5. Write the bundle in place.  Deleting the directory first would unlink
    # files a PDF viewer has open, and on a FUSE mount (this NTFS disk) each
    # one then lingers as a .fuse_hidden* file until the viewer closes.
    # Overwriting keeps the same file, and the viewer just reloads it.
    for src, _ in mapping:
        if not (PAPER / "figures" / src).exists():
            raise MissingResult(f"figures/{src} is missing; re-run its script.")
    OUT.mkdir(exist_ok=True)
    expected = ({MASTER, Path(MASTER).with_suffix(".pdf").name, "wlpeerj.cls",
                 "references.bib", "MANIFEST.txt"}
                | {dst for _, dst in mapping})
    for stale in OUT.iterdir():
        if (stale.is_file() and stale.name not in expected
                and not stale.name.startswith(".fuse_hidden")):
            stale.unlink()
    (OUT / MASTER).write_text(text)
    shutil.copy2(PAPER / "wlpeerj.cls", OUT / "wlpeerj.cls")
    shutil.copy2(PAPER / "references.bib", OUT / "references.bib")
    for src, dst in mapping:
        shutil.copy2(PAPER / "figures" / src, OUT / dst)

    manifest = ["Figure files, in order of first citation in the text:", ""]
    manifest += [f"  {dst:<15} {src}" for src, dst in mapping]
    manifest += [
        "",
        "Also in this directory:",
        f"  {MASTER:<15} the complete manuscript in one file",
        "  references.bib   bibliography source (the resolved entries are already",
        "                  inlined in the manuscript, so no BibTeX pass is needed)",
        "  wlpeerj.cls      PeerJ document class",
        "",
        "Note for submission staff (tables and bibliography are inside the TEX):",
        "",
        "  All tables are LaTeX tabular environments inside the manuscript file",
        "  rather than separate DOCX files, and the bibliography is included as a",
        "  resolved thebibliography environment (references.bib is supplied as",
        "  well).  Every number in the tables and text is written directly by the",
        "  analysis scripts that produced it, so re-keying them into DOCX would",
        "  break that provenance.  The file compiles with pdflatex alone, with no",
        "  BibTeX pass and no additional files beyond those listed above.",
        "",
        "Built by paper/make_submission.py -- edit the sources in paper/, not these.",
        "",
    ]
    (OUT / "MANIFEST.txt").write_text("\n".join(manifest))

    # 6. Compile in place, so the folder holds exactly what gets uploaded.
    if shutil.which("pdflatex") is None:
        print("warning: pdflatex not found; the bundle was not compiled.")
    else:
        for _ in range(3):
            subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", MASTER],
                cwd=OUT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        log = (OUT / "manuscript.log").read_text(errors="replace")
        errors = [ln for ln in log.splitlines() if ln.startswith("!")]
        if errors:
            raise RuntimeError("the bundle did not compile cleanly:\n"
                               + "\n".join(errors[:5]))
        for junk in ("aux", "log", "out", "synctex.gz"):
            (OUT / f"manuscript.{junk}").unlink(missing_ok=True)

    n_figs = len(mapping)
    n_refs = bbl.count("\\bibitem")
    print(f"submission/{MASTER}: {len(text.splitlines())} lines")
    print(f"  {n_macro_files} generated macro files inlined")
    print(f"  {n_tables} table bodies inlined")
    print(f"  {n_refs} bibliography entries inlined")
    print(f"  {n_figs} figures copied as figure1..figure{n_figs}.pdf")


if __name__ == "__main__":
    try:
        build()
    except (MissingResult, RuntimeError) as exc:
        sys.exit(f"error: {exc}")
