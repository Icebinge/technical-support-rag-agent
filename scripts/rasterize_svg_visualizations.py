from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

import resvg_py
import typer

from ts_rag_agent.application.svg_rasterizer import ResvgSvgRasterizer, rasterize_svg_files

app = typer.Typer(help="Rasterize project SVG charts with fixed resvg and bundled fonts.")


@app.command()
def main(
    input_dir: Annotated[Path, typer.Option("--input-dir")],
    output_dir: Annotated[Path, typer.Option("--output-dir")],
    manifest: Annotated[Path, typer.Option("--manifest")],
) -> None:
    svg_paths = tuple(sorted(input_dir.glob("*.svg")))
    if not svg_paths:
        raise typer.BadParameter(f"no SVG files found in {input_dir}", param_hint="--input-dir")

    rasterizer = ResvgSvgRasterizer.with_project_fonts()
    rendered = rasterize_svg_files(svg_paths, output_dir, rasterizer)
    payload = {
        "renderer": {
            "name": "resvg_py",
            "version": resvg_py.__version__,
            "skip_system_fonts": True,
            "font_family": rasterizer.font_family,
            "font_files": [
                {
                    "path": str(font_path.resolve()),
                    "sha256": _sha256(font_path),
                }
                for font_path in rasterizer.font_files
            ],
            "background": rasterizer.background,
            "fallback_enabled": False,
        },
        "input_directory": str(input_dir.resolve()),
        "output_directory": str(output_dir.resolve()),
        "rendered_count": len(rendered),
        "rendered": [asdict(item) for item in rendered],
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    typer.echo(json.dumps(payload, ensure_ascii=True, indent=2))
    typer.echo(f"Saved deterministic rasterization manifest: {manifest}")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    app()
