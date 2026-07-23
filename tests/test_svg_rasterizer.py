from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from ts_rag_agent.application.svg_rasterizer import (
    ResvgSvgRasterizer,
    rasterize_svg_files,
)


def _write_svg(path: Path, title: str = "Deterministic chart") -> Path:
    path.write_text(
        "\n".join(
            [
                '<svg xmlns="http://www.w3.org/2000/svg" width="320" height="120">',
                '<rect width="100%" height="100%" fill="#ffffff"/>',
                "<style>text{font-family:'Poppins';font-size:18px;font-weight:700}</style>",
                f'<text x="20" y="36">{title}</text>',
                '<rect x="20" y="60" width="180" height="24" fill="#2563eb"/>',
                "</svg>",
            ]
        ),
        encoding="utf-8",
    )
    return path


def test_resvg_rasterizer_is_deterministic_and_nonblank(tmp_path: Path) -> None:
    svg_path = _write_svg(tmp_path / "chart.svg")
    rasterizer = ResvgSvgRasterizer.with_project_fonts()

    first = rasterizer.render(svg_path, tmp_path / "first.png")
    second = rasterizer.render(svg_path, tmp_path / "second.png")

    assert first.sha256 == second.sha256
    assert (first.width, first.height) == (320, 120)
    assert first.non_background_pixel_count > 0
    assert first.byte_size > 0
    with Image.open(first.png_path) as image:
        assert image.format == "PNG"


def test_rasterize_svg_files_preserves_input_order(tmp_path: Path) -> None:
    svg_paths = [
        _write_svg(tmp_path / "first.svg", title="First"),
        _write_svg(tmp_path / "second.svg", title="Second"),
    ]

    rendered = rasterize_svg_files(
        svg_paths,
        tmp_path / "rendered",
        ResvgSvgRasterizer.with_project_fonts(),
    )

    assert [Path(item.png_path).name for item in rendered] == ["first.png", "second.png"]
    assert all(Path(item.png_path).exists() for item in rendered)


def test_resvg_rasterizer_rejects_missing_font(tmp_path: Path) -> None:
    svg_path = _write_svg(tmp_path / "chart.svg")
    rasterizer = ResvgSvgRasterizer(font_files=(tmp_path / "missing.ttf",))

    with pytest.raises(FileNotFoundError):
        rasterizer.render(svg_path, tmp_path / "chart.png")
