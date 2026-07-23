from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Protocol

from PIL import Image, ImageChops
from resvg_py import svg_to_bytes

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
PROJECT_FONT_FAMILY = "Poppins"
PROJECT_FONT_FILENAMES = ("Poppins-Regular.ttf", "Poppins-Bold.ttf")


@dataclass(frozen=True)
class SvgRasterization:
    """Verified metadata for one deterministic SVG-to-PNG conversion."""

    svg_path: str
    png_path: str
    width: int
    height: int
    byte_size: int
    sha256: str
    non_background_pixel_count: int


class SvgRasterizer(Protocol):
    """Port for deterministic SVG rasterization."""

    def render(self, svg_path: Path, png_path: Path) -> SvgRasterization:
        """Render and verify one SVG file."""


@dataclass(frozen=True)
class ResvgSvgRasterizer:
    """Deterministic resvg renderer with explicit project-owned fonts."""

    font_files: tuple[Path, ...]
    font_family: str = PROJECT_FONT_FAMILY
    background: str = "#ffffff"

    @classmethod
    def with_project_fonts(cls) -> ResvgSvgRasterizer:
        font_dir = Path(str(files("ts_rag_agent").joinpath("assets", "fonts", "poppins")))
        return cls(tuple(font_dir / filename for filename in PROJECT_FONT_FILENAMES))

    def render(self, svg_path: Path, png_path: Path) -> SvgRasterization:
        resolved_svg_path = svg_path.resolve(strict=True)
        resolved_font_files = tuple(path.resolve(strict=True) for path in self.font_files)
        svg = resolved_svg_path.read_text(encoding="utf-8")

        png = svg_to_bytes(
            svg_string=svg,
            background=self.background,
            skip_system_fonts=True,
            font_files=[str(path) for path in resolved_font_files],
            font_family=self.font_family,
            sans_serif_family=self.font_family,
            shape_rendering="geometric_precision",
            text_rendering="geometric_precision",
            image_rendering="optimize_quality",
        )
        if not png.startswith(PNG_SIGNATURE):
            raise ValueError(f"resvg did not return a PNG for {resolved_svg_path}")

        resolved_png_path = png_path.resolve()
        resolved_png_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_png_path.write_bytes(png)
        width, height, non_background_pixel_count = _inspect_png(
            resolved_png_path,
            background=self.background,
        )
        if non_background_pixel_count == 0:
            raise ValueError(f"rendered PNG is blank: {resolved_png_path}")

        return SvgRasterization(
            svg_path=str(resolved_svg_path),
            png_path=str(resolved_png_path),
            width=width,
            height=height,
            byte_size=len(png),
            sha256=hashlib.sha256(png).hexdigest(),
            non_background_pixel_count=non_background_pixel_count,
        )


def rasterize_svg_files(
    svg_paths: Iterable[Path],
    output_dir: Path,
    rasterizer: SvgRasterizer,
) -> tuple[SvgRasterization, ...]:
    """Rasterize SVG paths into a dedicated PNG directory."""

    return tuple(
        rasterizer.render(svg_path, output_dir / f"{svg_path.stem}.png") for svg_path in svg_paths
    )


def _inspect_png(png_path: Path, background: str) -> tuple[int, int, int]:
    with Image.open(png_path) as image:
        image.load()
        rgb_image = image.convert("RGB")
        background_rgb = _hex_to_rgb(background)
        background_image = Image.new("RGB", rgb_image.size, background_rgb)
        difference = ImageChops.difference(rgb_image, background_image)
        grayscale_histogram = difference.convert("L").histogram()
        non_background_pixel_count = rgb_image.width * rgb_image.height - grayscale_histogram[0]
        return rgb_image.width, rgb_image.height, non_background_pixel_count


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    normalized = color.removeprefix("#")
    if len(normalized) != 6:
        raise ValueError(f"expected a six-digit hex color, got {color!r}")
    return tuple(int(normalized[index : index + 2], 16) for index in (0, 2, 4))
