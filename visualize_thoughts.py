from __future__ import annotations

import argparse
import math
import struct
import zlib
from pathlib import Path
from typing import Sequence

import torch
from omegaconf import OmegaConf

from latent_pipeline import extract_reasoning_trace, initialize_hybrid_pipeline

DEFAULT_OUTPUT = Path("thought_trajectories.png")
DEFAULT_CORRECT_PROMPT = (
    "A student solves 12 * 7 and correctly concludes the answer is 84. "
    "Explain the reasoning."
)
DEFAULT_INCORRECT_PROMPT = (
    "A student solves 12 * 7 and incorrectly concludes the answer is 85. "
    "Explain the reasoning."
)
_CORRECT_COLOR = (32, 158, 119)
_INCORRECT_COLOR = (214, 57, 30)
_HANDOFF_COLOR = (240, 196, 25)
_BACKGROUND_COLOR = (247, 245, 240)
_AXIS_COLOR = (190, 186, 178)


def _load_cfg():
    return OmegaConf.load(Path(__file__).resolve().parent / "configs" / "main.yaml")


def _trajectory_points(trace: torch.Tensor) -> torch.Tensor:
    if trace.dim() != 4:
        raise ValueError("Expected continuous trajectory shape [steps, batch, seq, hidden]")
    return trace[:, 0, 0, :].to(torch.float32)


def _project_with_pca(trajectories: Sequence[torch.Tensor]) -> list[torch.Tensor]:
    flattened = [trajectory.to(torch.float32).cpu() for trajectory in trajectories]
    stacked = torch.cat(flattened, dim=0)
    centered = stacked - stacked.mean(dim=0, keepdim=True)

    if centered.shape[1] < 2:
        padded = torch.zeros(centered.shape[0], 2, dtype=centered.dtype)
        padded[:, : centered.shape[1]] = centered
        projected = padded
    else:
        _, _, vh = torch.linalg.svd(centered, full_matrices=False)
        projected = centered @ vh[:2].transpose(0, 1)

    splits = []
    cursor = 0
    for trajectory in flattened:
        length = trajectory.shape[0]
        splits.append(projected[cursor : cursor + length])
        cursor += length
    return splits


def _create_canvas(width: int, height: int, color: tuple[int, int, int]) -> bytearray:
    pixels = bytearray(width * height * 3)
    fill = bytes(color)
    for index in range(0, len(pixels), 3):
        pixels[index : index + 3] = fill
    return pixels


def _set_pixel(
    pixels: bytearray,
    width: int,
    height: int,
    x: int,
    y: int,
    color: tuple[int, int, int],
) -> None:
    if x < 0 or x >= width or y < 0 or y >= height:
        return
    index = (y * width + x) * 3
    pixels[index : index + 3] = bytes(color)


def _draw_line(
    pixels: bytearray,
    width: int,
    height: int,
    start: tuple[int, int],
    end: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    x0, y0 = start
    x1, y1 = end
    steps = max(abs(x1 - x0), abs(y1 - y0))
    if steps == 0:
        _draw_circle(pixels, width, height, x0, y0, thickness, color)
        return

    for step in range(steps + 1):
        t = step / steps
        x = round(x0 + (x1 - x0) * t)
        y = round(y0 + (y1 - y0) * t)
        _draw_circle(pixels, width, height, x, y, thickness, color)


def _draw_circle(
    pixels: bytearray,
    width: int,
    height: int,
    center_x: int,
    center_y: int,
    radius: int,
    color: tuple[int, int, int],
) -> None:
    for offset_y in range(-radius, radius + 1):
        for offset_x in range(-radius, radius + 1):
            if offset_x * offset_x + offset_y * offset_y <= radius * radius:
                _set_pixel(
                    pixels,
                    width,
                    height,
                    center_x + offset_x,
                    center_y + offset_y,
                    color,
                )


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    chunk = chunk_type + data
    crc = zlib.crc32(chunk) & 0xFFFFFFFF
    return struct.pack("!I", len(data)) + chunk + struct.pack("!I", crc)


def _write_png(path: Path, width: int, height: int, pixels: bytearray) -> None:
    raw_rows = bytearray()
    stride = width * 3
    for row in range(height):
        raw_rows.append(0)
        start = row * stride
        raw_rows.extend(pixels[start : start + stride])

    png = bytearray(b"\x89PNG\r\n\x1a\n")
    png.extend(_png_chunk(b"IHDR", struct.pack("!2I5B", width, height, 8, 2, 0, 0, 0)))
    png.extend(_png_chunk(b"IDAT", zlib.compress(bytes(raw_rows), level=9)))
    png.extend(_png_chunk(b"IEND", b""))
    path.write_bytes(bytes(png))


def _plot_trajectories(
    *,
    projected_correct: torch.Tensor,
    projected_incorrect: torch.Tensor,
    output_path: Path,
    width: int = 1200,
    height: int = 800,
    margin: int = 80,
) -> dict[str, float]:
    all_points = torch.cat([projected_correct, projected_incorrect], dim=0)
    min_x = float(all_points[:, 0].min().item())
    max_x = float(all_points[:, 0].max().item())
    min_y = float(all_points[:, 1].min().item())
    max_y = float(all_points[:, 1].max().item())

    if math.isclose(min_x, max_x):
        min_x -= 1.0
        max_x += 1.0
    if math.isclose(min_y, max_y):
        min_y -= 1.0
        max_y += 1.0

    pad_x = (max_x - min_x) * 0.12
    pad_y = (max_y - min_y) * 0.12
    min_x -= pad_x
    max_x += pad_x
    min_y -= pad_y
    max_y += pad_y

    pixels = _create_canvas(width, height, _BACKGROUND_COLOR)

    top = margin
    bottom = height - margin
    left = margin
    right = width - margin
    _draw_line(pixels, width, height, (left, top), (right, top), _AXIS_COLOR)
    _draw_line(pixels, width, height, (right, top), (right, bottom), _AXIS_COLOR)
    _draw_line(pixels, width, height, (right, bottom), (left, bottom), _AXIS_COLOR)
    _draw_line(pixels, width, height, (left, bottom), (left, top), _AXIS_COLOR)

    def to_pixel(point: torch.Tensor) -> tuple[int, int]:
        x = float(point[0].item())
        y = float(point[1].item())
        pixel_x = left + round((x - min_x) / (max_x - min_x) * (right - left))
        pixel_y = bottom - round((y - min_y) / (max_y - min_y) * (bottom - top))
        return pixel_x, pixel_y

    def draw_path(path: torch.Tensor, color: tuple[int, int, int]) -> None:
        pixel_points = [to_pixel(point) for point in path]
        for start, end in zip(pixel_points, pixel_points[1:]):
            _draw_line(pixels, width, height, start, end, color, thickness=2)
        for point in pixel_points[:-1]:
            _draw_circle(pixels, width, height, point[0], point[1], 3, color)
        handoff_x, handoff_y = pixel_points[-1]
        _draw_circle(pixels, width, height, handoff_x, handoff_y, 9, _HANDOFF_COLOR)
        _draw_circle(pixels, width, height, handoff_x, handoff_y, 5, color)

    draw_path(projected_correct, _CORRECT_COLOR)
    draw_path(projected_incorrect, _INCORRECT_COLOR)
    _write_png(output_path, width, height, pixels)

    return {
        "handoff_distance_2d": float(
            torch.linalg.vector_norm(projected_correct[-1] - projected_incorrect[-1]).item()
        ),
        "correct_path_extent": float(
            torch.linalg.vector_norm(projected_correct.max(dim=0).values - projected_correct.min(dim=0).values).item()
        ),
        "incorrect_path_extent": float(
            torch.linalg.vector_norm(
                projected_incorrect.max(dim=0).values - projected_incorrect.min(dim=0).values
            ).item()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Project latent reasoning trajectories to 2D and save a PNG plot."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--correct-prompt", default=DEFAULT_CORRECT_PROMPT)
    parser.add_argument("--incorrect-prompt", default=DEFAULT_INCORRECT_PROMPT)
    args = parser.parse_args()

    cfg = _load_cfg()
    initialize_hybrid_pipeline(cfg)

    correct_trace = extract_reasoning_trace(cfg, prompt=args.correct_prompt)
    incorrect_trace = extract_reasoning_trace(cfg, prompt=args.incorrect_prompt)

    projected_correct, projected_incorrect = _project_with_pca(
        [
            _trajectory_points(correct_trace["continuous_trajectory"]),
            _trajectory_points(incorrect_trace["continuous_trajectory"]),
        ]
    )
    stats = _plot_trajectories(
        projected_correct=projected_correct,
        projected_incorrect=projected_incorrect,
        output_path=args.output,
    )

    print(f"Wrote trajectory plot to {args.output}")
    print("Correct trajectory color: green")
    print("Incorrect trajectory color: red")
    print("Handoff point highlight: gold")
    print(f"Correct latent steps: {correct_trace['latent_trajectory_steps']}")
    print(f"Incorrect latent steps: {incorrect_trace['latent_trajectory_steps']}")
    print(f"Handoff distance in 2D PCA space: {stats['handoff_distance_2d']:.4f}")
    print(f"Correct path extent in 2D PCA space: {stats['correct_path_extent']:.4f}")
    print(f"Incorrect path extent in 2D PCA space: {stats['incorrect_path_extent']:.4f}")


if __name__ == "__main__":
    main()
