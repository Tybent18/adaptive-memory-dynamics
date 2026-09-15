"""Generate a deterministic repository demo without screen-recording dependencies."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def generate_demo(path: Path = Path("demos/adaptive-memory-demo.gif")) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    frames = []
    stages = [
        ("01", "Baseline retention", "Persistent-memory control", 0.18),
        ("02", "Passive decay", "Time-sensitive pathway decay", 0.43),
        ("03", "Reinforced retention", "Activation-protected pathways", 0.72),
        ("04", "Evidence bundle ready", "CSV · JSON · charts", 1.0),
    ]
    for active, (_number, title, detail, progress) in enumerate(stages):
        for pulse in range(4):
            image = Image.new("RGB", (960, 540), "#07111d")
            draw = ImageDraw.Draw(image)
            draw.text((48, 38), "ADAPTIVE MEMORY", fill="#32d5a4", font=font)
            draw.text((48, 64), "FORGETTING DYNAMICS LABORATORY", fill="#d9e7ef", font=font)
            draw.text(
                (48, 88),
                "Controlled retention · adaptation · continual learning",
                fill="#7f9bad",
                font=font,
            )
            for idx, (_, row_title, row_detail, _) in enumerate(stages):
                y = 138 + idx * 75
                fill = "#173348" if idx <= active else "#102235"
                draw.rounded_rectangle((48, y, 912, y + 58), radius=8, fill=fill)
                color = (
                    "#61edc5"
                    if idx == active and pulse % 2 == 0
                    else ("#32d5a4" if idx <= active else "#567083")
                )
                draw.text((68, y + 13), f"{idx + 1:02d}", fill=color, font=font)
                draw.text((112, y + 11), row_title, fill="#d9e7ef", font=font)
                draw.text((112, y + 31), row_detail, fill="#7f9bad", font=font)
            draw.rounded_rectangle((48, 465, 912, 480), radius=6, fill="#182b3b")
            draw.rounded_rectangle(
                (48, 465, 48 + int(864 * progress), 480), radius=6, fill="#32d5a4"
            )
            draw.text((48, 500), f"{title}: {detail}", fill="#7f9bad", font=font)
            frames.append(image)
    frames[0].save(
        path, save_all=True, append_images=frames[1:], duration=230, loop=0, optimize=True
    )
    return path


if __name__ == "__main__":
    print(generate_demo())
