#!/usr/bin/env python3
"""Generate social preview image for GitHub repo."""

from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageChops

W, H = 1280, 640
img = Image.new("RGB", (W, H), "#080808")
draw = ImageDraw.Draw(img)

cx, cy = W // 2, H // 2 - 40

# Subtle radial glow — much softer
glow = Image.new("RGB", (W, H), (0, 0, 0))
glow_draw = ImageDraw.Draw(glow)
for r in range(220, 0, -1):
    t = 1 - r / 220
    red = int(180 * t ** 2)
    green = int(20 * t ** 2)
    glow_draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(red, green, 0))
glow_soft = glow.point(lambda p: int(p * 0.06))
img = ImageChops.add(img, glow_soft)
draw = ImageDraw.Draw(img)

# Lobster emoji
lobster_size = 160
try:
    lobster_font = ImageFont.truetype("/System/Library/Fonts/Apple Color Emoji.ttc", lobster_size)
    draw.text((cx, cy - 10), "🦞", font=lobster_font, anchor="mm", embedded_color=True)
except Exception:
    draw.text((cx, cy - 10), "🦞", anchor="mm", fill="#ff4400")

# Fonts
try:
    title_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 48)
except Exception:
    title_font = ImageFont.load_default()
try:
    sub_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 20)
except Exception:
    sub_font = ImageFont.load_default()
try:
    small_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 16)
except Exception:
    small_font = ImageFont.load_default()

# Title
draw.text((cx, cy + 115), "VOICE MODE", font=title_font, anchor="mm", fill="#e0e0e0")

# Subtitle
draw.text((cx, cy + 155), "Local voice interface for OpenClaw  ·  Kokoro TTS", font=sub_font, anchor="mm", fill="#555555")

# Feature tags
tags = ["82M Params", "Real-time SSE", "Voice Input", "Runs Locally"]
tag_y = cy + 205
total_w = sum(len(t) * 9 + 28 for t in tags) + (len(tags) - 1) * 12
start_x = cx - total_w // 2

for tag in tags:
    tw = len(tag) * 8 + 28
    draw.rounded_rectangle(
        [start_x, tag_y - 13, start_x + tw, tag_y + 13],
        radius=10,
        fill="#111111",
        outline="#222222",
    )
    draw.text((start_x + tw // 2, tag_y), tag, font=small_font, anchor="mm", fill="#666666")
    start_x += tw + 12

img.save("social-preview.png", "PNG")
print(f"Saved social-preview.png ({W}x{H})")
