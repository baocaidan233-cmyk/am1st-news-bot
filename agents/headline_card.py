"""Renders the image for a poster post — a headline card used when Gettr has
no usable link preview to show.

Ported 2026-09-22 from russia_news/agents/headline_card.py (itself ported
from China_Scandal_News_EN), with the parts this channel does not have
deliberately left out:

  - **No OpenCV.** The source module imports cv2/numpy only for
    `_detect_faces`, which exists to filter keyword-searched Pexels stock
    photos. This channel does no stock-photo search (see the scoping note in
    agents/media_uploader.py), so the dependency buys nothing here and would
    add ~60MB to the venv for an unused code path.
  - No logo overlay, no ink stamp, no face-distortion filter — same
    reasoning as the russia_news port: those are editorial statements this
    channel has no signal to justify.

`make_photo_card` is ported even though the current trigger never reaches it:
the trigger is "the preview image is missing or broken", which by definition
means there is no photo to put behind the text. It is here because the open
question it serves — whether a photo-background card beats Gettr's own link
preview as the STANDARD format for every post — is the one untested format
lever left on this channel, and the renderer is the expensive half of finding
out. Measured 2026-09-22 across six Gettr news channels: native uploaded
images beat link previews on one (newsmax 45 vs 32) and lost on the other
three with enough variance to tell (rsbnetwork 43 vs 77, stevebannon 356 vs
441, jfradioshow 17 vs 17) — but none of those six posts RENDERED cards, they
post raw photos, so that measurement does not answer this question.
"""

from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

W, H = 1080, 1080          # Gettr renders square; matches the sibling bots since 2026-08-05
BG = (24, 26, 30)          # near-black, for the no-photo card
RED = (176, 34, 40)

# DejaVu Sans Bold is present on the deploy VM (Ubuntu); Liberation Sans Bold
# is the fallback in case that package ever changes. Both are system fonts —
# this module ships no font files.
_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
]


def _resolve_font() -> str:
    from pathlib import Path
    for path in _FONT_CANDIDATES:
        if Path(path).exists():
            return path
    raise RuntimeError(
        "headline_card: no usable bold font found; tried " + ", ".join(_FONT_CANDIDATES)
    )


FONT_PATH = _resolve_font()

_SMART_QUOTE_MAP = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'", "—": "-", "–": "-"})


def _font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_PATH, size)


def _wrap_by_pixel(draw: ImageDraw.ImageDraw, text: str, f: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    words = (text or "").translate(_SMART_QUOTE_MAP).split()
    lines: list[str] = []
    current = ""
    for word in words:
        trial = f"{current} {word}".strip()
        if draw.textlength(trial, font=f) <= max_width or not current:
            current = trial
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _draw_text_punchy(draw: ImageDraw.ImageDraw, xy, text: str, f, fill, shadow) -> None:
    x, y = xy
    draw.text((x + 3, y + 3), text, font=f, fill=shadow)
    draw.text((x, y), text, font=f, fill=fill)


def _fit_title(draw: ImageDraw.ImageDraw, title: str, max_width: int, max_lines: int, start_size: int) -> tuple:
    """Shrinks the headline until it fits in `max_lines`, rather than drawing
    the first N lines and silently dropping the rest.

    The source module wrapped at a fixed size and then drew `lines[:5]`. A
    real card went out on a sibling channel reading "...amid U.S. export",
    with "controls" discarded — the failure is invisible in code review
    because the card still looks well-formed. Only if the headline still will
    not fit at the smallest size is it cut, and then at a word boundary with
    an ellipsis, so it never stops mid-phrase.
    """
    for size in range(start_size, 37, -4):
        f = _font(size)
        lines = _wrap_by_pixel(draw, title, f, max_width)
        if len(lines) <= max_lines:
            return f, lines, size
    f = _font(40)
    lines = _wrap_by_pixel(draw, title, f, max_width)[:max_lines]
    if lines:
        while lines[-1] and draw.textlength(lines[-1] + "...", font=f) > max_width:
            lines[-1] = lines[-1].rsplit(" ", 1)[0]
        lines[-1] = lines[-1] + "..."
    return f, lines, 40


def _draw_tag_and_title(draw: ImageDraw.ImageDraw, title: str, tag_text: str, tag_y: int | None, on_photo: bool) -> None:
    """tag_y=None centres the tag+headline block vertically.

    The source module used a fixed tag_y for both card shapes. On a photo that
    is correct -- the text has to sit in the darkened lower half. On the plain
    card it is not: a three-line headline at the same fixed offset leaves the
    bottom third of a 1080x1080 square empty, which reads as a rendering bug
    rather than a design. Measuring the block first and centring it costs one
    extra wrap pass and nothing at runtime.
    """
    tag_font = _font(34)
    pad_x, pad_y = 26, 16
    tag_w = draw.textlength(tag_text, font=tag_font) + pad_x * 2
    tag_h = 34 + pad_y * 2
    tag_x = 64

    max_width = W - 64 * 2
    title_font, lines, size = _fit_title(draw, title, max_width, max_lines=5, start_size=64 if on_photo else 70)
    line_height = int(size * 1.22)
    if tag_y is None:
        block_h = tag_h + 34 + line_height * len(lines)
        tag_y = max(96, (H - block_h) // 2 - 40)   # -40 lifts it off dead centre, which sits low to the eye

    draw.rectangle([tag_x, tag_y, tag_x + tag_w, tag_y + tag_h], fill=RED)
    draw.text((tag_x + pad_x, tag_y + pad_y - 4), tag_text, font=tag_font, fill="white")

    y = tag_y + tag_h + 34
    for line in lines:
        if on_photo:
            _draw_text_punchy(draw, (64, y), line, title_font, "white", (0, 0, 0))
        else:
            draw.text((64, y), line, font=title_font, fill="white")
        y += line_height


def _draw_attribution(draw: ImageDraw.ImageDraw, text: str, fill, shadow=None) -> None:
    if not text:
        return
    f = _font(22)
    x, y = 64, H - 52
    if shadow:
        draw.text((x + 2, y + 2), text, font=f, fill=shadow)
    draw.text((x, y), text, font=f, fill=fill)


def make_card(title: str, out_path: str, tag_text: str, attribution: str = "") -> None:
    """Plain-background headline card — what this channel actually uses today,
    because the trigger (no usable preview image) means there is no photo."""
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, W, 14], fill=RED)
    _draw_tag_and_title(draw, title, tag_text, tag_y=None, on_photo=False)
    _draw_attribution(draw, attribution, "white", (0, 0, 0))
    draw.rectangle([0, H - 14, W, H], fill=RED)
    img.save(out_path, quality=92)


def make_photo_card(title: str, photo_path: str, out_path: str, tag_text: str, attribution: str = "") -> None:
    """Headline burned onto a photo background. Not reached by the current
    trigger — see the module docstring for why it is here anyway."""
    bg = Image.open(photo_path).convert("RGB")
    scale = max(W / bg.width, H / bg.height)
    bg = bg.resize((round(bg.width * scale), round(bg.height * scale)))
    left, top = (bg.width - W) // 2, (bg.height - H) // 2
    img = bg.crop((left, top, left + W, top + H))

    # Darkening gradient, so white text stays legible over an arbitrary photo.
    gradient = Image.new("L", (1, H), 0)
    grad_start, max_dark = int(H * 0.32), 210
    for y in range(H):
        gradient.putpixel((0, y), 0 if y < grad_start else int(max_dark * (((y - grad_start) / (H - grad_start)) ** 1.3)))
    img = Image.composite(Image.new("RGB", (W, H), (0, 0, 0)), img, gradient.resize((W, H)))

    draw = ImageDraw.Draw(img)
    _draw_tag_and_title(draw, title, tag_text, tag_y=522, on_photo=True)
    _draw_attribution(draw, attribution, "white", (0, 0, 0))
    draw.rectangle([0, H - 10, W, H], fill=RED)
    img.save(out_path, quality=92)
