import colorsys
import hashlib
import re
import struct
import unicodedata
import zlib
from functools import lru_cache
from pathlib import Path
from typing import Annotated, NamedTuple

import numpy as np
import skia
from fastapi import FastAPI, Form, HTTPException, Query, Request
from fastapi.responses import RedirectResponse
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from faker import Faker

BASE_DIR = Path(__file__).resolve().parent

fake = Faker()

app = FastAPI()
# Pages and their JSON (mostly the drawing's curve points) shrink several times over; PNGs do not.
app.add_middleware(GZipMiddleware, minimum_size=1000)

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

templates = Jinja2Templates(directory=BASE_DIR / "templates")


CANVAS_SIZE = 400
RENDER_SCALE = 2  # render at 2x for crisp output on high-DPI screens
CURVE_SCALE = 110

PHI = (1 + 5 ** 0.5) / 2
# The golden angle (~137.5 deg) divides a full turn in the golden ratio.
GOLDEN_ANGLE_DEG = 360 * (1 - 1 / PHI)
FIBONACCI = [1, 2, 3, 5, 8, 13, 21]
# The number of times the golden curve winds around the centre is a Fibonacci number.
FIBONACCI_TURNS = [5, 8, 13]
# Hue steps between colour layers: the golden angle (360/phi^2), then its own golden sections
# 360/phi^3 and 360/phi^4, giving palettes from contrasting (137.5 deg) to harmonious (52.5 deg).
GOLDEN_HUE_STEPS = [GOLDEN_ANGLE_DEG, 360 / PHI ** 3, 360 / PHI ** 4]
# Triangle offsets of the three colour layers: every other Fibonacci number, so each layer's
# triangles are ~phi^2 times larger than the previous layer's.
FIBONACCI_OFFSETS = [(2, 5, 13), (3, 8, 21), (5, 13, 34)]
# The curve has a Fibonacci number of points, F(19) ...
GOLDEN_SIZE = 4181
# ... and its nested waves only vibrate at Fibonacci frequencies, odd (1, 1, 3, 5, 13) and even
# (0, 0, 2, 8), like a chord built only from the notes of one scale.
FIBONACCI_FREQUENCIES = ((1, 1, 3, 5, 13), (0, 0, 2, 8))
# The golden curve is scaled so its outermost point lands at this radius.
FIT_RADIUS = 0.92 * CANVAS_SIZE / 2
# Curves whose radius barely varies are plain rings; they are rejected and re-drawn.
MIN_RADIAL_SPREAD = 0.3
# Curves whose drawn triangles reach too little of the canvas leave it nearly empty; also re-drawn.
MIN_COVERAGE = 0.55
MAX_CURVE_ATTEMPTS = 16
# Light is accumulated at 1/HDR_HEADROOM intensity so the additive blend never saturates.
HDR_HEADROOM = 32
# Triangles fainter than this are left out: large, faint triangles only add a haze
# that washes the colours out. (Drawn at 1/HDR_HEADROOM intensity, Skia used to drop them silently,
# as their alpha rounds to 0 in 8 bits; this keeps that look on purpose, and lets the page match it.)
MIN_GLOW = 1 / 16
# The centre holds the mandala itself again, shrunk and turned at each level (see golden_nest).
NEST_LEVELS = 1

# Longer names are refused (422): a name, not a document.
MAX_NAME_LENGTH = 100
Name = Annotated[str, Form(min_length=1, max_length=MAX_NAME_LENGTH)]
templates.env.globals["max_name_length"] = MAX_NAME_LENGTH


class Layer(NamedTuple):
    hue: float  # degrees
    offset: int  # index distance between the corners of each triangle
    saturation: int  # percent


class GoldenHash(NamedTuple):
    points: np.ndarray
    layers: list


def normalize_name(name):
    """Spellings of the same name give the same seed: NFC form, case folded, extra spaces removed."""
    return " ".join(unicodedata.normalize("NFC", name).casefold().split())


def name_to_seed(name):
    # Python's built-in hash() is randomized per process, so use a stable digest
    # to make the same name always produce the same visual hash.
    return int.from_bytes(hashlib.sha256(normalize_name(name).encode("utf-8")).digest()[:8], "big")


# A visual hash's id is its 64-bit seed in hexadecimal, in four groups: xxxx-xxxx-xxxx-xxxx. The
# same name always gets the same id, the image is drawn from the id alone (no name, no database),
# and the name never shows in a shared link.
ID_PATTERN = re.compile(r"[0-9a-f]{4}(?:-[0-9a-f]{4}){3}")


def hash_id(name):
    """The id of a name's visual hash, e.g. 1a2b-3c4d-5e6f-7a8b."""
    digits = f"{name_to_seed(name):016x}"
    return "-".join(digits[i:i + 4] for i in range(0, 16, 4))


def id_to_seed(visual_hash_id):
    """The seed an id stands for; a 404 if it is not an id."""
    if not ID_PATTERN.fullmatch(visual_hash_id):
        raise HTTPException(status_code=404)
    return int(visual_hash_id.replace("-", ""), 16)


def build_curve(rand, N, h, winding, size=GOLDEN_SIZE, frequencies=FIBONACCI_FREQUENCIES, chosen=None):
    """The curve of Gun Gun Febrianza's Visual Hashing, given its turn count N and amplitudes h.

    `winding` is the fraction of a turn the curve rotates per wave, `size` the number of points,
    and `frequencies` the odd and the even frequencies the nested waves draw from (odd ones keep the
    curve symmetric one way, even ones the other); the ones drawn are appended to `chosen`, if given.
    """

    def fi():
        return rand() < 0.5

    def rg(arr):
        i = int(len(arr) * rand())
        result = arr[i]
        arr[i] = arr[-1]
        arr.pop()
        if chosen is not None:
            chosen.append(result)
        return result

    h = list(h)
    for i in range(2, 8):
        if fi():
            h[i] *= -1

    ki, gu = (list(values) for values in frequencies)
    s = [None] * 8
    q = [0.0] * 8
    rand()  # where the original draws its winding; still drawn, so every name keeps its image
    pr = winding

    for i in range(2):
        if fi():
            s[i] = (np.cos, np.sin)
            q[i] = rg(ki) - pr
        else:
            s[i] = (np.sin, np.cos)
            q[i] = rg(gu) + pr

    for i in range(2, 8):
        use_cos = fi()
        if not ki:
            use_cos = False
        if not gu:
            use_cos = True
        q[i] = rg(ki) if use_cos else rg(gu)
        if fi():
            q[i] *= -1
        if i > 5:
            use_cos = not use_cos
        s[i] = np.cos if use_cos else np.sin

    n = [1 if fi() else -1 for _ in range(3)]

    step = (2 * np.pi / size) * N
    # cumsum adds the steps one by one, like `r += step` in the original JavaScript.
    r = np.concatenate(([0.0], np.cumsum(np.full(size - 1, step))))
    b = s[6](r * q[6] + s[3](r * q[3]) * h[5]) * n[0]
    a = 1 + b * h[0]
    d = s[7](r * q[7])
    e = -d
    d = d * (2 - a) * n[1]
    e = e * (2 - a) * n[2]
    c = (s[4](r * q[4] + s[5](r * q[5]) * h[7]) / 4) * h[6] * (a - (1 - h[0]))
    x = np.sin(r * pr + c) * a + s[0][0](r * q[0]) * h[2] * d + s[1][0](r * q[1]) * h[3] * e
    y = np.cos(r * pr + c) * a + s[0][1](r * q[0]) * h[2] * d + s[1][1](r * q[1]) * h[3] * e

    center = CANVAS_SIZE / 2
    return np.column_stack((x * CURVE_SCALE + center, y * CURVE_SCALE + center))


def coverage(points, offsets, cells=24, enough_light=0.5):
    """Fraction of the mandala's disc that visibly lights up, measured on a coarse grid.

    A cell counts when the triangles centred in it add up to `enough_light` alpha, so a few
    large, faint triangles do not count as filling the canvas.
    """
    light = np.zeros((cells, cells))
    for triangles, _, alphas in glow_triangles(points, [Layer(0, offset, 0) for offset in offsets]):
        cell = np.floor(triangles.mean(axis=1) / CANVAS_SIZE * cells).astype(int).clip(0, cells - 1)
        np.add.at(light, (cell[:, 1], cell[:, 0]), alphas)
    yy, xx = np.mgrid[:cells, :cells]
    disc = np.hypot(yy - (cells - 1) / 2, xx - (cells - 1) / 2) < cells * FIT_RADIUS / CANVAS_SIZE
    return (light[disc] >= enough_light).mean()


def radial_spread(points):
    r = np.hypot(*(points - CANVAS_SIZE / 2).T)
    return (r.max() - np.percentile(r, 5)) / r.max()


def golden_visual_hash(name):
    """The golden visual hash of a name (see golden_design)."""
    return golden_design(name_to_seed(name))[0]


def golden_signature(name):
    """A one-line account of the golden choices behind a name's image."""
    return golden_design(name_to_seed(name))[1]


def golden_harmonics(name):
    """The Fibonacci frequencies the curve's waves vibrate at; the page plays them as it draws."""
    return golden_design(name_to_seed(name))[2]


@lru_cache(maxsize=64)
def golden_design(seed):
    """Gun Gun Febrianza's Visual Hashing, re-tuned around the golden ratio phi ~ 1.618, for a seed
    (see name_to_seed): (GoldenHash, signature, harmonics).

    - Fibonacci turns: the curve winds around the centre N times, with N in {5, 8, 13}.
      This often gives N-fold symmetry, but the other frequencies can break it.
    - Golden winding: per wave the curve rotates F(j)/F(k) of a turn, a ratio of two Fibonacci
      numbers (5/8, 3/8, 8/13, ...) and so the best rational approximation of 1/phi^(k-j).
      F(k-1)/F(k) approximates the golden mean, the "most irrational" rotation, the same one that
      makes phyllotaxis pack evenly.
    - Golden amplitudes: each nested wave is ~1/phi (or 1/phi^2) the size of the one before.
      This is a proportion chosen for the eye, not something found in nature.
    - Golden hues: consecutive colour layers are 360/phi^k deg apart (k = 2, 3 or 4) on the colour
      wheel. k = 2 is the golden angle, the turn a plant makes between consecutive leaves; k = 3
      and 4 are golden sections of it, chosen for colour harmony rather than found in nature.
    - Fibonacci offsets: the triangles of consecutive layers grow by ~phi^2 (2, 5, 13 or 3, 8, 21 ...).
    - Golden nest: the centre holds the mandala itself, 1/phi^2 the size, turned by the golden angle
      and mirrored (see golden_nest), so the centre flows in the same light as the rest.

    Every curve is scaled to fill the canvas. Plain rings and curves that would leave the canvas
    nearly empty are re-drawn from the same generator, so the result stays deterministic.
    """
    rng = np.random.default_rng(seed)
    rand = rng.random

    center = CANVAS_SIZE / 2
    for _ in range(MAX_CURVE_ATTEMPTS):
        N = FIBONACCI_TURNS[int(rand() * len(FIBONACCI_TURNS))]
        k = FIBONACCI.index(N)
        winding = FIBONACCI[int(rand() * k)] / N
        h = [0.0] * 8
        h[0] = (0.8 + rand() * 0.2) / PHI
        h[2] = h[0] / PHI * (0.8 + rand() * 0.4)
        h[3] = h[2] / PHI ** 2 * (0.8 + rand() * 0.4)
        # The inner waves' strengths are powers of phi (1, phi, phi^2, phi^3), give or take 10%.
        h[5] = PHI ** int(rand() * 4) * (0.9 + rand() * 0.2)
        h[6] = PHI ** int(rand() * 2) * (0.9 + rand() * 0.2)
        h[7] = PHI ** int(rand() * 2) * (0.9 + rand() * 0.2)
        harmonics = []
        points = build_curve(rand, N, h, winding, GOLDEN_SIZE, FIBONACCI_FREQUENCIES, harmonics)
        radius = np.hypot(*(points - center).T).max()
        points = (points - center) * (FIT_RADIUS / radius) + center
        offsets = FIBONACCI_OFFSETS[int(rand() * len(FIBONACCI_OFFSETS))]
        if radial_spread(points) >= MIN_RADIAL_SPREAD and coverage(points, offsets) >= MIN_COVERAGE:
            break

    hue = rand() * 360
    hue_step = GOLDEN_HUE_STEPS[int(rand() * len(GOLDEN_HUE_STEPS))]
    layers = []
    for i, offset in enumerate(offsets):
        saturation = int(65 + rand() * 20)
        layers.append(Layer((hue + i * hue_step) % 360, offset, saturation))
    harmonics = sorted({f for f in harmonics if f})
    signature = " · ".join([
        f"{N} turns",
        f"winding {round(winding * N)}/{N}",
        "harmonics " + ", ".join(str(f) for f in harmonics),
        f"hues {hue_step:.1f}° apart",
        "triangles " + ", ".join(str(offset) for offset in offsets),
    ])
    return GoldenHash(points, layers), signature, harmonics


def golden_nest(points, levels=NEST_LEVELS):
    """The mandala and the copies of it nested in its centre, as (points, scale) pairs.

    Level k is 1/phi^(2k) the size of the mandala, the inner golden section of the level around it,
    and turned k golden angles, the turn a plant makes from one leaf to the next. Like the squares
    of a golden spiral, the same form keeps returning, smaller and turned, towards the centre.

    Every other level is mirrored, so its curves wind the other way, as the two families of spirals
    on a sunflower head do. (Mirroring alone would undo itself: level 2 would face exactly like the
    mandala. The turns add up instead, so no level faces like another.)
    """
    center = CANVAS_SIZE / 2
    for matrix, backwards, scale in nest_transforms(levels):
        offset = points - center
        if backwards:
            offset = offset[::-1]
        yield center + offset @ matrix.T, scale


def nest_transforms(levels=NEST_LEVELS):
    """How each level of the golden nest is made from the mandala, as (matrix, backwards, scale).

    The 2x2 matrix shrinks, turns and (for every other level) mirrors the curve about the centre.
    A mirror turns every triangle the other way round, and only counter-clockwise triangles glow,
    so a mirrored curve is also walked backwards to turn them back: the copy lights exactly the
    mirror image of the mandala's triangles.
    """
    for level in range(levels + 1):
        angle = level * np.radians(GOLDEN_ANGLE_DEG)
        turn = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        mirrored = level % 2 == 1
        mirror = np.diag([1, -1 if mirrored else 1])
        scale = PHI ** (-2 * level)
        yield turn @ mirror * scale, mirrored, scale


def glow_triangles(points, layers, scale=1.0, min_alpha=0.0):
    """Yield (triangles, layer, alphas) for every triangle the original fills.

    `scale` is the size of the curve relative to the full mandala: triangles are judged by their
    area relative to it, so a shrunken copy glows the same way. Triangles fainter than `min_alpha`
    are left out.
    """
    idx = np.arange(len(points))
    for layer in layers:
        corners = np.stack((idx, idx + layer.offset, idx + 2 * layer.offset), axis=1) % len(points)
        tri = points[corners]
        # Twice the signed area; only reasonably sized, counter-clockwise triangles are drawn.
        area = (
            tri[:, 0, 0] * (tri[:, 1, 1] - tri[:, 2, 1])
            + tri[:, 1, 0] * (tri[:, 2, 1] - tri[:, 0, 1])
            + tri[:, 2, 0] * (tri[:, 0, 1] - tri[:, 1, 1])
        )
        keep = (area > 45 * scale ** 2) & (area < 8000 * scale ** 2)
        # Small triangles are bright, large ones faint, so the edges of the curve glow.
        alpha = np.minimum(1.0, 55 * scale ** 2 / np.where(keep, area, 1))
        keep &= alpha >= min_alpha
        yield tri[keep], layer, alpha[keep]


def draw_glow(canvas, points, layers, alpha_scale=1.0, scale=1.0, min_alpha=0.0):
    # kPlus is additive blending, the same as the canvas "lighter" mode in the original:
    # overlapping translucent triangles build up into glowing highlights.
    paint = skia.Paint(AntiAlias=True, BlendMode=skia.BlendMode.kPlus)
    path = skia.Path()  # reused: thousands of triangles, so every call per triangle counts
    for triangles, layer, alphas in glow_triangles(points, layers, scale, min_alpha):
        r, g, b = colorsys.hls_to_rgb(layer.hue / 360, 0.4, layer.saturation / 100)
        paint.setColor4f(skia.Color4f(r, g, b, 1.0))
        for (p0, p1, p2), alpha in zip(triangles.tolist(), (alphas * alpha_scale).tolist()):
            paint.setAlphaf(alpha)
            path.rewind()
            path.moveTo(*p0)
            path.lineTo(*p1)
            path.lineTo(*p2)
            path.close()
            canvas.drawPath(path, paint)


def encode_png(rgb):
    """Encode an (h, w, 3) uint8 image as an opaque RGB PNG.

    About 4x faster than Skia's encoder (whose compression level cannot be set from Python)
    and slightly smaller: every row uses the "up" filter, then zlib level 3.
    """
    height, width, _ = rgb.shape
    rows = rgb.reshape(height, -1)
    up = np.vstack([rows[:1], rows[1:] - rows[:-1]])  # uint8 arithmetic wraps, as PNG expects
    raw = np.hstack([np.full((height, 1), 2, np.uint8), up]).tobytes()  # filter type 2 = up

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)  # 8-bit RGB, no interlace
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw, 3)) + chunk(b"IEND", b"")


# How long the page takes to replay the drawing, in seconds: a Fibonacci number. The pen eases in
# and out, so its first strokes and the finished image are slow enough to follow.
TIMELINE_SECONDS = 13

# The brightness tone mapping brings an image's brightest 1% to.
TONEMAP_KEY = 3.2


def auto_exposure(rgb, key=TONEMAP_KEY, percentile=99.0, peak=None):
    """The exposure that brings an image's brightest 1% to `key`, like a camera's auto-exposure."""
    if peak is None:
        peak = rgb.max(axis=2)
    lit = peak[peak > 0.01]
    return float(np.clip(key / np.percentile(lit, percentile), 0.8, 6.0)) if lit.size else 1.0


def tonemap(rgb, key=TONEMAP_KEY, percentile=99.0, exposure=None):
    """Map unbounded additive light to [0, 1] while keeping its colour.

    Exposure is set per image (see auto_exposure) unless given, so sparse and dense images
    are equally bright. Brightness is compressed with 1 - exp(-x) and every pixel keeps
    its hue, except the hottest cores, which fade towards white like a real glow.
    """
    peak = rgb.max(axis=2)
    if exposure is None:
        exposure = auto_exposure(rgb, key, percentile, peak)
    # Only lit pixels need mapping: where there is no light the result is black. (Most of the
    # canvas is dark, so this saves most of the work.)
    out = np.zeros_like(rgb)
    lit = np.flatnonzero(peak > 0)
    lit_peak = peak.ravel()[lit, None]
    exposed = lit_peak * exposure
    mapped = 1 - np.exp(-exposed)
    lit_out = rgb.reshape(-1, 3)[lit] * (mapped / np.maximum(lit_peak, 1e-9))
    # Only the hottest pixels fade towards white; elsewhere the whiteness is 0.
    whiteness = np.clip((exposed - 0.6 * key) / 4, 0, 0.5)
    hot = whiteness[:, 0] > 0
    lit_out[hot] += (mapped[hot] - lit_out[hot]) * whiteness[hot]
    out.reshape(-1, 3)[lit] = lit_out
    return out


def render_light(draw):
    """Run `draw` on a float canvas and return the unclipped RGB light it produced."""
    pixels = CANVAS_SIZE * RENDER_SCALE
    info = skia.ImageInfo.Make(pixels, pixels, skia.kRGBA_F32_ColorType, skia.kPremul_AlphaType)
    surface = skia.Surface.MakeRaster(info)
    canvas = surface.getCanvas()
    canvas.clear(skia.Color4f(0, 0, 0, 1))
    canvas.scale(RENDER_SCALE, RENDER_SCALE)
    draw(canvas)
    light = np.empty((pixels, pixels, 4), np.float32)
    surface.readPixels(info, light)
    # Skia clamps additive blending at 1, even on float surfaces, so everything is drawn
    # at 1/HDR_HEADROOM intensity and scaled back up here.
    return np.multiply(light[:, :, :3], HDR_HEADROOM, dtype=np.float64)


def nest_lights(points, layers):
    """The unclipped light of each level of the golden nest, each rendered on its own."""
    return [
        render_light(
            lambda canvas, nested=nested, scale=scale: draw_glow(canvas, nested, layers, 1 / HDR_HEADROOM, scale, MIN_GLOW)
        )
        for nested, scale in golden_nest(points)
    ]


def to_png(rgb):
    """Encode light in [0, 1] as an 8-bit PNG (rgb is changed in place)."""
    np.clip(rgb, 0, 1, out=rgb)
    rgb *= 255
    return encode_png(np.rint(rgb, out=rgb).astype(np.uint8))


def render_hdr_png(points, layers):
    """Render the mandala and its golden nest with unclipped light and tone mapping, so dense
    areas keep their colour."""
    return to_png(tonemap(sum(nest_lights(points, layers))))


def render_visual_hash_png(name):
    """The image of a name's visual hash, as a PNG."""
    return render_seed_png(name_to_seed(name))


# The same seed always gives the same image, so recent ones are kept rather than drawn again
# (about 0.5 MB per image).
@lru_cache(maxsize=64)
def render_seed_png(seed):
    return render_hdr_png(*golden_design(seed)[0])


def spin(level):
    """How fast level k of the golden nest turns in the animation, relative to the mandala: (-1/phi)^k.

    Neighbouring levels turn opposite ways, each 1/phi as fast as the one around it. As phi is the
    most irrational number, the levels never line up the same way twice: the animation never repeats.
    """
    return (-1 / PHI) ** level


class AnimationLayer(NamedTuple):
    png: bytes  # opaque, on black: the page blends the layers in "screen" mode
    size: float  # width, as a fraction of the full image's
    spin: float  # turning speed, relative to the mandala's


class Animation(NamedTuple):
    layers: list  # AnimationLayer
    exposure: float  # the exposure the image is tone-mapped with


def animation_layers(name):
    """The layers the page animates for a name (see animate)."""
    return animate(name_to_seed(name)).layers


@lru_cache(maxsize=32)
def animate(seed):
    """The layers the page animates: the mandala and each level of its nest apart, so each can
    turn at its own speed.

    The layers share one exposure, and each is tone-mapped on its own. Blending them in "screen" mode,
    1 - (1 - a)(1 - b), then gives back nearly the image tone-mapped as a whole, since
    1 - exp(-(x + y)) = 1 - exp(-x) exp(-y): the animation starts on the downloadable image.
    """
    points, layers = golden_design(seed)[0]
    lights = nest_lights(points, layers)
    exposure = auto_exposure(sum(lights))
    pixels = lights[0].shape[0]
    result = []
    for level, (light, (_, scale)) in enumerate(zip(lights, golden_nest(points))):
        # Crop each level to the square around it; inner levels are small.
        half = min(pixels // 2, int(np.ceil(FIT_RADIUS * scale * RENDER_SCALE)) + 4)
        crop = light[pixels // 2 - half:pixels // 2 + half, pixels // 2 - half:pixels // 2 + half]
        result.append(AnimationLayer(to_png(tonemap(crop, exposure=exposure)), 2 * half / pixels, spin(level)))
    return Animation(result, exposure)


def contrast_with_white(rgb):
    """WCAG contrast ratio of white text on an sRGB colour (channels 0 to 1)."""
    linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    luminance = 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]
    return 1.05 / (luminance + 0.05)


def page_colours(palette):
    """CSS colours for the page, taken from the mandala's own palette, so every name gets its own room.

    Every palette hue is darkened just enough for white button text to keep a 4.5:1 contrast (WCAG AA).
    Yellows have to darken a long way and turn olive or brown, so the accents are the two hues that
    stay brightest. The glow behind the artwork uses all three.
    """

    def readable(layer):
        """The lightest version of the layer's colour that white text is readable on, and its lightness."""
        saturation = min(layer.saturation, 80) / 100
        lightness = 0.5
        while True:
            # Check the colour as the page will get it: rounded to 8 bits per channel.
            rgb = [round(c * 255) for c in colorsys.hls_to_rgb(layer.hue / 360, lightness, saturation)]
            if contrast_with_white([c / 255 for c in rgb]) >= 4.5:
                return "#{:02x}{:02x}{:02x}".format(*rgb), lightness
            lightness -= 0.01

    brightest = sorted((readable(layer) for layer in palette), key=lambda colour: -colour[1])
    glow = [f"hsla({layer.hue:.0f}, {layer.saturation}%, 50%, 0.2)" for layer in palette]
    return {"accent": brightest[0][0], "accent-2": brightest[1][0], **{f"glow-{i}": c for i, c in enumerate(glow)}}


def timeline(points, palette, exposure):
    """What the page needs to replay how the image is drawn, triangle by triangle along the curve.

    The page applies the same rules as glow_triangles to these points (canvas units, 0 to 400) and
    to every level of the nest, so it grows exactly the triangles of the finished image. With WebGL
    it adds their light up unclipped and tone-maps it like tonemap, with the finished image's
    exposure, so the last frame is the finished image.
    """
    return {
        "points": np.round(points, 2).tolist(),
        "layers": [{"hue": layer.hue, "saturation": layer.saturation, "offset": layer.offset} for layer in palette],
        "levels": [{"matrix": np.round(matrix, 9).ravel().tolist(), "backwards": backwards, "scale": scale} for matrix, backwards, scale in nest_transforms()],
        "tonemap": {"exposure": exposure, "key": TONEMAP_KEY},
        "minAlpha": MIN_GLOW,
        "seconds": TIMELINE_SECONDS,
        # Without WebGL the page can only add light up to white, so the triangles are drawn fainter
        # there, to keep some of their colour while the drawing builds up.
        "gain": 0.3,
    }


@lru_cache(maxsize=32)
def visual_hash(visual_hash_id):
    """Everything the page shows for a visual hash, by its id: the layers it animates, the colours
    it takes on, and the timeline of how it is drawn. Cached: callers must not change it."""
    seed = id_to_seed(visual_hash_id)
    (points, palette), signature, harmonics = golden_design(seed)
    animation = animate(seed)
    layers = [
        # Fetched apart, as images: no base64 in the page, loaded in parallel, and kept by the
        # browser. The address carries a digest of the image, so a changed image gets a new one.
        {"src": f"/{visual_hash_id}/layers/{level}.png?v={hashlib.sha256(layer.png).hexdigest()[:12]}", "size": layer.size, "spin": layer.spin}
        for level, layer in enumerate(animation.layers)
    ]
    return {
        "id": visual_hash_id,
        "signature": signature,
        "harmonics": harmonics,
        "layers": layers,
        "colours": page_colours(palette),
        "timeline": timeline(points, palette, animation.exposure),
    }


@app.get("/", response_class=HTMLResponse)
def home(request: Request, name: Annotated[str | None, Query(max_length=MAX_NAME_LENGTH)] = None):
    """The page. A name (sent by the form without JavaScript) goes on to its visual hash's own page."""
    if name and name.strip():
        return RedirectResponse(request.url_for("visual_hash_page", visual_hash_id=hash_id(name)), status_code=303)
    return templates.TemplateResponse(request, "index.html", {"hash": None})


@app.get("/random-name")
def random_name():
    return fake.name()


@app.post("/api/visual-hash")
def visual_hash_api(name: Name):
    """A name's visual hash as JSON, so the page can swap in a new one without reloading."""
    return {**visual_hash(hash_id(name)), "name": name}


@app.get("/{visual_hash_id}.png")
def visual_hash_image(visual_hash_id: str, download: bool = False):
    """The image itself: for link previews (og:image), and to save. An id always gives the same image."""
    headers = {"Cache-Control": "public, max-age=31536000, immutable"}
    if download:
        headers["Content-Disposition"] = f'attachment; filename="visual-hash-{visual_hash_id}.png"'
    return Response(content=render_seed_png(id_to_seed(visual_hash_id)), media_type="image/png", headers=headers)


@app.get("/{visual_hash_id}/layers/{level}.png")
def animation_layer(visual_hash_id: str, level: int):
    """One layer of a visual hash's animation (see animate)."""
    layers = animate(id_to_seed(visual_hash_id)).layers
    if not 0 <= level < len(layers):
        raise HTTPException(status_code=404)
    # Its address carries a digest of its content (see visual_hash), so it can be kept for good.
    return Response(content=layers[level].png, media_type="image/png", headers={"Cache-Control": "public, max-age=31536000, immutable"})


@app.get("/{visual_hash_id}", response_class=HTMLResponse)
def visual_hash_page(request: Request, visual_hash_id: str):
    """A visual hash's own page, to share: /1a2b-3c4d-5e6f-7a8b. The name is not in it."""
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "hash": visual_hash(visual_hash_id),
            "share_url": str(request.url_for("visual_hash_page", visual_hash_id=visual_hash_id)),
            "image_url": str(request.url_for("visual_hash_image", visual_hash_id=visual_hash_id)),
        },
    )
