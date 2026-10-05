import colorsys
import hashlib
import itertools
import json
import os
import re
import subprocess
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pytest
import skia
from fastapi.testclient import TestClient

import main

client = TestClient(main.app)

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

NAMES = ["Budi", "Ani", "Alice", "李小龙 🐉", "x" * 200]


@pytest.mark.parametrize("name", NAMES)
def test_golden_parameters(name):
    points, layers = main.golden_visual_hash(name)
    assert points.shape == (main.GOLDEN_SIZE, 2)
    assert main.GOLDEN_SIZE == FIBONACCI[19]
    assert len(layers) == 3
    # Colour layers are spaced by one of the golden hue steps (360 / phi^k).
    hue_steps = [(b.hue - a.hue) % 360 for a, b in zip(layers, layers[1:])]
    assert hue_steps[0] == pytest.approx(hue_steps[1])
    assert any(hue_steps[0] == pytest.approx(step) for step in main.GOLDEN_HUE_STEPS)
    # Triangle offsets are every other Fibonacci number, so they grow by ~phi^2 per layer.
    offsets = [layer.offset for layer in layers]
    assert tuple(offsets) in main.FIBONACCI_OFFSETS
    assert offsets[2] / offsets[1] == pytest.approx(main.PHI ** 2, rel=0.05)


FIBONACCI = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181]


def test_golden_curve_is_built_from_phi(monkeypatch):
    calls = []
    build_curve = main.build_curve

    def spy(rand, N, h, winding, size=main.GOLDEN_SIZE, frequencies=main.FIBONACCI_FREQUENCIES, chosen=None):
        calls.append((N, h, winding, size, frequencies))
        return build_curve(rand, N, h, winding, size, frequencies, chosen)

    monkeypatch.setattr(main, "build_curve", spy)
    main.golden_design.cache_clear()  # so every curve is built again, through the spy
    for i in range(100):
        main.golden_visual_hash(f"name {i}")
    for N, h, winding, size, frequencies in calls:
        assert N in main.FIBONACCI_TURNS
        # Winding: a ratio of two Fibonacci numbers.
        assert round(winding * N) in main.FIBONACCI and round(winding * N) < N
        # A Fibonacci number of points, waving only at Fibonacci frequencies ...
        assert size in FIBONACCI
        odd, even = frequencies
        assert set(odd) | set(even) <= set(FIBONACCI)
        assert all(f % 2 for f in odd) and not any(f % 2 for f in even)
        # ... with inner waves as strong as a power of phi, give or take 10%.
        for amplitude in (h[5], h[6], h[7]):
            power = round(np.log(amplitude) / np.log(main.PHI))
            assert amplitude / main.PHI ** power == pytest.approx(1, abs=0.1)
    # The golden mean itself, 1/phi ~ F(k-1)/F(k), is among the windings.
    assert any(winding == pytest.approx(1 / main.PHI, abs=0.02) for _, _, winding, _, _ in calls)


@pytest.mark.parametrize(
    "a, b",
    [("José", unicodedata.normalize("NFD", "José")), ("Budi Santoso", "  Budi   Santoso "), ("Budi", "Budi\t"), ("Budi", "BUDI"), ("Straße", "STRASSE")],
)
def test_spellings_of_the_same_name_give_the_same_golden_image(a, b):
    assert main.render_visual_hash_png(a) == main.render_visual_hash_png(b)
    assert main.hash_id(a) == main.hash_id(b)


def test_golden_curves_fill_the_canvas_and_are_never_plain_rings():
    for i in range(60):
        points, _ = main.golden_visual_hash(f"name {i}")
        radius = np.hypot(*(points - main.CANVAS_SIZE / 2).T)
        assert radius.max() == pytest.approx(main.FIT_RADIUS)
        assert main.radial_spread(points) >= main.MIN_RADIAL_SPREAD


def test_golden_images_are_never_nearly_empty():
    # These names used to give a mandala with no visible triangles, or only a small one.
    for name in ["Nicole Gutierrez", "Robert Sullivan"] + [f"name {i}" for i in range(60)]:
        points, layers = main.golden_visual_hash(name)
        assert main.coverage(points, [layer.offset for layer in layers]) >= main.MIN_COVERAGE


def test_tonemap_keeps_hue_and_stays_in_range():
    rng = np.random.default_rng(0)
    rgb = rng.random((64, 64, 3)) * rng.choice([0.1, 1, 10, 100], size=(64, 64, 1))
    out = main.tonemap(rgb)
    assert out.min() >= 0 and out.max() <= 1
    # Brighter input never gets darker output.
    ramp = main.tonemap(np.linspace(0, 50, 200)[:, None, None] * [[[1.0, 0.5, 0.2]]])[:, 0, :]
    assert np.diff(ramp, axis=0).min() > -1e-12
    # Dim pixels keep their colour ratios exactly.
    dim = np.array([[[0.02, 0.01, 0.004]]])
    mapped = main.tonemap(np.concatenate([rgb, np.broadcast_to(dim, (64, 1, 3))], axis=1))[0, -1]
    assert mapped / mapped[0] == pytest.approx(dim[0, 0] / dim[0, 0, 0])


def test_golden_render_does_not_clip_to_white():
    for name in NAMES:
        image = skia.Image.MakeFromEncoded(main.render_visual_hash_png(name))
        pixels = np.array(image.toarray())[:, :, :3]
        assert (pixels.min(axis=2) >= 250).mean() < 0.001


def test_golden_nest_repeats_the_mandala_smaller_turned_and_mirrored():
    points, _ = main.golden_visual_hash("Budi")
    nest = list(main.golden_nest(points))
    assert len(nest) == main.NEST_LEVELS + 1
    center = main.CANVAS_SIZE / 2
    z = (points[:, 0] - center) + 1j * (points[:, 1] - center)
    for level, (copy, scale) in enumerate(nest):
        z_copy = (copy[:, 0] - center) + 1j * (copy[:, 1] - center)
        # Every other level is the mirror image (walked backwards), so its curves wind the other way.
        source = np.conj(z[::-1]) if level % 2 else z
        # Level k sits at 1/phi^(2k) of the distance, turned k golden angles (~137.5 deg each).
        expected = np.exp(1j * level * np.radians(main.GOLDEN_ANGLE_DEG)) / main.PHI ** (2 * level)
        assert z_copy / source == pytest.approx(np.full(len(z), expected))
        assert scale == pytest.approx(main.PHI ** (-2 * level))


def test_no_level_of_the_nest_faces_like_another():
    # Mirroring alone would undo itself every second level; the turns must keep adding up.
    points, _ = main.golden_visual_hash("Budi")
    shapes = [(copy - main.CANVAS_SIZE / 2) / scale for copy, scale in main.golden_nest(points, levels=4)]
    for a, b in itertools.combinations(range(len(shapes)), 2):
        assert not np.allclose(shapes[a], shapes[b], atol=1e-6), (a, b)


def test_every_level_of_the_nest_lights_the_same_triangles():
    # The mirrored levels must glow like the mandala too, not light only what it leaves dark.
    for i in range(40):
        points, layers = main.golden_visual_hash(f"name {i}")
        counts = [
            [len(triangles) for triangles, _, _ in main.glow_triangles(copy, layers, scale)]
            for copy, scale in main.golden_nest(points)
        ]
        assert all(count == counts[0] for count in counts)


def test_nest_levels_spin_at_powers_of_minus_one_over_phi():
    assert [main.spin(level) for level in range(3)] == pytest.approx([1, -1 / main.PHI, 1 / main.PHI ** 2])
    # Neighbouring levels turn opposite ways, each 1/phi as fast; no two ever turn at a rational
    # ratio, so the animation never comes back to the same arrangement.
    for level in range(4):
        assert main.spin(level + 1) / main.spin(level) == pytest.approx(-1 / main.PHI)
    layers = main.animation_layers("Budi")
    assert [layer.spin for layer in layers] == [main.spin(level) for level in range(main.NEST_LEVELS + 1)]
    assert [layer.size for layer in layers] == sorted((layer.size for layer in layers), reverse=True)


def decode(png):
    return np.array(skia.Image.MakeFromEncoded(png).toarray(colorType=skia.kRGBA_8888_ColorType))[:, :, :3] / 255


@pytest.mark.parametrize("name", NAMES)
def test_animation_starts_on_the_downloadable_image(name):
    # The page blends the layers in "screen" mode: 1 - (1 - a)(1 - b) ...
    still = decode(main.render_visual_hash_png(name))
    pixels = still.shape[0]
    dark = np.ones_like(still)
    for layer in main.animation_layers(name):
        image = decode(layer.png)
        half = image.shape[0] // 2
        assert image.shape[0] == round(layer.size * pixels)
        placed = np.zeros_like(still)
        placed[pixels // 2 - half:pixels // 2 + half, pixels // 2 - half:pixels // 2 + half] = image
        dark *= 1 - placed
    # ... which gives back the image tone-mapped as a whole, up to a little rounding.
    assert np.abs((1 - dark) - still).mean() < 1 / 255


def test_shrunken_copies_glow_the_same_way():
    # A copy shrunk by `scale` keeps exactly the triangles, and the brightness, of the full mandala.
    points, layers = main.golden_visual_hash("Budi")
    scale = 1 / main.PHI ** 2
    small = (points - main.CANVAS_SIZE / 2) * scale + main.CANVAS_SIZE / 2
    for (big_tris, _, big_alpha), (small_tris, _, small_alpha) in zip(
        main.glow_triangles(points, layers), main.glow_triangles(small, layers, scale)
    ):
        assert len(big_tris) == len(small_tris)
        assert small_alpha == pytest.approx(big_alpha)


def test_golden_nest_lights_the_centre(monkeypatch):
    def centre_light(name):
        # Drawn afresh (past the cache), as golden_nest is swapped out below.
        seed = main.name_to_seed(name)
        image = np.array(skia.Image.MakeFromEncoded(main.render_hdr_png(*main.golden_design(seed)[0])).toarray())[:, :, :3]
        size = image.shape[0]
        r = int(size / 2 * main.FIT_RADIUS / (main.CANVAS_SIZE / 2) / main.PHI ** 2)
        return image[size // 2 - r:size // 2 + r, size // 2 - r:size // 2 + r].astype(float).mean()

    nested = [centre_light(name) for name in NAMES]
    golden_nest = main.golden_nest
    monkeypatch.setattr(main, "golden_nest", lambda points: golden_nest(points, levels=0))
    bare = [centre_light(name) for name in NAMES]
    assert all(with_nest > without for with_nest, without in zip(nested, bare))


def test_same_name_gives_same_image():
    assert main.render_visual_hash_png("Budi") == main.render_visual_hash_png("Budi")


def test_different_names_give_different_images():
    assert main.render_visual_hash_png("Budi") != main.render_visual_hash_png("Ani")


def test_seed_is_stable_across_processes():
    # hash() would differ between processes; the seed must not.
    code = "import main; print(main.name_to_seed('Budi'))"
    outputs = {
        subprocess.check_output([sys.executable, "-c", code], cwd=main.BASE_DIR, text=True)
        for _ in range(2)
    }
    assert outputs == {f"{main.name_to_seed('Budi')}\n"}


def test_render_is_a_glowing_image_on_black():
    image = skia.Image.MakeFromEncoded(main.render_visual_hash_png("Budi"))
    pixels = np.array(image.toarray())
    size = main.CANVAS_SIZE * main.RENDER_SCALE
    assert pixels.shape[:2] == (size, size)
    assert (pixels[0, 0, :3] == 0).all()  # black corner
    assert pixels[:, :, :3].max() > 200  # bright highlights where triangles overlap


def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert "Visual Hashing" in response.text


def test_random_name():
    response = client.get("/random-name")
    assert response.status_code == 200
    assert isinstance(response.json(), str)


@pytest.mark.parametrize("name", NAMES + ["Denise Arias", "José"])
def test_ids_are_four_groups_of_four_hex_digits_from_the_seed(name):
    visual_hash_id = main.hash_id(name)
    assert re.fullmatch(r"[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}", visual_hash_id)
    assert main.id_to_seed(visual_hash_id) == main.name_to_seed(name)  # the id is the seed itself


def test_ids_of_different_names_differ():
    names = [f"name {i}" for i in range(2000)]
    assert len({main.hash_id(name) for name in names}) == len(names)


def visual_hash_page(name):
    page = client.get(f"/{main.hash_id(name)}")
    assert page.status_code == 200
    return page, json.loads(re.search(r"data-hash='([^']*)'", page.text).group(1))


def test_a_visual_hash_page_is_drawn_from_its_id_alone():
    visual_hash_id = main.hash_id("Denise Arias")
    page, hash_ = visual_hash_page("Denise Arias")
    assert hash_ == main.visual_hash(visual_hash_id)
    assert hash_["id"] == visual_hash_id and "name" not in hash_
    assert "Denise" not in page.text  # a shared link does not give the name away
    assert f"<title>{visual_hash_id} · Visual Hashing</title>" in page.text
    # The page takes on the mandala's colours, and offers its image to save.
    for key, value in hash_["colours"].items():
        assert f"--{key}: {value};" in page.text
    assert f'href="/{visual_hash_id}.png?download=1"' in page.text


def test_the_form_without_javascript_goes_to_the_ids_page():
    response = client.get("/", params={"name": "Denise Arias"}, follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"] == f"http://testserver/{main.hash_id('Denise Arias')}"


def test_api_returns_the_visual_hash_with_the_name():
    api = client.post("/api/visual-hash", data={"name": "Denise Arias"})
    assert api.status_code == 200
    assert api.json() == {**main.visual_hash(main.hash_id("Denise Arias")), "name": "Denise Arias"}


def test_api_escapes_nothing_into_pages():
    response = client.post("/api/visual-hash", data={"name": '"><script>alert(1)</script>'})
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"


def test_layers_and_image_by_id():
    visual_hash_id = main.hash_id("Budi")
    hash_ = main.visual_hash(visual_hash_id)
    expected = main.animation_layers("Budi")
    assert [(layer["size"], layer["spin"]) for layer in hash_["layers"]] == [(layer.size, layer.spin) for layer in expected]
    for level, (layer, png) in enumerate(zip(hash_["layers"], (layer.png for layer in expected))):
        assert layer["src"] == f"/{visual_hash_id}/layers/{level}.png?v={hashlib.sha256(png).hexdigest()[:12]}"
        response = client.get(layer["src"])
        assert response.headers["content-type"] == "image/png"
        # The address changes with the image, so keeping it for good is safe.
        assert "immutable" in response.headers["cache-control"]
        assert response.content == png
    assert client.get(f"/{visual_hash_id}/layers/9.png").status_code == 404
    image = client.get(f"/{visual_hash_id}.png")
    assert image.headers["content-type"] == "image/png" and "content-disposition" not in image.headers
    assert image.content == main.render_visual_hash_png("Budi")
    saved = client.get(f"/{visual_hash_id}.png", params={"download": 1})
    assert saved.headers["content-disposition"] == f'attachment; filename="visual-hash-{visual_hash_id}.png"'
    assert saved.content.startswith(PNG_SIGNATURE)


@pytest.mark.parametrize("path", ["/not-an-id", "/1234-5678", "/ABCD-1234-5678-9abc", "/zzzz-zzzz-zzzz-zzzz", "/zzzz-zzzz-zzzz-zzzz.png", "/1234-5678.png"])
def test_anything_else_is_not_found(path):
    assert client.get(path).status_code == 404


def test_link_previews_show_the_image():
    visual_hash_id = main.hash_id("Budi Santoso")
    page, _ = visual_hash_page("Budi Santoso")
    assert f'<meta property="og:url" content="http://testserver/{visual_hash_id}" />' in page.text
    image_url = re.search(r'<meta property="og:image" content="([^"]*)"', page.text).group(1)
    assert image_url == f"http://testserver/{visual_hash_id}.png"
    assert client.get(image_url).content == main.render_visual_hash_png("Budi Santoso")


@pytest.mark.parametrize("name", NAMES + [f"name {i}" for i in range(40)])
def test_page_colours_come_from_the_palette_and_stay_readable(name):
    _, palette = main.golden_visual_hash(name)
    colours = main.page_colours(palette)
    for key in ("accent", "accent-2"):
        rgb = [int(colours[key][i:i + 2], 16) / 255 for i in (1, 3, 5)]
        # White button text stays readable (WCAG AA) whatever the hue ...
        assert main.contrast_with_white(rgb) >= 4.5
        # ... and the colour is one of the palette's hues.
        hue = colorsys.rgb_to_hls(*rgb)[0] * 360
        match = min(palette, key=lambda layer: abs((hue - layer.hue + 180) % 360 - 180))
        assert abs((hue - match.hue + 180) % 360 - 180) < 3


def test_timeline_replays_the_triangles_of_the_finished_image():
    # The page draws the timeline with the rules of glow_triangles; check its data gives them back.
    points, palette = main.golden_visual_hash("Budi")
    hash_ = main.visual_hash(main.hash_id("Budi"))
    data = hash_["timeline"]
    assert np.allclose(data["points"], points, atol=0.005)
    # The page leaves out the same faint triangles as the server, and tone-maps like it.
    assert data["minAlpha"] == main.MIN_GLOW
    assert data["seconds"] == main.TIMELINE_SECONDS and main.TIMELINE_SECONDS in FIBONACCI
    assert data["tonemap"] == {"exposure": main.animate(main.name_to_seed("Budi")).exposure, "key": main.TONEMAP_KEY}
    assert hash_["harmonics"] == main.golden_harmonics("Budi")
    assert [(layer["hue"], layer["saturation"], layer["offset"]) for layer in data["layers"]] == [
        (layer.hue, layer.saturation, layer.offset) for layer in palette
    ]
    center = main.CANVAS_SIZE / 2
    expected = list(main.golden_nest(points))
    assert len(data["levels"]) == len(expected)
    for level, (nested, scale) in zip(data["levels"], expected):
        source = np.array(data["points"])[::-1] if level["backwards"] else np.array(data["points"])
        replayed = center + (source - center) @ np.array(level["matrix"]).reshape(2, 2).T
        assert np.allclose(replayed, nested, atol=0.01)
        assert level["scale"] == pytest.approx(scale)


@pytest.mark.parametrize("name", ["Budi", "Siti Rahma", "Ani"])
def test_server_light_is_all_the_triangles_light(name):
    # Every triangle that is meant to glow adds exactly its light: nothing is dropped silently (as
    # Skia does with alphas that round to 0 in 8 bits), so the page's WebGL replay can match it.
    points, layers = main.golden_visual_hash(name)
    measured = sum(light.sum(axis=(0, 1)) for light in main.nest_lights(points, layers))
    expected = np.zeros(3)
    for nested, scale in main.golden_nest(points):
        for triangles, layer, alphas in main.glow_triangles(nested, layers, scale, main.MIN_GLOW):
            x, y = np.moveaxis(triangles, 2, 0)
            area = 0.5 * np.abs(x[:, 0] * (y[:, 1] - y[:, 2]) + x[:, 1] * (y[:, 2] - y[:, 0]) + x[:, 2] * (y[:, 0] - y[:, 1]))
            colour = np.array(colorsys.hls_to_rgb(layer.hue / 360, 0.4, layer.saturation / 100))
            expected += colour * (alphas * area).sum() * main.RENDER_SCALE ** 2
    assert measured == pytest.approx(expected, rel=0.02)


@pytest.mark.parametrize("name", NAMES)
def test_golden_signature_tells_the_golden_choices(name):
    signature = main.golden_signature(name)
    turns, winding, harmonics, hues, triangles = signature.split(" · ")
    points, layers = main.golden_visual_hash(name)
    assert int(turns.split()[0]) in main.FIBONACCI_TURNS
    numerator, denominator = map(int, winding.split()[1].split("/"))
    assert numerator in main.FIBONACCI and denominator == int(turns.split()[0])
    assert {int(f) for f in harmonics.split(" ", 1)[1].split(", ")} <= set(FIBONACCI)
    assert float(hues.split()[1].rstrip("°")) in [round(step, 1) for step in main.GOLDEN_HUE_STEPS]
    assert [int(t) for t in triangles.split(" ", 1)[1].split(", ")] == [layer.offset for layer in layers]
    assert main.visual_hash(main.hash_id(name))["signature"] == signature


def test_pages_are_compressed():
    page = client.get(f"/{main.hash_id('Budi')}", headers={"Accept-Encoding": "gzip"})
    assert page.headers["content-encoding"] == "gzip"
    assert "base64" not in page.text  # the images come apart
    api = client.post("/api/visual-hash", data={"name": "Budi"}, headers={"Accept-Encoding": "gzip"})
    assert api.headers["content-encoding"] == "gzip"


def test_page_needs_nothing_from_other_sites():
    page = client.get(f"/{main.hash_id('Budi')}").text
    assert "googleapis" not in page and "gstatic" not in page
    for font in ("cormorant-garamond-500.woff2", "cormorant-garamond-400-italic.woff2", "OFL.txt"):
        assert client.get(f"/static/fonts/{font}").status_code == 200


def test_names_are_at_most_100_characters():
    longest = "x" * main.MAX_NAME_LENGTH
    assert client.post("/api/visual-hash", data={"name": longest}).status_code == 200
    for request in (
        lambda name: client.post("/api/visual-hash", data={"name": name}),
        lambda name: client.get("/", params={"name": name}),
    ):
        assert request(longest + "x").status_code == 422
    assert f'maxlength="{main.MAX_NAME_LENGTH}"' in client.get("/").text


def test_visual_hashes_are_cached():
    main.visual_hash.cache_clear()
    main.render_seed_png.cache_clear()
    visual_hash_id = main.hash_id("Cache Test")
    first = main.visual_hash(visual_hash_id)
    assert main.visual_hash(visual_hash_id) is first
    assert main.render_visual_hash_png("Cache Test") is main.render_visual_hash_png("Cache Test")
    assert main.visual_hash.cache_info().hits == 1


def test_home_has_no_visual_hash_yet():
    page = client.get("/")
    assert "data-hash='null'" in page.text
    assert 'aria-disabled="true"' in page.text  # nothing to save yet


# Fingerprints of known images, so an upgrade of numpy or Skia (or an accidental change to the
# algorithm) cannot silently change what a name looks like. After an intended change, regenerate
# them with: UPDATE_SNAPSHOTS=1 pytest -k snapshot
SNAPSHOTS = Path(__file__).parent / "fixtures" / "snapshots.json"
SNAPSHOT_NAMES = ["Budi", "Ani", "李小龙 🐉", "Gun Gun Feb"]


def fingerprint(png, blocks=32):
    """Average colour of each block of a blocks x blocks grid: small, and stable to rounding."""
    pixels = np.array(skia.Image.MakeFromEncoded(png).toarray(colorType=skia.kRGBA_8888_ColorType))[:, :, :3]
    size = pixels.shape[0] // blocks
    return pixels.reshape(blocks, size, blocks, size, 3).mean(axis=(1, 3))


def test_snapshots():
    current = {name: fingerprint(main.render_visual_hash_png(name)) for name in SNAPSHOT_NAMES}
    if os.environ.get("UPDATE_SNAPSHOTS"):
        SNAPSHOTS.write_text(json.dumps({key: np.round(value).astype(int).tolist() for key, value in current.items()}))
    expected = json.loads(SNAPSHOTS.read_text())
    assert set(expected) == set(current)
    for key, value in current.items():
        difference = np.abs(value - np.array(expected[key]))
        assert difference.max() <= 3, f"{key} looks different from its snapshot"
