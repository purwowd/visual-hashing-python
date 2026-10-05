"""The page in a real browser: generating, the drawing's timeline, its colours and its transitions.

Runs Chrome (or Playwright's Chromium) against the app on a free port, with Playwright's clock, so
a 13-second drawing is checked in milliseconds. Skipped where Playwright or a browser is missing.
"""
import socket
import threading
import time

import numpy as np
import pytest
import skia
import uvicorn

import main

sync_api = pytest.importorskip("playwright.sync_api")


@pytest.fixture(scope="module")
def server():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    app_server = uvicorn.Server(uvicorn.Config(main.app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=app_server.run, daemon=True)
    thread.start()
    while not app_server.started:
        time.sleep(0.02)
    yield f"http://127.0.0.1:{port}"
    app_server.should_exit = True
    thread.join()


@pytest.fixture(scope="module")
def browser():
    with sync_api.sync_playwright() as playwright:
        for options in ({"channel": "chrome"}, {}):
            try:
                launched = playwright.chromium.launch(**options)
                break
            except Exception:
                launched = None
        if launched is None:
            pytest.skip("no Chrome or Chromium for Playwright")
        yield launched
        launched.close()


@pytest.fixture
def page(browser, server):
    context = browser.new_context(viewport={"width": 1100, "height": 760}, accept_downloads=True)
    context.grant_permissions(["clipboard-read", "clipboard-write"], origin=server)
    page = context.new_page()
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.clock.install()
    page.add_init_script(FRAME_SAMPLER)
    page.server = server
    yield page
    assert errors == []
    context.close()


# Samples the canvas right after each frame is drawn (installed over the clock's frames). Reading it
# between frames instead, with toDataURL, now and then catches it blank under Playwright's clock.
# It keeps the latest frame only, shrunk four times each way (averaging each 4 x 4 block).
FRAME_SAMPLER = """
window.__frame = null;
const nextFrame = window.requestAnimationFrame.bind(window);
window.requestAnimationFrame = (callback) => nextFrame((time) => {
  callback(time);
  const canvas = document.getElementById('visual-hash-canvas');
  if (!window.__sampling || !canvas) return;
  const size = canvas.width, data = canvas.getContext('2d').getImageData(0, 0, size, size).data;
  const small = Math.floor(size / 4), rgb = new Array(small * small * 3).fill(0);
  for (let y = 0; y < small * 4; y++)
    for (let x = 0; x < small * 4; x++)
      for (let c = 0; c < 3; c++) rgb[((y >> 2) * small + (x >> 2)) * 3 + c] += data[(y * size + x) * 4 + c] / 16;
  window.__frame = [small, rgb, size];
});
"""


def last_frame(page):
    """The latest frame drawn, sampled right after drawing it, as an (h, w, 3) array."""
    small, rgb, _ = page.evaluate("window.__frame")
    return np.array(rgb, float).reshape(small, small, 3)


def finished_image(name, size):
    """The downloadable image as the sampler sees the canvas: at its size, in 4 x 4 block averages."""
    image = skia.Image.MakeFromEncoded(main.render_visual_hash_png(name)).resize(size * 4, size * 4)
    pixels = np.array(image.toarray(colorType=skia.kRGBA_8888_ColorType))[:, :, :3].astype(float)
    return pixels.reshape(size, 4, size, 4, 3).mean(axis=(1, 3))


def open_hash(page, name):
    page.goto(f"{page.server}/{main.hash_id(name)}")
    page.wait_for_function("document.getElementById('visual-hash-canvas').classList.contains('visible')")
    page.evaluate("window.__sampling = true")
    page.clock.run_for(2000)


def has_webgl_light(page):
    return page.evaluate("createLightRenderer() !== null")


def test_generate_swaps_in_a_new_visual_hash_without_reloading(page):
    page.goto(page.server)
    page.evaluate("window.notReloaded = true")
    page.fill("#name", "Budi Santoso")
    page.click("#generate-btn")
    page.wait_for_function("document.getElementById('signature').textContent.length > 0")
    assert page.evaluate("window.notReloaded") is True
    assert page.url == f"{page.server}/{main.hash_id('Budi Santoso')}"  # the id, not the name
    assert page.text_content("#hash-id") == main.hash_id("Budi Santoso")
    assert page.get_attribute("#save-btn", "href") == f"/{main.hash_id('Budi Santoso')}.png?download=1"
    assert page.get_attribute("#save-btn", "download") == "budi-santoso_visual_hash.png"
    assert page.title() == "Budi Santoso · Visual Hashing"
    assert page.text_content("#signature") == main.golden_signature("Budi Santoso")
    assert page.is_enabled("#save-btn") and page.is_enabled("#share-btn")
    colours = main.visual_hash(main.hash_id("Budi Santoso"))["colours"]
    assert page.evaluate("document.body.style.getPropertyValue('--accent')") == colours["accent"]


def test_timeline_ends_on_the_finished_image(page):
    open_hash(page, "Siti Rahma")
    if not has_webgl_light(page):
        pytest.skip("no WebGL2 with float colour buffers")
    page.click("#visual-hash-canvas")
    page.clock.run_for(1000)
    early = last_frame(page)
    page.clock.run_for(main.TIMELINE_SECONDS * 1000 - 1050)
    last = last_frame(page)
    final = finished_image("Siti Rahma", last.shape[0])
    lit = final.max(axis=-1) > 30
    # It grows from the first point ...
    assert (early.max(axis=-1) > 30).mean() < 0.2 * lit.mean()
    # ... into exactly the finished image's colours (the light adds up and is tone-mapped as on the server).
    assert np.abs(last - final).mean() < 6
    assert np.abs(last[lit].mean(axis=0) - final[lit].mean(axis=0)).max() < 6


def test_timeline_melts_into_the_turning_image_smoothly(page):
    open_hash(page, "Siti Rahma")
    page.click("#visual-hash-canvas")
    page.clock.run_for(main.TIMELINE_SECONDS * 1000 - 500)
    frames = []
    for _ in range(30):
        page.clock.run_for(100)
        frames.append(last_frame(page))
    brightness = [frame[frame.max(axis=-1) > 25].mean() for frame in frames]
    # No dimming while the drawing cross-fades into the turning image ...
    assert min(brightness) > 0.97 * brightness[0]
    # ... which starts turning gently, then picks up speed.
    motion = [np.abs(b - a).mean() for a, b in zip(frames, frames[1:])]
    start = next(i for i, m in enumerate(motion) if m > 0.2)
    assert motion[start] < 0.25 * max(motion)
    assert page.evaluate("document.getElementById('visual-hash-canvas').classList.contains('playing')") is False


def test_scrubbing_and_pausing(page):
    open_hash(page, "Budi")
    page.click("#play-btn")  # pause
    page.evaluate("""() => {
        const scrubber = document.getElementById('scrubber');
        scrubber.value = 500;
        scrubber.dispatchEvent(new Event('input'));
        scrubber.dispatchEvent(new Event('change'));
    }""")
    page.clock.run_for(1500)
    assert page.get_attribute("#play-btn", "aria-label") == "Play"
    assert page.input_value("#scrubber") == "500"  # paused: the drawing stays where it was
    half = last_frame(page)
    final = finished_image("Budi", half.shape[0])
    assert 0.2 < (half.max(axis=-1) > 30).mean() / (final.max(axis=-1) > 30).mean() < 0.95
    page.click("#play-btn")
    page.clock.run_for(1500)
    assert int(page.input_value("#scrubber")) > 500


def test_share_copies_the_link(page):
    open_hash(page, "Budi Santoso")
    page.click("#share-btn")
    page.wait_for_function("document.getElementById('share-btn').textContent === 'Link copied'")
    assert page.evaluate("navigator.clipboard.readText()") == f"{page.server}/{main.hash_id('Budi Santoso')}"


def test_record_saves_the_drawing_as_a_video(page):
    open_hash(page, "Ani")
    if not page.evaluate("'MediaRecorder' in window"):
        pytest.skip("no MediaRecorder")
    with page.expect_download() as download:
        page.click("#record-btn")
        page.clock.run_for(main.TIMELINE_SECONDS * 1000 + 2000)
    # Opened by its id, the page does not know the name: the file is named by the id.
    stem = f"visual-hash-{main.hash_id('Ani')}"
    assert download.value.suggested_filename in (f"{stem}.webm", f"{stem}.mp4")
    assert download.value.path().stat().st_size > 0
