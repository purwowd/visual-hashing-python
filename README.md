# visual-hashing-python

![👀](https://visitor-badge.laobi.icu/badge?page_id=purwowd.visual-hashing-python)

A Python/FastAPI take on [Visual Hashing](https://github.com/gungunfebrianza/Visual-Hashing) by Gun Gun
Febrianza, re-tuned around the golden ratio: turn any name into a unique, deterministic glowing mandala.
The same name always produces the same image.

## Features

- Generate a glowing, symmetric "mandala" from any text, shown on a rotating canvas
- Download the result as an 800×800 PNG
- A page that follows φ as well: form and artwork panels 1 : φ wide, a title φ² the body text,
  Fibonacci spacing (8, 13, 21, 34, 55 px), and accents and a halo in each mandala's own colours
  (kept readable: white button text always has at least 4.5:1 contrast)
- New mandalas replace the old one in place, with a cross-fade and without reloading the page
  (`POST /api/visual-hash`); without JavaScript the plain form still works
- Click the artwork to watch it being drawn: from the curve's first point, the glowing triangles
  appear in the order the curve reaches them, on every level of the nest at once, with a pen of light
  at each tip, over 13 seconds (a Fibonacci number), easing in and out so the first strokes are slow
  enough to follow; when it is complete it melts back into the finished image, which starts turning
  gently, easing up to speed over φ² ≈ 2.6 seconds. Click again to skip.
  With WebGL the light adds up unclipped and is tone-mapped exactly as on the server, so the
  drawing's colours are the finished image's (a 2D canvas fallback adds light up to white)
- A player under the artwork: play / pause, a timeline to scrub through the drawing, sound (the
  curve's own Fibonacci harmonics over a root note set by its first colour, brighter as the pen
  swings out), and saving the drawing as a video (WebM)
- A golden signature under each image, e.g. *8 turns · winding 2/8 · harmonics 1, 2, 3, 5, 8, 13 ·
  hues 137.5° apart · triangles 3, 8, 21*
- Every visual hash has its own address to share, made of its id: `/cbfa-828e-c4c5-53dc`, with a link
  preview (`og:image`, served by `GET /cbfa-828e-c4c5-53dc.png`). The id is the image's 64-bit seed
  in hexadecimal, so the same name always gets the same id, the image is drawn from the id alone (no
  database), and a shared link does not give the name away
- Light and dark themes (following the system), a serif title, and no motion until asked for when
  the system asks for reduced motion
- Random name generator (via [Faker](https://faker.readthedocs.io/))

## How it works

1. **Curve**: a pseudo-random generator seeded by the name picks frequencies, amplitudes and phases for
   a nested sine/cosine curve of 4,181 points that winds around the centre N times.
2. **Glow**: for three colour layers, every point is joined with the points `offset` and `2 × offset`
   further along the curve into a translucent triangle. Small triangles are bright and large ones faint
   (alpha = 55 / area), and they are blended additively on black, so overlaps build up into light.
   Rendering uses [skia-python](https://github.com/kyamagu/skia-python), the same engine behind the
   browser canvas the original draws on.

### The golden ratio

The original seeds its generator from the characters of the name, which different names can share
(`Abcdefghijklb` and `Qbcdefghijklc` drew the same image). This one seeds the generator with SHA-256 of the name and uses φ throughout. Names that look the same give the
same image: the name is first normalised to Unicode NFC, case-folded (`Budi` = `BUDI`) and its extra
spaces are removed.

- **Fibonacci turns**: the curve winds around the centre N ∈ {5, 8, 13} times. This often gives
  N-fold symmetry, but not always: the other frequencies of the curve can break it.
- **Fibonacci harmonics**: the curve has F(19) = 4,181 points (the original has 4,620), and its
  nested waves only vibrate at Fibonacci frequencies, odd (1, 3, 5, 13) or even (0, 2, 8), like
  a chord built only from the notes of one scale.
- **Golden winding**: per wave the curve rotates F(j)/F(k) of a turn, a ratio of two Fibonacci numbers
  (3/5, 5/8, 8/13, …). These are the best rational approximations of powers of 1/φ; F(k−1)/F(k)
  approaches the golden mean itself, the same "most irrational" rotation that drives phyllotaxis.
- **Golden amplitudes**: each nested wave is about 1/φ (then 1/φ²) the size of the one before, and
  the inner waves are as strong as a power of φ (1, φ, φ², φ³), give or take 10%.
- **Golden hues**: the three colour layers sit 360°/φᵏ apart (k = 2, 3, 4: 137.5°, 85° or 52.5°),
  giving palettes from contrasting to harmonious. k = 2 is the golden angle.
- **Fibonacci offsets**: the triangle offsets of the three layers are every other Fibonacci number
  (2, 5, 13 or 3, 8, 21 or 5, 13, 34), so each layer's triangles are ~φ² larger than the last.
- **Golden nest**: the centre holds the mandala itself again, 1/φ² the size (the inner golden
  section of its radius), turned by the golden angle (137.5°, the turn a plant makes from one leaf
  to the next) and mirrored, so its curves wind the other way, as the two families of spirals on a
  sunflower head do. It is drawn with the same glowing triangles as the rest, so the centre flows in
  the same light. A mirror turns every triangle the other way round, and only counter-clockwise
  triangles glow, so the mirrored copy walks the curve backwards: it lights exactly the mirror image
  of the mandala's triangles, never fewer. (A second, smaller copy inside it was tried and left out:
  at 1/φ⁴ of the size it gave every image the same bright centre, and names became harder to tell
  apart; see the scoreboard below.)

  On the page the levels turn as well, each at its own speed: level k turns (−1/φ)ᵏ as fast as the
  mandala, so the core turns against it at 0.618×. φ is
  the most irrational number, so the levels never line up the same way twice: the animation never
  repeats. The server sends each level as its own image, tone-mapped with one shared exposure, and
  the page blends them in "screen" mode, 1 − (1 − a)(1 − b), which matches the tone mapping
  (1 − e^−(x+y) = 1 − e^−x · e^−y), so the animation starts on the image you download.

What is and is not from nature: the golden angle and Fibonacci numbers are how plants grow
(phyllotaxis). The golden amplitudes and the hue steps other than the golden angle are golden
proportions chosen for the eye. Despite the popular claim, nautilus shells and spiral galaxies are
logarithmic spirals that do not grow by φ.

It also fixes weaknesses of the original's rendering:

- **HDR glow**: light is accumulated without clipping and then tone-mapped with per-image
  auto-exposure, so dense areas keep their colour instead of blowing out to white.
- **No plain rings or empty canvases**: curves whose radius barely varies, or whose visible
  triangles would light up less than 55% of the disc, are re-drawn (deterministically).
- **Consistent size**: every curve is scaled to fill the canvas.

Scored against the original algorithm (which this app used to offer as a second style) over 1,000
random names, with 90% intervals; a difference only counts when the intervals do not overlap:

| Measure                                        | Original              | Golden                | Better   |
| ---------------------------------------------- | --------------------- | --------------------- | -------- |
| Distinct (median nearest-neighbour distance)   | 1.177 [1.191–1.211]   | 1.208 [1.214–1.239]   | golden   |
| Distinct by shape alone (brightness only)      | 1.064 [1.073–1.086]   | 1.114 [1.119–1.129]   | golden   |
| Distinct for colour-blind eyes (deuteranopia)  | 0.925 [0.939–0.954]   | 0.952 [0.963–0.977]   | golden   |
| Distinct at 64 px (avatar size)                | 1.039 [1.055–1.074]   | 1.036 [1.050–1.065]   | tie      |
| Distinct at 32 px                              | 0.935 [0.944–0.965]   | 0.953 [0.963–0.980]   | tie      |
| Canvas lit by the least-filled 5%              | 0.270 [0.248–0.295]   | 0.263 [0.256–0.268]   | tie      |
| Lit pixels blown out to white                  | 6.6%                  | 0.0%                  | golden   |
| Colourfulness (chroma)                         | 0.322 [0.317–0.327]   | 0.339 [0.336–0.342]   | golden   |
| First render (best of 3; then cached)          | 66.7 ms [63.2–81.9]   | 68.0 ms [65.7–71.9]   | tie      |

The golden ratio version is better on five measures and as good on the other four. Its choices were made
with the scoreboard: of the ideas tried, the one inner copy instead of two and a coverage threshold
of 55% instead of 35% helped; continuous golden amplitudes, hues turned per level, a larger inner copy
(1/φ) and fainter triangles did not, and were left out.

## Running locally

Requires Python 3.12+.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```

Then open http://localhost:8000.

## Running with Docker

```bash
docker compose up --build
```

## Endpoints

| Method | Path                           | Description                                                        |
| ------ | ------------------------------ | ------------------------------------------------------------------ |
| GET    | `/`                            | Web UI; `/?name=…` (the form without JavaScript) redirects to the id |
| GET    | `/{id}`                        | The page of a visual hash, e.g. `/cbfa-828e-c4c5-53dc`, to share   |
| GET    | `/{id}.png`                    | The image itself (link previews); `?download=1` to save it         |
| GET    | `/{id}/layers/{level}.png`     | One layer of the page's animation                                  |
| GET    | `/random-name`                 | A random name (JSON string)                                        |
| POST   | `/api/visual-hash`             | Form field `name`; the id, animation layers, page colours and drawing timeline as JSON |

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

`test_snapshots` compares a few names' images with fingerprints in `tests/fixtures/snapshots.json`, so
an upgrade of numpy or Skia cannot silently change what a name looks like. After an intended change
to the images, regenerate them with `UPDATE_SNAPSHOTS=1 pytest -k snapshot`.

`tests/test_browser.py` drives the page in Chrome with Playwright (generating, the drawing's timeline
and its colours, the cross-fade into the turning image, scrubbing, sharing, recording), on Playwright's
clock so a 13-second drawing is checked in milliseconds. It uses the installed Chrome; without one, run
`playwright install chromium`. Without Playwright it is skipped.

Every platform draws the same pixels for the same name: macOS (arm64) and the Docker image on Linux
arm64 and x86-64 were checked image by image. `scripts/digests.py` repeats the check (see its docstring).

To score the visual hash on every measure above (about a minute, on all cores):

```bash
python scripts/scoreboard.py --count 1000
```

`scripts/evaluate.py` is a quicker look.
