document.addEventListener("DOMContentLoaded", function () {
  const $ = (id) => document.getElementById(id);
  const canvas = $("visual-hash-canvas");
  const form = $("generate-form");
  const inputBox = $("name");
  const generateButton = $("generate-btn");
  const saveButton = $("save-btn");
  const shareButton = $("share-btn");
  const player = $("player");
  const playButton = $("play-btn");
  const scrubber = $("scrubber");
  const soundButton = $("sound-btn");
  const recordButton = $("record-btn");
  const signature = $("signature");
  const hashId = $("hash-id");
  const placeholder = document.querySelector(".placeholder");
  const status = $("status");
  const ctx = canvas.getContext("2d");
  const fadeBuffer = document.createElement("canvas"); // the incoming image, during a cross-fade

  const CANVAS_UNITS = 400; // the server's coordinates
  const PHI = (1 + Math.sqrt(5)) / 2;
  const degreesPerSecond = 9; // the mandala's turn; every layer turns at its own multiple (its spin)
  const fadeMilliseconds = 700;
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  // The visual hash on show, and the scene drawing it now (and the one fading out behind it).
  // Each scene keeps two clocks: `age`, wall time since it appeared (for cross-fades), and
  // `elapsed`, which stops while paused or scrubbing (for its own motion).
  let hash = null;
  let current = null;
  let previous = null;
  let paused = reducedMotion; // with reduced motion the image stands still until asked to move
  let scrubbing = false;
  let lastTime = null;
  let sound = null; // the drawing's music, once switched on (see createSound)
  let recording = null;
  let lightRenderer; // WebGL light (see light.js), made on first use; null where unsupported

  function getLightRenderer() {
    if (lightRenderer === undefined) lightRenderer = createLightRenderer();
    return lightRenderer;
  }

  // The stem of a saved file's name: the person's name when the page knows it, otherwise the id.
  function fileStem(visualHash) {
    return visualHash.name ? `${slug(visualHash.name)}_visual_hash` : `visual-hash-${visualHash.id}`;
  }

  function slug(name) {
    // As main.safe_filename on the server.
    return name.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "") || "visual-hash";
  }

  function loadLayers(layers) {
    return Promise.all(
      layers.map(
        (layer) =>
          new Promise((resolve, reject) => {
            const image = new Image();
            image.onload = () => resolve({ ...layer, image });
            image.onerror = reject;
            image.src = layer.src;
          }),
      ),
    );
  }

  function resize() {
    // Match the canvas to its on-screen size and pixel density for a sharp image.
    const size = Math.round(canvas.clientWidth * window.devicePixelRatio);
    if (size === canvas.width) return;
    canvas.width = canvas.height = size;
    fadeBuffer.width = fadeBuffer.height = size;
    if (current && current.resize) current.resize();
  }

  // How far the mandala has turned after `elapsed` ms, in degrees. It starts still, so its first
  // frame is the downloadable image, and eases up to full speed over phi^2 (about 2.6) seconds.
  const spinUpMilliseconds = PHI * PHI * 1000;
  function turnedAfter(elapsed) {
    const u = elapsed / spinUpMilliseconds;
    const ramp = u < 1 ? u ** 3 - u ** 4 / 2 : u - 0.5; // the integral of a smoothstep speed-up
    return (degreesPerSecond * spinUpMilliseconds * ramp) / 1000;
  }

  // Scene: the finished image, its layers turning at their own speeds.
  function spinScene(layers) {
    return {
      kind: "spin",
      draw(c, alpha) {
        const size = canvas.width;
        const turned = turnedAfter(this.elapsed);
        c.globalAlpha = alpha;
        for (const layer of layers) {
          const width = size * layer.size;
          c.save();
          c.translate(size / 2, size / 2);
          c.rotate((turned * layer.spin * Math.PI) / 180);
          c.drawImage(layer.image, -width / 2, -width / 2, width, width);
          c.restore();
        }
      },
    };
  }

  // The triangles of the image, in the order the curve reaches them: the same rules as the
  // server's glow_triangles, applied to every level of the golden nest.
  function timelineTriangles(timeline) {
    const center = CANVAS_UNITS / 2;
    const count = timeline.points.length;
    const triangles = [];
    const paths = [];
    for (const level of timeline.levels) {
      const [a, b, c, d] = level.matrix;
      const source = level.backwards ? [...timeline.points].reverse() : timeline.points;
      const points = source.map(([x, y]) => [center + a * (x - center) + b * (y - center), center + c * (x - center) + d * (y - center)]);
      paths.push(points);
      const area2 = level.scale * level.scale;
      for (const layer of timeline.layers) {
        const colour = `hsl(${layer.hue}, ${layer.saturation}%, 40%)`;
        const rgb = hlsToRgb(layer.hue / 360, 0.4, layer.saturation / 100);
        for (let i = 0; i < count; i++) {
          const p = points[i];
          const q = points[(i + layer.offset) % count];
          const r = points[(i + 2 * layer.offset) % count];
          const area = p[0] * (q[1] - r[1]) + q[0] * (r[1] - p[1]) + r[0] * (p[1] - q[1]);
          const strength = Math.min(1, (55 * area2) / area);
          if (area > 45 * area2 && area < 8000 * area2 && strength >= timeline.minAlpha) {
            const light = rgb.map((channel) => channel * strength);
            triangles.push({ i, p, q, r, light, colour, alpha: strength * timeline.gain });
          }
        }
      }
    }
    triangles.sort((x, y) => x.i - y.i);
    return { triangles, paths, count };
  }

  // Scene: the image drawn from its first point, triangle by triangle along the curve, with a pen
  // of light at the tip of every level of the nest. With WebGL the light adds up and is tone-mapped
  // exactly as on the server, so the last frame is the finished image; otherwise a 2D canvas adds
  // it up to white.
  function timelineScene(timeline) {
    const { triangles, paths, count } = timelineTriangles(timeline);
    const duration = timeline.seconds * 1000;
    // The pen eases in and out: its first strokes and the last ones are slow enough to follow.
    const ease = (t) => (1 - Math.cos(Math.PI * t)) / 2;
    const renderer = getLightRenderer();
    const vertices = renderer ? lightVertices(triangles) : null;
    const ink = renderer ? renderer.canvas : document.createElement("canvas");
    const inkCtx = renderer ? null : ink.getContext("2d");
    let drawn = 0; // triangles added to the ink so far
    let reachedSoFar = 0;

    function clearInk() {
      drawn = 0;
      reachedSoFar = 0;
      if (renderer) {
        renderer.reset(canvas.width, vertices);
        return;
      }
      ink.width = ink.height = canvas.width;
      inkCtx.fillStyle = "#000";
      inkCtx.fillRect(0, 0, ink.width, ink.height);
    }

    function inkUpTo(reached) {
      // Scrubbing back means starting the drawing again.
      if (reached < reachedSoFar) clearInk();
      reachedSoFar = reached;
      let end = drawn;
      while (end < triangles.length && triangles[end].i < reached) end++;
      if (renderer) {
        renderer.add(drawn, end - drawn);
        renderer.present(timeline.tonemap);
        drawn = end;
        return;
      }
      const k = ink.width / CANVAS_UNITS;
      inkCtx.setTransform(k, 0, 0, k, 0, 0);
      inkCtx.globalCompositeOperation = "lighter";
      for (; drawn < end; drawn++) {
        const { p, q, r, colour, alpha } = triangles[drawn];
        inkCtx.globalAlpha = alpha;
        inkCtx.fillStyle = colour;
        inkCtx.beginPath();
        inkCtx.moveTo(p[0], p[1]);
        inkCtx.lineTo(q[0], q[1]);
        inkCtx.lineTo(r[0], r[1]);
        inkCtx.closePath();
        inkCtx.fill();
      }
    }

    function drawPens(c, reached, progress, alpha) {
      const k = canvas.width / CANVAS_UNITS;
      // The pens and their thread fade out over the last part, leaving only the finished image.
      alpha *= Math.min(1, (1 - progress) / 0.15);
      c.save();
      c.setTransform(k, 0, 0, k, 0, 0);
      // The thread the pen has drawn so far, along the outer curve.
      c.globalAlpha = 0.3 * alpha;
      c.strokeStyle = "#fff";
      c.lineWidth = 0.5;
      c.beginPath();
      paths[0].slice(0, reached + 1).forEach(([x, y], i) => (i ? c.lineTo(x, y) : c.moveTo(x, y)));
      c.stroke();
      // A pen of light at the tip of each level.
      c.globalAlpha = alpha;
      paths.forEach((points, level) => {
        const [x, y] = points[Math.min(reached, count - 1)];
        const radius = 6 / (level + 1);
        const glow = c.createRadialGradient(x, y, 0, x, y, radius);
        glow.addColorStop(0, "rgba(255, 255, 255, 1)");
        glow.addColorStop(1, "rgba(255, 255, 255, 0)");
        c.fillStyle = glow;
        c.beginPath();
        c.arc(x, y, radius, 0, 2 * Math.PI);
        c.fill();
      });
      c.restore();
    }

    clearInk();
    return {
      kind: "timeline",
      // How far through its time the drawing is, 0 to 1 (what the timeline shows).
      time() {
        return Math.min(1, this.elapsed / duration);
      },
      // How much of the curve is drawn, 0 to 1.
      progress() {
        return ease(this.time());
      },
      seek(time) {
        this.elapsed = time * duration;
      },
      // How far the pen of the outer curve is from the centre, 0 to 1 (for the music).
      reach() {
        const [x, y] = paths[0][Math.min(Math.floor(this.progress() * count), count - 1)];
        return Math.hypot(x - CANVAS_UNITS / 2, y - CANVAS_UNITS / 2) / (CANVAS_UNITS / 2);
      },
      resize() {
        clearInk(); // inked again up to where it was on the next frame
      },
      draw(c, alpha) {
        const progress = this.progress();
        const reached = Math.floor(progress * count);
        inkUpTo(progress >= 1 ? count : reached);
        c.globalAlpha = alpha;
        c.drawImage(ink, 0, 0, canvas.width, canvas.height);
        if (progress < 1) drawPens(c, reached, progress, alpha);
      },
    };
  }

  function play(scene, { fade = true } = {}) {
    previous = fade ? current : null;
    current = scene;
    current.age = fade ? 0 : fadeMilliseconds;
    current.elapsed = 0;
    canvas.classList.toggle("playing", scene.kind === "timeline");
  }

  function setPaused(value) {
    paused = value;
    player.classList.toggle("paused", paused);
    playButton.setAttribute("aria-label", paused ? "Play" : "Pause");
  }

  // Draw a scene on black, its layers adding up their light the way the server's tone mapping does.
  function paint(c, scene, alpha) {
    c.globalCompositeOperation = "source-over";
    c.globalAlpha = 1;
    c.fillStyle = "#000";
    c.fillRect(0, 0, canvas.width, canvas.height);
    c.globalCompositeOperation = "screen";
    scene.draw(c, alpha);
  }

  function frame(time) {
    const dt = lastTime === null ? 0 : Math.min(100, time - lastTime);
    lastTime = time;
    for (const scene of [current, previous]) {
      if (!scene) continue;
      scene.age += dt;
      if (!paused && !scrubbing) scene.elapsed += dt;
    }

    const fade = Math.min(1, current.age / fadeMilliseconds);
    if (previous && fade < 1) {
      // Cross-fade by mixing the two images, (1 - f) old + f new. Fading both while adding their
      // light ("screen") would dim the picture halfway through, as two near-identical images do.
      paint(ctx, previous, 1);
      paint(fadeBuffer.getContext("2d"), current, 1);
      ctx.globalCompositeOperation = "source-over";
      ctx.globalAlpha = fade;
      ctx.drawImage(fadeBuffer, 0, 0);
    } else {
      previous = null;
      paint(ctx, current, fade); // the first image fades in from black
    }

    const drawing = current.kind === "timeline";
    if (!scrubbing) scrubber.value = drawing ? Math.round(current.time() * 1000) : 1000;
    if (sound) sound.update(drawing && !paused && !scrubbing ? current.reach() : null);
    // When the drawing is complete it melts into the finished, turning image.
    if (drawing && current.time() >= 1 && !paused && !scrubbing) {
      play(spinScene(hash.layers));
      if (recording) recording.finish();
    }
    requestAnimationFrame(frame);
  }

  function startDrawing() {
    setPaused(false);
    play(timelineScene(hash.timeline));
  }

  // Clicking the image replays how it was drawn; clicking again skips to the end.
  function toggleTimeline() {
    if (!hash || recording) return;
    if (current.kind === "timeline") play(spinScene(hash.layers));
    else startDrawing();
  }

  // The drawing's music: the curve's own Fibonacci harmonics over a root note set by its first
  // colour. The further the pen swings from the centre, the brighter the chord.
  function createSound(harmonics, hue) {
    const audio = new AudioContext();
    const master = audio.createGain();
    master.gain.value = 0;
    master.connect(audio.destination);
    const stream = audio.createMediaStreamDestination(); // for recordings
    master.connect(stream);
    const root = 110 * 2 ** (hue / 360); // somewhere between A2 and A3
    const partials = harmonics.map((harmonic) => {
      const oscillator = audio.createOscillator();
      oscillator.frequency.value = root * harmonic;
      const gain = audio.createGain();
      gain.gain.value = 0;
      oscillator.connect(gain).connect(master);
      oscillator.start();
      return { harmonic, gain };
    });
    return {
      stream,
      update(reach) {
        const now = audio.currentTime;
        master.gain.setTargetAtTime(reach === null ? 0 : 0.12, now, 0.08);
        if (reach === null) return;
        for (const { harmonic, gain } of partials) {
          // Higher harmonics only open up as the pen swings out: (reach)^(steps up the phi scale).
          const brightness = reach ** (Math.log(harmonic) / Math.log(PHI) / 2);
          gain.gain.setTargetAtTime(brightness / harmonic ** 0.7 / partials.length, now, 0.05);
        }
      },
      close() {
        audio.close();
      },
    };
  }

  function setSound(on) {
    if (sound) sound.close();
    sound = on && hash ? createSound(hash.harmonics, hash.timeline.layers[0].hue) : null;
    soundButton.setAttribute("aria-pressed", String(on));
    player.classList.toggle("sound-on", on);
  }

  // Record the drawing, from its first point to a moment of the finished image, as a video.
  function record() {
    if (!hash || recording) return;
    const type = ["video/webm;codecs=vp9", "video/webm", "video/mp4"].find((t) => MediaRecorder.isTypeSupported(t));
    const stream = canvas.captureStream(60);
    if (sound) sound.stream.stream.getAudioTracks().forEach((track) => stream.addTrack(track));
    const recorder = new MediaRecorder(stream, { mimeType: type, videoBitsPerSecond: 8_000_000 });
    const chunks = [];
    const stem = fileStem(hash);
    recorder.ondataavailable = (event) => chunks.push(event.data);
    recorder.onstop = () => {
      const link = document.createElement("a");
      link.href = URL.createObjectURL(new Blob(chunks, { type }));
      link.download = `${stem}.${type.startsWith("video/mp4") ? "mp4" : "webm"}`;
      link.click();
      URL.revokeObjectURL(link.href);
      recording = null;
      recordButton.disabled = false;
      player.classList.remove("recording");
      status.textContent = "Video saved";
    };
    recording = {
      finish() {
        setTimeout(() => recorder.stop(), 1500); // and a moment of the finished image
      },
    };
    recordButton.disabled = true;
    player.classList.add("recording");
    status.textContent = "Recording the drawing…";
    setPaused(false);
    play(timelineScene(hash.timeline), { fade: false });
    recorder.start();
  }

  async function share() {
    const url = `${location.origin}/${hash.id}`;
    // Phones get their share sheet; elsewhere the link is copied.
    if (navigator.share && window.matchMedia("(pointer: coarse)").matches) {
      navigator.share({ title: `A golden-ratio visual hash · ${hash.id}`, url }).catch(() => {});
      return;
    }
    await navigator.clipboard.writeText(url);
    shareButton.textContent = "Link copied";
    status.textContent = "Link copied";
    setTimeout(() => (shareButton.textContent = "Share link"), 2000);
  }

  async function show(newHash) {
    const layers = await loadLayers(newHash.layers);
    hash = { ...newHash, layers };
    play(spinScene(layers));
    placeholder.hidden = true;
    player.hidden = false;
    signature.textContent = hash.signature;
    hashId.textContent = hash.id;
    canvas.classList.add("visible");
    canvas.tabIndex = 0;
    if (sound) setSound(true); // the new curve's harmonics
    if (lastTime === null && !show.started) {
      show.started = true;
      resize();
      window.addEventListener("resize", resize);
      requestAnimationFrame(frame);
    }
  }

  function applyColours(colours) {
    for (const [key, value] of Object.entries(colours)) {
      document.body.style.setProperty(`--${key}`, value);
    }
  }

  async function generate(event) {
    event.preventDefault();
    if (!form.reportValidity() || recording) return;
    generateButton.disabled = true;
    generateButton.classList.add("loading");
    status.textContent = "Generating…";
    try {
      const response = await fetch("/api/visual-hash", { method: "POST", body: new FormData(form) });
      if (!response.ok) throw new Error(response.statusText);
      const newHash = await response.json();
      await show(newHash);
      applyColours(newHash.colours);
      saveButton.href = `/${newHash.id}.png?download=1`;
      saveButton.download = `${fileStem(newHash)}.png`;
      saveButton.removeAttribute("aria-disabled");
      shareButton.disabled = false;
      canvas.setAttribute("aria-label", `Visual hash of ${newHash.name}. Press to watch it being drawn.`);
      // Its own address, ready to share: the id, not the name.
      history.replaceState(null, "", `/${newHash.id}`);
      document.title = `${newHash.name} · Visual Hashing`;
      status.textContent = "";
    } catch (error) {
      // Fall back to the plain form, which renders the whole page on the server.
      form.submit();
    } finally {
      generateButton.disabled = false;
      generateButton.classList.remove("loading");
    }
  }

  form.addEventListener("submit", generate);
  $("random-name-btn").addEventListener("click", async () => {
    const response = await fetch("/random-name");
    inputBox.value = await response.json();
  });
  shareButton.addEventListener("click", share);
  canvas.addEventListener("click", toggleTimeline);
  canvas.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      toggleTimeline();
    }
  });
  playButton.addEventListener("click", () => setPaused(!paused));
  soundButton.addEventListener("click", () => setSound(!sound));
  scrubber.addEventListener("input", () => {
    if (!hash || recording) return;
    if (!scrubbing) {
      scrubbing = true;
      if (current.kind !== "timeline") play(timelineScene(hash.timeline));
    }
    current.seek(scrubber.value / 1000);
  });
  scrubber.addEventListener("change", () => (scrubbing = false));
  if (window.MediaRecorder && canvas.captureStream) recordButton.addEventListener("click", record);
  else recordButton.hidden = true;

  setPaused(paused);
  const initial = JSON.parse(canvas.dataset.hash);
  if (initial) show(initial);
});
