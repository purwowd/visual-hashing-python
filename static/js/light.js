// Adds up light the way the server does: unclipped, in floating point, then tone-mapped with the
// same formula (main.tonemap) and the finished image's exposure. A 2D canvas can only add light
// up to white, one byte per channel, which washes colours out and loses the faintest triangles.
//
// createLightRenderer() returns null where WebGL2 with float colour buffers is not available.

// colorsys.hls_to_rgb, as the server colours its triangles.
function hlsToRgb(h, l, s) {
  if (s === 0) return [l, l, l];
  const m2 = l <= 0.5 ? l * (1 + s) : l + s - l * s;
  const m1 = 2 * l - m2;
  const channel = (hue) => {
    hue = ((hue % 1) + 1) % 1;
    if (hue < 1 / 6) return m1 + (m2 - m1) * hue * 6;
    if (hue < 0.5) return m2;
    if (hue < 2 / 3) return m1 + (m2 - m1) * (2 / 3 - hue) * 6;
    return m1;
  };
  return [channel(h + 1 / 3), channel(h), channel(h - 1 / 3)];
}

const ADD_VERTEX = `#version 300 es
in vec2 position;  // canvas units, 0 to 400, y down
in vec3 light;     // colour times alpha
out vec3 vLight;
void main() {
  vLight = light;
  gl_Position = vec4(position.x / 200.0 - 1.0, 1.0 - position.y / 200.0, 0.0, 1.0);
}`;

const ADD_FRAGMENT = `#version 300 es
precision highp float;
in vec3 vLight;
out vec4 colour;
void main() { colour = vec4(vLight, 1.0); }`;

const TONEMAP_VERTEX = `#version 300 es
void main() {
  // One triangle that covers the screen.
  vec2 corner = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  gl_Position = vec4(corner * 2.0 - 1.0, 0.0, 1.0);
}`;

// main.tonemap, per pixel; each screen pixel averages the 2x2 light pixels under it (antialiasing).
const TONEMAP_FRAGMENT = `#version 300 es
precision highp float;
uniform sampler2D light;
uniform bool hdr;
uniform float exposure;
uniform float key;
out vec4 colour;
vec3 tonemap(vec3 rgb) {
  if (!hdr) return min(rgb, 1.0);
  float peak = max(rgb.r, max(rgb.g, rgb.b));
  float exposed = peak * exposure;
  float mapped = 1.0 - exp(-exposed);
  vec3 kept = peak > 0.0 ? rgb * (mapped / max(peak, 1e-9)) : vec3(0.0);
  float whiteness = clamp((exposed - 0.6 * key) / 4.0, 0.0, 0.5);
  return kept + (mapped - kept) * whiteness;
}
void main() {
  ivec2 base = ivec2(gl_FragCoord.xy) * 2;
  vec3 sum = vec3(0.0);
  for (int dy = 0; dy < 2; dy++)
    for (int dx = 0; dx < 2; dx++)
      sum += tonemap(texelFetch(light, base + ivec2(dx, dy), 0).rgb);
  colour = vec4(sum / 4.0, 1.0);
}`;

function compile(gl, vertexSource, fragmentSource) {
  const program = gl.createProgram();
  for (const [type, source] of [[gl.VERTEX_SHADER, vertexSource], [gl.FRAGMENT_SHADER, fragmentSource]]) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader));
    gl.attachShader(program, shader);
  }
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
  return program;
}

function createLightRenderer() {
  const canvas = document.createElement("canvas");
  const gl = canvas.getContext("webgl2", { antialias: false, premultipliedAlpha: false });
  if (!gl || !gl.getExtension("EXT_color_buffer_float")) return null;
  // 32-bit float light keeps the faintest triangles; blending it needs EXT_float_blend.
  const format = gl.getExtension("EXT_float_blend") ? gl.RGBA32F : gl.RGBA16F;
  let add;
  let tonemap;
  try {
    add = compile(gl, ADD_VERTEX, ADD_FRAGMENT);
    tonemap = compile(gl, TONEMAP_VERTEX, TONEMAP_FRAGMENT);
  } catch (error) {
    return null;
  }
  const buffer = gl.createBuffer();
  const vertexArray = gl.createVertexArray();
  const texture = gl.createTexture();
  const framebuffer = gl.createFramebuffer();
  let size = 0;

  gl.bindVertexArray(vertexArray);
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  const position = gl.getAttribLocation(add, "position");
  const light = gl.getAttribLocation(add, "light");
  gl.enableVertexAttribArray(position);
  gl.vertexAttribPointer(position, 2, gl.FLOAT, false, 20, 0);
  gl.enableVertexAttribArray(light);
  gl.vertexAttribPointer(light, 3, gl.FLOAT, false, 20, 8);

  return {
    canvas,

    // Start a new drawing at `pixels` wide, whose triangles are `vertices`: x, y, r, g, b each.
    reset(pixels, vertices) {
      size = pixels;
      canvas.width = canvas.height = size;
      gl.bindTexture(gl.TEXTURE_2D, texture);
      gl.texImage2D(gl.TEXTURE_2D, 0, format, size * 2, size * 2, 0, gl.RGBA, gl.FLOAT, null);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
      gl.clearColor(0, 0, 0, 1);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
      gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);
    },

    // Add the light of triangles first .. first + count - 1.
    add(first, count) {
      if (count <= 0) return;
      gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
      gl.viewport(0, 0, size * 2, size * 2);
      gl.useProgram(add);
      gl.bindVertexArray(vertexArray);
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.ONE, gl.ONE);
      gl.drawArrays(gl.TRIANGLES, first * 3, count * 3);
      gl.disable(gl.BLEND);
    },

    // Tone-map the light so far onto the canvas: { exposure, key }, or null to clip at white.
    present(settings) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.viewport(0, 0, size, size);
      gl.useProgram(tonemap);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, texture);
      gl.uniform1i(gl.getUniformLocation(tonemap, "light"), 0);
      gl.uniform1i(gl.getUniformLocation(tonemap, "hdr"), settings ? 1 : 0);
      gl.uniform1f(gl.getUniformLocation(tonemap, "exposure"), settings ? settings.exposure : 1);
      gl.uniform1f(gl.getUniformLocation(tonemap, "key"), settings ? settings.key : 1);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    },
  };
}

// The vertices of the timeline's triangles for createLightRenderer: x, y, r, g, b per vertex, with
// the colour already multiplied by the triangle's alpha (light adds up premultiplied).
function lightVertices(triangles) {
  const vertices = new Float32Array(triangles.length * 15);
  triangles.forEach(({ p, q, r, light }, t) => {
    [p, q, r].forEach(([x, y], v) => vertices.set([x, y, ...light], t * 15 + v * 5));
  });
  return vertices;
}
