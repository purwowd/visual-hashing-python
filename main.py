import numpy as np
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse

from fastapi.templating import Jinja2Templates
import base64
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
from scipy.interpolate import CubicSpline


app = FastAPI()

templates = Jinja2Templates(directory="templates")


WIDTH = 500
HEIGHT = 500
CELL_SIZE = -1

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "image": None})


def random_colormap(rng):
    colormaps = [
        'viridis', 'plasma', 'inferno', 'magma', 'cividis',
        'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
        'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu', 'GnBu', 'PuBu',
        'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn',
        'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu', 'RdYlGn', 'Spectral',
        'coolwarm', 'bwr', 'seismic',
        'twilight', 'twilight_shifted', 'hsv',
        'Pastel1', 'Pastel2', 'Paired', 'Accent', 'Dark2', 'Set1', 'Set2', 'Set3',
        'tab10', 'tab20', 'tab20b', 'tab20c',
    ]
    return rng.choice(colormaps)


def generate_colormap(rng, n_points):
    colormap_name = random_colormap(rng)
    colormap = cm.get_cmap(colormap_name)
    idx = np.linspace(0, 1, n_points)
    rng.shuffle(idx)
    return colormap(idx)

def generate_visual_hash_points_and_draw_curved(name):
    rng = np.random.default_rng(hash(name) % (2**32))

    def rand():
        return rng.random()

    def rg(arr):
        i = rng.integers(0, len(arr))
        result = arr[i]
        arr[i] = arr[-1]
        arr.pop()
        return result

    def fi():
        return rand() < 0.5

    N = 7 if fi() else 11
    size = 4620
    step = (2 * np.pi / size) * N

    h = np.zeros(8)
    h[0] = 0.4 + rand() * 0.2
    h[2] = 0.3 + rand() * 0.2
    h[3] = 0.1 + rand() * 0.1
    h[5] = 1 + rand() * 4
    h[6] = 1 + rand()
    h[7] = 1 + rand()
    for i in range(2, 8):
        if fi():
            h[i] *= -1

    ki = [1, 3, 5, 7, 9, 11]
    gu = [0, 0, 2, 4, 6, 8, 10]
    q = np.zeros(8)
    s = [None] * 8
    pr = ((1 + rand() * (N - 1)) // 1) / N

    for i in range(2):
        if fi():
            s[i] = [np.cos, np.sin]
            q[i] = rg(ki) - pr
        else:
            s[i] = [np.sin, np.cos]
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
        s[i] = np.cos if use_cos else np.sin

    n = [1 if fi() else -1 for _ in range(3)]

    points = []
    r = 0
    for _ in range(size):
        b = s[6](r * q[6] + s[3](r * q[3]) * h[5]) * n[0]
        a = 1 + b * h[0]
        d = s[7](r * q[7])
        e = -d
        d *= (2 - a) * n[1]
        e *= (2 - a) * n[2]
        c = (s[4](r * q[4] + s[5](r * q[5]) * h[7]) / 4) * h[6] * (a - (1 - h[0]))
        x = np.sin(r * pr + c) * a + s[0][0](r * q[0]) * h[2] * d + s[1][0](r * q[1]) * h[3] * e
        y = np.cos(r * pr + c) * a + s[0][1](r * q[0]) * h[2] * d + s[1][1](r * q[1]) * h[3] * e
        points.append((x * 110 + 200, y * 110 + 200))
        r += step

    size = 4620
    colors = generate_colormap(rng, size)

    x, y = zip(*points)
    x = np.array(x)
    y = np.array(y)

    x = np.concatenate((x, [x[0]]))
    y = np.concatenate((y, [y[0]]))

    t = np.linspace(0, 1, len(x))
    cs_x = CubicSpline(t, x, bc_type='periodic')
    cs_y = CubicSpline(t, y, bc_type='periodic')

    num_points = 5000
    t_new = np.linspace(0, 1, num_points)
    x_smooth = cs_x(t_new)
    y_smooth = cs_y(t_new)

    fig, ax = plt.subplots()
    
    ax.set_facecolor('black')
    fig.set_facecolor('black')

    for i in range(len(x_smooth) - 1):
        ax.plot(x_smooth[i:i + 2], y_smooth[i:i + 2], c=colors[i % len(colors)], lw=2)

    ax.set_xticks(range(0, WIDTH, CELL_SIZE))
    ax.set_yticks(range(0, HEIGHT, CELL_SIZE))
    plt.grid()

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')

    plt.box(False)

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=80)
    plt.close(fig)
    buf.seek(0)
    img = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_UNCHANGED)[:, :, :3]

    return img
    

@app.post("/generate-plot", response_class=HTMLResponse)
async def generate_visual_hash(request: Request, name: str = Form(...)):
    img = generate_visual_hash_points_and_draw_curved(name)

    img_pil = Image.fromarray(img)

    buffer = BytesIO()
    img_pil.save(buffer, format="PNG")
    buffer.seek(0)

    img_base64 = base64.b64encode(buffer.read()).decode()
    img_data_url = f"data:image/png;base64,{img_base64}"

    return templates.TemplateResponse("index.html", {"request": request, "image": img_data_url, "name": name})
