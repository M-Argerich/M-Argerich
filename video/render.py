"""Procedural render: red sports car cruising through a neon city at night.
5 s, 30 fps, seamless loop (all motion is periodic over the clip length)."""
import math, os, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

W, H = 1920, 1080
FPS, DUR = 30, 5.0
NF = int(FPS * DUR)
SPEED = 20.0            # m/s  -> 100 m per clip
PERIOD = SPEED * DUR    # facade/road texture period (m) => seamless loop
SUB = 8                 # motion-blur subsamples
SHUTTER = 1.0 / 45
OUT = sys.argv[1] if len(sys.argv) > 1 else "frames"
ONLY = int(sys.argv[2]) if len(sys.argv) > 2 else None
rng = np.random.default_rng(7)

# ---------------- camera ----------------
C = np.array([-6.2, 4.3, 1.45])
target = np.array([0.15, 0.0, 0.55])
f = target - C; f /= np.linalg.norm(f)
r = np.cross(f, [0, 0, 1.0]); r /= np.linalg.norm(r)
u = np.cross(r, f)
FL = 1500.0
CX, CY = W * 0.5, H * 0.47

def project(p):
    d = np.asarray(p, float) - C
    x, y, z = d @ r, d @ u, d @ f
    return CX + FL * x / z, CY - FL * y / z

# ---------------- facade texture ----------------
PPM = 40                      # px per metre
FY = -15.0                    # facade plane y
TW, TH = int(PERIOD * PPM), 34 * PPM
emis = Image.new("RGB", (TW, TH), (0, 0, 0))
alb = Image.new("RGBA", (TW, TH), (0, 0, 0, 0))
de, da = ImageDraw.Draw(emis), ImageDraw.Draw(alb)
NEON = [(190, 60, 255), (255, 40, 200), (40, 230, 255), (140, 80, 255), (255, 70, 150)]

def rect(d, x0, z0, x1, z1, col):
    for off in (0, -TW, TW):
        d.rectangle([x0 * PPM + off, TH - z1 * PPM, x1 * PPM + off, TH - z0 * PPM], fill=col)

x = 0.0
while x < PERIOD - 0.5:
    bw = min(rng.uniform(9, 19), PERIOD - x)
    if PERIOD - x - bw < 6: bw = PERIOD - x
    bh = rng.uniform(13, 32)
    base = tuple(int(c) for c in rng.uniform([14, 12, 26], [30, 24, 46]))
    rect(da, x, 0, x + bw, bh, base + (255,))
    # windows
    for wz in np.arange(5.5, bh - 1.2, 1.6):
        for wx in np.arange(x + 0.8, x + bw - 1.2, 1.5):
            if rng.random() < 0.28:
                c = NEON[rng.integers(5)] if rng.random() < 0.2 else (
                    tuple(int(v) for v in rng.uniform([90, 70, 110], [170, 140, 200])))
                c = tuple(int(v * rng.uniform(0.35, 0.8)) for v in c)
                rect(de, wx, wz, wx + 0.8, wz + 0.9, c)
    # storefront
    sc = NEON[rng.integers(5)]
    rect(de, x + 0.6, 0.3, x + bw - 0.6, 3.2, tuple(int(v * 0.55) for v in sc))
    for mx in np.arange(x + 0.6, x + bw - 0.6, 2.2):
        rect(de, mx, 0.3, mx + 0.12, 3.2, (0, 0, 0))
    rect(de, x + 0.4, 3.4, x + bw - 0.4, 4.3, (0, 0, 0))
    rect(de, x + 0.4, 3.4, x + bw - 0.4, 3.52, sc)
    rect(de, x + 0.4, 4.18, x + bw - 0.4, 4.3, sc)
    for gx in np.arange(x + 1.0, x + bw - 1.2, 0.7):
        if rng.random() < 0.7:
            rect(de, gx, 3.65, gx + 0.45, 4.05, sc)
    # vertical blade signs
    for _ in range(rng.integers(1, 3)):
        sx = x + rng.uniform(0.5, bw - 1.8)
        sz = rng.uniform(5.5, 8); sh = rng.uniform(5, 10)
        c = NEON[rng.integers(5)]
        rect(de, sx, sz, sx + 1.3, sz + sh, c)
        rect(de, sx + 0.12, sz + 0.12, sx + 1.18, sz + sh - 0.12, (10, 0, 20))
        for cz in np.arange(sz + 0.5, sz + sh - 0.9, 1.0):
            rect(de, sx + 0.35, cz, sx + 0.95, cz + 0.65, c)
    # horizontal neon tubes
    if rng.random() < 0.6:
        c = NEON[rng.integers(5)]
        hz = rng.uniform(9, min(bh - 2, 18))
        rect(de, x + 1, hz, x + bw - 1, hz + 0.18, c)
    x += bw + rng.uniform(0.0, 0.6)

emis_np = np.asarray(emis, np.float32) / 255
glow = np.asarray(emis.filter(ImageFilter.GaussianBlur(18)), np.float32) / 255
glow2 = np.asarray(emis.filter(ImageFilter.GaussianBlur(60)), np.float32) / 255
alb_np = np.asarray(alb, np.float32) / 255
facade = alb_np[..., :3] * 0.5 + emis_np * 1.7 + glow * 1.3 + glow2 * 0.5
facade_a = np.maximum(alb_np[..., 3:], np.clip(glow2.max(-1, keepdims=True) * 1.2, 0, 1))
# blurred copy for wet-road / paint reflections (vertical streaks)
refl_img = Image.fromarray((np.clip(facade / 3, 0, 1) * 255).astype(np.uint8))
refl_img = refl_img.resize((TW // 4, TH // 4)).filter(ImageFilter.GaussianBlur(3))
refl_img = refl_img.resize((TW // 4, TH // 40)).resize((TW // 4, TH // 4), Image.BILINEAR)
refl = np.asarray(refl_img.filter(ImageFilter.GaussianBlur(4)), np.float32) / 255 * 3
paint_img = Image.fromarray((np.clip(facade / 3, 0, 1) * 255).astype(np.uint8)).resize((TW // 8, TH // 8))
paint_refl = np.asarray(paint_img.filter(ImageFilter.GaussianBlur(2)), np.float32) / 255 * 3

def sample(tex, uu, vv, ppm):
    th, tw = tex.shape[:2]
    px = (uu * ppm) % tw
    py = np.clip(th - 1 - vv * ppm, 0, th - 1.001)
    x0 = np.floor(px).astype(np.int32); y0 = np.floor(py).astype(np.int32)
    fx = (px - x0)[..., None]; fy = (py - y0)[..., None]
    x0 %= tw; x1 = (x0 + 1) % tw; y1 = y0 + 1
    return ((tex[y0, x0] * (1 - fx) + tex[y0, x1] * fx) * (1 - fy) +
            (tex[y1, x0] * (1 - fx) + tex[y1, x1] * fx) * fy)

# ---------------- per-pixel rays (static geometry) ----------------
jj, ii = np.mgrid[0:H, 0:W].astype(np.float32)
D = (f[None, None] + ((ii - CX) / FL)[..., None] * r + (-(jj - CY) / FL)[..., None] * u)
D /= np.linalg.norm(D, axis=-1, keepdims=True)
dx, dy, dz = D[..., 0], D[..., 1], D[..., 2]
with np.errstate(divide="ignore", invalid="ignore"):
    tf = np.where(dy < -1e-4, (FY - C[1]) / dy, np.inf)
    tr = np.where(dz < -1e-4, -C[2] / dz, np.inf)
hit_f = tf < tr
hit_r = (tr <= tf) & np.isfinite(tr)
fu = np.where(hit_f, C[0] + tf * dx, 0); fv = np.where(hit_f, C[2] + tf * dz, 0)
rx = np.where(hit_r, C[0] + tr * dx, 0); ry = np.where(hit_r, C[1] + tr * dy, 0)
# reflection ray off the road
with np.errstate(divide="ignore", invalid="ignore"):
    t2 = np.where(hit_r & (dy < -1e-4), (FY - ry) / dy, np.inf)
ru = np.where(np.isfinite(t2), rx + t2 * dx, 0)
rv = np.where(np.isfinite(t2), -dz * t2, 99)
dist = np.where(hit_f, tf, np.where(hit_r, tr, 200.0))
haze = (1 - np.exp(-dist / 70.0))[..., None]
haze_col = np.array([0.10, 0.04, 0.16], np.float32)
sky = np.stack([0.05 + 0.06 * (jj / H), 0.02 + 0.02 * (jj / H), 0.09 + 0.08 * (jj / H)], -1)
fres = np.clip(0.25 + 0.75 * (1 - np.abs(dz)) ** 5, 0, 1)[..., None]
# keep the lower frame calm for lyrics
calm = np.clip((jj - H * 0.66) / (H * 0.16), 0, 1)[..., None] ** 0.8
sidewalk = hit_r & (ry < -8.5)
rnoise = rng.random((64, 4000)).astype(np.float32)

def background(t):
    acc = np.zeros((H, W, 3), np.float32)
    for k in range(SUB):
        off = SPEED * (t + SHUTTER * k / SUB)
        fc = sample(facade, fu + off, fv, PPM); fa = sample(facade_a, fu + off, fv, PPM)
        img = sky * (1 - fa) + fc * fa
        # road
        rr = sample(refl, ru + off, rv, PPM / 4)
        dash = (((rx + off) % 10.0) < 3.0) & (np.abs(ry + 2.0) < 0.08) & hit_r
        edge = (np.abs(ry + 8.5) < 0.12) & hit_r
        nz = rnoise[(np.abs(ry * 6).astype(np.int32)) % 64, ((rx + off) * 40).astype(np.int32) % 4000]
        base = np.where(sidewalk, 0.05, 0.022)[..., None] * (0.7 + 0.6 * nz[..., None]) * np.array([0.9, 0.85, 1.1])
        road = base + rr * fres * (0.55 - 0.48 * calm)
        road = np.where(dash[..., None], road * 0.4 + np.array([0.55, 0.5, 0.6]), road)
        road = np.where(edge[..., None], road * 0.5 + np.array([0.25, 0.22, 0.3]), road)
        img = np.where(hit_r[..., None], road, img)
        acc += img
    acc /= SUB
    return acc * (1 - haze * 0.75) + haze_col * haze * 0.75

# ---------------- car ----------------
SS = 2
def car_layers(t):
    bob = 0.012 * math.sin(2 * math.pi * 3 * t / DUR) + 0.006 * math.sin(2 * math.pi * 7 * t / DUR + 1)
    pitch = 0.004 * math.sin(2 * math.pi * 4 * t / DUR)
    def P(x, y, z):
        z = z + bob + pitch * x if z > 0.3 else z
        a, b = project((x, y, z)); return (a * SS, b * SS)
    col = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
    paint = Image.new("L", (W * SS, H * SS), 0)
    glass = Image.new("L", (W * SS, H * SS), 0)
    light = Image.new("RGB", (W * SS, H * SS), (0, 0, 0))
    dc, dp, dg, dl = (ImageDraw.Draw(i) for i in (col, paint, glass, light))
    def poly(pts, c, kind=None):
        dc.polygon(pts, fill=c)
        if kind == "paint": dp.polygon(pts, fill=255)
        else: dp.polygon(pts, fill=0)
        if kind == "glass": dg.polygon(pts, fill=255)
        else: dg.polygon(pts, fill=0)

    BW, GW = 0.98, 0.70
    XR, XF = -2.30, 2.30
    # profile of the belt / greenhouse (x, z)
    hood = [(1.05, 0.96), (1.6, 0.86), (2.15, 0.74), (XF, 0.56)]
    # hood + front (mostly occluded, far first)
    poly([P(1.05, -BW, 0.96), P(2.15, -BW, 0.74), P(2.15, BW, 0.74), P(1.05, BW, 0.96)], (120, 8, 16), "paint")
    # windshield
    poly([P(0.25, -GW, 1.26), P(1.05, -BW * 0.92, 0.97), P(1.05, BW * 0.92, 0.97), P(0.25, GW, 1.26)], (10, 8, 18), "glass")
    # roof
    poly([P(-0.65, -GW, 1.25), P(0.25, -GW, 1.26), P(0.25, GW, 1.26), P(-0.65, GW, 1.25)], (150, 10, 20), "paint")
    # rear window (fastback)
    poly([P(-1.75, -GW * 1.05, 1.0), P(-0.65, -GW, 1.25), P(-0.65, GW, 1.25), P(-1.75, GW * 1.05, 1.0)], (12, 10, 22), "glass")
    # trunk deck
    poly([P(XR, -BW, 0.95), P(-1.75, -BW, 1.0), P(-1.75, BW, 1.0), P(XR, BW, 0.95)], (170, 12, 22), "paint")
    # left shoulder (between body side and greenhouse)
    poly([P(-1.75, BW, 1.0), P(-1.75, GW * 1.05, 1.0), P(-0.65, GW, 1.25), P(0.25, GW, 1.26),
          P(1.05, BW * 0.92, 0.97), P(1.05, BW, 0.96), P(0.3, BW, 0.99), P(-0.8, BW, 1.02)], (165, 12, 22), "paint")
    # left body side
    side = [(XR, 0.28), (XR - 0.03, 0.95), (-1.75, 1.0), (-0.8, 1.02), (0.3, 0.99), (1.05, 0.96)] + hood + [(XF, 0.30), (2.05, 0.22), (-2.1, 0.22)]
    poly([P(x_, BW, z_) for x_, z_ in side], (155, 8, 18), "paint")
    # side intake / character line shadow
    # side windows (dark glass with frame)
    poly([P(-1.6, GW * 1.04, 1.03), P(-0.65, GW + 0.01, 1.23), P(0.25, GW + 0.01, 1.24), P(0.95, GW + 0.05, 0.99)], (8, 6, 14), "glass")
    poly([P(-0.2, GW + 0.02, 1.24), P(-0.13, GW + 0.02, 1.24), P(-0.13, GW + 0.04, 1.0), P(-0.2, GW + 0.04, 1.0)], (3, 2, 5), None)
    # mirror
    poly([P(0.95, BW + 0.02, 1.02), P(0.75, BW + 0.2, 1.06), P(0.75, BW + 0.2, 0.95), P(0.95, BW + 0.02, 0.95)], (140, 8, 18), "paint")
    # rear face
    poly([P(XR, -BW, 0.95), P(XR, BW, 0.95), P(XR - 0.03, BW, 0.26), P(XR, -BW, 0.26)], (120, 6, 14), "paint")
    # spoiler lip
    poly([P(XR - 0.06, -BW, 0.99), P(XR - 0.06, BW, 0.99), P(XR + 0.08, BW, 0.95), P(XR + 0.08, -BW, 0.95)], (25, 5, 10), None)
    # tail-light bar
    tl = [P(XR - 0.01, -BW * 0.95, 0.84), P(XR - 0.01, BW * 0.97, 0.84), P(XR - 0.02, BW * 0.97, 0.77), P(XR - 0.01, -BW * 0.95, 0.77)]
    poly(tl, (255, 20, 25), None); dl.polygon(tl, fill=(255, 8, 12))
    # plate recess + diffuser
    poly([P(XR - 0.01, -0.3, 0.65), P(XR - 0.01, 0.3, 0.65), P(XR - 0.01, 0.3, 0.5), P(XR - 0.01, -0.3, 0.5)], (20, 8, 12), None)
    poly([P(XR - 0.02, -BW, 0.40), P(XR - 0.02, BW, 0.40), P(XR - 0.03, BW, 0.24), P(XR - 0.02, -BW, 0.24)], (12, 10, 14), None)
    for ex in (-0.55, 0.55):
        pts = [P(XR - 0.03, ex + dyy * 0.14, 0.33 + dzz * 0.06) for dyy, dzz in
               [(math.cos(a), math.sin(a)) for a in np.linspace(0, 2 * math.pi, 16)]]
        poly(pts, (5, 5, 6), None)
    # wheels (left side)
    rot = 24 * (2 * math.pi / 5) * t / DUR   # 24 spoke-steps per clip: loops, no strobe freeze
    WR = 0.35
    for wx in (-1.45, 1.42):
        arch = [P(wx + 0.47 * math.cos(a), BW + 0.005, 0.34 + 0.47 * math.sin(a)) for a in np.linspace(0, math.pi, 24)]
        poly(arch, (6, 4, 8), None)
        tire = [P(wx + WR * math.cos(a), BW - 0.02, 0.34 + WR * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 40)]
        poly(tire, (14, 13, 16), None)
        rim = [P(wx + 0.24 * math.cos(a), BW, 0.34 + 0.24 * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 40)]
        poly(rim, (55, 55, 62), None)
        # rotational blur: many faint spokes
        for s in range(5):
            for k in range(6):
                a = rot + s * 2 * math.pi / 5 + k * 0.07
                p0 = P(wx + 0.04 * math.cos(a), BW + 0.01, 0.34 + 0.04 * math.sin(a))
                p1 = P(wx + 0.23 * math.cos(a), BW + 0.01, 0.34 + 0.23 * math.sin(a))
                dc.line([p0, p1], fill=(150, 150, 160, 70), width=int(5 * SS))
        hub = [P(wx + 0.05 * math.cos(a), BW + 0.01, 0.34 + 0.05 * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 16)]
        poly(hub, (30, 30, 34), None)
        cal = [P(wx + 0.2 * math.cos(a), BW - 0.005, 0.34 + 0.2 * math.sin(a)) for a in np.linspace(math.pi * 0.55, math.pi * 0.95, 8)]
        dc.line(cal, fill=(200, 30, 30, 120), width=int(4 * SS))
    def down(im, mode):
        return np.asarray(im.resize((W, H), Image.LANCZOS), np.float32) / 255
    return down(col, "RGBA"), down(paint, "L"), down(glass, "L"), down(light, "RGB")

# car contact shadow (static)
sh = Image.new("L", (W, H), 0)
pts = [project((x_, y_, 0)) for x_, y_ in [(-2.6, -1.3), (-2.6, 1.3), (2.6, 1.3), (2.6, -1.3)]]
ImageDraw.Draw(sh).polygon(pts, fill=235)
shadow = np.asarray(sh.filter(ImageFilter.GaussianBlur(28)), np.float32)[..., None] / 255

xn = (ii / W)[..., None]; yn = (jj / H)[..., None]

def bloom(img):
    small = Image.fromarray((np.clip(img - 0.75, 0, 1) * 255).astype(np.uint8)).resize((W // 4, H // 4))
    b1 = np.asarray(small.filter(ImageFilter.GaussianBlur(6)).resize((W, H), Image.BILINEAR), np.float32) / 255
    b2 = np.asarray(small.filter(ImageFilter.GaussianBlur(22)).resize((W, H), Image.BILINEAR), np.float32) / 255
    return img + b1 * 0.6 + b2 * 0.6

def tonemap(x):
    x = np.maximum(x, 0)
    return (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14)

def frame(n):
    t = n / FPS
    img = background(t)
    img *= 1 - shadow * 0.9
    col, paint, glass, light = car_layers(t)
    a = col[..., 3:]
    car = col[..., :3] ** 2.2 * 1.3
    # neon reflections gliding over paint & glass (periodic over the clip)
    off = SPEED * t
    pu = -xn[..., 0] * 70 + yn[..., 0] * 9 + off
    pv = 4 + (1 - yn[..., 0]) * 14
    pr = sample(paint_refl, pu, pv, PPM / 8)
    pr2 = sample(paint_refl, pu * 0.6 + 13, pv * 0.8 + 3, PPM / 8)
    car = car + paint[..., None] * (np.clip(pr - 0.15, 0, None) ** 1.6 * 0.6 + np.clip(pr2 - 0.25, 0, None) ** 1.6 * 0.3)
    car = car + glass[..., None] * (pr * 0.45 + 0.01)
    # soft rim / sky reflection on upper surfaces
    car = car + paint[..., None] * np.clip(0.55 - yn, 0, 1) * np.array([0.12, 0.05, 0.22])
    car = car + light * 2.2
    img = img * (1 - a) + car * a
    # tail-light reflection on wet road (static, under the car)
    img = bloom(img)
    # vignette + grade
    vig = 1 - 0.45 * (((xn - 0.5) * 1.3) ** 2 + ((yn - 0.45) * 1.5) ** 2)
    img = img * vig * 1.15
    img = tonemap(img)
    img = img ** (1 / 1.05)
    img += (rng.random((H, W, 1)).astype(np.float32) - 0.5) * 0.012
    return Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))

os.makedirs(OUT, exist_ok=True)
frames = [ONLY] if ONLY is not None else range(NF)
for n in frames:
    frame(n).save(f"{OUT}/f{n:04d}.png")
    print(n, flush=True)
