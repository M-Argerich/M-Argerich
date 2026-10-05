"""Night neon-city tracking shot: the reference car photo (cut out, relit, repainted red)
composited into a 3D street whose building walls are rectified real night photos.
5 s, 30 fps, seamless loop: every moving element has a period that divides the clip."""
import math, os, sys, colorsys
import numpy as np
from PIL import Image, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 1920, 1080
FPS, DUR = 30, 5.0
NF = int(FPS * DUR)
L = 64.0                      # facade period (m) == distance travelled per clip
SPEED = L / DUR               # 12.8 m/s
SUB, SHUTTER = 10, 1.0 / 50
OUT = sys.argv[1] if len(sys.argv) > 1 else "frames2"
WORKER = int(sys.argv[2]) if len(sys.argv) > 2 else 0
NWORK = int(sys.argv[3]) if len(sys.argv) > 3 else 1
ONLY = int(sys.argv[4]) if len(sys.argv) > 4 else None
rng = np.random.default_rng(3)

def load(p): return np.asarray(Image.open(os.path.join(HERE, p)).convert("RGB"), np.float32) / 255

def bilinear(img, px, py, wrap_x=False):
    h, w = img.shape[:2]
    if wrap_x: px = px % w
    else: px = np.clip(px, 0, w - 1.001)
    py = np.clip(py, 0, h - 1.001)
    x0 = np.floor(px).astype(np.int32); y0 = np.floor(py).astype(np.int32)
    if wrap_x: x0 %= w
    fx = (px - x0)[..., None]; fy = (py - y0)[..., None]
    x1 = (x0 + 1) % w if wrap_x else np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    return ((img[y0, x0] * (1 - fx) + img[y0, x1] * fx) * (1 - fy) +
            (img[y1, x0] * (1 - fx) + img[y1, x1] * fx) * fy)

# ------------------------------------------------------------------ facade texture
# Source photo: Kabukicho building row (Wikimedia Commons, CC BY-SA 4.0).
# Level camera, so rectify with a 1-D projective map along the wall + per-column scale.
src = load("cand/c1.jpg")
S = src.shape[1] / 1600.0          # measurements below were taken at 1600 px width
XVP, YH, YB = -3210.0, 1170.0, 1190.0
KV = 36.0 / (1600 - XVP)           # vertical px/m per px of distance from the VP
A = (1600 - XVP) ** 2 / (36.0 * 0.85)
B = A / (1600 - XVP)
PPM, TEXH, XF = 40, 38.0, 6.0      # px/m, wall height (m), crossfade length (m)
uu = np.arange(int((L + XF) * PPM)) / PPM
hh = (np.arange(int(TEXH * PPM))[::-1] + 0.5) / PPM
U_, H_ = np.meshgrid(uu, hh)
X_ = XVP + A / (U_ + B)
Y_ = YH + (YB - YH) * (X_ - XVP) / (1600 - XVP) - H_ * KV * (X_ - XVP)
raw = bilinear(src, X_ * S, Y_ * S)
n = int(L * PPM); nx = int(XF * PPM)
w = (np.arange(nx) / nx)[None, :, None]
tex = raw[:, :n].copy()
tex[:, :nx] = raw[:, :nx] * w + raw[:, n:n + nx] * (1 - w)

# neon grade: remap hues toward purple / magenta / cyan, keep it photographic
def regrade(img):
    im = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).convert("HSV")
    a = np.asarray(im).astype(np.float32)
    h = a[..., 0] * 360 / 255
    knots_in = [0, 30, 60, 110, 150, 200, 250, 290, 330, 360]
    knots_out = [320, 330, 275, 192, 186, 195, 250, 275, 305, 320]
    h2 = np.interp(h, knots_in, knots_out) % 360
    a[..., 0] = h2 * 255 / 360
    a[..., 1] = np.clip(a[..., 1] * 1.1, 0, 255)
    return np.asarray(Image.fromarray(a.astype(np.uint8), "HSV").convert("RGB"), np.float32) / 255
tex = regrade(tex)
lin = tex ** 2.2 * 1.4                          # to linear-ish radiance
glow = np.asarray(Image.fromarray((np.clip(tex, 0, 1) * 255).astype(np.uint8))
                  .filter(ImageFilter.GaussianBlur(10)), np.float32) / 255
facade = lin + (glow ** 2.2) * 0.6
# vertically smeared copy for wet-road reflections, softer copy for car reflections
def smear(img, sx, sy):
    im = Image.fromarray((np.clip(img / 2, 0, 1) * 255).astype(np.uint8))
    w0, h0 = im.size
    im = im.resize((max(1, w0 // sx), max(1, h0 // sy)), Image.BOX).resize((w0 // 2, h0 // 2), Image.BILINEAR)
    return np.asarray(im.filter(ImageFilter.GaussianBlur(3)), np.float32) / 255 * 2
road_refl = smear(facade, 3, 24)
car_refl = smear(facade, 10, 10)
TW = facade.shape[1]
def blur_wrap(img, k):
    """mip level k: box-downsample by 2**k (with wrap padding), bilinear back up to full size"""
    f = 2 ** k; pad = f * 2
    big = np.concatenate([img[:, -pad:], img, img[:, :pad]], 1)
    h0, w0 = big.shape[:2]
    out = []
    for c in range(3):
        im = Image.fromarray(np.ascontiguousarray(big[..., c]), "F")
        im = im.resize((max(1, w0 // f), h0), Image.BOX).resize((w0, h0), Image.BILINEAR)
        out.append(np.asarray(im, np.float32))
    return np.stack(out, -1)[:, pad:-pad]
MIPS = [facade] + [blur_wrap(facade, k) for k in range(1, 8)]

# ------------------------------------------------------------------ camera
FL = 1400.0
CX, CY = W / 2, H / 2
CAM = np.array([0.0, 0.0, 1.55])
VPX, VPY = 675.0, 381.0              # where the street's +x direction should land
def basis(yaw, pitch):
    f = np.array([math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), math.sin(pitch)])
    r = np.cross(f, [0, 0, 1.0]); r /= np.linalg.norm(r)
    return r, np.cross(r, f), f
def vp_of(yaw, pitch):
    r, u, f = basis(yaw, pitch)
    X = np.array([1.0, 0, 0])
    return CX + FL * (X @ r) / (X @ f), CY - FL * (X @ u) / (X @ f)
best = None
for yw in np.linspace(-0.6, 0.6, 241):
    for pt in np.linspace(-0.4, 0.4, 161):
        px, py = vp_of(yw, pt); e = (px - VPX) ** 2 + (py - VPY) ** 2
        if best is None or e < best[0]: best = (e, yw, pt)
yaw, pitch = best[1], best[2]
r, u, f = basis(yaw, pitch)

jj, ii = np.mgrid[0:H, 0:W].astype(np.float32)
D = f[None, None] + ((ii - CX) / FL)[..., None] * r + (-(jj - CY) / FL)[..., None] * u
D /= np.linalg.norm(D, axis=-1, keepdims=True)
dx, dy, dz = D[..., 0], D[..., 1], D[..., 2]

YL, YR = 5.8, -8.6                   # building walls (left / right)
def walls(ox, oy, oz, dx, dy, dz):
    with np.errstate(divide="ignore", invalid="ignore"):
        tl = np.where(dy > 1e-5, (YL - oy) / dy, np.inf)
        tr = np.where(dy < -1e-5, (YR - oy) / dy, np.inf)
    t = np.minimum(tl, tr); left = tl < tr
    hx = ox + t * dx; hz = oz + t * dz
    return t, left, hx, hz
with np.errstate(divide="ignore", invalid="ignore"):
    tg = np.where(dz < -1e-5, -CAM[2] / dz, np.inf)
tw, wleft, wx, wz = walls(CAM[0], CAM[1], CAM[2], dx, dy, dz)
hit_g = tg < tw
hit_w = ~hit_g & (wz < TEXH)
gx = np.where(hit_g, CAM[0] + tg * dx, 0); gy = np.where(hit_g, CAM[1] + tg * dy, 0)
t2, rleft, rx_, rz_ = walls(gx, gy, 0 * gx, dx, dy, -dz)       # mirror ray off wet road
rvis = hit_g & (rz_ < TEXH)
# texture LOD for the walls: footprint of one pixel on the wall, in texels (+ motion smear is separate)
cosn = np.clip(np.abs(dy), 0.02, 1)
foot = tw / FL / cosn * PPM          # along-street footprint (anisotropic)
LOD = np.clip(np.log2(np.maximum(np.where(np.isfinite(foot), foot, 1), 1e-3)), 0, 8 - 1.001)
LOD0 = np.floor(LOD).astype(np.int32); LODF = (LOD - LOD0)[..., None]
dist = np.where(hit_g, tg, np.where(hit_w, tw, 300.0))

def wall_u(x, left, off):
    return np.where(left, -(x + off) + 17.0, x + off)            # left wall reads mirrored-correct
def sample_wall(tex_, x, z, left, off, scale=1):
    ppm = PPM / scale
    return bilinear(tex_, wall_u(x, left, off) * ppm, (TEXH - z) * ppm - 0.5, wrap_x=True)

sky = np.stack([0.010 + 0.02 * (jj / H), 0.006 + 0.008 * (jj / H), 0.022 + 0.03 * (jj / H)], -1)
haze_col = np.array([0.07, 0.025, 0.11], np.float32)
haze = (1 - np.exp(-dist / 55.0))[..., None] * 0.85
fres = (0.12 + 0.88 * (1 - np.clip(-dz, 0, 1)) ** 5)[..., None]
calm = np.clip((jj - H * 0.73) / (H * 0.10), 0, 1)[..., None]
asphalt = rng.random((256, 2048)).astype(np.float32)
asphalt = np.asarray(Image.fromarray((asphalt * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.7)), np.float32) / 255
puddle = np.asarray(Image.fromarray((rng.random((32, 128)) * 255).astype(np.uint8)).resize((2048, 256), Image.BICUBIC)
                    .filter(ImageFilter.GaussianBlur(6)), np.float32) / 255
sidewalk = hit_g & ((gy > 3.1) | (gy < -5.9))

def background(t):
    acc = np.zeros((H, W, 3), np.float32)
    for k in range(SUB):
        off = SPEED * (t + SHUTTER * k / SUB)
        wallc = np.zeros((H, W, 3), np.float32)
        for lv in range(8):
            sel = hit_w & ((LOD0 == lv) | (LOD0 == lv - 1))
            if not sel.any(): continue
            c = sample_wall(MIPS[lv], wx[sel], wz[sel], wleft[sel], off)
            wgt = np.where(LOD0[sel] == lv, 1 - LODF[sel, 0], LODF[sel, 0])[..., None]
            wallc[sel] += c * wgt
        img = np.where(hit_w[..., None], wallc, sky)
        # wet asphalt
        ax = ((gx + off) * 60).astype(np.int64) % 2048
        ay = (np.abs(gy) * 60).astype(np.int64) % 256
        nz = asphalt[ay, ax][..., None]
        wet = 0.55 + 0.45 * puddle[(np.abs(gy) * 8).astype(np.int64) % 256, ((gx + off) * 8).astype(np.int64) % 2048][..., None]
        base = np.where(sidewalk[..., None], 0.030, 0.014) * (0.6 + 0.8 * nz) * np.array([0.95, 0.9, 1.1])
        refl = np.where(rvis[..., None], sample_wall(road_refl, rx_, rz_, rleft, off, 2), sky * 0.5)
        road = base + refl * fres * wet * (0.8 - 0.72 * calm)
        dash = (((gx + off) % 9.0) < 3.0) & (np.abs(gy + 0.45) < 0.07)
        edge = (np.abs(gy - 2.9) < 0.08) | (np.abs(gy + 5.6) < 0.08) | ((((gx + off) % 9.0) < 3.0) & (np.abs(gy + 3.85) < 0.07))
        paint = (dash | edge) & hit_g
        road = np.where(paint[..., None], road * 0.5 + 0.20 * (0.7 + 0.3 * nz) * (1 - 0.6 * calm), road)
        img = np.where(hit_g[..., None], road, img)
        acc += img
    acc /= SUB
    return acc * (1 - haze) + haze_col * haze

# ------------------------------------------------------------------ car plate (photo cut-out)
ref = load("ref.jpg")
mask = np.asarray(Image.open(os.path.join(HERE, "mask.png")).convert("L"), np.float32) / 255
SC = 1.25
AX_R, AY_R = 930.0, 607.0            # anchor in the reference (rear centre on the ground)
AX_O, AY_O = 1250.0, 780.0           # where it lands in the output
x0r, y0r, x1r, y1r = 600, 325, 1105, 625
crop = ref[y0r:y1r, x0r:x1r]; cm = mask[y0r:y1r, x0r:x1r]
cw, ch = int((x1r - x0r) * SC), int((y1r - y0r) * SC)
ox, oy = int(round(AX_O + (x0r - AX_R) * SC)), int(round(AY_O + (y0r - AY_R) * SC))
def up(a, mode=Image.LANCZOS):
    im = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))
    return np.asarray(im.resize((cw, ch), mode), np.float32) / 255
car = up(crop); calpha = up(cm)[..., None] if up(cm).ndim == 2 else up(cm)
cyy, cxx = np.mgrid[0:ch, 0:cw].astype(np.float32)
rx_ref = x0r + cxx / SC; ry_ref = y0r + cyy / SC     # reference coords of each car pixel

def poly_mask(pts):
    from PIL import ImageDraw
    im = Image.new("L", (cw, ch), 0)
    ImageDraw.Draw(im).polygon([((x - x0r) * SC, (y - y0r) * SC) for x, y in pts], fill=255)
    return np.asarray(im.filter(ImageFilter.GaussianBlur(1.2)), np.float32) / 255
glass = np.maximum(poly_mask([(758, 362), (960, 352), (1003, 420), (795, 422)]),
                   poly_mask([(677, 350), (740, 347), (743, 410), (680, 400)]))
plate = poly_mask([(868, 523), (990, 523), (990, 553), (868, 553)])
lum = (car @ np.array([0.3, 0.59, 0.11], np.float32))[..., None]
red_bar = ((car[..., 0] > 0.42) & (car[..., 1] < 0.22) & (car[..., 2] < 0.25)).astype(np.float32)
red_bar = np.asarray(Image.fromarray((red_bar * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1)), np.float32)[..., None] / 255
sat = (car.max(-1) - car.min(-1))[..., None]
paint = np.clip(1 - glass[..., None] - plate[..., None] - red_bar * 2, 0, 1) * np.clip(1 - sat * 3, 0, 1)
# repaint: luminance -> deep glossy red, highlights stay whitish
lin_car = car ** 2.2
l_lin = lin_car @ np.array([0.3, 0.59, 0.11], np.float32)
red = l_lin[..., None] * np.array([2.6, 0.10, 0.13]) * 1.0
hi = np.clip((lum - 0.55) / 0.35, 0, 1) ** 2
repaint = red * (1 - hi) + lin_car * hi
body = lin_car * (1 - paint) + repaint * paint
# night relight: kill daylight, cool ambient; tyres/glass/plate darker still
body = body * np.array([0.38, 0.33, 0.45]) * (1 - 0.6 * glass[..., None]) * (1 + 0.6 * plate[..., None])
spec = np.clip(lum * 1.2, 0.08, 1.0) ** 1.5 * (paint + glass[..., None] * 0.8)
# the rear face mostly mirrors the dark road behind: weaker there than on the flank
spec = spec * np.clip(1.0 - (rx_ref[..., None] - 720) / 120, 0.25, 1.0)
tail = red_bar * np.array([3.0, 0.10, 0.15])

# wheels (reference ellipses: centre, rx, ry)
WHEELS = [(677.0, 533.0, 21.0, 74.0), (626.0, 467.0, 12.0, 54.0)]

def car_frame(t):
    off = SPEED * t
    # suspension: body bobs a little, wheels stay planted
    bob = 1.6 * math.sin(2 * math.pi * 3 * t / DUR) + 0.8 * math.sin(2 * math.pi * 5 * t / DUR + 1.3)
    img = body.copy()
    # rotating wheels: rotational blur in ellipse-normalised space, phase loops over the clip
    rot = 2 * math.pi * 15 * t / DUR
    for cxr, cyr, a, b in WHEELS:
        nxp = (rx_ref - cxr) / a; nyp = (ry_ref - cyr) / b
        rr = np.sqrt(nxp ** 2 + nyp ** 2); th = np.arctan2(nyp, nxp)
        inside = (rr < 0.97)[..., None]
        accw = np.zeros_like(img)
        for k in range(8):
            ang = th - rot - k * 0.09
            sx = (cxr + rr * np.cos(ang) * a - x0r) * SC
            sy = (cyr + rr * np.sin(ang) * b - y0r) * SC
            accw += bilinear(body, sx, sy)
        img = np.where(inside, accw / 8, img)
    # gliding neon reflections (move rearwards along the body = left -> right in frame)
    ru = off - (ox + cxx) / 34.0
    rv = 4.0 + (H - (oy + cyy)) / 55.0
    nr = bilinear(car_refl, ((ru % L) * PPM / 2), (TEXH - rv) * PPM / 2 - 0.5, wrap_x=True)
    nr2 = bilinear(car_refl, (((off * 0.7 - (ox + cxx) / 20.0 + 23) % L) * PPM / 2),
                   (TEXH - (8 + (H - (oy + cyy)) / 30.0)) * PPM / 2, wrap_x=True)
    img = img + spec * (np.clip(nr - 0.08, 0, None) * 0.85 + np.clip(nr2 - 0.12, 0, None) * 0.45)
    img = img + tail
    # integer-free vertical shift of the body above the axle line
    shift = np.clip((AY_O - 60 - (oy + cyy)) / 120, 0, 1)[..., None] * bob
    if abs(bob) > 1e-3:
        sy = cyy - shift[..., 0]
        img = bilinear(img, cxx, sy); al = bilinear(calpha, cxx, sy)
    else:
        al = calpha
    return img, al

# contact shadow + tail-light glow on the road (static, under the car)
shadow = np.zeros((H, W), np.float32)
sh_im = Image.new("L", (W, H), 0)
from PIL import ImageDraw
ImageDraw.Draw(sh_im).ellipse([ox + 0.10 * cw, AY_O - 40, ox + 0.98 * cw, AY_O + 22], fill=230)
shadow = np.asarray(sh_im.filter(ImageFilter.GaussianBlur(22)), np.float32)[..., None] / 255
tl_im = Image.new("L", (W, H), 0)
ImageDraw.Draw(tl_im).rectangle([ox + 0.30 * cw, AY_O + 5, ox + 0.92 * cw, AY_O + 95], fill=255)
tl_refl = np.asarray(tl_im.filter(ImageFilter.GaussianBlur(26)), np.float32)[..., None] / 255
tl_refl = tl_refl * np.array([0.55, 0.02, 0.04]) * np.exp(-np.clip(jj - AY_O, 0, None) / 70)[..., None]

xn = (ii / W)[..., None]; yn = (jj / H)[..., None]
def bloom(img):
    small = Image.fromarray((np.clip((img - 0.6) * 0.8, 0, 1) * 255).astype(np.uint8)).resize((W // 4, H // 4), Image.BOX)
    b1 = np.asarray(small.filter(ImageFilter.GaussianBlur(5)).resize((W, H), Image.BILINEAR), np.float32) / 255
    b2 = np.asarray(small.filter(ImageFilter.GaussianBlur(24)).resize((W, H), Image.BILINEAR), np.float32) / 255
    return img + b1 * 0.5 + b2 * 0.7
def tonemap(x):
    x = np.maximum(x, 0)
    return (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14)

def frame(nf):
    t = nf / FPS
    img = background(t)
    img = img * (1 - shadow * 0.85) + tl_refl
    cimg, cal = car_frame(t)
    reg = img[oy:oy + ch, ox:ox + cw]
    img[oy:oy + ch, ox:ox + cw] = reg * (1 - cal) + cimg * cal
    img = bloom(img)
    vig = 1 - 0.40 * (((xn - 0.5) * 1.2) ** 2 + ((yn - 0.45) * 1.4) ** 2)
    img = tonemap(img * vig * 1.6)
    img = np.clip(img, 0, 1) ** (1 / 1.0)
    img = img * np.array([1.0, 0.97, 1.04])
    img += (rng.random((H, W, 1)).astype(np.float32) - 0.5) * 0.018
    return Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))

os.makedirs(OUT, exist_ok=True)
todo = [ONLY] if ONLY is not None else range(WORKER, NF, NWORK)
for nf in todo:
    frame(nf).save(f"{OUT}/f{nf:04d}.png")
    print(nf, flush=True)
