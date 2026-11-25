"""Rebuild combined_walk.gif: crossfade + bg/char layer split (constant shifts)."""
import numpy as np
from PIL import Image, ImageSequence, ImageDraw

BASE = r"C:\Users\Charles\Downloads\techstack"
S_W = 145        # world (bg+terrain) shift: bg-phase optimal
S_C = 140        # char shift: keeps helmet centre continuous (315.5 -> 315)
N_FADE = 8

def load(n):
    im = Image.open(f"{BASE}\\{n}.gif")
    return [np.asarray(f.convert("RGB"), dtype=np.float64)
            for f in ImageSequence.Iterator(im)]

A = load("left_walk")
B = load("right_walk")
H, W, _ = A[0].shape

def fg_mask(a):
    mx = a.max(axis=2); mn = a.min(axis=2); lum = a.mean(axis=2)
    return ((mx - mn) > 25) | (lum < 100) | (lum > 225)

def fix_rect(a):
    """left_walk's checkerboard is half-tile phase-flipped inside the rect
    x>=450, y>=245 (the game draws that zone with its own texture origin);
    right_walk has no such flip.  Crossfading the two there cancels the
    checker (washed-out patch), so invert the rect's tones in every A frame
    so both clips share one lattice.  The rect holds only background in A
    (the char never reaches x=450 until it has left the rect again), and
    fg pixels are excluded from the gate anyway."""
    out = a.copy()
    reg = out[245:, 450:]
    lum = reg.mean(axis=2)
    chroma = reg.max(axis=2) - reg.min(axis=2)
    gate = (lum >= 160) & (lum <= 214) & (chroma <= 12) & ~fg_mask(reg)
    reg[gate] = (372.0, 370.0, 370.0) - reg[gate]
    return out

A = [fix_rect(a) for a in A]      # make A's checker lattice uniform (see fix_rect)

# ---- bg period ----
best = None
for P in range(56, 73):
    errs = [np.abs(B[i][:, :40] - B[i][:, P:P + 40]).mean() for i in range(0, 30, 5)]
    m = float(np.mean(errs))
    if best is None or m < best[0]:
        best = (m, P)
P = best[1]
print(f"bg period P={P} (err {best[0]:.2f})")

# ---- terrain start (left edge of right-side structure) ----
def terrain_start(a):
    mx = a.max(axis=2); mn = a.min(axis=2); lum = a.mean(axis=2)
    bg = ((mx - mn) <= 14) & (lum >= 152) & (lum <= 216)
    cols = (~bg).sum(axis=0)
    edge = None
    for i in range(639, 0, -1):
        if cols[i] > 60:
            edge = i
        elif edge is not None and cols[i] < 8:
            break
    if edge is None:
        return None
    i = edge
    while i > 0 and cols[i - 1] > 60:
        i -= 1
    return int(i)

# ---- helpers ----
def dilate(m, r=3):
    out = m.copy()
    for _ in range(r):
        d = out.copy()
        d[1:, :] |= out[:-1, :]; d[:-1, :] |= out[1:, :]
        d[:, 1:] |= out[:, :-1]; d[:, :-1] |= out[:, 1:]
        out = d
    return out

def char_mask(b, ts):
    m = fg_mask(b)
    m[:, :96] = False          # char never left of x=96
    m[:, ts - 2:] = False      # never right of terrain start
    return m                   # undilated; dilate at the use site

def fill_footprint(b, mask):
    """Replace masked px with bg tiled from x-P, left->right (progressive)."""
    cols = np.where(mask.any(axis=0))[0]
    out = b.copy()
    for x in cols:
        src = x - P
        if src < 0:
            continue
        rows = mask[:, x]
        out[rows, x] = out[rows, src]
    return out

def shift_world(img, g):
    canvas = np.empty_like(img)
    idx = (np.arange(g) - g) % P
    canvas[:, :g] = img[:, idx]
    canvas[:, g:] = img[:, :W - g]
    return canvas

def shift_right(img, s):
    canvas = np.zeros_like(img)
    canvas[:, s:] = img[:, :W - s]
    return canvas

def composite(b, g, s):
    ts = terrain_start(b)
    m = char_mask(b, ts)
    mask = dilate(m, 3)
    world = fill_footprint(b, mask)
    ws = shift_world(world, g)
    out = ws.copy()
    cl = np.where(mask[..., None], b, 0.0)
    cm = shift_right(mask.astype(np.float64), s) > 0.5
    out[cm] = shift_right(cl, s)[cm]
    # the outer dilation ring around the char holds background: keep it on
    # the world layer (shift g).  Pasting it at the char shift s would jump
    # the checker phase by (g-s) there -> a halo around the character.
    core = shift_right(dilate(m, 1).astype(np.float64), s) > 0.5
    band = cm & ~core
    out[band] = ws[band]
    return out

# ---- per-frame world shift: cancel B's camera pan during the fade ----
# p[f] = how far B[f]'s content moved left vs B[0] (measured on terrain edge,
# a sharp feature). g[f] = S_W + p[f] keeps bg phase locked to A29, which puts
# the terrain at constant x=S_W+ts[0] (off-screen) for the whole fade.
ts0 = terrain_start(B[0])
ts = [terrain_start(B[f]) for f in range(30)]
g = [S_W + (ts0 - ts[f]) for f in range(30)]
g[N_FADE:] = [g[N_FADE - 1]] * (30 - N_FADE)
print(f"terrain starts: {ts[:12]} ... g: {g[:12]}")

# ---- assemble ----
compB = [composite(B[f], g[f], S_C) for f in range(30)]
frames = [A[i] for i in range(30)]
for f in range(N_FADE):
    a = (f + 1) / N_FADE
    frames.append(A[29] * (1 - a) + compB[f] * a)
frames.extend(compB[N_FADE:])
frames = [np.clip(np.rint(f), 0, 255).astype(np.uint8) for f in frames]
print("total frames:", len(frames))

def mad(f1, f2):
    return float(np.abs(f1.astype(float) - f2.astype(float)).mean())

# ---- diagnostics ----
fa = fg_mask(A[29][:25])
emoji_cols = np.where(fa.any(axis=0))[0]
print("A29 fg cols rows0-24:", emoji_cols.min(), emoji_cols.max())
a_ref = A[29][:25]
print("\nfade-frame bg MAD vs A29 (rows0-24, x>=145, fg excluded):")
for f in range(N_FADE):
    o = frames[30 + f].astype(float)
    m = (~fa[:, 145:]) & (~fg_mask(o[:25, 145:]))
    d = (a_ref[:, 145:] - o[:25, 145:])[m]
    print(f"  fade {f}: {np.abs(d - d.mean()).mean():.2f}")

print("\nrect region (y>=245, x>=450) luminance std (checker contrast, "
      "~18 = healthy, ~6 = washed out):")
for i in list(range(28, 46)) + [50, 59]:
    o = frames[i].astype(float)
    print(f"  {i}:{o[245:, 450:].mean(axis=2).std():5.1f}", end="")
print()

def green(a):
    r, g_, b_ = a[..., 0], a[..., 1], a[..., 2]
    return (g_ > r + 8) & (g_ > b_ + 8) & (g_ > 45) & (g_ < 140)
def helmet_c(a, xmax=640):
    m = green(a).copy(); m[:, xmax:] = False
    ys = np.where(m.any(axis=1))[0]
    if not len(ys): return None
    y0 = ys[0]; rows = set(ys.tolist()); y = y0
    while (y + 1) in rows and y - y0 < 70: y += 1
    xs = np.where(m[y0:y + 1].any(axis=0))[0]
    return (int(xs.min()) + int(xs.max())) / 2

print("\nhelmet centre (A29=315.5):")
for f in (0, 3, 7, 15, 29):
    print(f"  B{f}: {helmet_c(compB[f])}")

print("\nframe-to-frame MAD 28..42:")
for i in range(28, 42):
    print(f"  {i}->{i+1}: {mad(frames[i], frames[i+1]):.2f}")
print("ref A5->6:", round(mad(frames[5], frames[6]), 2))

# footprint check: composited frame vs plain world shift with the same g
# (should differ only inside the char footprint)
b8 = B[8]
plain = shift_world(fill_footprint(b8, dilate(char_mask(b8, terrain_start(b8)), 3)), g[8])
chk = np.abs(compB[8] - plain).mean(axis=2)
print("\ncompB8 vs plain world@ g[8]: mean diff left of char (x<250):",
      round(float(chk[:, :250].mean()), 2),
      "| right of char (x>470):",
      round(float(chk[:, 470:].mean()), 2))

# ---- save ----
sample = np.concatenate([f[::8, ::8].reshape(-1, 3) for f in frames], axis=0)
palette = Image.fromarray(sample.reshape(-1, 1, 3).astype(np.uint8)).quantize(
    colors=256, method=Image.Quantize.MEDIANCUT)
imgs = [Image.fromarray(f).quantize(palette=palette, dither=Image.Dither.NONE)
        for f in frames]
imgs[0].save(f"{BASE}\\frames_tmp\\combined_new.gif", save_all=True,
             append_images=imgs[1:], duration=100, loop=0, disposal=2)
print("saved frames_tmp/combined_new.gif")

# ---- montage ----
idx = list(range(26, 44))
cols, w, h = 6, 256, 144
rows = (len(idx) + cols - 1) // cols
sheet = Image.new("RGB", (cols * w, rows * h), "white")
d = ImageDraw.Draw(sheet)
for k, i in enumerate(idx):
    x, y = (k % cols) * w, (k // cols) * h
    sheet.paste(Image.fromarray(frames[i]).resize((w, h), Image.NEAREST), (x, y))
    d.rectangle([x + 2, y + 2, x + 52, y + 18], fill="black")
    d.text((x + 5, y + 5), f"#{i}", fill="yellow")
sheet.save(f"{BASE}\\frames_tmp\\transition.png")
# full-size post-fade frame for artifact check
Image.fromarray(frames[40]).save(f"{BASE}\\frames_tmp\\new_40.png")
print("saved transition.png, new_40.png")
