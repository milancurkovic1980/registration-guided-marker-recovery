"""Hand-marking of marker centres (click on the centre of the white dot).

    python tools/annotate_markers.py <photo> [<photo> ...]      or      --list photos.txt

left click: certain marker, right click: uncertain marker, middle click or d: delete the nearest mark,
wheel: zoom, r: whole photograph, n / p: next / previous photograph, q: quit.
Every change is saved to work/annotations/<photo>.json."""
import os, sys, json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from markerdet import load_rgb
PHOTO_DIR = os.path.join(ROOT, "photos")
OUT = os.path.join(ROOT, "work", "annotations"); os.makedirs(OUT, exist_ok=True)
ARGS = [a for a in sys.argv[1:] if not a.isdigit()]
if ARGS[:1] == ["--list"]:
    PHOTOS = [l.strip() for l in open(ARGS[1]) if l.strip()]
else:
    PHOTOS = ARGS
if not PHOTOS:
    sys.exit(__doc__)


def photo_file(name):
    for ext in (".jpg", ".JPG", ".png", ".bmp"):
        if os.path.exists(os.path.join(PHOTO_DIR, name + ext)):
            return os.path.join(PHOTO_DIR, name + ext)
    return name
for k in ("keymap.back", "keymap.forward", "keymap.home", "keymap.pan", "keymap.zoom", "keymap.save",
          "keymap.quit", "keymap.fullscreen", "keymap.grid", "keymap.yscale", "keymap.xscale"):
    if k in plt.rcParams:
        plt.rcParams[k] = []


class Annotator:
    def __init__(self):
        self.i = 0
        self.fig, self.ax = plt.subplots(figsize=(14, 8.5))
        self.fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.01)
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("scroll_event", self.on_scroll)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.load()

    def path(self):
        return os.path.join(OUT, os.path.basename(PHOTOS[self.i]) + ".json")

    def load(self):
        rel = PHOTOS[self.i]
        self.img = load_rgb(photo_file(rel))
        self.H, self.W = self.img.shape[:2]
        self.marks = {"sure": [], "unsure": []}
        if os.path.exists(self.path()):
            d = json.load(open(self.path()))
            self.marks = {"sure": d.get("certain", d.get("sure", [])), "unsure": d.get("uncertain", d.get("unsure", []))}
        self.ax.clear(); self.ax.imshow(self.img); self.ax.set_axis_off()
        self.ax.set_xlim(0, self.W); self.ax.set_ylim(self.H, 0)
        self.art = []; self.redraw()

    def save(self):
        json.dump({"photo": os.path.splitext(os.path.basename(PHOTOS[self.i]))[0],
                   "image_size": [self.W, self.H], "certain": self.marks["sure"], "uncertain": self.marks["unsure"]},
                  open(self.path(), "w"), indent=1)

    def redraw(self):
        for a in self.art:
            a.remove()
        self.art = []
        for key, col in (("sure", "#00e000"), ("unsure", "#ff9900")):
            if self.marks[key]:
                p = np.array(self.marks[key])
                self.art.append(self.ax.scatter(p[:, 0], p[:, 1], s=90, facecolors="none", edgecolors=col, linewidths=1.6))
        self.ax.set_title(f"[{self.i + 1}/{len(PHOTOS)}] {PHOTOS[self.i]}    certain: {len(self.marks['sure'])}   "
                          f"uncertain: {len(self.marks['unsure'])}      left = certain, right = uncertain, "
                          f"middle/d = delete, wheel = zoom, r = reset, n/p = next/prev, q = quit", fontsize=9)
        self.fig.canvas.draw_idle()

    def nearest(self, x, y):
        best = None
        for key in ("sure", "unsure"):
            for j, (u, v) in enumerate(self.marks[key]):
                d = (u - x) ** 2 + (v - y) ** 2
                if best is None or d < best[0]:
                    best = (d, key, j)
        return best

    def delete_near(self, x, y):
        b = self.nearest(x, y)
        if b is not None:
            self.marks[b[1]].pop(b[2]); self.save(); self.redraw()

    def on_click(self, e):
        tb = getattr(self.fig.canvas, "toolbar", None)
        if e.inaxes != self.ax or e.xdata is None or (tb is not None and getattr(tb, "mode", "")):
            return
        x, y = float(e.xdata), float(e.ydata)
        if e.button == 1:
            self.marks["sure"].append([round(x, 1), round(y, 1)])
        elif e.button == 3:
            self.marks["unsure"].append([round(x, 1), round(y, 1)])
        elif e.button == 2:
            self.delete_near(x, y); return
        self.save(); self.redraw()

    def on_scroll(self, e):
        if e.inaxes != self.ax or e.xdata is None:
            return
        f = 0.7 if e.button == "up" else 1 / 0.7
        x0, x1 = self.ax.get_xlim(); y0, y1 = self.ax.get_ylim()
        nx0 = e.xdata - (e.xdata - x0) * f; nx1 = e.xdata + (x1 - e.xdata) * f
        ny0 = e.ydata - (e.ydata - y0) * f; ny1 = e.ydata + (y1 - e.ydata) * f
        if nx1 - nx0 > self.W:
            nx0, nx1, ny0, ny1 = 0, self.W, self.H, 0
        self.ax.set_xlim(nx0, nx1); self.ax.set_ylim(ny0, ny1); self.fig.canvas.draw_idle()

    def on_key(self, e):
        if e.key in ("n", "right") and self.i < len(PHOTOS) - 1:
            self.save(); self.i += 1; self.load()
        elif e.key in ("p", "left") and self.i > 0:
            self.save(); self.i -= 1; self.load()
        elif e.key == "r":
            self.ax.set_xlim(0, self.W); self.ax.set_ylim(self.H, 0); self.fig.canvas.draw_idle()
        elif e.key == "d" and e.xdata is not None:
            self.delete_near(e.xdata, e.ydata)
        elif e.key == "q":
            self.save(); plt.close(self.fig)


if __name__ == "__main__":
    nums = [a for a in sys.argv[1:] if a.isdigit()]
    if nums:
        start = int(nums[0]) - 1
    else:
        start = next((k for k, p in enumerate(PHOTOS) if not os.path.exists(os.path.join(OUT, os.path.basename(p) + ".json"))), 0)
    a = Annotator(); a.i = start; a.load()
    plt.show()
