#!/usr/bin/env python3
"""
ASTRID-Science: System 1 (LLM + conformal controller) and System 2 (RL) in one
journal-style figure.

How the figure is built
-----------------------
* Base layer = the System 1 figure, with its processing-order circles 1-8 on the
  same components (dark navy, matching \\fstep{1..8} in the paper).
* Overlay = System 2, drawn entirely in purple (#6A3D9A, matching \\rstep{1..6}).
  Purple circles keep the numbering of the stand-alone System 2 figure:
    1 mission-world simulator   2 offline RL training   3 certification
    4 release package           5 RL query policy        6 science team
* System 2 keeps the deterministic planner in the outer loop and replaces only
  Gate B's selection rule; Gate B's checks become the shield around the policy.
* Remove everything purple and the System 1 figure is recovered.

Usage
-----
    python astrid_science_combined_architecture.py

Writes astrid_science_combined_architecture.{png,pdf} next to this script.
Requires Python 3.8+ and matplotlib >= 3.6 (no GUI needed).

Editing tips
------------
* Geometry lives on a 100 x GRID_H grid (x to the right, y up); 1 unit = 0.07 in.
* Font sizes are in points. After any text or layout edit, rerun the script: it
  prints a warning for every label that no longer fits inside its box.
"""
import math
from pathlib import Path

import matplotlib as mpl
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.path import Path as MPath

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
SHOW_STEPS = True          # navy 1-8 (processing order) and purple 1-6 (System 2)
FIG_WIDTH_IN = 7.0         # same width as the System 1 figure
OUT_STEM = "astrid_science_combined_architecture"
FORMATS = ("png", "pdf")   # add "svg", "eps" or "tiff" if needed
PNG_DPI = 600

GRID_W, GRID_H = 100.0, 82.6
FIG_HEIGHT_IN = FIG_WIDTH_IN * GRID_H / GRID_W

# First installed family wins; Liberation Sans and TeX Gyre Heros are
# metric-compatible stand-ins for Arial and Helvetica.
FONT_CHAIN = ["Arial", "Helvetica", "Liberation Sans", "TeX Gyre Heros", "DejaVu Sans"]

FS = dict(band=6.2, group=7.0, group_sub=5.7, title=6.9, body=5.8, cell=5.5,
          chip=5.3, note=5.5, step=5.4, legend=5.8)  # font sizes in points
LINE = 1.22                                            # line-spacing multiple
PITCH, LINE_H = 1.10, 0.94  # measured baseline pitch and single-line height, in em

NAVY, RL = "#173452", "#6A3D9A"  # badge colours: System 1 order, System 2 lifecycle
STYLE = {
    "new":      dict(fc="#DDE9F6", ec="#2F5E8E", tc="#173452", lw=0.8),
    "rl":       dict(fc="#ECE3F6", ec=RL, tc="#3B1F5E", lw=0.9),
    "existing": dict(fc="#EEEEEE", ec="#808080", tc="#2A2A2A", lw=0.8),
    "boundary": dict(fc="#DCEEDA", ec="#3B7B36", tc="#1D4A1A", lw=1.1),
    "group":    dict(fc="#F6F9FC", ec="#A8BCD0", tc="#173452", lw=0.6),
    "rlgroup":  dict(fc="#FAF7FD", ec="#B9A3D3", tc=RL, lw=0.6),
    "cell":     dict(fc="#FFFFFF", ec="#ABABAB", tc="#2A2A2A", lw=0.5),
    "chip":     dict(fc="#FFFFFF", ec="#2F5E8E", tc="#173452", lw=0.6),
    "people":   dict(fc="#FFFFFF", ec="#808080", tc="#2A2A2A", lw=0.8),
}
DASH = (0, (3.2, 1.8))
DOT = (0, (1.0, 1.5))
FLOW = {  # arrow styles
    "cmd":  dict(color="#1E1E1E", lw=0.85, ls="solid"),  # request / decision flow
    "data": dict(color="#2F5E8E", lw=0.85, ls="solid"),  # observation / record flow
    "cfg":  dict(color="#6F6F6F", lw=0.9, ls=DOT),       # configuration / downlink
    "rl":   dict(color=RL, lw=0.9, ls="solid"),          # System 2 only
}
BODY_C, NOTE_C = "#333333", "#3C3C3C"
BAND_EC, STRIP_FC, STRIP_TC = "#BEBEBE", "#EDEDED", "#4D4D4D"
CYL_TOP_FC, WHITE = "#C9DBEE", "#FFFFFF"

BX0, BX1, STRIP_W = 1.0, 99.6, 3.8  # band extents and side-label strip width
CYL_RY = 1.25                        # half-height of the ledger's elliptical rims

_FIT = []  # (text artist, box bounds, name) for the overflow check


# ----------------------------------------------------------------------------
# Drawing primitives (same look as the System 1 and System 2 figures)
# ----------------------------------------------------------------------------
def pt(v):
    """Points -> grid units."""
    return v / 72.0 * GRID_W / FIG_WIDTH_IN


def _lines_h(n, fs):
    """Rendered height (grid units) of n text lines at size fs (used for centring)."""
    return ((n - 1) * PITCH + LINE_H) * pt(fs) if n else 0.0


def _as_line(item):
    """Body lines are plain strings or (text, colour, style) tuples."""
    if isinstance(item, tuple):
        text, color, style = (tuple(item) + (BODY_C, "normal"))[:3]
        return text, color, style
    return item, BODY_C, "normal"


def block(ax, bounds, title=None, lines=(), kind="new", tfs=None, bfs=None, gap=0.35,
          valign="center", pad_top=0.45, name=None, z=5.0):
    """Bold title over body lines, laid out exactly as in the System 1 figure.

    Body lines may carry their own colour: each colour is drawn as a separate text
    with the other lines left blank, so matplotlib lays every copy out identically
    and the coloured lines land exactly where they belong.
    """
    x0, y0, x1, y1 = bounds
    tfs, bfs = tfs or FS["title"], bfs or FS["body"]
    bl = [_as_line(item) for item in lines]
    n_t = title.count("\n") + 1 if title else 0
    h_t, h_b = _lines_h(n_t, tfs), _lines_h(len(bl), bfs)
    g = gap if (title and bl) else 0.0
    if valign == "center":
        top = (y0 + y1) / 2 + (h_t + g + h_b) / 2
    else:
        top = y1 - pad_top
    xc = (x0 + x1) / 2
    label = name or (title.splitlines()[0] if title else bl[0][0])
    if title:
        t = ax.text(xc, top, title, ha="center", va="top", fontsize=tfs, fontweight="bold",
                    color=STYLE[kind]["tc"], linespacing=LINE, zorder=z)
        _FIT.append((t, bounds, label))
    if bl:
        looks = []
        for _, c, st in bl:
            if (c, st) not in looks:
                looks.append((c, st))
        for c, st in looks:
            txt = "\n".join(s if (cc, ss) == (c, st) else "" for s, cc, ss in bl)
            b = ax.text(xc, top - h_t - g, txt, ha="center", va="top", fontsize=bfs,
                        color=c, fontstyle=st, linespacing=LINE, zorder=z)
            _FIT.append((b, bounds, label))


def rbox(ax, x0, y0, x1, y1, kind, dashed=False, r=0.55, z=2.0):
    s = STYLE[kind]
    ax.add_patch(FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0, boxstyle=f"round,pad=0,rounding_size={r}",
        fc=s["fc"], ec=s["ec"], lw=s["lw"], ls=DASH if dashed else "solid", zorder=z))


def component(ax, x0, y0, x1, y1, title, lines=(), kind="new", dashed=False, z=2.0, **kw):
    rbox(ax, x0, y0, x1, y1, kind, dashed=dashed, z=z)
    block(ax, (x0, y0, x1, y1), title, lines, kind, z=z + 3.0, **kw)


def cylinder(ax, x0, y0, x1, y1, kind="new", z=2.0):
    s, ry = STYLE[kind], CYL_RY
    w, cx = x1 - x0, (x0 + x1) / 2
    ax.add_patch(Ellipse((cx, y0 + ry), w, 2 * ry, fc=s["fc"], ec=s["ec"], lw=s["lw"], zorder=z))
    ax.add_patch(Rectangle((x0, y0 + ry), w, y1 - y0 - 2 * ry, fc=s["fc"], ec="none", zorder=z + 0.01))
    for x in (x0, x1):
        ax.plot([x, x], [y0 + ry, y1 - ry], color=s["ec"], lw=s["lw"],
                solid_capstyle="butt", zorder=z + 0.02)
    ax.add_patch(Ellipse((cx, y1 - ry), w, 2 * ry, fc=CYL_TOP_FC, ec=s["ec"], lw=s["lw"], zorder=z + 0.03))


def rim_y(x, x0, x1, y_mid, upper=True):
    """y on a cylinder rim (ellipse centred at y_mid) at abscissa x."""
    a, cx = (x1 - x0) / 2, (x0 + x1) / 2
    dy = CYL_RY * math.sqrt(max(0.0, 1.0 - ((x - cx) / a) ** 2))
    return y_mid + dy if upper else y_mid - dy


HEAD = "head_length=0.45,head_width=0.2"
HEAD_SCALE = 9


def _head(ax, p_from, p_to, f, z):
    """Solid arrowhead on the final stretch of a segment (keeps dotted links crisp)."""
    (x0, y0), (x1, y1) = p_from, p_to
    d = math.hypot(x1 - x0, y1 - y0)
    k = min(0.85, d) / d
    ax.add_patch(FancyArrowPatch(
        (x1 - (x1 - x0) * k, y1 - (y1 - y0) * k), (x1, y1), arrowstyle=f"-|>,{HEAD}",
        mutation_scale=HEAD_SCALE, color=f["color"], lw=f["lw"], shrinkA=0, shrinkB=0,
        zorder=z + 0.01))


def flow(ax, pts, kind="cmd", both=False, z=3.0):
    """Orthogonal poly-line arrow through `pts` (head at the last point)."""
    f = FLOW[kind]
    if f["ls"] == "solid":
        head = "<|-|>" if both else "-|>"
        path = MPath(pts, [MPath.MOVETO] + [MPath.LINETO] * (len(pts) - 1))
        ax.add_patch(FancyArrowPatch(
            path=path, arrowstyle=f"{head},{HEAD}", mutation_scale=HEAD_SCALE,
            color=f["color"], lw=f["lw"], joinstyle="miter", capstyle="butt", zorder=z))
        return
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=f["color"], lw=f["lw"], ls=f["ls"], dash_capstyle="butt",
            solid_joinstyle="miter", zorder=z)
    _head(ax, pts[-2], pts[-1], f, z)
    if both:
        _head(ax, pts[1], pts[0], f, z)


def note(ax, x, y, s, ha="center", va="center", fs=None, bg=WHITE, rot=0,
         style="italic", color=NOTE_C, weight="normal", z=6.0):
    kw = dict(bbox=dict(boxstyle="square,pad=0.12", fc=bg, ec="none")) if bg else {}
    return ax.text(x, y, s, ha=ha, va=va, fontsize=fs or FS["note"], fontstyle=style,
                   color=color, rotation=rot, fontweight=weight, zorder=z, **kw)


def badge(ax, x, y, n, fc=NAVY, r=0.95):
    """Numbered circle: navy = System 1 processing order, purple = System 2 lifecycle."""
    if not SHOW_STEPS:
        return
    ax.add_patch(Circle((x, y), r, fc=fc, ec=WHITE, lw=0.6, zorder=8))
    ax.text(x, y - 0.05, str(n), ha="center", va="center", fontsize=FS["step"],
            fontweight="bold", color=WHITE, zorder=9)


def band(ax, y0, y1, label):
    ax.add_patch(Rectangle((BX0, y0), BX1 - BX0, y1 - y0, fc=WHITE, ec=BAND_EC, lw=0.6, zorder=0))
    ax.add_patch(Rectangle((BX0, y0), STRIP_W, y1 - y0, fc=STRIP_FC, ec=BAND_EC, lw=0.6, zorder=0.1))
    ax.text(BX0 + STRIP_W / 2, (y0 + y1) / 2, label, rotation=90, ha="center", va="center",
            fontsize=FS["band"], fontweight="bold", color=STRIP_TC, linespacing=1.1, zorder=1)


def group(ax, x0, y0, x1, y1, title, subtitle):
    rbox(ax, x0, y0, x1, y1, "group", r=0.8, z=1.0)
    ax.text(x0 + 0.9, y1 - 0.55, title, ha="left", va="top", fontsize=FS["group"],
            fontweight="bold", color=STYLE["group"]["tc"], zorder=5)
    ax.text(x0 + 0.9, y1 - 0.55 - pt(FS["group"]) * 1.3, subtitle, ha="left", va="top",
            fontsize=FS["group_sub"], fontstyle="italic", color=NOTE_C, zorder=5)


# ----------------------------------------------------------------------------
# Figure content
# ----------------------------------------------------------------------------
LEDGER = (36.0, 44.0, 57.6, 57.0)   # x0, y0, x1, y1 of the mission ledger
TEAM = (6.0, 62.0, 22.4, 72.0)      # science & engineering team
PKG = (26.4, 62.0, 99.0, 72.0)      # versioned release package


def draw_ground(ax):
    band(ax, 61.0, 82.2, "GROUND\n(all learning happens here)")

    # --- shared: team and release package (as in the System 1 figure) -------------
    tx0, ty0, tx1, ty1 = TEAM
    component(ax, tx0, ty0, tx1, ty1, "Science & engineering\nteam",
              ["sets goals & priorities", "ontology · error tolerances",
               "reviews downlinked data", ("+ reward weights · deferrals", RL, "italic")],
              kind="people")
    badge(ax, tx0, ty1, 6, fc=RL)

    x0, y0, x1, y1 = PKG
    rbox(ax, x0, y0, x1, y1, "existing")
    ax.text((x0 + x1) / 2, y1 - 0.6, "Versioned release package · frozen onboard, "
            "replaced only at a release boundary", ha="center", va="top", fontsize=FS["title"],
            fontweight="bold", color=STYLE["existing"]["tc"], zorder=5)
    cells = [("Science", ["goals · ontology", "STM–MCDM version"]),
             ("Evidence", ["models · familiarity", "conformal cutoffs"]),
             ("Actions", ["templates · roles", "resource bounds"]),
             ("Supervisor", ["LLM · prompt", "document corpus"]),
             ("Diagnostics", ["full-set support", "table · thresholds",
                              (r"+ RL policy $\pi_\theta$", RL, "italic")]),
             ("Assurance", ["data splits", "risk tests · approval"])]
    gap, cx0, cx1, cy0, cy1 = 0.8, x0 + 1.0, x1 - 1.0, 63.9, 69.8
    cw = (cx1 - cx0 - gap * (len(cells) - 1)) / len(cells)
    for i, (head, text) in enumerate(cells):
        a = cx0 + i * (cw + gap)
        component(ax, a, cy0, a + cw, cy1, head, text, kind="cell", tfs=FS["cell"] + 0.5,
                  bfs=FS["cell"], gap=0.15, valign="top", pad_top=0.3)
    ax.text((x0 + x1) / 2, 62.95, r"built and risk-tested on independent data partitions: "
            r"$D_{\mathrm{fit}}$, $D_{\mathrm{set}}$, $D_{\mathrm{dev}}$, $D_{\mathrm{risk}}$, "
            r"$D_{\mathrm{final}}$", ha="center", va="center", fontsize=FS["body"],
            color=BODY_C, zorder=5)
    badge(ax, x0, y1, 4, fc=RL)
    flow(ax, [(tx1, 67.0), (x0, 67.0)], "cfg")

    # --- System 2: RL lifecycle (ground only) ------------------------------------
    gy0, gy1 = 74.2, 81.0
    rbox(ax, 6.0, gy0, 99.0, gy1, "rlgroup", r=0.8, z=1.0)
    note(ax, 52.0, gy1, "System 2 · RL lifecycle (ground only)", bg=WHITE, style="italic",
         weight="bold", color=RL, fs=FS["group_sub"] + 0.3)
    ry0, ry1 = 75.0, 80.2
    component(ax, 6.8, ry0, 31.0, ry1, "Mission-world simulator",
              ["hidden-truth worlds from paired", "analog & prior-mission data"], kind="rl")
    component(ax, 36.0, ry0, 60.0, ry1, "Offline RL training",
              ["reward: correct decision − cost", "− delay − wrong-commit penalty"], kind="rl")
    component(ax, 65.0, ry0, 98.2, ry1, "Certification",
              ["shield verified · off-policy & simulator evaluation",
               r"whole-policy risk test on $D_{\mathrm{risk}}$"], kind="rl")
    for x, n in ((6.8, 1), (36.0, 2), (65.0, 3)):
        badge(ax, x, ry1, n, fc=RL)
    ym = (ry0 + ry1) / 2
    flow(ax, [(31.0, ym), (36.0, ym)], "rl")
    flow(ax, [(60.0, ym), (65.0, ym)], "rl")
    flow(ax, [(94.5, ry0), (94.5, y1)], "rl")                              # certified policy
    note(ax, 93.8, 73.1, r"certified $\pi_\theta$", ha="right", color=RL)
    flow(ax, [(14.2, ty1), (14.2, ry0)], "rl")                             # labels -> simulator
    note(ax, 14.9, 73.1, "labels, new data", ha="left", color=RL)


def draw_onboard(ax):
    band(ax, 20.0, 59.0, "ONBOARD AUTONOMY\n(hosted in OnAIR)")

    # --- outer loop: proposal policy ------------------------------------------------
    group(ax, 6.0, 31.0, 30.4, 58.0, "Proposal policy", "outer loop: what to observe next")
    component(ax, 7.4, 43.4, 29.0, 53.4, "Bounded LLM supervisor",
              ["reads approved documents and the", "mission record via read-only tools;",
               "proposes ≤ 3 SURVEY steps", "as a typed plan (no commands)",
               ("outer loop of System 1", NAVY, "italic")])
    badge(ax, 7.4, 53.4, 1)
    note(ax, 18.2, 42.4, "or", bg=STYLE["group"]["fc"])
    component(ax, 7.4, 32.6, 29.0, 41.4, "Deterministic planner",
              ["objective-aware comparator and", "fallback; same request interface",
               ("outer loop of System 2", RL, "italic")])

    # --- shared state ------------------------------------------------------------------
    lx0, ly0, lx1, ly1 = LEDGER
    cylinder(ax, lx0, ly0, lx1, ly1)
    block(ax, (lx0, ly0 + CYL_RY, lx1, ly1 - 2 * CYL_RY), "Mission ledger",
          ["active release · evidence · outcomes", "requests · attempts · reservations"])

    # --- admission boundary --------------------------------------------------------------
    component(ax, 6.0, 21.0, 60.6, 29.0, "Mission executive · command admission boundary",
              ["schema, ID and snapshot validation · resource reservation · idempotent dispatch",
               "origin permissions:  LLM → SURVEY only;  local controller → DIAGNOSTIC, COMMIT"],
              kind="boundary")
    badge(ax, 6.0, 29.0, 2)

    # --- inner loop: local science controller -------------------------------------------
    group(ax, 63.6, 21.0, 99.0, 58.0, "Local science controller",
          "inner loop: what to do about this target")
    component(ax, 64.6, 45.4, 78.8, 53.4, "Episode outcome",
              ["COMMIT_REQUESTED", "CONTINUE", "CONTINUE_FLAGGED",
               ("DEFERRED_TO_GROUND", RL, "normal")], bfs=FS["cell"])
    component(ax, 84.1, 45.4, 97.9, 53.4, "Pinned OBAI MCDM",
              ["existing STM-weighted", "COMMIT vs CONTINUE;", "one query per $h$"],
              kind="existing")
    draw_gate_b(ax, 64.6, 29.0, 78.8, 42.2)
    component(ax, 82.0, 33.4, 97.9, 42.2, "Gate A",
              ["$d(h;s)$ for every $h \\in H$", "unanimous → act", "split → Gate B"])
    component(ax, 82.0, 22.2, 97.9, 30.6, "Evidence pipeline",
              ["quality checks · analysis", "conformal prediction set $C$",
               "familiarity · association", r"fuse:  $H \leftarrow H \cap C$"],
              bfs=FS["cell"] + 0.1)
    badge(ax, 82.0, 30.6, 5)
    badge(ax, 82.0, 42.2, 6)
    badge(ax, 64.6, 42.2, 7)
    badge(ax, 64.6, 53.4, 8)

    gbg = STYLE["group"]["fc"]
    flow(ax, [(89.95, 30.6), (89.95, 33.4)], "data")                       # evidence -> Gate A
    note(ax, 90.7, 32.0, "$H$", ha="left", bg=gbg, style="normal")
    flow(ax, [(91.5, 42.2), (91.5, 45.4)], "data", both=True)              # Gate A <-> MCDM
    note(ax, 92.2, 43.8, "per-$h$ query", ha="left", bg=gbg)
    flow(ax, [(82.0, 39.0), (78.8, 39.0)], "cmd")                          # split
    note(ax, 80.4, 40.0, "split", bg=gbg)
    flow(ax, [(83.3, 42.2), (83.3, 49.4), (78.8, 49.4)], "cmd")            # unanimous
    flow(ax, [(71.9, 42.2), (71.9, 45.4)], "cmd")                          # to outcome
    note(ax, 72.6, 44.35, "none eligible", ha="left", bg=gbg)
    note(ax, 72.6, 43.2, "defer / stop", ha="left", bg=gbg, color=RL)
    flow(ax, [(70.0, 29.0), (70.0, 26.4), (60.6, 26.4)], "cmd")            # DIAGNOSTIC
    note(ax, 70.7, 27.7, "DIAGNOSTIC", ha="left", bg=gbg, style="normal", fs=FS["note"] - 0.3)
    flow(ax, [(82.0, 35.0), (80.4, 35.0), (80.4, 23.4), (60.6, 23.4)], "cmd")  # COMMIT
    note(ax, 79.7, 25.0, "COMMIT", ha="right", bg=gbg, style="normal", fs=FS["note"] - 0.3)
    flow(ax, [(64.6, 49.4), (lx1, 49.4)], "data")                          # outcome -> ledger
    note(ax, 61.1, 49.4, "record", bg=WHITE)

    # --- links through the centre ---------------------------------------------------------
    flow(ax, [(lx0, 50.6), (30.4, 50.6)], "data")                          # ledger -> policy
    note(ax, 33.2, 52.5, "read-only\nsnapshot", bg=WHITE)
    xm = (lx0 + lx1) / 2                                                   # executive <-> ledger
    flow(ax, [(xm, 29.0), (xm, rim_y(xm, lx0, lx1, ly0 + CYL_RY, upper=False))], "data", both=True)
    note(ax, xm + 1.25, 36.5, "status · reservations", rot=90, bg=WHITE)
    flow(ax, [(17.6, 31.0), (17.6, 29.0)], "cmd")                          # SURVEY request
    note(ax, 18.3, 30.0, "SURVEY only", ha="left", bg=WHITE, style="normal",
         fs=FS["note"] - 0.3)

    # --- ground <-> onboard ------------------------------------------------------------------
    top = ly1 - CYL_RY
    flow(ax, [(50.0, PKG[1]), (50.0, rim_y(50.0, lx0, lx1, top))], "cfg")  # install
    note(ax, 50.7, 60.0, "install", ha="left")
    xs = 42.0
    path = [(xs, rim_y(xs, lx0, lx1, top)), (xs, 60.0), (14.2, 60.0), (14.2, TEAM[1])]
    flow(ax, path, "cfg")                                                  # downlink
    _head(ax, path[1], path[0], FLOW["rl"], 3.0)                           # System 2 uplink
    note(ax, 18.6, 60.0, "downlink")
    note(ax, 31.0, 60.0, "uplink: ground decisions", color=RL)


def draw_gate_b(ax, x0, y0, x1, y1):
    """Gate B: shared eligibility checks (the System 2 shield) around two selectors."""
    rbox(ax, x0, y0, x1, y1, "new")
    block(ax, (x0, y0, x1, y1), "Gate B",
          ["full-set support table", r"$r_{\mathrm{wrong}} \leq \eta_{\max}$, budget"],
          valign="top", pad_top=0.5, bfs=FS["cell"] + 0.1, z=5.0)
    m, h, sep, b0 = 0.55, 3.1, 1.3, 0.5
    c2 = (x0 + m, y0 + b0, x1 - m, y0 + b0 + h)
    c1 = (x0 + m, c2[3] + sep, x1 - m, c2[3] + sep + h)
    component(ax, *c1, r"System 1 · rule, $K{=}1$",
              [r"$r_{\mathrm{good}} \geq \rho_{\min}$, best $r_{\mathrm{good}}/c$"],
              kind="chip", tfs=FS["chip"], bfs=FS["chip"], gap=0.15, z=3.0,
              valign="top", pad_top=0.35)
    note(ax, (x0 + x1) / 2, c2[3] + sep / 2, "or", bg=STYLE["new"]["fc"], fs=FS["chip"])
    component(ax, *c2, r"System 2 · RL $\pi_\theta$",
              ["diagnose · defer · stop"], kind="rl", tfs=FS["chip"], bfs=FS["chip"],
              gap=0.15, z=3.0, valign="top", pad_top=0.35)
    badge(ax, c2[0], c2[3], 5, fc=RL)


def draw_execution(ax):
    band(ax, 5.2, 18.0, "EXISTING\nASTRA STACK")
    component(ax, 6.0, 6.2, 30.6, 16.8, "Task planner & ReCAP auction",
              ["schedules admitted jobs and", "allocates assets; may decline"], kind="existing")
    component(ax, 34.6, 6.2, 58.6, 16.8, "Behavior-tree executor",
              ["flight software · pre-dispatch re-check", "status & abort · independent safety"],
              kind="existing")
    x0, x1, y0, y1 = 62.6, 99.0, 6.2, 16.8
    rbox(ax, x0, y0, x1, y1, "existing")
    ax.text((x0 + x1) / 2, y1 - 0.55, "Instruments & assets", ha="center", va="top",
            fontsize=FS["title"], fontweight="bold", color=STYLE["existing"]["tc"], zorder=5)
    subs = [("Survey\nmode", ["scout sensors"]),
            ("Diagnostic\nmodes", [r"$D_1, \ldots, D_m$"]),
            ("Targeted\nfollow-up", ["high-SWaP asset"])]
    gap, a0, a1 = 1.0, x0 + 1.0, x1 - 1.0
    w = (a1 - a0 - gap * (len(subs) - 1)) / len(subs)
    for i, (head, text) in enumerate(subs):
        a = a0 + i * (w + gap)
        component(ax, a, 7.0, a + w, 14.0, head, text, kind="cell",
                  tfs=FS["cell"] + 0.6, bfs=FS["cell"])
    badge(ax, 6.0, 16.8, 3)
    badge(ax, 62.6, 16.8, 4)

    flow(ax, [(17.6, 21.0), (17.6, 16.8)], "cmd")                     # admitted jobs
    note(ax, 18.3, 19.0, "admitted jobs", ha="left")
    flow(ax, [(30.6, 11.5), (34.6, 11.5)], "cmd")
    flow(ax, [(58.6, 11.5), (62.6, 11.5)], "cmd")
    flow(ax, [(89.95, 16.8), (89.95, 22.2)], "data")                  # observations
    note(ax, 89.2, 19.0, "observations + footprint/time", ha="right")


def draw_legend(ax, fig):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    def width(t):
        return t.get_window_extent(renderer).width / fig.dpi / FIG_WIDTH_IN * GRID_W

    def run(y, items):
        x = BX0 + 0.4
        for kind, key, label in items:
            if kind == "box":
                rbox(ax, x, y - 0.75, x + 2.8, y + 0.75, key, r=0.3)
                x += 2.8
            elif kind == "line":
                flow(ax, [(x, y), (x + 4.4, y)], key)
                x += 4.4
            else:
                badge(ax, x + 0.95, y, 1, fc=key)
                x += 1.9
            t = ax.text(x + 0.8, y, label, ha="left", va="center", fontsize=FS["legend"],
                        color=BODY_C, zorder=5)
            x += 0.8 + width(t) + 2.6
        if x - 2.6 > BX1:
            print(f"[fit] legend row at y={y} is wider than the figure ({x - 2.6:.1f})")

    row1 = [("box", "new", "New in this work"),
            ("box", "rl", "System 2 (RL) addition"),
            ("box", "existing", "Existing capability / approved input"),
            ("box", "boundary", "Admission boundary")]
    row2 = [("line", "cmd", "Request / decision"),
            ("line", "data", "Observation / record"),
            ("line", "cfg", "Install / downlink"),
            ("line", "rl", "System 2 link")]
    if SHOW_STEPS:
        row2 += [("step", NAVY, "Processing order"), ("step", RL, "System 2 lifecycle")]
    run(3.35, row1)
    run(1.15, row2)


def check_fit(fig, ax, margin=0.25):
    """Warn about any registered label that spills outside its box."""
    fig.canvas.draw()
    renderer, inv = fig.canvas.get_renderer(), ax.transData.inverted()
    bad = 0
    for art, (x0, y0, x1, y1), name in _FIT:
        bb = art.get_window_extent(renderer)
        (a0, b0), (a1, b1) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
        if a0 < x0 + margin or a1 > x1 - margin or b0 < y0 + 0.6 * margin or b1 > y1 - 0.6 * margin:
            bad += 1
            print(f"[fit] '{name.splitlines()[0]}': text x {a0:.1f}-{a1:.1f}, y {b0:.1f}-{b1:.1f} "
                  f"vs box x {x0:.1f}-{x1:.1f}, y {y0:.1f}-{y1:.1f}")
    print("[fit] all box labels fit" if not bad else f"[fit] {bad} label(s) need attention")
    return bad


def build():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": FONT_CHAIN,
        "mathtext.fontset": "custom",
        "mathtext.rm": "sans",
        "mathtext.it": "sans:italic",
        "mathtext.bf": "sans:bold",
        "mathtext.cal": "sans:italic",
        "mathtext.sf": "sans",
        "mathtext.fallback": "stixsans",
        "pdf.fonttype": 42,     # embed TrueType, no Type 3 fonts (IEEE PDF eXpress)
        "ps.fonttype": 42,
        "svg.fonttype": "none",  # keep SVG text editable
    })
    _FIT.clear()
    fig = Figure(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN), dpi=PNG_DPI)
    FigureCanvasAgg(fig)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, GRID_W)
    ax.set_ylim(0, GRID_H)
    ax.axis("off")

    draw_ground(ax)
    draw_onboard(ax)
    draw_execution(ax)
    draw_legend(ax, fig)
    return fig, ax


def main():
    fig, ax = build()
    check_fit(fig, ax)
    try:
        out_dir = Path(__file__).resolve().parent
    except NameError:  # pasted into a notebook
        out_dir = Path.cwd()
    for ext in FORMATS:
        path = out_dir / f"{OUT_STEM}.{ext}"
        fig.savefig(path, dpi=PNG_DPI if ext in ("png", "tiff") else None,
                    facecolor="white")
        print("wrote", path)


if __name__ == "__main__":
    main()