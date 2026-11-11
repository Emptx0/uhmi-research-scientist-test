"""Карти та графіки для презентації (аудиторія — фермери)."""
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib import font_manager as fm

for f in fm.findSystemFonts():
    if "/inter/" in f.lower():
        fm.fontManager.addfont(f)
plt.rcParams.update({"font.family": "Inter", "font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 100, "savefig.dpi": 220})

INK, MUTED, GRID = "#22211f", "#6b6a66", "#e4e2dc"
# Класи SPEI: дивергентна шкала (коричневий = сухо, синьо-зелений = волого, сірий = норма)
BOUNDS = [-9, -2, -1.5, -1, 1, 1.5, 9]
COLORS = ["#7a2e0e", "#c0561c", "#eba04f", "#efeeea", "#7fbfb6", "#2b7a72"]
LABELS = ["екстремальна\nпосуха", "сильна\nпосуха", "помірна\nпосуха", "норма", "волого", "дуже\nволого"]
cmap, norm = ListedColormap(COLORS), BoundaryNorm(BOUNDS, len(COLORS))

g = np.load("out/spei_grids.npz")
lat, lon, years, mask = g["lat"], g["lon"], g["years"], g["mask"]
spei6 = g["spei6"]
df = pd.read_csv("out/drought_years_table.csv")
adm = gpd.read_file("data_ukr_adm1.geojson" if False else "ukr_adm1.geojson")
obl = adm[adm.shapeName.str.contains("Mykolaiv")]
ext = [lon.min() - 0.05, lon.max() + 0.05, lat.min() - 0.03, lat.max() + 0.03]
dlon, dlat = np.diff(lon).mean() / 2, np.diff(lat).mean() / 2
extent = [lon.min() - dlon, lon.max() + dlon, lat.min() - dlat, lat.max() + dlat]

CITIES = {"Миколаїв": (31.99, 46.97), "Первомайськ": (30.85, 48.05), "Вознесенськ": (31.33, 47.57),
          "Баштанка": (32.44, 47.41), "Очаків": (31.54, 46.62), "Новий Буг": (32.51, 47.69),
          "Южноукраїнськ": (31.18, 47.82)}
DROUGHT = set(df[df.drought_year].year)


def base_map(ax, data, cities=False, lw=0.8):
    adm.boundary.plot(ax=ax, color="#c9c7c0", lw=0.4, zorder=1)
    ax.imshow(np.where(mask, data, np.nan), origin="lower", extent=extent, cmap=cmap, norm=norm,
              interpolation="nearest", zorder=2)
    obl.boundary.plot(ax=ax, color=INK, lw=lw, zorder=3)
    ax.set_xlim(30.05, 33.85); ax.set_ylim(46.25, 48.35)
    ax.set_aspect(1 / np.cos(np.deg2rad(47.3)))
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(""); ax.set_ylabel("")
    for s in ax.spines.values():
        s.set_visible(False)
    if cities:
        for n, (x, y) in CITIES.items():
            ax.plot(x, y, "o", ms=4, color=INK, mec="white", mew=0.8, zorder=4)
            ax.annotate(n, (x, y), xytext=(4, 3), textcoords="offset points", fontsize=9, color=INK,
                        zorder=5, path_effects=[matplotlib.patheffects.withStroke(linewidth=2.5, foreground="white")])


import matplotlib.patheffects  # noqa: E402


def legend(fig, y=0.02, x0=0.2, w=0.6, fs=9):
    lax = fig.add_axes([x0, y, w, 0.035])
    for i, (c, l) in enumerate(zip(COLORS, LABELS)):
        lax.add_patch(plt.Rectangle((i + 0.03, 0), 0.94, 1, color=c, ec="#bdbab2" if c == "#efeeea" else c, lw=0.6))
        lax.text(i + 0.5, -0.25, l, ha="center", va="top", fontsize=fs, color=INK, linespacing=1.0)
    lax.set_xlim(0, len(COLORS)); lax.set_ylim(-2.2, 1); lax.axis("off")


# ---------- 1. Часовий ряд ----------
fig, ax = plt.subplots(figsize=(12, 4.6))
cols = [COLORS[np.digitize(v, BOUNDS[1:-1])] for v in df.spei6_mean]
ax.bar(df.year, df.spei6_mean, color=cols, width=0.78, edgecolor=["#bdbab2" if c == "#efeeea" else c for c in cols], lw=0.6)
ax.axhline(0, color=MUTED, lw=0.8)
for y in [-1, -1.5, -2]:
    ax.axhline(y, color=GRID, lw=0.8, ls="--", zorder=0)
ax.text(1980.3, -1.03, "межа посухи (−1)", va="top", fontsize=9, color=MUTED)
for _, r in df[df.drought_year].iterrows():
    off = 0.30 if int(r.year) == 2018 else 0.08
    ax.text(r.year, r.spei6_mean - off, str(int(r.year)), ha="center", va="top", fontsize=10, fontweight="bold", color=INK)
ax.set_ylim(-2.6, 2.4); ax.set_xlim(1980, 2026)
ax.set_ylabel("SPEI-6, квітень–вересень\n(середнє по області)", color=INK)
ax.tick_params(colors=MUTED); ax.grid(False)
ax.text(1980.5, 2.25, "↑ волого", fontsize=10, color="#2b7a72"); ax.text(1980.5, -2.5, "↓ сухо", fontsize=10, color="#7a2e0e")
plt.tight_layout(); plt.savefig("out/fig1_timeline.png", bbox_inches="tight"); plt.close()

# ---------- 2. Мала мультиплікація: усі роки ----------
nc, nr = 9, 5
fig, axs = plt.subplots(nr, nc, figsize=(15, 8.3))
for k, ax in enumerate(axs.ravel()):
    if k >= len(years):
        ax.axis("off"); continue
    base_map(ax, spei6[k], lw=0.6)
    yv = int(years[k])
    ax.set_title(str(yv), fontsize=12, color=INK if yv in DROUGHT else MUTED,
                 fontweight="bold" if yv in DROUGHT else "normal", pad=2)
    if yv in DROUGHT:
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=False, ec="#7a2e0e", lw=1.8))
plt.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.12, wspace=0.04, hspace=0.22)
legend(fig, y=0.035, x0=0.22, w=0.56, fs=10)
plt.savefig("out/fig2_all_years.png", bbox_inches="tight"); plt.close()

# ---------- 3. Посушливі роки крупно ----------
dy = sorted(DROUGHT)
fig, axs = plt.subplots(2, 4, figsize=(14, 6.6))
for ax, yv in zip(axs.ravel(), dy):
    k = list(years).index(yv)
    base_map(ax, spei6[k], cities=(yv == dy[0]))
    r = df[df.year == yv].iloc[0]
    ax.set_title(f"{yv}  ·  опади IV–IX: {r.p_aprsep_mm:.0f} мм", fontsize=11.5, color=INK, pad=3)
# остання клітинка — частота посух 2003–2025
ax = axs.ravel()[-1]
fr = g["freq_late"]
fcmap = ListedColormap(["#f6dcc0", "#eba04f", "#c0561c", "#7a2e0e"])
fnorm = BoundaryNorm([6.5, 7.5, 8.5, 9.5, 11.5], 4)
adm.boundary.plot(ax=ax, color="#c9c7c0", lw=0.4)
im = ax.imshow(np.where(mask, fr, np.nan), origin="lower", extent=extent, cmap=fcmap, norm=fnorm, interpolation="nearest")
obl.boundary.plot(ax=ax, color=INK, lw=0.8)
ax.set_xlim(30.05, 33.85); ax.set_ylim(46.25, 48.35); ax.set_aspect(1 / np.cos(np.deg2rad(47.3)))
ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(""); ax.set_ylabel(""); [s.set_visible(False) for s in ax.spines.values()]
ax.set_title("Скільки разів за 2003–2025 рр.\nбула посуха (SPEI-6 ≤ −1)", fontsize=11, color=INK, pad=3)
cb = plt.colorbar(im, ax=ax, orientation="horizontal", fraction=0.05, pad=0.02, ticks=[7, 8, 9, 10.5])
cb.ax.set_xticklabels(["7", "8", "9", "10–11"]); cb.ax.tick_params(labelsize=9, colors=INK); cb.outline.set_visible(False)
plt.subplots_adjust(left=0.01, right=0.99, top=0.94, bottom=0.13, wspace=0.05, hspace=0.28)
legend(fig, y=0.03, x0=0.15, w=0.5, fs=9.5)
plt.savefig("out/fig3_drought_years.png", bbox_inches="tight"); plt.close()

# ---------- 4. Сезонність: весна / літо ----------
fig, ax = plt.subplots(figsize=(12, 3.0))
rows = [("Весна (III–V)\nозимі, ранні ярі", "spei3_spring_mean"), ("Літо (VI–VIII)\nсоняшник, кукурудза", "spei3_summer_mean")]
for i, (lab, c) in enumerate(rows):
    for _, r in df.iterrows():
        v = r[c]; col = COLORS[np.digitize(v, BOUNDS[1:-1])]
        ax.add_patch(plt.Rectangle((r.year - 0.45, 1 - i - 0.42), 0.9, 0.84, color=col,
                                   ec="#bdbab2" if col == "#efeeea" else col, lw=0.5))
ax.set_yticks([1, 0]); ax.set_yticklabels([r[0] for r in rows], fontsize=10, color=INK)
ax.set_xlim(1980.4, 2025.6); ax.set_ylim(-0.6, 1.6)
ax.set_xticks(range(1981, 2026, 4)); ax.tick_params(colors=MUTED, length=0)
for s in ax.spines.values(): s.set_visible(False)
plt.subplots_adjust(left=0.17, right=0.99, top=0.98, bottom=0.30)
legend(fig, y=0.1, x0=0.22, w=0.7, fs=9)
plt.savefig("out/fig4_seasons.png", bbox_inches="tight"); plt.close()

# ---------- 5. Тренд опадів і випаровуваності ----------
fig, ax = plt.subplots(figsize=(6.5, 3.6))
deficit = df.pet_aprsep_mm - df.p_aprsep_mm
ax.bar(df.year, deficit, color=["#c0561c" if y in DROUGHT else "#d9d6cf" for y in df.year], width=0.75)
z = np.polyfit(df.year, deficit, 1)
ax.plot(df.year, np.polyval(z, df.year), color=INK, lw=2)
ax.set_ylabel("дефіцит вологи IV–IX, мм\n(випаровуваність − опади)", color=INK, fontsize=10)
ax.set_ylim(0, 850); ax.tick_params(colors=MUTED)
ax.text(1981, 800, f"+{z[0]*10:.0f} мм кожні 10 років", fontsize=12, fontweight="bold", color=INK)
plt.tight_layout(); plt.savefig("out/fig5_deficit_trend.png", bbox_inches="tight"); plt.close()
print("ok", z[0] * 10, deficit[df.year <= 2002].mean(), deficit[df.year > 2002].mean())
