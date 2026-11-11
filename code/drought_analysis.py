"""
Просторовий аналіз посух у Миколаївській області, 1981–2025.

Дані:  TerraClimate (Abatzoglou et al., 2018), ~4 км, щомісячні:
       ppt  – опади, мм; pet – еталонна евапотранспірація (Penman–Monteith), мм;
       PDSI – індекс Палмера (для перехресної перевірки); soil – вологість ґрунту, мм.
Межі:  geoBoundaries gbOpen UKR ADM1 (OSM, ODbL).

Індикатор: SPEI-6 за вересень (кліматичний водний баланс P–PET, сума квітень–вересень),
           розподіл log-logistic, параметри методом зважених ймовірнісних моментів
           (Vicente-Serrano et al., 2010), калібрування 1981–2025 для кожної клітинки.
"""
import json
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from shapely.geometry import Point
from shapely.prepared import prep
from scipy.special import gamma as G
from scipy.stats import norm

RAW = "raw/mykolaiv_terraclimate_{}_1981_2025.nc"
Y0, Y1 = 1981, 2025


def load(v):
    da = xr.open_dataset(RAW.format(v))[v]
    return da.sortby("lat")


ppt, pet, pdsi, soil = (load(v) for v in ["ppt", "pet", "PDSI", "soil"])

# ---------- маска області ----------
adm = gpd.read_file("ukr_adm1.geojson")
obl = adm[adm.shapeName.str.contains("Mykolaiv")].geometry.union_all()
P = prep(obl)
lon2d, lat2d = np.meshgrid(ppt.lon.values, ppt.lat.values)
mask = np.array([P.contains(Point(x, y)) for x, y in zip(lon2d.ravel(), lat2d.ravel())]).reshape(lon2d.shape)
# лише суходіл з даними (лимани та море в TerraClimate — пропуски)
mask &= np.isfinite(ppt.isel(time=0).values) & np.isfinite(pet.isel(time=0).values)
# вага клітинки ~ cos(lat)
w = np.cos(np.deg2rad(lat2d)) * mask
print("клітинок в області:", mask.sum())


# ---------- SPEI ----------
def loglogistic_spei(x):
    """x: (n_years, ...) — накопичений P-PET. Повертає SPEI тієї ж форми.

    Як у R-пакеті SPEI (Beguería & Vicente-Serrano): ймовірнісно-зважені моменти
    з plotting position (i-0.35)/n -> L-моменти -> параметри узагальненого
    логістичного розподілу (log-logistic у параметризації Hosking, GLO)."""
    n = x.shape[0]
    xs = np.sort(x, axis=0)
    i = np.arange(1, n + 1).reshape((-1,) + (1,) * (x.ndim - 1))
    F = (i - 0.35) / n
    b0 = xs.mean(0)
    b1 = (F * xs).mean(0)
    b2 = (F ** 2 * xs).mean(0)
    l1, l2, l3 = b0, 2 * b1 - b0, 6 * b2 - 6 * b1 + b0
    t3 = l3 / l2
    k = -t3
    with np.errstate(invalid="ignore", divide="ignore"):
        kp = k * np.pi
        alpha = l2 * np.sin(kp) / kp
        xi = l1 - alpha * (1 / k - np.pi / np.sin(kp))
        arg = 1 - k * (x - xi) / alpha
        y = -np.log(arg) / k
        cdf = 1 / (1 + np.exp(-y))
        # поза межами носія розподілу
        cdf = np.where(arg <= 0, np.where(k > 0, 1.0, 0.0), cdf)
    cdf = np.where(np.isnan(x), np.nan, cdf)
    cdf = np.clip(cdf, 1e-6, 1 - 1e-6)
    return norm.ppf(cdf), (xi, alpha, k)


D = (ppt - pet)
gs = D.where(D["time.month"].isin(range(4, 10))).groupby("time.year").sum("time", min_count=6)  # квіт–вер
yr = D.groupby("time.year").sum("time", min_count=12)  # календарний рік (SPEI-12 грудня)

spei6, par6 = loglogistic_spei(gs.values)
spr = D.where(D["time.month"].isin([3, 4, 5])).groupby("time.year").sum("time", min_count=3)   # весна
smr = D.where(D["time.month"].isin([6, 7, 8])).groupby("time.year").sum("time", min_count=3)   # літо
spei3_spr, _ = loglogistic_spei(spr.values)
spei3_smr, _ = loglogistic_spei(smr.values)
spei3_spr = np.where(mask[None], spei3_spr, np.nan)
spei3_smr = np.where(mask[None], spei3_smr, np.nan)
spei12, _ = loglogistic_spei(yr.values)
years = gs.year.values

spei6 = np.where(mask[None], spei6, np.nan)
spei12 = np.where(mask[None], spei12, np.nan)


def oblast_mean(a):
    return np.nansum(a * w[None], axis=(1, 2)) / w.sum()


def area_share(a, thr):
    return np.nansum((a <= thr) * w[None], axis=(1, 2)) / w.sum() * 100


# Перехресна перевірка: PDSI серпня та аномалія вологості ґрунту в липні
pdsi_aug = pdsi.where(pdsi["time.month"] == 8, drop=True).values
pdsi_aug = np.where(mask[None], pdsi_aug, np.nan)
soil_jul = soil.where(soil["time.month"] == 7, drop=True).values
soil_jul = np.where(mask[None], soil_jul, np.nan)
soil_anom = (soil_jul - np.nanmean(soil_jul, 0)) / np.nanstd(soil_jul, 0)
gs_p = ppt.where(ppt["time.month"].isin(range(4, 10))).groupby("time.year").sum("time").values
gs_pet = pet.where(pet["time.month"].isin(range(4, 10))).groupby("time.year").sum("time").values

df = pd.DataFrame({
    "year": years,
    "spei6_mean": oblast_mean(spei6),
    "area_moderate_pct": area_share(spei6, -1.0),
    "area_severe_pct": area_share(spei6, -1.5),
    "area_extreme_pct": area_share(spei6, -2.0),
    "spei12_mean": oblast_mean(spei12),
    "spei3_spring_mean": oblast_mean(spei3_spr),
    "spei3_summer_mean": oblast_mean(spei3_smr),
    "pdsi_aug_mean": oblast_mean(pdsi_aug),
    "soil_jul_anom": oblast_mean(np.nan_to_num(soil_anom) * mask[None]) ,
    "p_aprsep_mm": oblast_mean(np.where(mask[None], gs_p, np.nan)),
    "pet_aprsep_mm": oblast_mean(np.where(mask[None], gs_pet, np.nan)),
})


def cls(s):
    if s <= -2: return "екстремальна"
    if s <= -1.5: return "сильна"
    if s <= -1.0: return "помірна"
    return ""


df["class"] = df.spei6_mean.apply(cls)
# Рік вважаємо посушливим, якщо середній SPEI-6 ≤ -1 АБО ≥ 50 % площі мали SPEI-6 ≤ -1
df["drought_year"] = (df.spei6_mean <= -1) | (df.area_moderate_pct >= 50)
df["local_drought"] = (~df.drought_year) & (df.area_moderate_pct >= 20)
df.round(2).to_csv("out/drought_years_table.csv", index=False)

# частота посух у кожній клітинці
freq_all = np.where(mask, (spei6 <= -1).sum(0), np.nan)
early = years <= 2002
freq_early = np.where(mask, (spei6[early] <= -1).sum(0), np.nan)
freq_late = np.where(mask, (spei6[~early] <= -1).sum(0), np.nan)

np.savez_compressed("out/spei_grids.npz", spei6=spei6, spei12=spei12, years=years,
                    lat=ppt.lat.values, lon=ppt.lon.values, mask=mask,
                    freq_all=freq_all, freq_early=freq_early, freq_late=freq_late,
                    pdsi_aug=pdsi_aug, spei3_spr=spei3_spr, spei3_smr=spei3_smr)

# кореляції для перевірки узгодженості
print(df.round(2).to_string())
print("\nr(SPEI6, PDSI серп.) =", round(df.spei6_mean.corr(df.pdsi_aug_mean), 2))
print("r(SPEI6, ґрунт. волога лип.) =", round(df.spei6_mean.corr(df.soil_jul_anom), 2))
print("Посушливі роки:", df[df.drought_year].year.tolist())
print("Весняні посухи (SPEI-3 трав. ≤ -1):", df[df.spei3_spring_mean<=-1].year.tolist())
print("Літні посухи (SPEI-3 серп. ≤ -1):", df[df.spei3_summer_mean<=-1].year.tolist())
print("Локальні посухи (20–50 % площі):", df[df.local_drought].year.tolist())
print("Частота (клітинка) 1981–2002 vs 2003–2025: середнє",
      np.nanmean(freq_early), np.nanmean(freq_late), "max", np.nanmax(freq_all), "min", np.nanmin(freq_all))
# тренд P і PET
for c in ["p_aprsep_mm", "pet_aprsep_mm", "spei6_mean"]:
    s = np.polyfit(df.year, df[c], 1)[0] * 10
    print(f"тренд {c}: {s:.2f} за 10 років")
json.dump({"drought_years": df[df.drought_year].year.tolist()}, open("out/summary.json", "w"))
