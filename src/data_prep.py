"""Loading, cleaning, splitting, and the price-tier label."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

SEED = 42
DATA_PATH = "data/bina_az_sale.csv"

# Target-derived columns. unit_price = price/area, total_price == price.
LEAKAGE_COLUMNS: list[str] = ["unit_price", "total_price"]

_NUM = re.compile(r"[\d]+(?:[.,][\d]+)?")


def _num(s) -> float:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return np.nan
    m = _NUM.search(str(s).replace(" ", ""))
    if not m:
        return np.nan
    try:
        return float(m.group(0).replace(",", "."))
    except ValueError:
        return np.nan


def load_raw(path: str = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8", engine="python")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # one listing = one row (scrape repeats)
    if "estate_rel_url" in df.columns:
        df = df.sort_values("datetime_scrape_x", na_position="first")
        df = df.drop_duplicates(subset="estate_rel_url", keep="last")
    price = pd.to_numeric(df["price"], errors="coerce")
    df["price"] = price
    df = df.dropna(subset=["price"])
    df = df[(df["price"] >= 10_000) & (df["price"] <= 15_000_000)]
    def find(*subs):
        for c in df.columns:
            if all(s in c for s in subs):
                return c
        return None

    c_area = find("Sah")
    c_rooms = find("Otaq")
    c_floor = find("rt", "b")  # Mərtəbə
    c_land = find("Torpaq")
    c_cat = find("Kateqoriya")
    c_rep = find("mir")  # Təmir
    c_deed = find("xar")  # Çıxarış
    c_mort = find("poteka")  # İpoteka

    area = df[c_area].map(_num) if c_area else np.nan
    df["area_m2"] = pd.to_numeric(area, errors="coerce")
    df = df.dropna(subset=["area_m2"])
    df = df[(df["area_m2"] >= 10) & (df["area_m2"] <= 3000)]

    df["rooms"] = pd.to_numeric(df[c_rooms], errors="coerce") if c_rooms else np.nan
    # "7 / 9" -> floor, total
    fl, tot = [], []
    if c_floor:
        for v in df[c_floor].fillna(""):
            p = str(v).split("/")
            try:
                fl.append(float(p[0].strip()))
                tot.append(float(p[1].strip()) if len(p) > 1 else np.nan)
            except ValueError:
                fl.append(np.nan)
                tot.append(np.nan)
    else:
        fl = [np.nan] * len(df)
        tot = [np.nan] * len(df)
    df["floor"], df["floors_total"] = fl, tot

    land = df[c_land].map(_num) if c_land else np.nan
    df["land_sot"] = pd.to_numeric(land, errors="coerce").fillna(0.0)
    df["has_land"] = (df["land_sot"] > 0).astype(int)

    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lng"] = pd.to_numeric(df["lng"], errors="coerce")
    df = df.dropna(subset=["lat", "lng"])
    df = df[(df["lat"].between(38.5, 42.0)) & (df["lng"].between(46.0, 51.0))]

    # short categoricals, missing kept as its own level
    def cat(col, name):
        s = df[col].fillna("unknown").astype(str).str.strip() if col else "unknown"
        s = s.replace({"": "unknown", "nan": "unknown"})
        df[name] = s

    cat("location", "district")
    cat(c_cat, "category")
    cat(c_rep, "repair_status")
    cat(c_deed, "deed")
    cat(c_mort, "mortgage_flag")
    df["is_repaired"] = (df["repair"] == "T\u0259mirli").astype(int) if "repair" in df.columns else 0
    df["has_deed"] = df["bill_of_sale"].notna().astype(int) if "bill_of_sale" in df.columns else 0
    df["mortgage_ok"] = df["mortgage"].notna().astype(int) if "mortgage" in df.columns else 0
    df["is_vip"] = (df["vip"] == "vipped").astype(int) if "vip" in df.columns else 0
    df["is_featured"] = (df["featured"] == "featured").astype(int) if "featured" in df.columns else 0

    drop = [c for c in LEAKAGE_COLUMNS if c in df.columns]
    df = df.drop(columns=drop, errors="ignore")
    return df.reset_index(drop=True)


def make_features(df: pd.DataFrame, top_districts: int = 15):
    num = ["area_m2", "rooms", "floor", "floors_total", "land_sot",
           "has_land", "lat", "lng", "is_repaired", "has_deed",
           "mortgage_ok", "is_vip", "is_featured"]
    for c in num:
        if c not in df.columns:
            df[c] = 0
    # leak-free fill: constants + flags
    X_num = pd.DataFrame({
        "area_m2": df["area_m2"].astype(float),
        "rooms": df["rooms"].fillna(0).astype(float),
        "floor": df["floor"].fillna(-1).astype(float),
        "floors_total": df["floors_total"].fillna(-1).astype(float),
        "land_sot": df["land_sot"].astype(float),
        "has_land": df["has_land"].astype(float),
        "lat": df["lat"].astype(float),
        "lng": df["lng"].astype(float),
        "is_repaired": df["is_repaired"].astype(float),
        "has_deed": df["has_deed"].astype(float),
        "mortgage_ok": df["mortgage_ok"].astype(float),
        "is_vip": df["is_vip"].astype(float),
        "is_featured": df["is_featured"].astype(float),
    })
    top = df["district"].value_counts().head(top_districts).index
    dist = df["district"].where(df["district"].isin(top), "other")
    dummies = pd.get_dummies(dist, prefix="dist")
    for c in ["category", "repair_status", "deed", "mortgage_flag"]:
        dummies = pd.concat([dummies, pd.get_dummies(df[c].fillna("unknown"), prefix=c[:4])], axis=1)
    X = pd.concat([X_num, dummies.astype(float)], axis=1)
    y = df["price"].to_numpy(dtype=float)
    return X.to_numpy(dtype=float), y, list(X.columns)


def make_tier_label(y_price, threshold=None):
    y_price = np.asarray(y_price, dtype=float)
    if threshold is None:
        threshold = float(np.median(y_price))
    return (y_price > threshold).astype(int), float(threshold)


def train_val_test_split(X, y, val_size=0.15, test_size=0.15, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(X)
    idx = rng.permutation(n)
    n_te = int(n * test_size)
    n_va = int(n * val_size)
    # stratify by price quintile so splits match on value
    q = pd.qcut(y, 5, labels=False, duplicates="drop")
    order = sorted(idx, key=lambda i: (q[i], rng.random()))
    te, va, tr = [], [], []
    for k, i in enumerate(order):
        m = k % 20
        if m < int(20 * test_size):
            te.append(i)
        elif m < int(20 * (test_size + val_size)):
            va.append(i)
        else:
            tr.append(i)
    tr, va, te = map(np.array, (tr, va, te))
    return X[tr], y[tr], X[va], y[va], X[te], y[te]


def standardize(X_tr, *others):
    mu = X_tr.mean(axis=0)
    sd = X_tr.std(axis=0)
    sd[sd == 0] = 1.0
    out = [(X_tr - mu) / sd]
    out += [(X - mu) / sd for X in others]
    return out[0] if len(out) == 1 else tuple(out)
