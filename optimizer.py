# ============================================================================
# SINTER BURDEN OPTIMIZER v37.0 — STREAMLIT BACKEND MODULE
# ============================================================================
# This is the v37.0 Colab optimizer with all ipywidgets/Colab-only UI code
# removed (file upload widgets, HTML/print dashboards, on-click handlers).
# The LP model, constraints, chemistry, thermal and costing logic are
# unchanged. Three thin helpers have been added at the bottom so the
# Streamlit dashboard (app.py) has everything it imports:
#   - TARGETS               (module-level dict; app.py does `opt.TARGETS`)
#   - quality_table(...)    (small achieved-vs-target DataFrame helper)
#   - solve_manual_scenario (Manual Burden Control "practical scenario" solve)
#   - what_if_analysis      (Scenario Analysis "material shortage" sweep)
# ============================================================================

import pandas as pd
import pulp
import numpy as np
import re
import copy
from io import BytesIO

# ============================================================================
# FUEL SOURCE SELECTION — KSL COKE vs LOCAL COKE
# ============================================================================
FUEL_SOURCE_SELECTION_MODE = "OPTIMIZER"
MANUAL_KSL_FUEL_PCT = 50.0
MANUAL_LOCAL_FUEL_PCT = 50.0
FUEL_DIVERSITY_WEIGHT = 0.30
OPTIMIZER_MIN_COKE_SOURCE_PCT = 40.0
OPTIMIZER_MAX_COKE_SOURCE_PCT = 60.0


def _find_coke_sources(fuels):
    normalized = {
        re.sub(r"[^A-Z0-9]+", "_", str(m).strip().upper()).strip("_"): m
        for m in fuels
    }
    ksl = normalized.get("KSL_COKE")
    local = normalized.get("LOCAL_COKE")
    return ksl, local


# ============================================================================
# DYNAMIC METALLURGICAL ORE / FUEL SUITABILITY WEIGHTS
# ============================================================================
DYNAMIC_ORE_W_FE = 1.00
DYNAMIC_ORE_W_SIO2 = 1.00
DYNAMIC_ORE_W_AL2O3 = 0.90
DYNAMIC_ORE_W_CAO = 0.35
DYNAMIC_ORE_W_MGO = 0.30
DYNAMIC_ORE_W_LOI = 0.35
DYNAMIC_ORE_W_MOISTURE = 0.05
DYNAMIC_ORE_W_COST = 0.10
DYNAMIC_ORE_SUITABILITY_WEIGHT = 1.0
DYNAMIC_ORE_DIVERSITY_WEIGHT = 0.30


def _fuel_source_suitability_table(df):
    fuels = _fuel_materials(df)
    if not fuels:
        return pd.DataFrame()
    prices = pd.to_numeric(df.loc[fuels, "Price_Rs_t"], errors="coerce").fillna(0.0)
    pmin, pmax = float(prices.min()), float(prices.max())
    rows = []
    for m in fuels:
        r = df.loc[m]
        fc = float(r["FC_Pct"]); cv = float(r["CV_kcal_kg"]); moisture = float(r["Moisture_Pct"])
        ash_ar = float(r.get("Ash_AR_Pct", 0.0)); ash_si = float(r.get("Ash_SiO2_Pct", 0.0))
        ash_al = float(r.get("Ash_Al2O3_Pct", 0.0)); ash_ca = float(r.get("Ash_CaO_Pct", 0.0))
        price = float(r["Price_Rs_t"])
        fc_score = np.clip((fc - 68.0) / 15.0, 0.0, 1.0)
        cv_score = np.clip((cv - 6000.0) / 1200.0, 0.0, 1.0)
        ash_score = np.clip((16.0 - ash_ar) / 16.0, 0.0, 1.0)
        ash_si_score = np.clip((60.0 - ash_si) / 60.0, 0.0, 1.0)
        ash_al_score = np.clip((30.0 - ash_al) / 30.0, 0.0, 1.0)
        ash_ca_score = np.clip(ash_ca / 8.0, 0.0, 1.0)
        moisture_score = np.clip((12.0 - moisture) / 12.0, 0.0, 1.0)
        cost_score = 1.0 if pmax <= pmin else 1.0 - (price - pmin) / (pmax - pmin)
        score = (0.40*fc_score + 0.25*cv_score + 0.12*ash_score + 0.08*ash_si_score
                 + 0.06*ash_al_score + 0.02*ash_ca_score + 0.02*moisture_score + 0.05*cost_score)
        rows.append({"Material": m, "Suitability": 100.0*score, "FC %": fc, "CV kcal/kg": cv,
                      "Ash AR %": ash_ar, "Ash SiO2 %": ash_si, "Ash Al2O3 %": ash_al,
                      "Ash CaO %": ash_ca, "Moisture %": moisture, "Price Rs/t": price,
                      "Available t": float(r["Available_Tonnes"])})
    return pd.DataFrame(rows).sort_values(["Suitability", "FC %"], ascending=[False, False]).reset_index(drop=True)


def _range_score(value, lo, hi):
    v = float(value)
    if lo <= v <= hi:
        center = (lo + hi) / 2.0
        half = max((hi - lo) / 2.0, 1e-9)
        return max(0.0, 1.0 - abs(v - center) / half)
    if v < lo:
        return max(0.0, 1.0 - (lo - v) / max(hi - lo, 1e-9))
    return max(0.0, 1.0 - (v - hi) / max(hi - lo, 1e-9))


def _ore_suitability_table(df):
    ores = _eligible_iron_ores(df)
    if not ores:
        return pd.DataFrame()
    prices = pd.to_numeric(df.loc[ores, "Price_Rs_t"], errors="coerce").fillna(0.0)
    pmin, pmax = float(prices.min()), float(prices.max())
    rows = []
    for m in ores:
        r = df.loc[m]
        fe = float(r["Fe"]); sio2 = float(r["SiO2"]); al = float(r["Al2O3"])
        cao = float(r["CaO"]); mgo = float(r["MgO"]); loi = float(r["LOI"])
        moisture = float(r["Moisture_Pct"]); price = float(r["Price_Rs_t"])
        fe_score = np.clip((fe - 50.0) / 15.0, 0.0, 1.0)
        sio2_score = np.clip((6.5 - sio2) / 6.5, 0.0, 1.0)
        al_score = np.clip((4.5 - al) / 4.5, 0.0, 1.0)
        loi_score = np.clip((8.0 - loi) / 8.0, 0.0, 1.0)
        moisture_score = np.clip((10.0 - moisture) / 10.0, 0.0, 1.0)
        al_sio2_ratio = al / max(sio2, 1e-9)
        al_sio2_score = np.clip((0.60 - al_sio2_ratio) / 0.60, 0.0, 1.0)
        cao_score = _range_score(cao, 0.0, 3.0)
        mgo_score = _range_score(mgo, 0.0, 2.5)
        cost_score = 1.0 - (price - pmin) / (pmax - pmin) if pmax > pmin else 1.0
        weighted = (DYNAMIC_ORE_W_FE*fe_score + DYNAMIC_ORE_W_SIO2*sio2_score
                    + DYNAMIC_ORE_W_AL2O3*al_score + DYNAMIC_ORE_W_AL2O3*al_sio2_score
                    + DYNAMIC_ORE_W_CAO*cao_score + DYNAMIC_ORE_W_MGO*mgo_score
                    + DYNAMIC_ORE_W_LOI*loi_score + DYNAMIC_ORE_W_MOISTURE*moisture_score
                    + DYNAMIC_ORE_W_COST*cost_score)
        denom = (DYNAMIC_ORE_W_FE + DYNAMIC_ORE_W_SIO2 + 2*DYNAMIC_ORE_W_AL2O3 + DYNAMIC_ORE_W_CAO
                 + DYNAMIC_ORE_W_MGO + DYNAMIC_ORE_W_LOI + DYNAMIC_ORE_W_MOISTURE + DYNAMIC_ORE_W_COST)
        rows.append({"Material": m, "Suitability": 100.0*weighted/max(denom, 1e-9), "Fe %": fe,
                      "SiO2 %": sio2, "Al2O3 %": al, "Al2O3/SiO2": al_sio2_ratio, "CaO %": cao,
                      "MgO %": mgo, "LOI %": loi, "Moisture %": moisture, "Price Rs/t": price,
                      "Available t": float(r["Available_Tonnes"])})
    result = pd.DataFrame(rows)
    return result.sort_values(["Suitability", "Fe %"], ascending=[False, False]).reset_index(drop=True)


def get_iron_ore_ranking(df):
    table = _ore_suitability_table(df)
    return table["Material"].tolist() if not table.empty else []


FLUX_RANK = {"LIMESTONE": 1, "DOLOMITE": 2, "QUICKLIME": 3, "LIME_POWDER": 4}
MILL_SCALE_MIN_BURDEN_PCT = 0.05
MILL_SCALE_MAX_BURDEN_PCT = 0.15
PRODUCTION_CONTINUITY_MODE = True

IRON_ORE_GENERIC_MAX_PCT = 0.80
IRON_ORE_GENERIC_MAX_PCT_RELAXED = 0.95
IRON_ORE_GENERIC_MAX_PCT_CRISIS = 0.95
IRON_ORE_MIN_PCT = {}
MAX_IRON_ORE_PORTION = 0.80
MAX_IRON_ORE_PORTION_CRISIS = 0.95

FLUX_MAX_PCT_BASE = {"LIMESTONE": 0.60, "DOLOMITE": 0.45, "QUICKLIME": 0.60, "LIME_POWDER": 0.50}
FLUX_MAX_PCT_RELAXED = {"LIMESTONE": 0.75, "DOLOMITE": 0.55, "QUICKLIME": 0.80, "LIME_POWDER": 0.65}
FLUX_MAX_PCT_CRISIS = {"LIMESTONE": 0.95, "DOLOMITE": 0.95, "QUICKLIME": 0.95, "LIME_POWDER": 0.95}
FLUX_MIN_PCT = {"LIMESTONE": 0.05, "DOLOMITE": 0.05, "QUICKLIME": 0.02, "LIME_POWDER": 0.0}
FLUX_MIN_PCT_QUALITY_RELAXED = {"LIMESTONE": 0.0, "DOLOMITE": 0.0, "QUICKLIME": 0.0, "LIME_POWDER": 0.0}
MAX_FLUX_PORTION = 0.25
MAX_FLUX_PORTION_CRISIS = 0.40

SIO2_MAX_SHORTAGE = 6.2
FE_TARGET = 54.0
FE_LOWER = 52.5
FE_UPPER = 54.5
FE_TOLERANCE = 1.5
FE_CENTER_WEIGHT = 2.0

DEVIATION_WEIGHTS = {"Fe": 5.0, "Basicity": 6.0, "CaO": 5.0, "MgO": 4.0,
                      "Al2O3": 3.0, "SiO2": 2.0, "Al2O3_SiO2_ratio": 2.0}

PIN_TOLERANCE = 1e-3
FLUX_BASELINE_INCREASE_CAP = 0.05

ADJUSTMENT_RANGES = {"Iron_ore": 0.15, "Flux": 0.10, "Recycle": 0.00, "Fuel": 0.10,
                      "IOL_Fines_Mandate": 0.00, "BF_Returns_Mandate": 0.00}

NUM_ALT_ORE_SLOTS = 0

IOL_FINES_NOMINAL_PCT = 0.08
BF_RETURNS_NOMINAL_PCT = 0.17
TOTAL_RS_NOMINAL_PCT = IOL_FINES_NOMINAL_PCT + BF_RETURNS_NOMINAL_PCT

IOL_FINES_FALLBACK_MIN = IOL_FINES_NOMINAL_PCT
IOL_FINES_FALLBACK_MAX = IOL_FINES_NOMINAL_PCT
BF_RETURNS_FALLBACK_MIN = BF_RETURNS_NOMINAL_PCT
BF_RETURNS_FALLBACK_MAX = BF_RETURNS_NOMINAL_PCT
PIN_BAND = 0.0

DEFAULT_OM_COST_RS_T = 750.0
DEFAULT_COKE_CV_KCAL_KG = 6800.0
DEFAULT_COKE_FC_PCT = 71.35
DEFAULT_HEAT_LATENT_MOISTURE = 540.0
DEFAULT_HEAT_CALCINATION_PER_LOI_KG = 420.0
DEFAULT_HEAT_MELTING_PER_KG_SINTER = 60.0
DEFAULT_HEAT_LOSS_FRACTION = 0.12
DEFAULT_FIRING_RATIO_MAX = 1.10
ENFORCE_FIRING_RATIO_MAX = False

DEFAULT_COKE_MIN_KG_T = 55.0
DEFAULT_COKE_MAX_KG_T = 85.0
DEFAULT_FEO_MIN_PCT = 8.5
DEFAULT_FEO_TARGET_PCT = 9.2
DEFAULT_FEO_MAX_PCT = 10.0
DEFAULT_FEO_REFERENCE_THERMAL_SURPLUS_KCAL = 189180.0
DEFAULT_FEO_REFERENCE_PCT = 8.6
DEFAULT_FEO_THERMAL_SLOPE_PCT_PER_10K_KCAL = 0.35
DEFAULT_REFERENCE_COKE_CV_KCAL_KG = 6800.0
DEFAULT_REFERENCE_COKE_FC_PCT = 71.35

PRODUCTIVITY_ADJUSTMENT_PCT = 0.01
DEFAULT_KSL_SPLIT_PCT = 50.0

DEFAULT_FUEL_ASH_SETTINGS = {
    "KSL_COKE": {"Ash_AR_Pct": 14.0, "Ash_SiO2_Pct": 52.5, "Ash_Al2O3_Pct": 25.0,
                 "Ash_CaO_Pct": 5.5, "Ash_MgO_Pct": 3.0, "Ash_Fe2O3_Pct": 10.0},
    "LOCAL_COKE": {"Ash_AR_Pct": 14.0, "Ash_SiO2_Pct": 52.5, "Ash_Al2O3_Pct": 25.0,
                   "Ash_CaO_Pct": 5.5, "Ash_MgO_Pct": 3.0, "Ash_Fe2O3_Pct": 10.0},
}
ACTIVE_FUEL_ASH_SETTINGS = copy.deepcopy(DEFAULT_FUEL_ASH_SETTINGS)


def _normalise_fuel_ash_settings(settings=None, df=None):
    base = copy.deepcopy(DEFAULT_FUEL_ASH_SETTINGS)
    if settings:
        for mat, vals in settings.items():
            base.setdefault(mat, {}).update({k: float(v) for k, v in vals.items()})
    if df is not None:
        fuel_mats = _fuel_materials(df)
        for m in fuel_mats:
            if m not in base:
                base[m] = copy.deepcopy(base.get("KSL_COKE", next(iter(base.values()))))
    return base


def set_active_fuel_ash_settings(settings):
    global ACTIVE_FUEL_ASH_SETTINGS
    ACTIVE_FUEL_ASH_SETTINGS = _normalise_fuel_ash_settings(settings)
    return copy.deepcopy(ACTIVE_FUEL_ASH_SETTINGS)


def get_active_fuel_ash_settings(df=None):
    return _normalise_fuel_ash_settings(ACTIVE_FUEL_ASH_SETTINGS, df=df)


def sanitize_material_name(raw_name):
    name = raw_name.strip()
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^A-Za-z0-9_]", "", name)
    return name


# ============================================================================
# DEFAULT CHEMISTRY
# ============================================================================
def get_default_chemistry():
    data = {
        "Material": ["MILL_SCALE", "LLOYD", "DIOM", "SIOM", "KIOM",
                     "INTERNAL_FINES", "IOL_Fines", "FLUE_DUST", "IRON_POWDER", "BF_Returns",
                     "DOLOMITE", "LIMESTONE", "QUICKLIME", "LIME_POWDER", "KSL_COKE", "LOCAL_COKE"],
        "Group": ["Iron_ore", "Iron_ore", "Iron_ore", "Iron_ore", "Iron_ore",
                  "Recycle", "IOL_Fines_Mandate", "Recycle", "Recycle", "BF_Returns_Mandate",
                  "Flux", "Flux", "Flux", "Flux", "Fuel", "Fuel"],
        "Fe":    [68.34, 63.52, 57.17, 59.34, 58.41, 50.0, 60.00, 47.02, 47.02, 52.5, 0.54, 0.88, 0.01, 0.50, 0, 0],
        "SiO2":  [2.00, 3.86, 12.39, 6.92, 5.75, 6.00, 5.00, 7.07, 7.07, 5.62, 4.72, 4.48, 2.50, 3.00, 0.0, 0.0],
        "Al2O3": [2.72, 2.27, 2.93, 3.72, 5.48, 4.50, 3.00, 4.50, 4.50, 3.20, 0.95, 1.19, 0.61, 0.80, 0, 0],
        "CaO":   [0, 0.022, 0.058, 0.256, 0.157, 1.122, 8.79, 1.10, 1.10, 10.74, 30.02, 48.71, 89.00, 52.00, 0, 0],
        "MgO":   [0, 0.034, 0.114, 0.331, 0.018, 0.06, 1.52, 0.29, 0.29, 2.30, 18.75, 2.59, 1.57, 1.00, 0, 0],
        "LOI":   [2.50, 2.29, 4.00, 3.45, 4.62, 3.00, 3.00, 15.00, 15.00, 3.00, 42.00, 40.00, 5.00, 38.00, 0.00, 0.00],
        "Moisture_Pct": [6.0, 5.0, 6.0, 6.0, 6.0, 1.1, 4.13, 9.4, 9.4, 0.0, 2.0, 2.0, 0.0, 2.0, 11.27, 11.27],
        "Tech_Min": [0, 0, 0, 0, 0, 30, 0, 25, 25, 0, 30, 0, 40, 0, 0, 0],
        "Tech_Max": [220, 200, 200, 200, 300, 30, 999, 25, 25, 999, 200, 250, 65, 200, 999, 999],
        "Available_Tonnes": [2000, 10000, 6000, 8000, 5000, 5000, 5000, 3000, 3000, 5000, 10000, 15000, 5000, 5000, 6000, 6000],
        "Price_Rs_t": [7800, 7820, 4600, 4600, 4900, 1000, 5577, 500, 500, 0, 1340, 1355, 9200, 1200, 15022, 12500],
        "CV_kcal_kg": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, DEFAULT_COKE_CV_KCAL_KG, DEFAULT_COKE_CV_KCAL_KG],
        "FC_Pct": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, DEFAULT_COKE_FC_PCT, DEFAULT_COKE_FC_PCT],
    }
    df = pd.DataFrame(data).set_index("Material")
    df["Fines_Pct"] = 0.0
    for mat in df[df["Group"] == "Recycle"].index:
        fixed_rate = df.loc[mat, "Tech_Min"]
        df.loc[mat, "Tech_Min"] = fixed_rate
        df.loc[mat, "Tech_Max"] = fixed_rate
    return df


# ============================================================================
# HELPERS
# ============================================================================
# ============================================================================
# R/S SHORTFALL COMPENSATION
# ============================================================================
# The normal user-selected R/S split remains strict when the required IOL
# Fines and BF Returns quantities are physically available.
#
# If either source is unavailable or its inventory/Tech Max cannot support its
# nominal share for the requested production, the model enters R/S shortfall
# compensation mode. The physically usable R/S is consumed first and the
# missing R/S is compensated by the iron-bearing burden through the LP's
# existing chemistry, inventory, Tech Min/Max, flux and thermal constraints.
RS_SHORTFALL_COMPENSATION_ENABLED = True
RS_SHORTFALL_TOLERANCE_KG_T = 0.01
RS_SHORTFALL_MAX_IRON_ORE_PORTION = 0.95


def _rs_source_capacity_kg_t(df, mat, nominal_pct, production_tonnes):
    """Return the maximum usable nominal-share quantity of an R/S source, kg/t."""
    if mat not in df.index:
        return 0.0
    available_t = max(float(df.loc[mat, "Available_Tonnes"]), 0.0)
    tech_max = max(float(df.loc[mat, "Tech_Max"]), 0.0)
    if available_t <= 0.0 or tech_max <= 0.0:
        return 0.0
    nominal_kg_t = float(nominal_pct) * 1000.0
    inventory_cap = (available_t / max(float(production_tonnes), 1e-12)) * 1000.0
    return max(0.0, min(nominal_kg_t, tech_max, inventory_cap))


def get_rs_shortfall_info(df, production_tonnes,
                          iol_nominal=IOL_FINES_NOMINAL_PCT,
                          bf_nominal=BF_RETURNS_NOMINAL_PCT):
    """Calculate normal R/S target, usable source capacities and shortfall."""
    iol_target = float(iol_nominal) * 1000.0
    bfr_target = float(bf_nominal) * 1000.0
    iol_cap = _rs_source_capacity_kg_t(df, "IOL_Fines", iol_nominal, production_tonnes)
    bfr_cap = _rs_source_capacity_kg_t(df, "BF_Returns", bf_nominal, production_tonnes)
    target = iol_target + bfr_target
    available = iol_cap + bfr_cap
    shortfall = max(0.0, target - available)
    return {
        "iol_target_kg_t": iol_target,
        "bfr_target_kg_t": bfr_target,
        "rs_target_kg_t": target,
        "iol_capacity_kg_t": iol_cap,
        "bfr_capacity_kg_t": bfr_cap,
        "rs_available_kg_t": available,
        "rs_shortfall_kg_t": shortfall,
        "iol_shortfall_kg_t": max(0.0, iol_target - iol_cap),
        "bfr_shortfall_kg_t": max(0.0, bfr_target - bfr_cap),
        "active": bool(shortfall > RS_SHORTFALL_TOLERANCE_KG_T),
    }


def _rs_source_is_available(df, mat):
    return (
        mat in df.index
        and float(df.loc[mat, "Available_Tonnes"]) > 0.0
        and float(df.loc[mat, "Tech_Max"]) > 0.0
    )


def get_mandate_shortfall_triggers(df):
    triggers = 0
    reasons = []
    for mat in ["IOL_Fines", "BF_Returns"]:
        if mat in df.index:
            if df.loc[mat, "Available_Tonnes"] <= 0 or df.loc[mat, "Tech_Max"] == 0:
                triggers += 1
                reasons.append(mat)
    return triggers, reasons


def get_iron_ore_tier(df, iron_ores, extra_missing=0, extra_reasons=None):
    extra_reasons = extra_reasons or []
    unavailable = [m for m in iron_ores
                   if float(df.loc[m, "Available_Tonnes"]) <= 0 or float(df.loc[m, "Tech_Max"]) <= 0]
    n = len(unavailable) + extra_missing
    display_list = unavailable + [f"{r}(mandate-shortfall)" for r in extra_reasons]
    if n == 0:
        caps = {m: IRON_ORE_GENERIC_MAX_PCT for m in iron_ores}
        return caps, unavailable, "✅ All iron ores available — common 80% concentration ceiling.", "base"
    elif n == 1:
        caps = {m: IRON_ORE_GENERIC_MAX_PCT_RELAXED for m in iron_ores}
        return caps, unavailable, f"⚠️ One iron-ore shortfall: {', '.join(display_list)} — ceiling relaxed.", "relaxed"
    else:
        caps = {m: IRON_ORE_GENERIC_MAX_PCT_CRISIS for m in iron_ores}
        return caps, unavailable, f"🔥 {n} iron-ore shortfalls: {', '.join(display_list)} — crisis ceiling relaxed.", "crisis"


def get_flux_tier(df, fluxes, extra_missing=0, extra_reasons=None):
    extra_reasons = extra_reasons or []
    unavailable = [m for m in fluxes if df.loc[m, "Available_Tonnes"] <= 0 or df.loc[m, "Tech_Max"] == 0]
    n = len(unavailable) + extra_missing
    display_list = unavailable + [f"{r}(mandate-shortfall)" for r in extra_reasons]
    if n == 0:
        return FLUX_MAX_PCT_BASE.copy(), unavailable, "✅ All fluxes available - using base flux caps", "base"
    elif n == 1:
        return FLUX_MAX_PCT_RELAXED.copy(), unavailable, f"⚠️ {n} flux shortfall: {', '.join(display_list)} — RELAXED", "relaxed"
    else:
        return FLUX_MAX_PCT_CRISIS.copy(), unavailable, f"🔥 {n} flux shortfalls: {', '.join(display_list)} — CRISIS", "crisis"


def check_fuel_gate(df):
    fuels = _fuel_materials(df)
    problems = []
    for mat in fuels:
        tech_min = df.loc[mat, "Tech_Min"]; available = df.loc[mat, "Available_Tonnes"]; tech_max = df.loc[mat, "Tech_Max"]
        if tech_min > 0 and (available <= 0 or tech_max <= 0):
            problems.append(f"{mat} (requires >= {tech_min} kg/t, but Available_Tonnes={available}, Tech_Max={tech_max})")
    usable_fuels = [m for m in fuels if df.loc[m, "Available_Tonnes"] > 0 and df.loc[m, "Tech_Max"] > 0]
    if fuels and not usable_fuels:
        problems.append("No Fuel-group material is available at all (all coke sources exhausted).")
    return (False, problems) if problems else (True, [])


def build_bounds(df, production_tonnes):
    bounds = {}
    for mat in df.index:
        tech_min = df.loc[mat, "Tech_Min"]; tech_max = df.loc[mat, "Tech_Max"]; available = df.loc[mat, "Available_Tonnes"]
        if available <= 0 or tech_max == 0:
            bounds[mat] = (0, 0)
        else:
            inv_cap = (available / production_tonnes) * 1000
            eff_max = min(tech_max, inv_cap)
            bounds[mat] = (tech_min, eff_max)
    return bounds


# ============================================================================
# MANDATE + STRUCTURAL CONSTRAINTS
# ============================================================================
def add_mandate_constraints(prob, x, df, OUT, mandate_mode="pinned",
                             iol_nominal=IOL_FINES_NOMINAL_PCT, bf_nominal=BF_RETURNS_NOMINAL_PCT,
                             iol_fb_min=IOL_FINES_FALLBACK_MIN, iol_fb_max=IOL_FINES_FALLBACK_MAX,
                             bf_fb_min=BF_RETURNS_FALLBACK_MIN, bf_fb_max=BF_RETURNS_FALLBACK_MAX,
                             rs_shortfall_active=False, rs_source_caps=None):
    """
    Apply the IOL/BF Returns burden rules.

    Normal operation:
        IOL Fines = iol_nominal of total burden
        BF Returns = bf_nominal of total burden
        Combined R/S = (iol_nominal + bf_nominal) of total burden

    R/S-shortfall operation:
        Each source is capped at its physically usable quantity. The missing
        R/S is deliberately NOT forced into the LP. This prevents the model
        from demanding material that is not in stock and allows the iron-
        bearing compensation logic to rebalance the burden.
    """
    total_burden = pulp.lpSum(x[m] for m in df.index)
    rs_source_caps = rs_source_caps or {}

    if rs_shortfall_active:
        # Use every physically available part of the nominal R/S allocation.
        # The variable upper bound already enforces inventory; the explicit
        # upper/lower bound here also prevents the cost objective from
        # voluntarily leaving available R/S unused during a shortage.
        if "IOL_Fines" in x:
            cap = float(rs_source_caps.get("IOL_Fines", 0.0))
            prob += x["IOL_Fines"] == min(cap, float(iol_nominal) * OUT), "IOL_Fines_RS_Shortfall_Use"
        if "BF_Returns" in x:
            cap = float(rs_source_caps.get("BF_Returns", 0.0))
            prob += x["BF_Returns"] == min(cap, float(bf_nominal) * OUT), "BF_Returns_RS_Shortfall_Use"
        # No combined 25% equality here: the difference is the explicit
        # R/S shortfall and is compensated through iron-bearing materials.
        return

    # Normal case: preserve the existing strict split.
    for mat, nominal in [("IOL_Fines", iol_nominal), ("BF_Returns", bf_nominal)]:
        if mat not in x:
            continue
        prob += x[mat] == nominal * total_burden, f"{mat}_STRICT_BURDEN_PCT"

    if "IOL_Fines" in x and "BF_Returns" in x:
        combined_nominal = iol_nominal + bf_nominal
        prob += (
            x["IOL_Fines"] + x["BF_Returns"]
            == combined_nominal * total_burden
        ), "RETURN_SINTER_TOTAL_STRICT"


def add_structural_constraints(prob, x, df, bounds, iron_ores, fluxes, iron_ore_max_pct,
                                unavailable_iron, flux_max_pct, unavailable_flux, OUT,
                                baseline_flux_portion=None, iron_tier="base", flux_tier="base",
                                flux_min_pct_override=None, mandate_mode="pinned",
                                iol_nominal=IOL_FINES_NOMINAL_PCT, bf_nominal=BF_RETURNS_NOMINAL_PCT,
                                iol_fb_min=IOL_FINES_FALLBACK_MIN, iol_fb_max=IOL_FINES_FALLBACK_MAX,
                                bf_fb_min=BF_RETURNS_FALLBACK_MIN, bf_fb_max=BF_RETURNS_FALLBACK_MAX,
                                fuel_ash_settings=None, rs_shortfall_active=False, rs_source_caps=None):
    non_fuel = [m for m in x if df.loc[m, "Group"] != "Fuel"]
    mass = pulp.lpSum(x[m] * (1 - df.loc[m, "LOI"] / 100) for m in non_fuel)
    ash_coeff = _fuel_ash_coefficients(df, fuel_ash_settings)
    mass += pulp.lpSum(x[m] * ash_coeff[m]["Ash"] for m in x if m in ash_coeff)
    prob += mass >= OUT - 2, "Mass_Balance_Lower"
    prob += mass <= OUT + 2, "Mass_Balance_Upper"

    total_iron_ore = pulp.lpSum(x[m] for m in iron_ores)
    total_flux = pulp.lpSum(x[m] for m in fluxes)
    total_burden = pulp.lpSum(x[m] for m in df.index)

    for mat in iron_ores:
        if mat in unavailable_iron:
            prob += x[mat] == 0, f"{mat}_unavailable"
        else:
            max_pct = iron_ore_max_pct.get(mat, 0.29)
            prob += x[mat] <= max_pct * total_iron_ore + 0.001, f"{mat}_max_pct"
            prob += x[mat] >= IRON_ORE_MIN_PCT.get(mat, 0.03) * total_iron_ore - 0.001, f"{mat}_min_pct"

    if "MILL_SCALE" in x and "MILL_SCALE" not in unavailable_iron:
        # NOTE: the "unavailable" case is already covered by the generic
        # iron-ore loop above (which adds a "MILL_SCALE_unavailable"
        # constraint); adding it again here duplicated the constraint name
        # and crashed PuLP whenever Mill Scale was toggled unavailable.
        prob += x["MILL_SCALE"] >= MILL_SCALE_MIN_BURDEN_PCT * total_burden, "MILL_SCALE_Burden_Min"
        prob += x["MILL_SCALE"] <= MILL_SCALE_MAX_BURDEN_PCT * total_burden, "MILL_SCALE_Burden_Max"

    # R/S shortfall compensation may require additional iron-bearing burden.
    # Allow up to the approved 95% common iron-ore ceiling in this mode; this
    # is a maximum, not a target. The LP still decides the actual blend.
    if rs_shortfall_active:
        iron_ore_portion_cap = RS_SHORTFALL_MAX_IRON_ORE_PORTION
    else:
        iron_ore_portion_cap = MAX_IRON_ORE_PORTION_CRISIS if iron_tier == "crisis" else MAX_IRON_ORE_PORTION
    prob += total_iron_ore <= iron_ore_portion_cap * OUT, "Max_Iron_Ore_Portion"

    min_pct_source = flux_min_pct_override if flux_min_pct_override is not None else FLUX_MIN_PCT
    default_min_pct = 0.0 if flux_min_pct_override is not None else 0.02
    for mat in fluxes:
        if mat in unavailable_flux:
            prob += x[mat] == 0, f"{mat}_unavailable"
        else:
            max_pct = flux_max_pct.get(mat, 0.5)
            min_pct = min_pct_source.get(mat, default_min_pct)
            prob += x[mat] <= max_pct * total_flux + 0.001, f"{mat}_max_pct"
            prob += x[mat] >= min_pct * total_flux - 0.001, f"{mat}_min_pct"

    flux_portion_cap = MAX_FLUX_PORTION_CRISIS if flux_tier == "crisis" else MAX_FLUX_PORTION
    prob += total_flux <= flux_portion_cap * OUT, "Max_Flux_Portion"

    mandate_active = any(m in df.index and df.loc[m, "Available_Tonnes"] > 0 and df.loc[m, "Tech_Max"] > 0
                          for m in ["IOL_Fines", "BF_Returns"])

    if baseline_flux_portion is not None and flux_tier not in ("crisis", "quality_relaxed") and not mandate_active:
        prob += total_flux <= (baseline_flux_portion + FLUX_BASELINE_INCREASE_CAP) * OUT, "Flux_Baseline_Cap"

    add_mandate_constraints(prob, x, df, OUT, mandate_mode=mandate_mode, iol_nominal=iol_nominal,
                             bf_nominal=bf_nominal, iol_fb_min=iol_fb_min, iol_fb_max=iol_fb_max,
                             bf_fb_min=bf_fb_min, bf_fb_max=bf_fb_max,
                             rs_shortfall_active=rs_shortfall_active, rs_source_caps=rs_source_caps)
    return total_iron_ore, total_flux, total_burden


# ============================================================================
# CHEMISTRY / FUEL ASH
# ============================================================================
def _fuel_ash_coefficients(df, fuel_ash_settings=None):
    settings = _normalise_fuel_ash_settings(fuel_ash_settings, df=df)
    coeff = {}
    for m in _fuel_materials(df):
        vals = settings.get(m, settings.get("KSL_COKE", {}))
        moisture = float(df.loc[m, "Moisture_Pct"]) / 100.0
        ash_ar = float(vals.get("Ash_AR_Pct", 0.0)) / 100.0
        ash_dry_fraction = ash_ar / (1.0 - moisture) if moisture < 1.0 else 0.0
        coeff[m] = {
            "Ash": ash_dry_fraction,
            "SiO2": ash_dry_fraction * float(vals.get("Ash_SiO2_Pct", 0.0)) / 100.0,
            "Al2O3": ash_dry_fraction * float(vals.get("Ash_Al2O3_Pct", 0.0)) / 100.0,
            "CaO": ash_dry_fraction * float(vals.get("Ash_CaO_Pct", 0.0)) / 100.0,
            "MgO": ash_dry_fraction * float(vals.get("Ash_MgO_Pct", 0.0)) / 100.0,
            "Fe2O3": ash_dry_fraction * float(vals.get("Ash_Fe2O3_Pct", 0.0)) / 100.0,
        }
    return coeff


def _lp_chemistry_sums(x, df, fuel_ash_settings=None):
    nonfuel = [m for m in x if str(df.loc[m, "Group"]).strip().lower() != "fuel"]
    fuels = [m for m in x if str(df.loc[m, "Group"]).strip().lower() == "fuel"]
    coeff = _fuel_ash_coefficients(df, fuel_ash_settings)
    Fe_sum = pulp.lpSum(x[m] * float(df.loc[m, "Fe"]) / 100.0 for m in nonfuel)
    SiO2_sum = (pulp.lpSum(x[m] * float(df.loc[m, "SiO2"]) / 100.0 for m in nonfuel)
                + pulp.lpSum(x[m] * coeff[m]["SiO2"] for m in fuels if m in coeff))
    Al2O3_sum = (pulp.lpSum(x[m] * float(df.loc[m, "Al2O3"]) / 100.0 for m in nonfuel)
                 + pulp.lpSum(x[m] * coeff[m]["Al2O3"] for m in fuels if m in coeff))
    CaO_sum = (pulp.lpSum(x[m] * float(df.loc[m, "CaO"]) / 100.0 for m in nonfuel)
               + pulp.lpSum(x[m] * coeff[m]["CaO"] for m in fuels if m in coeff))
    MgO_sum = (pulp.lpSum(x[m] * float(df.loc[m, "MgO"]) / 100.0 for m in nonfuel)
               + pulp.lpSum(x[m] * coeff[m]["MgO"] for m in fuels if m in coeff))
    return Fe_sum, SiO2_sum, Al2O3_sum, CaO_sum, MgO_sum


def compute_fuel_ash_contribution(blend, df, fuel_ash_settings=None):
    settings = _normalise_fuel_ash_settings(fuel_ash_settings, df=df)
    totals = {"Ash_kg": 0.0, "SiO2_kg": 0.0, "Al2O3_kg": 0.0, "CaO_kg": 0.0, "MgO_kg": 0.0}
    per_fuel = {}
    for m in _fuel_materials(df):
        if m not in blend:
            continue
        dry_kg = float(blend[m])
        vals = settings.get(m, settings.get("KSL_COKE", {}))
        moisture = float(df.loc[m, "Moisture_Pct"]) / 100.0
        ash_ar = float(vals.get("Ash_AR_Pct", 0.0)) / 100.0
        ash_dry_pct = ash_ar / (1.0 - moisture) * 100.0 if moisture < 1.0 else 0.0
        ash_kg = dry_kg * ash_ar / (1.0 - moisture) if moisture < 1.0 else 0.0
        si = ash_kg * float(vals.get("Ash_SiO2_Pct", 0.0)) / 100.0
        al = ash_kg * float(vals.get("Ash_Al2O3_Pct", 0.0)) / 100.0
        ca = ash_kg * float(vals.get("Ash_CaO_Pct", 0.0)) / 100.0
        mg = ash_kg * float(vals.get("Ash_MgO_Pct", 0.0)) / 100.0
        fe2o3 = ash_kg * float(vals.get("Ash_Fe2O3_Pct", 0.0)) / 100.0
        totals["Ash_kg"] += ash_kg; totals["SiO2_kg"] += si; totals["Al2O3_kg"] += al
        totals["CaO_kg"] += ca; totals["MgO_kg"] += mg
        per_fuel[m] = {"Dry_Fuel_kg": dry_kg, "Ash_AR_Pct": ash_ar*100.0, "Ash_Dry_Pct": ash_dry_pct,
                        "Ash_kg": ash_kg, "SiO2_kg": si, "Al2O3_kg": al, "CaO_kg": ca, "MgO_kg": mg, "Fe2O3_kg": fe2o3}
    totals["Per_Fuel"] = per_fuel
    return totals


def compute_achieved(blend, df, OUT, fuel_ash_settings=None):
    nonfuel = [m for m in blend if str(df.loc[m, "Group"]).strip().lower() != "fuel"]
    Fe_mass = sum(blend[m] * float(df.loc[m, "Fe"]) / 100.0 for m in nonfuel)
    SiO2_mass = sum(blend[m] * float(df.loc[m, "SiO2"]) / 100.0 for m in nonfuel)
    Al2O3_mass = sum(blend[m] * float(df.loc[m, "Al2O3"]) / 100.0 for m in nonfuel)
    CaO_mass = sum(blend[m] * float(df.loc[m, "CaO"]) / 100.0 for m in nonfuel)
    MgO_mass = sum(blend[m] * float(df.loc[m, "MgO"]) / 100.0 for m in nonfuel)
    ash = compute_fuel_ash_contribution(blend, df, fuel_ash_settings)
    SiO2_mass += ash["SiO2_kg"]; Al2O3_mass += ash["Al2O3_kg"]; CaO_mass += ash["CaO_kg"]; MgO_mass += ash["MgO_kg"]
    Fe = Fe_mass / OUT * 100.0; SiO2 = SiO2_mass / OUT * 100.0; Al2O3 = Al2O3_mass / OUT * 100.0
    CaO = CaO_mass / OUT * 100.0; MgO = MgO_mass / OUT * 100.0
    achieved = {"Fe": Fe, "SiO2": SiO2, "Al2O3": Al2O3, "CaO": CaO, "MgO": MgO,
                "Fuel_Ash_kg": ash["Ash_kg"], "Fuel_Ash_SiO2_kg": ash["SiO2_kg"],
                "Fuel_Ash_Al2O3_kg": ash["Al2O3_kg"], "Fuel_Ash_CaO_kg": ash["CaO_kg"], "Fuel_Ash_MgO_kg": ash["MgO_kg"]}
    if SiO2 > 0:
        achieved["Basicity"] = CaO / SiO2
        achieved["Al2O3/SiO2"] = Al2O3 / SiO2
        achieved["B4"] = (CaO + MgO) / (SiO2 + Al2O3)
    else:
        achieved["Basicity"] = 0; achieved["Al2O3/SiO2"] = 0; achieved["B4"] = 0
    return achieved


def build_soft_vars_and_constraints(prob, xr, df, OUT, targets, fe_lo, fe_hi, suffix="", fuel_ash_settings=None):
    Fe_s, SiO2_s, Al2O3_s, CaO_s, MgO_s = _lp_chemistry_sums(xr, df, fuel_ash_settings)
    Fe_under = pulp.LpVariable(f"Fe_under{suffix}", lowBound=0)
    Fe_over = pulp.LpVariable(f"Fe_over{suffix}", lowBound=0)
    SiO2_over = pulp.LpVariable(f"SiO2_over{suffix}", lowBound=0)
    Al2O3_over = pulp.LpVariable(f"Al2O3_over{suffix}", lowBound=0)
    ratio_over = pulp.LpVariable(f"ratio_over{suffix}", lowBound=0)
    Bas_under = pulp.LpVariable(f"Bas_under{suffix}", lowBound=0)
    Bas_over = pulp.LpVariable(f"Bas_over{suffix}", lowBound=0)
    MgO_under = pulp.LpVariable(f"MgO_under{suffix}", lowBound=0)
    MgO_over = pulp.LpVariable(f"MgO_over{suffix}", lowBound=0)
    CaO_under = pulp.LpVariable(f"CaO_under{suffix}", lowBound=0)
    CaO_over = pulp.LpVariable(f"CaO_over{suffix}", lowBound=0)
    Fe_center_dev = pulp.LpVariable(f"Fe_center_dev{suffix}", lowBound=0)
    prob += Fe_s + Fe_under >= fe_lo, f"Fe_lo_soft{suffix}"
    prob += Fe_s - Fe_over <= fe_hi, f"Fe_hi_soft{suffix}"
    prob += SiO2_s - SiO2_over <= targets["SiO2_max"] * OUT / 100, f"SiO2_soft{suffix}"
    prob += Al2O3_s - Al2O3_over <= targets["Al2O3_max"] * OUT / 100, f"Al2O3_soft{suffix}"
    prob += (Al2O3_s - targets["Al2O3_SiO2_max"] * SiO2_s) - ratio_over <= 0, f"Ratio_soft{suffix}"
    prob += (CaO_s - targets["Basicity_min"] * SiO2_s) + Bas_under >= 0, f"Basicity_lo_soft{suffix}"
    prob += (CaO_s - targets["Basicity_max"] * SiO2_s) - Bas_over <= 0, f"Basicity_hi_soft{suffix}"
    prob += MgO_s + MgO_under >= targets["MgO_min"] * OUT / 100, f"MgO_lo_soft{suffix}"
    prob += MgO_s - MgO_over <= targets["MgO_max"] * OUT / 100, f"MgO_hi_soft{suffix}"
    prob += CaO_s + CaO_under >= targets["CaO_min"] * OUT / 100, f"CaO_lo_soft{suffix}"
    prob += CaO_s - CaO_over <= targets["CaO_max"] * OUT / 100, f"CaO_hi_soft{suffix}"
    prob += Fe_s - (FE_TARGET * OUT / 100) <= Fe_center_dev, f"Fe_center_pos{suffix}"
    prob += (FE_TARGET * OUT / 100) - Fe_s <= Fe_center_dev, f"Fe_center_neg{suffix}"
    slacks = {"Fe_under": Fe_under, "Fe_over": Fe_over, "SiO2_over": SiO2_over, "Al2O3_over": Al2O3_over,
              "ratio_over": ratio_over, "Bas_under": Bas_under, "Bas_over": Bas_over, "MgO_under": MgO_under,
              "MgO_over": MgO_over, "CaO_under": CaO_under, "CaO_over": CaO_over, "Fe_center_dev": Fe_center_dev}
    sums = {"Fe": Fe_s, "SiO2": SiO2_s, "Al2O3": Al2O3_s, "CaO": CaO_s, "MgO": MgO_s}
    return slacks, sums


def weighted_deviation_expr(slacks, targets, OUT):
    W = DEVIATION_WEIGHTS
    return (W["Fe"]*((slacks["Fe_under"]+slacks["Fe_over"])/(FE_TOLERANCE*OUT/100))
            + W["SiO2"]*(slacks["SiO2_over"]/(targets["SiO2_max"]*OUT/100))
            + W["Al2O3"]*(slacks["Al2O3_over"]/(targets["Al2O3_max"]*OUT/100))
            + W["Al2O3_SiO2_ratio"]*(slacks["ratio_over"]/max(targets["Al2O3_SiO2_max"]*OUT/100, 1e-6))
            + W["Basicity"]*((slacks["Bas_under"]+slacks["Bas_over"])/max((targets["Basicity_max"]-targets["Basicity_min"])*OUT/100, 1e-6))
            + W["MgO"]*((slacks["MgO_under"]+slacks["MgO_over"])/max((targets["MgO_max"]-targets["MgO_min"])*OUT/100, 1e-6))
            + W["CaO"]*((slacks["CaO_under"]+slacks["CaO_over"])/max((targets["CaO_max"]-targets["CaO_min"])*OUT/100, 1e-6))
            + FE_CENTER_WEIGHT*(slacks["Fe_center_dev"]/(FE_TOLERANCE*OUT/100)))


def _report_compensation(blend, df, iron_ores, fluxes, unavailable_iron, unavailable_flux,
                          iron_ore_max_pct, flux_max_pct, iron_tier, flux_tier, OUT, mandate_reasons=None):
    diagnostics = []
    if unavailable_iron or (mandate_reasons and iron_tier != "base"):
        iron_ore_total = sum(blend[m] for m in iron_ores)
        diag_msg = f"\n   ✅ Iron Ore Compensation Result (Tier: {iron_tier}):"
        diag_msg += f"\n   Iron Ore Portion: {iron_ore_total:.1f} kg ({iron_ore_total/OUT*100:.1f}% of burden)"
        for mat in iron_ores:
            if mat in unavailable_iron:
                diag_msg += f"\n      {mat}: UNAVAILABLE"
            else:
                pct = blend[mat] / iron_ore_total * 100 if iron_ore_total > 0 else 0
                max_pct = iron_ore_max_pct.get(mat, 0.29) * 100
                diag_msg += f"\n      {mat}: {blend[mat]:.1f} kg ({pct:.1f}%) [Max {max_pct:.0f}%]"
        diagnostics.append(diag_msg)
    if unavailable_flux or (mandate_reasons and flux_tier != "base"):
        flux_total = sum(blend[m] for m in fluxes)
        diag_msg = f"\n   ✅ Flux Compensation Result (Tier: {flux_tier}):"
        for mat in fluxes:
            if mat in unavailable_flux:
                diag_msg += f"\n      {mat}: UNAVAILABLE"
            else:
                pct = blend[mat] / flux_total * 100 if flux_total > 0 else 0
                max_pct = flux_max_pct.get(mat, 0.5) * 100
                diag_msg += f"\n      {mat}: {blend[mat]:.1f} kg ({pct:.1f}%) [Max {max_pct:.0f}%]"
        diagnostics.append(diag_msg)
    total_burden_actual = sum(blend[m] for m in blend if m in df.index)
    if total_burden_actual > 0:
        iol_actual = blend.get("IOL_Fines", 0.0); bf_actual = blend.get("BF_Returns", 0.0)
        rs_actual = iol_actual + bf_actual
        rs_target_pct = (IOL_FINES_NOMINAL_PCT + BF_RETURNS_NOMINAL_PCT) * 100.0
        rs_actual_pct = rs_actual / total_burden_actual * 100.0
        diagnostics.append(
            f"\n   🔄 R/S TARGET / ACTUAL: "
            f"IOL Fines = {iol_actual:.2f} kg ({iol_actual/total_burden_actual*100:.2f}%), "
            f"BF Returns = {bf_actual:.2f} kg ({bf_actual/total_burden_actual*100:.2f}%), "
            f"Combined R/S = {rs_actual_pct:.2f}% (normal target {rs_target_pct:.2f}%)."
        )
        if mandate_reasons:
            diagnostics.append(
                "   ⚠️ R/S SHORTFALL: unavailable/insufficient R/S was not forced. "
                "The missing quantity was made available for iron-bearing compensation."
            )
    if mandate_reasons:
        diagnostics.append(f"\n   🔗 NOTE: {', '.join(mandate_reasons)} shortfall detected.")
    if iron_tier == "crisis" or flux_tier == "crisis":
        diagnostics.append("\n   💰 NOTE: CRISIS MODE – usage caps loosened; cost may be higher.")
    return diagnostics


def _report_fines_loading(blend, df, OUT):
    fine_materials = [m for m in ["IOL_Fines", "BF_Returns", "MILL_SCALE"] if m in blend]
    total_fines = sum(blend.get(m, 0) for m in fine_materials)
    total_burden = sum(blend[m] for m in blend if m in df.index)
    pct = total_fines / total_burden * 100 if total_burden > 0 else 0.0
    msg = f"\n   🧱 COMBINED FINES LOADING (IOL_Fines + BF_Returns + Mill Scale): {total_fines:.1f} kg ({pct:.1f}% of burden)"
    if pct > 30:
        msg += "\n   ⚠️ HIGH FINES LOADING (>30%) – verify permeability."
    return msg


# ============================================================================
# COSTING TABLES
# ============================================================================
def compute_dry_cost_table(blend, df, om_cost):
    total_raw_input = sum(blend.values())
    rm_cost = sum(blend[m] * df.loc[m, "Price_Rs_t"] / 1000 for m in blend)
    table = pd.DataFrame({
        "Group": [df.loc[m, "Group"] for m in blend],
        "Dry kg / t sinter": [blend[m] for m in blend],
        "% of Burden": [(blend[m] / total_raw_input) * 100 if total_raw_input else 0 for m in blend],
        "Dry Cost (Rs/t)": [round(blend[m] * df.loc[m, "Price_Rs_t"] / 1000, 2) for m in blend],
    }, index=blend.keys())
    total_row = pd.DataFrame({"Group": ["TOTAL"], "Dry kg / t sinter": [total_raw_input],
                               "% of Burden": [100.0], "Dry Cost (Rs/t)": [round(rm_cost, 2)]}, index=["TOTAL"])
    table = pd.concat([table, total_row])
    total_sinter_cost = rm_cost + om_cost
    return table, rm_cost, total_sinter_cost


def compute_wet_cost_table(blend, df, om_cost):
    if "Moisture_Pct" not in df.columns:
        df = df.copy(); df["Moisture_Pct"] = 0.0
    rows = []; wet_rm_cost = 0; total_dry_kg = 0; total_wet_kg = 0
    for m in blend:
        moisture = df.loc[m, "Moisture_Pct"] / 100 if df.loc[m, "Moisture_Pct"] < 100 else 0.0
        dry_kg = blend[m]
        wet_kg = dry_kg / (1 - moisture) if moisture < 1 else dry_kg
        wet_cost = wet_kg * df.loc[m, "Price_Rs_t"] / 1000
        wet_rm_cost += wet_cost; total_dry_kg += dry_kg; total_wet_kg += wet_kg
        rows.append({"Material": m, "Group": df.loc[m, "Group"], "Dry kg / t sinter": dry_kg,
                      "Moisture %": round(moisture * 100, 2), "Wet (As-Received) kg": round(wet_kg, 2),
                      "Wet Cost (Rs/t)": round(wet_cost, 2)})
    table = pd.DataFrame(rows).set_index("Material")
    total_row = pd.DataFrame({"Group": ["TOTAL"], "Dry kg / t sinter": [total_dry_kg], "Moisture %": [0.0],
                               "Wet (As-Received) kg": [total_wet_kg], "Wet Cost (Rs/t)": [round(wet_rm_cost, 2)]}, index=["TOTAL"])
    table = pd.concat([table, total_row])
    table["Moisture %"] = table["Moisture %"].astype(float)
    total_sinter_cost_wet = wet_rm_cost + om_cost
    return table, wet_rm_cost, total_sinter_cost_wet


def compute_coke_heat_balance_diagnostic(blend, df, OUT,
                                          latent_heat=DEFAULT_HEAT_LATENT_MOISTURE,
                                          calcination_heat=DEFAULT_HEAT_CALCINATION_PER_LOI_KG,
                                          melting_heat=DEFAULT_HEAT_MELTING_PER_KG_SINTER,
                                          loss_fraction=DEFAULT_HEAT_LOSS_FRACTION,
                                          feo_min=DEFAULT_FEO_MIN_PCT, feo_target=DEFAULT_FEO_TARGET_PCT,
                                          feo_max=DEFAULT_FEO_MAX_PCT,
                                          feo_ref_surplus=DEFAULT_FEO_REFERENCE_THERMAL_SURPLUS_KCAL,
                                          feo_ref_pct=DEFAULT_FEO_REFERENCE_PCT,
                                          feo_thermal_slope=DEFAULT_FEO_THERMAL_SLOPE_PCT_PER_10K_KCAL,
                                          ref_coke_cv=DEFAULT_REFERENCE_COKE_CV_KCAL_KG,
                                          ref_coke_fc=DEFAULT_REFERENCE_COKE_FC_PCT):
    fuel_mats = [m for m in _fuel_materials(df) if m in blend]
    if not fuel_mats:
        return None
    per_fuel = {}; Q_fuel = 0.0; total_fuel_kg = 0.0
    for m in fuel_mats:
        kg = blend.get(m, 0.0)
        cv = float(df.loc[m, "CV_kcal_kg"]) if "CV_kcal_kg" in df.columns else DEFAULT_COKE_CV_KCAL_KG
        fc = float(df.loc[m, "FC_Pct"]) if "FC_Pct" in df.columns else DEFAULT_COKE_FC_PCT
        if cv <= 0: cv = DEFAULT_COKE_CV_KCAL_KG
        if fc <= 0: fc = DEFAULT_COKE_FC_PCT
        q = kg * (fc / 100) * cv
        Q_fuel += q; total_fuel_kg += kg
        per_fuel[m] = {"kg": kg, "CV": cv, "FC": fc, "Q_kcal": q}
    total_wet_mass = 0.0
    for m in blend:
        if m in fuel_mats: continue
        moisture = df.loc[m, "Moisture_Pct"] / 100 if "Moisture_Pct" in df.columns else 0.0
        wet_kg = blend[m] / (1 - moisture) if moisture < 1 else blend[m]
        total_wet_mass += wet_kg
    dry_nonfuel = sum(blend[m] for m in blend if m not in fuel_mats)
    moisture_mass = max(total_wet_mass - dry_nonfuel, 0.0)
    Q_moisture = moisture_mass * latent_heat
    total_loi_mass = sum(blend[m] * df.loc[m, "LOI"] / 100 for m in blend if m not in fuel_mats)
    Q_calcination = total_loi_mass * calcination_heat
    Q_melting = OUT * melting_heat
    Q_required_before_loss = Q_moisture + Q_calcination + Q_melting
    Q_required = Q_required_before_loss / (1 - loss_fraction) if (1 - loss_fraction) > 0 else 0.0
    firing_ratio = Q_fuel / Q_required if Q_required > 0 else 0.0
    thermal_surplus = Q_fuel - Q_required
    weighted_fc = sum(per_fuel[m]["kg"]*per_fuel[m]["FC"] for m in per_fuel)/total_fuel_kg if total_fuel_kg > 0 else DEFAULT_COKE_FC_PCT
    weighted_cv = sum(per_fuel[m]["kg"]*per_fuel[m]["CV"] for m in per_fuel)/total_fuel_kg if total_fuel_kg > 0 else DEFAULT_COKE_CV_KCAL_KG
    effective_coke_kg_t = total_fuel_kg * (weighted_fc / ref_coke_fc) * (weighted_cv / ref_coke_cv)
    feo_est = feo_ref_pct + feo_thermal_slope * ((thermal_surplus - feo_ref_surplus) / 10000.0)
    if feo_est < feo_min:
        suggestion = f"FeO {feo_est:.2f}% below minimum {feo_min:.2f}% → increase coke / review thermal conditions"
    elif feo_est > feo_max:
        suggestion = f"FeO {feo_est:.2f}% above maximum {feo_max:.2f}% → reduce coke / review thermal conditions"
    elif abs(feo_est - feo_target) <= 0.15:
        suggestion = f"FeO {feo_est:.2f}% is close to target {feo_target:.2f}% – no adjustment suggested"
    elif feo_est < feo_target:
        suggestion = f"FeO {feo_est:.2f}% is below target {feo_target:.2f}% but within operating band"
    else:
        suggestion = f"FeO {feo_est:.2f}% is above target {feo_target:.2f}% but within operating band"
    return {"Per_Fuel": per_fuel, "Total_Fuel_kg": total_fuel_kg, "Weighted_CV": weighted_cv, "Weighted_FC": weighted_fc,
            "Q_fuel_kcal": Q_fuel, "Q_required_kcal": Q_required, "Thermal_Surplus_kcal": thermal_surplus,
            "Firing_Ratio": firing_ratio, "FeO_Estimate_Pct": feo_est, "FeO_Min_Pct": feo_min,
            "FeO_Target_Pct": feo_target, "FeO_Max_Pct": feo_max, "Reference_Thermal_Surplus_kcal": feo_ref_surplus,
            "Thermal_Slope_Pct_per_10k_kcal": feo_thermal_slope, "Effective_Coke_kg_t": effective_coke_kg_t,
            "Controller_Suggestion": suggestion,
            "note": "⚠️ PROVISIONAL FeO thermal-state model – calibrate coefficients against plant Coke/FeO history."}


# ============================================================================
# ROLE-BASED ORE HELPERS
# ============================================================================
ROLE_COLUMN = "Material_Role"


def _ensure_material_role(df):
    d = df.copy()
    if ROLE_COLUMN not in d.columns:
        legacy = {"SKME", "THAKUR", "RBSSN", "SMIORE"}
        d[ROLE_COLUMN] = ["Alternative_Iron_Ore" if str(m).strip().upper() in legacy
                           else ("Primary_Iron_Ore" if str(d.loc[m, "Group"]).strip() == "Iron_ore" else "Other")
                           for m in d.index]
    if "CV_kcal_kg" not in d.columns: d["CV_kcal_kg"] = 0.0
    if "FC_Pct" not in d.columns: d["FC_Pct"] = 0.0
    if "Fines_Pct" not in d.columns: d["Fines_Pct"] = 0.0
    return d


def _primary_iron_ores(df):
    d = _ensure_material_role(df)
    return [m for m in d.index if str(d.loc[m, "Group"]).strip() == "Iron_ore"
            and str(d.loc[m, ROLE_COLUMN]).strip() == "Primary_Iron_Ore"]


def _alternative_iron_ores(df):
    d = _ensure_material_role(df)
    return [m for m in d.index if str(d.loc[m, "Group"]).strip() == "Iron_ore"
            and str(d.loc[m, ROLE_COLUMN]).strip() == "Alternative_Iron_Ore"]


def _eligible_iron_ores(df):
    d = _ensure_material_role(df)
    return _primary_iron_ores(d) + [m for m in _alternative_iron_ores(d)
                                     if float(d.loc[m, "Available_Tonnes"]) > 0 and float(d.loc[m, "Tech_Max"]) > 0]


def _fuel_materials(df):
    return [m for m in df.index if str(df.loc[m, "Group"]).strip() == "Fuel"]


# ============================================================================
# MAIN SOLVER
# ============================================================================
def solve_blend_with_compensation(df, production_tonnes, targets, baseline_blend=None,
                                   enforce_b4=False, b4_min=1.8, b4_max=2.0,
                                   iol_nominal=IOL_FINES_NOMINAL_PCT, bf_nominal=BF_RETURNS_NOMINAL_PCT,
                                   iol_fb_min=IOL_FINES_FALLBACK_MIN, iol_fb_max=IOL_FINES_FALLBACK_MAX,
                                   bf_fb_min=BF_RETURNS_FALLBACK_MIN, bf_fb_max=BF_RETURNS_FALLBACK_MAX,
                                   latent_heat=DEFAULT_HEAT_LATENT_MOISTURE,
                                   calcination_heat=DEFAULT_HEAT_CALCINATION_PER_LOI_KG,
                                   melting_heat=DEFAULT_HEAT_MELTING_PER_KG_SINTER,
                                   loss_fraction=DEFAULT_HEAT_LOSS_FRACTION,
                                   firing_ratio_max=DEFAULT_FIRING_RATIO_MAX,
                                   enforce_firing_ratio_max=ENFORCE_FIRING_RATIO_MAX,
                                   coke_min_rate=DEFAULT_COKE_MIN_KG_T, coke_max_rate=DEFAULT_COKE_MAX_KG_T,
                                   feo_min=DEFAULT_FEO_MIN_PCT, feo_target=DEFAULT_FEO_TARGET_PCT, feo_max=DEFAULT_FEO_MAX_PCT,
                                   feo_ref_surplus=DEFAULT_FEO_REFERENCE_THERMAL_SURPLUS_KCAL,
                                   feo_ref_pct=DEFAULT_FEO_REFERENCE_PCT,
                                   feo_thermal_slope=DEFAULT_FEO_THERMAL_SLOPE_PCT_PER_10K_KCAL,
                                   ref_coke_cv=DEFAULT_REFERENCE_COKE_CV_KCAL_KG, ref_coke_fc=DEFAULT_REFERENCE_COKE_FC_PCT,
                                   manual_override=False, manual_coke_rate=65.0, ksl_split_pct=DEFAULT_KSL_SPLIT_PCT,
                                   fuel_ash_settings=None):
    OUT = 1000
    df = _ensure_material_role(df)
    fuel_ash_settings = get_active_fuel_ash_settings(df) if fuel_ash_settings is None else _normalise_fuel_ash_settings(fuel_ash_settings, df=df)
    iron_ores = _eligible_iron_ores(df)
    fluxes = [m for m in df.index if df.loc[m, "Group"] == "Flux"]
    fuels = _fuel_materials(df)

    fuel_ok, fuel_problems = check_fuel_gate(df)
    if not fuel_ok:
        diagnostics = ["🚫 PRODUCTION IMPOSSIBLE: Fuel requirement cannot be met."] + [f"   {p}" for p in fuel_problems]
        return "No_Production", None, None, None, diagnostics, False

    # Build inventory-aware R/S shortfall information before selecting the
    # iron-ore compensation tier.
    rs_info = get_rs_shortfall_info(
        df, production_tonnes, iol_nominal=iol_nominal, bf_nominal=bf_nominal
    )
    rs_shortfall_active = (
        RS_SHORTFALL_COMPENSATION_ENABLED and rs_info["active"]
    )
    rs_source_caps = {
        "IOL_Fines": rs_info["iol_capacity_kg_t"],
        "BF_Returns": rs_info["bfr_capacity_kg_t"],
    }

    mandate_reasons = []
    for mat, nominal in [("IOL_Fines", iol_nominal), ("BF_Returns", bf_nominal)]:
        if mat not in df.index:
            mandate_reasons.append(f"{mat} missing from chemistry master")
        elif rs_shortfall_active:
            cap = rs_source_caps.get(mat, 0.0)
            target = nominal * 1000.0
            if cap + RS_SHORTFALL_TOLERANCE_KG_T < target:
                mandate_reasons.append(
                    f"{mat} shortfall {target-cap:.1f} kg/t "
                    f"(usable {cap:.1f} vs target {target:.1f} kg/t)"
                )

    # A Return Sinter shortfall is an additional iron-bearing compensation
    # trigger. Existing unavailable iron ores are counted as before.
    extra_missing = 1 if rs_shortfall_active else 0
    extra_reasons = list(mandate_reasons) if rs_shortfall_active else []
    iron_ore_max_pct, unavailable_iron, iron_msg, iron_tier = get_iron_ore_tier(
        df, iron_ores, extra_missing=extra_missing, extra_reasons=extra_reasons
    )
    flux_max_pct, unavailable_flux, flux_msg, flux_tier = get_flux_tier(df, fluxes)

    diagnostics = [iron_msg, flux_msg]
    if rs_shortfall_active:
        diagnostics.append(
            f"⚠️ R/S SHORTFALL COMPENSATION ACTIVE — target {rs_info['rs_target_kg_t']:.1f} kg/t "
            f"({rs_info['rs_target_kg_t']/10:.1f}%), usable R/S {rs_info['rs_available_kg_t']:.1f} kg/t "
            f"({rs_info['rs_available_kg_t']/10:.1f}%), shortfall {rs_info['rs_shortfall_kg_t']:.1f} kg/t "
            f"({rs_info['rs_shortfall_kg_t']/10:.1f}%)."
        )
    if mandate_reasons:
        diagnostics += ["⚠️ R/S AVAILABILITY ISSUE — production-continuity compensation is active."] + [f"   {r}" for r in mandate_reasons]

    bounds = build_bounds(df, production_tonnes)
    if unavailable_iron:
        diagnostics.append(f"🔄 Compensating for missing ore(s): {', '.join(unavailable_iron)} (Iron tier: {iron_tier})")
    if unavailable_flux:
        diagnostics.append(f"🔄 Compensating for missing flux(es): {', '.join(unavailable_flux)} (Flux tier: {flux_tier})")

    fe_lo = FE_LOWER * OUT / 100
    fe_hi = FE_UPPER * OUT / 100

    baseline_flux_portion = None
    if baseline_blend:
        baseline_flux_portion = sum(baseline_blend.get(m, 0) for m in fluxes) / OUT

    shortage_targets = None
    if iron_tier != "base":
        shortage_targets = dict(targets)
        shortage_targets["SiO2_max"] = SIO2_MAX_SHORTAGE
        shortage_targets["CaO_max"] = round(targets["Basicity_max"] * SIO2_MAX_SHORTAGE, 3)

    moisture_factor = {}; loi_factor = {}
    for m in df.index:
        if m in fuels: continue
        mois = df.loc[m, "Moisture_Pct"] / 100 if "Moisture_Pct" in df.columns else 0.0
        moisture_factor[m] = mois / (1 - mois) if mois < 1 else 0.0
        loi_factor[m] = df.loc[m, "LOI"] / 100

    fuel_cv = {m: (float(df.loc[m, "CV_kcal_kg"]) if float(df.loc[m, "CV_kcal_kg"]) > 0 else DEFAULT_COKE_CV_KCAL_KG) for m in fuels}
    fuel_fc = {m: (float(df.loc[m, "FC_Pct"]) if float(df.loc[m, "FC_Pct"]) > 0 else DEFAULT_COKE_FC_PCT) for m in fuels}

    def _add_recovery_mandate_constraints(prob, x):
        total_burden = pulp.lpSum(x[m] for m in df.index)
        if rs_shortfall_active:
            if "IOL_Fines" in x:
                prob += x["IOL_Fines"] == min(
                    rs_source_caps.get("IOL_Fines", 0.0), iol_nominal * OUT
                ), "IOL_Fines_Recovery_RS_Shortfall"
            if "BF_Returns" in x:
                prob += x["BF_Returns"] == min(
                    rs_source_caps.get("BF_Returns", 0.0), bf_nominal * OUT
                ), "BF_Returns_Recovery_RS_Shortfall"
            return

        for mat, nominal in [("IOL_Fines", iol_nominal), ("BF_Returns", bf_nominal)]:
            if mat not in x:
                continue
            available = _rs_source_is_available(df, mat)
            if available:
                prob += x[mat] == nominal * total_burden, f"{mat}_Recovery_Strict"
            else:
                prob += x[mat] == 0, f"{mat}_Recovery_Unavailable"

        if "IOL_Fines" in x and "BF_Returns" in x:
            combined_nominal = iol_nominal + bf_nominal
            prob += (
                x["IOL_Fines"] + x["BF_Returns"]
                == combined_nominal * total_burden
            ), "RETURN_SINTER_TOTAL_RECOVERY_STRICT"


    def _fuel_q_expr(x):
        return pulp.lpSum(x[m] * (fuel_fc[m] / 100) * fuel_cv[m] for m in fuels if m in x)

    def _add_fuel_split_constraint(prob, x, tag_suffix=""):
        ksl, local = _find_coke_sources(fuels)
        if not (ksl and local and ksl in x and local in x):
            return
        total_fuel = x[ksl] + x[local]
        mode = FUEL_SOURCE_SELECTION_MODE.upper()
        if mode == "MANUAL":
            ksl_pct = float(MANUAL_KSL_FUEL_PCT); local_pct = float(MANUAL_LOCAL_FUEL_PCT)
            if abs((ksl_pct + local_pct) - 100.0) > 1e-6:
                raise ValueError("Manual KSL + Local Coke percentages must total 100%.")
            prob += x[ksl] == (ksl_pct / 100.0) * total_fuel, f"Manual_Fuel_Split_KSL{tag_suffix}"
            prob += x[local] == (local_pct / 100.0) * total_fuel, f"Manual_Fuel_Split_Local{tag_suffix}"
        elif mode == "OPTIMIZER":
            ksl_available = float(df.loc[ksl, "Available_Tonnes"]) > 0 and float(df.loc[ksl, "Tech_Max"]) > 0
            local_available = float(df.loc[local, "Available_Tonnes"]) > 0 and float(df.loc[local, "Tech_Max"]) > 0
            if ksl_available and local_available:
                lo = OPTIMIZER_MIN_COKE_SOURCE_PCT / 100.0; hi = OPTIMIZER_MAX_COKE_SOURCE_PCT / 100.0
                prob += x[ksl] >= lo * total_fuel, f"Optimizer_KSL_MinShare{tag_suffix}"
                prob += x[ksl] <= hi * total_fuel, f"Optimizer_KSL_MaxShare{tag_suffix}"
                prob += x[local] >= lo * total_fuel, f"Optimizer_Local_MinShare{tag_suffix}"
                prob += x[local] <= hi * total_fuel, f"Optimizer_Local_MaxShare{tag_suffix}"

    def _build_and_solve(flux_min_pct_override, flux_tier_label, tag, use_targets=None, mandate_mode="pinned"):
        t = use_targets if use_targets is not None else targets
        prob = pulp.LpProblem(f"Sinter_Burden_Opt_{tag}", pulp.LpMinimize)
        x = {m: pulp.LpVariable(f"x{tag}_{m}", lowBound=bounds[m][0], upBound=bounds[m][1]) for m in df.index}
        add_structural_constraints(prob, x, df, bounds, iron_ores, fluxes, iron_ore_max_pct, unavailable_iron,
                                    flux_max_pct, unavailable_flux, OUT, baseline_flux_portion, iron_tier, flux_tier_label,
                                    flux_min_pct_override=flux_min_pct_override, mandate_mode=mandate_mode,
                                    iol_nominal=iol_nominal, bf_nominal=bf_nominal, iol_fb_min=iol_fb_min,
                                    iol_fb_max=iol_fb_max, bf_fb_min=bf_fb_min, bf_fb_max=bf_fb_max,
                                    fuel_ash_settings=fuel_ash_settings,
                                    rs_shortfall_active=rs_shortfall_active, rs_source_caps=rs_source_caps)
        _add_fuel_split_constraint(prob, x, tag_suffix=f"_{tag}")
        Fe_sum, SiO2_sum, Al2O3_sum, CaO_sum, MgO_sum = _lp_chemistry_sums(x, df, fuel_ash_settings)
        prob += Fe_sum >= fe_lo, "Fe_min_hard"
        prob += Fe_sum <= fe_hi, "Fe_max_hard"
        prob += SiO2_sum <= t["SiO2_max"] * OUT / 100, "SiO2_max_hard"
        prob += Al2O3_sum <= t["Al2O3_max"] * OUT / 100, "Al2O3_max_hard"
        prob += (Al2O3_sum - t["Al2O3_SiO2_max"] * SiO2_sum <= 0), "Al2O3_SiO2_max_hard"
        prob += (CaO_sum >= t["Basicity_min"] * SiO2_sum), "Basicity_min_hard"
        prob += (CaO_sum <= t["Basicity_max"] * SiO2_sum), "Basicity_max_hard"
        prob += MgO_sum >= t["MgO_min"] * OUT / 100, "MgO_min_hard"
        prob += MgO_sum <= t["MgO_max"] * OUT / 100, "MgO_max_hard"
        prob += CaO_sum >= t["CaO_min"] * OUT / 100, "CaO_min_hard"
        prob += CaO_sum <= t["CaO_max"] * OUT / 100, "CaO_max_hard"
        if enforce_b4:
            prob += ((CaO_sum + MgO_sum) - b4_min * (SiO2_sum + Al2O3_sum) >= 0), "B4_min"
            prob += ((CaO_sum + MgO_sum) - b4_max * (SiO2_sum + Al2O3_sum) <= 0), "B4_max"

        cost_expr = pulp.lpSum(x[m] * df.loc[m, "Price_Rs_t"] / 1000 for m in x)

        ksl_fuel = next((m for m in fuels if str(m).strip().upper() == "KSL_COKE"), None)
        local_fuel = next((m for m in fuels if str(m).strip().upper() == "LOCAL_COKE"), None)
        if ksl_fuel and local_fuel and FUEL_SOURCE_SELECTION_MODE.upper() == "MANUAL":
            manual_sum = MANUAL_KSL_FUEL_PCT + MANUAL_LOCAL_FUEL_PCT
            if abs(manual_sum - 100.0) > 1e-6:
                raise ValueError("Manual KSL + Local Coke percentages must total 100%.")
            total_fuel_expr = pulp.lpSum(x[m] for m in fuels if m in x)
            prob += x[ksl_fuel] == (MANUAL_KSL_FUEL_PCT / 100.0) * total_fuel_expr, f"Manual_KSL_Fuel_Share_{tag}"
            prob += x[local_fuel] == (MANUAL_LOCAL_FUEL_PCT / 100.0) * total_fuel_expr, f"Manual_Local_Fuel_Share_{tag}"

        suitability_table = _ore_suitability_table(df)
        ore_score = {r["Material"]: float(r["Suitability"])/100.0 for _, r in suitability_table.iterrows()} if not suitability_table.empty else {}
        ore_suitability_expr = pulp.lpSum(x[m] * ore_score.get(m, 0.0) for m in iron_ores if m in x)
        fuel_suitability_table = _fuel_source_suitability_table(df)
        fuel_score = {r["Material"]: float(r["Suitability"])/100.0 for _, r in fuel_suitability_table.iterrows()} if not fuel_suitability_table.empty else {}
        fuel_suitability_expr = pulp.lpSum(x[m] * fuel_score.get(m, 0.0) for m in fuels if m in x)

        ore_max_share = pulp.LpVariable(f"Ore_Max_Share_{tag}", lowBound=0.0, upBound=1.0)
        for m in iron_ores:
            prob += x[m] <= ore_max_share * OUT, f"Ore_Max_Share_Constraint_{tag}_{m}"

        if fuels:
            total_fuel = pulp.lpSum(x[m] for m in fuels if m in x)
            prob += total_fuel >= coke_min_rate, "Coke_Practical_Min"
            prob += total_fuel <= coke_max_rate, "Coke_Practical_Max"
            Q_fuel = _fuel_q_expr(x)
            Q_moisture = latent_heat * pulp.lpSum(x[m] * moisture_factor.get(m, 0.0) for m in x if m not in fuels)
            Q_calcination = calcination_heat * pulp.lpSum(x[m] * loi_factor.get(m, 0.0) for m in x if m not in fuels)
            Q_melting = melting_heat * OUT
            Q_required = (Q_moisture + Q_calcination + Q_melting) / (1 - loss_fraction)
            thermal_surplus = Q_fuel - Q_required
            if manual_override:
                prob += total_fuel == manual_coke_rate, "Manual_Coke_Override"
            else:
                prob += Q_fuel >= Q_required, "Heat_Balance_Min"
                if enforce_firing_ratio_max:
                    prob += (Q_fuel <= Q_required * firing_ratio_max), "Firing_Ratio_Max"
            FeO_pred = feo_ref_pct + feo_thermal_slope * ((thermal_surplus - feo_ref_surplus) / 10000.0)
            prob += FeO_pred >= feo_min, "FeO_Min_Hard"
            prob += FeO_pred <= feo_max, "FeO_Max_Hard"
            feo_dev = pulp.LpVariable(f"FeO_Target_Deviation_{tag}", lowBound=0)
            prob += (feo_dev >= FeO_pred - feo_target), f"FeO_Target_Dev_Pos_{tag}"
            prob += (feo_dev >= feo_target - FeO_pred), f"FeO_Target_Dev_Neg_{tag}"
            if manual_override:
                prob.setObjective(cost_expr)
            else:
                prob.setObjective(feo_dev)
                prob.solve(pulp.PULP_CBC_CMD(msg=0))
                if pulp.LpStatus[prob.status] != "Optimal":
                    return prob, x, pulp.LpStatus[prob.status]
                min_dev = max(0.0, float(pulp.value(feo_dev) or 0.0))
                prob += (feo_dev <= min_dev + 1e-6), f"FeO_Target_Dev_Pin_{tag}"

        prob.sense = pulp.LpMaximize
        combined_suitability_expr = ore_suitability_expr + fuel_suitability_expr
        prob.setObjective(combined_suitability_expr)
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        if pulp.LpStatus[prob.status] != "Optimal":
            return prob, x, pulp.LpStatus[prob.status]
        best_suitability = float(pulp.value(combined_suitability_expr) or 0.0)
        suitability_tol = max(0.01, abs(best_suitability) * 0.002)
        prob += (combined_suitability_expr >= best_suitability - suitability_tol), f"Combined_Material_Suitability_Pin_{tag}"

        prob.sense = pulp.LpMinimize
        prob.setObjective(ore_max_share)
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        if pulp.LpStatus[prob.status] != "Optimal":
            return prob, x, pulp.LpStatus[prob.status]
        best_max_share = float(pulp.value(ore_max_share) or 0.0)
        share_tol = 0.005
        prob += (ore_max_share <= best_max_share + share_tol), f"Ore_Diversity_Pin_{tag}"

        prob.sense = pulp.LpMinimize
        prob.setObjective(cost_expr)
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        return prob, x, pulp.LpStatus[prob.status]

    def _add_diagnostic_coke_constraints(prob, x, suffix=""):
        if not fuels: return
        _add_fuel_split_constraint(prob, x, tag_suffix=suffix)
        if manual_override:
            total_fuel = pulp.lpSum(x[m] for m in fuels if m in x)
            prob += (total_fuel == manual_coke_rate), f"Manual_Coke_Diagnostic{suffix}"
            return
        total_fuel = pulp.lpSum(x[m] for m in fuels if m in x)
        prob += total_fuel >= coke_min_rate, f"Coke_Practical_Min{suffix}"
        prob += total_fuel <= coke_max_rate, f"Coke_Practical_Max{suffix}"
        Q_fuel = _fuel_q_expr(x)
        Q_moisture = latent_heat * pulp.lpSum(x[m] * moisture_factor.get(m, 0.0) for m in x if m not in fuels)
        Q_calcination = calcination_heat * pulp.lpSum(x[m] * loi_factor.get(m, 0.0) for m in x if m not in fuels)
        Q_melting = melting_heat * OUT
        Q_required = (Q_moisture + Q_calcination + Q_melting) / (1 - loss_fraction)
        prob += Q_fuel >= Q_required, f"Heat_Balance_Min{suffix}"
        if enforce_firing_ratio_max:
            prob += (Q_fuel <= Q_required * firing_ratio_max), f"Firing_Ratio_Max{suffix}"
        thermal_surplus = Q_fuel - Q_required
        FeO_pred = feo_ref_pct + feo_thermal_slope * ((thermal_surplus - feo_ref_surplus) / 10000.0)
        prob += FeO_pred >= feo_min, f"FeO_Min_Hard{suffix}"
        prob += FeO_pred <= feo_max, f"FeO_Max_Hard{suffix}"

    def _finalize(prob, x, tag_label, note=None):
        blend = {m: round(x[m].value(), 2) for m in x}
        total_cost = pulp.value(prob.objective)
        achieved = compute_achieved(blend, df, OUT, fuel_ash_settings)
        diag = list(diagnostics)
        if note: diag.append(note)
        diag += _report_compensation(blend, df, iron_ores, fluxes, unavailable_iron, unavailable_flux,
                                      iron_ore_max_pct, flux_max_pct, iron_tier, flux_tier, OUT, mandate_reasons)
        diag.append(_report_fines_loading(blend, df, OUT))
        heat_info = compute_coke_heat_balance_diagnostic(blend, df, OUT, latent_heat, calcination_heat, melting_heat,
                                                           loss_fraction, feo_min, feo_target, feo_max, feo_ref_surplus,
                                                           feo_ref_pct, feo_thermal_slope, ref_coke_cv, ref_coke_fc)
        if heat_info:
            diag.append(f"\n🔥 Optimised Total Fuel: {heat_info['Total_Fuel_kg']:.1f} kg/t" if not manual_override
                        else f"\n🔥 MANUAL OVERRIDE: Total fuel forced to {manual_coke_rate:.1f} kg/t")
            diag.append(f"   Firing Ratio: {heat_info['Firing_Ratio']:.3f} (max {firing_ratio_max})")
            diag.append(f"   Predicted FeO: {heat_info['FeO_Estimate_Pct']:.2f}% (target {feo_target:.2f}%, band {feo_min:.2f}-{feo_max:.2f}%)")
            diag.append(f"   {heat_info['Controller_Suggestion']}")
        return "Optimal", blend, total_cost, achieved, diag, False

    probA, xA, statusA = _build_and_solve(None, flux_tier, "A", mandate_mode="pinned")
    if statusA == "Optimal":
        return _finalize(probA, xA, "A", "✅ Mandates met at nominal.")

    diagnostics.append("⚠️ Base flux floors infeasible with pinned mandates – relaxing flux floors...")
    probB, xB, statusB = _build_and_solve(FLUX_MIN_PCT_QUALITY_RELAXED, "quality_relaxed", "B", mandate_mode="pinned")
    if statusB == "Optimal":
        return _finalize(probB, xB, "B", "✅ Resolved with relaxed flux floors.")

    if shortage_targets is not None:
        probC, xC, statusC = _build_and_solve(FLUX_MIN_PCT_QUALITY_RELAXED, "quality_relaxed", "C", use_targets=shortage_targets, mandate_mode="pinned")
        if statusC == "Optimal":
            return _finalize(probC, xC, "C", "✅ Resolved with widened ceilings – R/S shortfall compensated where required.")

    probD, xD, statusD = _build_and_solve(None, flux_tier, "D", mandate_mode="pinned")
    if statusD == "Optimal":
        return _finalize(probD, xD, "D", "✅ Solved with chemistry relaxation; R/S shortfall compensation remained active where required.")

    probE, xE, statusE = _build_and_solve(FLUX_MIN_PCT_QUALITY_RELAXED, "quality_relaxed", "E", mandate_mode="pinned")
    if statusE == "Optimal":
        return _finalize(probE, xE, "E", "✅ Solved with relaxed chemistry/flux floors; R/S shortfall compensation remained active where required.")

    if shortage_targets is not None:
        probF, xF, statusF = _build_and_solve(FLUX_MIN_PCT_QUALITY_RELAXED, "quality_relaxed", "F", use_targets=shortage_targets, mandate_mode="pinned")
        if statusF == "Optimal":
            return _finalize(probF, xF, "F", "✅ Solved using permitted chemistry relaxations; R/S shortfall compensation remained active where required.")

    if PRODUCTION_CONTINUITY_MODE:
        diagnostics.append("🟡 Production-continuity recovery: searching for a quality-compliant blend at higher cost if necessary.")
        probR = pulp.LpProblem("Production_Continuity_Recovery", pulp.LpMinimize)
        xR = {m: pulp.LpVariable(f"xR_{m}", lowBound=bounds[m][0], upBound=bounds[m][1]) for m in df.index}
        add_structural_constraints(probR, xR, df, bounds, iron_ores, fluxes, iron_ore_max_pct, unavailable_iron,
                                    flux_max_pct, unavailable_flux, OUT, baseline_flux_portion, iron_tier, "quality_relaxed",
                                    flux_min_pct_override=FLUX_MIN_PCT_QUALITY_RELAXED, mandate_mode="recovery",
                                    iol_nominal=iol_nominal, bf_nominal=bf_nominal, iol_fb_min=iol_fb_min,
                                    iol_fb_max=iol_fb_max, bf_fb_min=bf_fb_min, bf_fb_max=bf_fb_max,
                                    fuel_ash_settings=fuel_ash_settings,
                                    rs_shortfall_active=rs_shortfall_active, rs_source_caps=rs_source_caps)
        for cname in ["IOL_Fines_STRICT_BURDEN_PCT", "BF_Returns_STRICT_BURDEN_PCT"]:
            if cname in probR.constraints: del probR.constraints[cname]
        _add_recovery_mandate_constraints(probR, xR)
        Fe_sumR, SiO2_sumR, Al2O3_sumR, CaO_sumR, MgO_sumR = _lp_chemistry_sums(xR, df, fuel_ash_settings)
        probR += Fe_sumR >= fe_lo, "Recovery_Fe_min"
        probR += Fe_sumR <= fe_hi, "Recovery_Fe_max"
        probR += SiO2_sumR <= targets["SiO2_max"] * OUT / 100, "Recovery_SiO2_max"
        probR += Al2O3_sumR <= targets["Al2O3_max"] * OUT / 100, "Recovery_Al2O3_max"
        probR += Al2O3_sumR <= targets["Al2O3_SiO2_max"] * SiO2_sumR, "Recovery_Al2O3_SiO2"
        probR += CaO_sumR >= targets["Basicity_min"] * SiO2_sumR, "Recovery_Basicity_min"
        probR += CaO_sumR <= targets["Basicity_max"] * SiO2_sumR, "Recovery_Basicity_max"
        probR += MgO_sumR >= targets["MgO_min"] * OUT / 100, "Recovery_MgO_min"
        probR += MgO_sumR <= targets["MgO_max"] * OUT / 100, "Recovery_MgO_max"
        probR += CaO_sumR >= targets["CaO_min"] * OUT / 100, "Recovery_CaO_min"
        probR += CaO_sumR <= targets["CaO_max"] * OUT / 100, "Recovery_CaO_max"
        _add_diagnostic_coke_constraints(probR, xR, suffix="_recovery")
        probR += pulp.lpSum(xR[m] * df.loc[m, "Price_Rs_t"] / 1000 for m in xR), "Recovery_Total_Cost"
        probR.solve(pulp.PULP_CBC_CMD(msg=0))
        if pulp.LpStatus[probR.status] == "Optimal":
            blendR = {m: round(xR[m].value(), 2) for m in xR}
            costR = sum(blendR[m] * df.loc[m, "Price_Rs_t"] / 1000 for m in blendR)
            achievedR = compute_achieved(blendR, df, OUT, fuel_ash_settings)
            rec_diag = list(diagnostics)
            rec_diag.append("🟢 Production-continuity recovery found a quality-compliant blend.")
            rec_diag.append(f"💰 Recovery cost: Rs {costR:,.2f}/t — cost may be higher because available materials are constrained.")
            if mandate_reasons:
                rec_diag.append("⚠️ Mandate exception: an unavailable IOL/BF material could not be physically loaded.")
            rec_diag += _report_compensation(blendR, df, iron_ores, fluxes, unavailable_iron, unavailable_flux,
                                              iron_ore_max_pct, flux_max_pct, iron_tier, flux_tier, OUT, mandate_reasons)
            return "Recovery", blendR, costR, achievedR, rec_diag, False

    diagnostics.append("⚠️ All hard constraints infeasible – generating closest achievable reference.")
    prob1 = pulp.LpProblem("Phase1_MinDeviation", pulp.LpMinimize)
    x1 = {m: pulp.LpVariable(f"x1_{m}", lowBound=bounds[m][0], upBound=bounds[m][1]) for m in df.index}
    add_structural_constraints(prob1, x1, df, bounds, iron_ores, fluxes, iron_ore_max_pct, unavailable_iron,
                                flux_max_pct, unavailable_flux, OUT, baseline_flux_portion, iron_tier, "quality_relaxed",
                                flux_min_pct_override=FLUX_MIN_PCT_QUALITY_RELAXED, mandate_mode="pinned",
                                iol_nominal=iol_nominal, bf_nominal=bf_nominal, iol_fb_min=iol_fb_min,
                                iol_fb_max=iol_fb_max, bf_fb_min=bf_fb_min, bf_fb_max=bf_fb_max,
                                fuel_ash_settings=fuel_ash_settings,
                                rs_shortfall_active=rs_shortfall_active, rs_source_caps=rs_source_caps)
    _add_diagnostic_coke_constraints(prob1, x1, suffix="_p1")
    slacks1, sums1 = build_soft_vars_and_constraints(prob1, x1, df, OUT, targets, fe_lo, fe_hi, suffix="_p1", fuel_ash_settings=fuel_ash_settings)
    obj1 = weighted_deviation_expr(slacks1, targets, OUT)
    prob1 += obj1, "Total_Weighted_Deviation"
    prob1.solve(pulp.PULP_CBC_CMD(msg=0))
    if pulp.LpStatus[prob1.status] != "Optimal":
        diagnostics.append("❌ No feasible blend even in diagnostic mode. Check inventory.")
        return "Production_Risk", None, None, None, diagnostics, True
    blend1 = {m: round(x1[m].value(), 2) for m in x1}
    achieved1 = compute_achieved(blend1, df, OUT, fuel_ash_settings)
    phase1_slack_values = {k: (v.value() or 0.0) for k, v in slacks1.items()}
    dev_report = {"Fe": phase1_slack_values["Fe_under"]+phase1_slack_values["Fe_over"],
                  "SiO2": phase1_slack_values["SiO2_over"], "Al2O3": phase1_slack_values["Al2O3_over"],
                  "Al2O3/SiO2": phase1_slack_values["ratio_over"],
                  "Basicity": phase1_slack_values["Bas_under"]+phase1_slack_values["Bas_over"],
                  "MgO": phase1_slack_values["MgO_under"]+phase1_slack_values["MgO_over"],
                  "CaO": phase1_slack_values["CaO_under"]+phase1_slack_values["CaO_over"]}
    binding_spec = max(dev_report, key=dev_report.get)
    diagnostics.append(f"📊 Most binding spec: {binding_spec} (deviation = {dev_report[binding_spec]:.3f})")
    diagnostics.append("🏃 Phase 2: minimising cost at same deviation...")

    prob2 = pulp.LpProblem("Phase2_MinCost", pulp.LpMinimize)
    x2 = {m: pulp.LpVariable(f"x2_{m}", lowBound=bounds[m][0], upBound=bounds[m][1]) for m in df.index}
    add_structural_constraints(prob2, x2, df, bounds, iron_ores, fluxes, iron_ore_max_pct, unavailable_iron,
                                flux_max_pct, unavailable_flux, OUT, baseline_flux_portion, iron_tier, "quality_relaxed",
                                flux_min_pct_override=FLUX_MIN_PCT_QUALITY_RELAXED, mandate_mode="pinned",
                                iol_nominal=iol_nominal, bf_nominal=bf_nominal, iol_fb_min=iol_fb_min,
                                iol_fb_max=iol_fb_max, bf_fb_min=bf_fb_min, bf_fb_max=bf_fb_max,
                                fuel_ash_settings=fuel_ash_settings,
                                rs_shortfall_active=rs_shortfall_active, rs_source_caps=rs_source_caps)
    _add_diagnostic_coke_constraints(prob2, x2, suffix="_p2")
    slacks2, sums2 = build_soft_vars_and_constraints(prob2, x2, df, OUT, targets, fe_lo, fe_hi, suffix="_p2", fuel_ash_settings=fuel_ash_settings)
    for key, var in slacks2.items():
        p1_val = phase1_slack_values.get(key, 0.0)
        prob2 += var <= p1_val + PIN_TOLERANCE, f"Pin_{key}"
    prob2 += pulp.lpSum(x2[m] * df.loc[m, "Price_Rs_t"] / 1000 for m in x2), "Total_Cost"
    prob2.solve(pulp.PULP_CBC_CMD(msg=0))
    if pulp.LpStatus[prob2.status] == "Optimal":
        blend2 = {m: round(x2[m].value(), 2) for m in x2}
        cost2 = sum(blend2[m] * df.loc[m, "Price_Rs_t"] / 1000 for m in blend2)
        achieved2 = compute_achieved(blend2, df, OUT, fuel_ash_settings)
        diagnostics.append(f"✅ Phase 2 complete. Reference cost: Rs {cost2:,.2f}/t")
        return "Production_Risk", blend2, cost2, achieved2, diagnostics, True
    else:
        diagnostics.append("⚠️ Phase 2 issue – returning Phase 1 blend.")
        return "Production_Risk", blend1, None, achieved1, diagnostics, True


def get_baseline_blend(df, targets, enforce_b4=False):
    result = solve_blend_with_compensation(df, 1000, targets, baseline_blend=None, enforce_b4=enforce_b4)
    status, blend, cost, achieved = result[0], result[1], result[2], result[3]
    if status in ("Optimal", "Recovery", "Production_Risk") and blend is not None:
        return blend, cost, achieved
    return None, None, None


# ============================================================================
# PRODUCTIVITY
# ============================================================================
def calculate_productivity_metrics(total_dry_burden_kg_t, feed_rate_t_h, productivity_target=None):
    total_dry_burden_kg_t = float(total_dry_burden_kg_t)
    feed_rate_t_h = float(feed_rate_t_h)
    if not np.isfinite(total_dry_burden_kg_t) or total_dry_burden_kg_t <= 0:
        raise ValueError("Total dry burden must be greater than zero.")
    if not np.isfinite(feed_rate_t_h) or feed_rate_t_h <= 0:
        raise ValueError("Feed rate must be greater than zero.")
    sinter_yield = 1000.0 / total_dry_burden_kg_t
    gross_sinter_t_h = sinter_yield * feed_rate_t_h
    net_sinter_t_h = gross_sinter_t_h - (0.20 * gross_sinter_t_h)
    if net_sinter_t_h <= 0:
        raise ValueError("Calculated net sinter must be greater than zero.")
    gross_net_ratio = gross_sinter_t_h / net_sinter_t_h
    base_productivity = gross_sinter_t_h / net_sinter_t_h
    final_productivity = base_productivity - (PRODUCTIVITY_ADJUSTMENT_PCT * base_productivity)
    result = {"Total_Dry_Burden_kg_t": total_dry_burden_kg_t, "Sinter_Yield": sinter_yield,
              "Sinter_Yield_Pct": sinter_yield * 100.0, "Feed_Rate_t_h": feed_rate_t_h,
              "Gross_Sinter_t_h": gross_sinter_t_h, "Gross_Loss_20pct_t_h": 0.20 * gross_sinter_t_h,
              "Net_Sinter_t_h": net_sinter_t_h, "Gross_Net_Ratio": gross_net_ratio,
              "Base_Productivity": base_productivity, "Final_Productivity": final_productivity}
    if productivity_target is not None:
        target = float(productivity_target)
        if not np.isfinite(target) or target <= 0:
            raise ValueError("Productivity target must be greater than zero.")
        result["Productivity_Target"] = target
        result["Achievement_Pct"] = final_productivity / target * 100.0
        result["Target_Achieved"] = final_productivity >= target
    return result


# ============================================================================
# PLANT VALIDATION
# ============================================================================
def compute_validation_chemistry(blend, df, finished_sinter_kg, recovery=None, fuel_ash_settings=None):
    if finished_sinter_kg <= 0:
        raise ValueError("Finished sinter quantity must be greater than zero.")
    if recovery is None:
        recovery = {"Fe": 1.0, "SiO2": 1.0, "Al2O3": 1.0, "CaO": 1.0, "MgO": 1.0}
    for key in ["Fe", "SiO2", "Al2O3", "CaO", "MgO"]:
        if key not in recovery or recovery[key] < 0:
            raise ValueError(f"Invalid recovery factor for {key}.")
    nonfuel = [m for m in blend if str(df.loc[m, "Group"]).strip().lower() != "fuel"]
    fe_mass_input = sum(blend[m] * float(df.loc[m, "Fe"]) / 100.0 for m in nonfuel)
    sio2_mass_input = sum(blend[m] * float(df.loc[m, "SiO2"]) / 100.0 for m in nonfuel)
    al2o3_mass_input = sum(blend[m] * float(df.loc[m, "Al2O3"]) / 100.0 for m in nonfuel)
    cao_mass_input = sum(blend[m] * float(df.loc[m, "CaO"]) / 100.0 for m in nonfuel)
    mgo_mass_input = sum(blend[m] * float(df.loc[m, "MgO"]) / 100.0 for m in nonfuel)
    loi_mass = sum(blend[m] * float(df.loc[m, "LOI"]) / 100.0 for m in nonfuel)
    ash = compute_fuel_ash_contribution(blend, df, fuel_ash_settings)
    sio2_mass_input += ash["SiO2_kg"]; al2o3_mass_input += ash["Al2O3_kg"]
    cao_mass_input += ash["CaO_kg"]; mgo_mass_input += ash["MgO_kg"]
    total_dry_burden = sum(blend.values())
    actual_yield_pct = finished_sinter_kg / total_dry_burden * 100.0 if total_dry_burden > 0 else np.nan
    fe_mass_sinter = fe_mass_input * recovery["Fe"]
    sio2_mass_sinter = sio2_mass_input * recovery["SiO2"]
    al2o3_mass_sinter = al2o3_mass_input * recovery["Al2O3"]
    cao_mass_sinter = cao_mass_input * recovery["CaO"]
    mgo_mass_sinter = mgo_mass_input * recovery["MgO"]
    Fe = fe_mass_sinter / finished_sinter_kg * 100.0
    SiO2 = sio2_mass_sinter / finished_sinter_kg * 100.0
    Al2O3 = al2o3_mass_sinter / finished_sinter_kg * 100.0
    CaO = cao_mass_sinter / finished_sinter_kg * 100.0
    MgO = mgo_mass_sinter / finished_sinter_kg * 100.0
    Basicity = CaO / SiO2 if SiO2 > 0 else np.nan
    Al2O3_SiO2 = Al2O3 / SiO2 if SiO2 > 0 else np.nan
    return {"Fe": Fe, "SiO2": SiO2, "Al2O3": Al2O3, "CaO": CaO, "MgO": MgO, "Basicity": Basicity,
            "Al2O3/SiO2": Al2O3_SiO2, "Fe_mass_input_kg": fe_mass_input, "SiO2_mass_input_kg": sio2_mass_input,
            "Al2O3_mass_input_kg": al2o3_mass_input, "CaO_mass_input_kg": cao_mass_input, "MgO_mass_input_kg": mgo_mass_input,
            "Fuel_Ash_kg": ash["Ash_kg"], "Fuel_Ash_SiO2_kg": ash["SiO2_kg"], "Fuel_Ash_Al2O3_kg": ash["Al2O3_kg"],
            "Fuel_Ash_CaO_kg": ash["CaO_kg"], "Fuel_Ash_MgO_kg": ash["MgO_kg"], "Total_dry_burden_kg": total_dry_burden,
            "Finished_sinter_kg": finished_sinter_kg, "Actual_yield_pct": actual_yield_pct, "LOI_mass_kg": loi_mass,
            "Fe_recovery": recovery["Fe"], "SiO2_recovery": recovery["SiO2"], "Al2O3_recovery": recovery["Al2O3"],
            "CaO_recovery": recovery["CaO"], "MgO_recovery": recovery["MgO"], "Fe_mass_sinter_kg": fe_mass_sinter,
            "SiO2_mass_sinter_kg": sio2_mass_sinter, "Al2O3_mass_sinter_kg": al2o3_mass_sinter,
            "CaO_mass_sinter_kg": cao_mass_sinter, "MgO_mass_sinter_kg": mgo_mass_sinter}


# ============================================================================
# EXCEL MASTER LOADER
# ============================================================================
MASTER_REQUIRED_COLUMNS = ["Material", "Group", "Material_Role", "Fe_%", "SiO2_%", "Al2O3_%", "CaO_%",
                            "MgO_%", "LOI_%", "Moisture_%", "Tech_Min_kg_t", "Tech_Max_kg_t",
                            "Available_Stock_t", "Price_Rs_t", "Fines_%", "CV_kcal_kg", "FC_Pct"]

MASTER_COLUMN_ALIASES = {
    "Material": ["Material", "Material_Name", "Material Name"],
    "Group": ["Group", "Material_Group", "Material Type", "Material_Type"],
    "Material_Role": ["Material_Role", "Material Role", "Role"],
    "Fe_%": ["Fe_%", "Fe", "Fe %", "Fe_Pct", "Fe_Pct_%"],
    "SiO2_%": ["SiO2_%", "SiO2", "SiO2 %", "SiO2_Pct", "SiO₂", "SiO₂ %"],
    "Al2O3_%": ["Al2O3_%", "Al2O3", "Al2O3 %", "Al2O3_Pct", "Al₂O₃", "Al₂O₃ %"],
    "CaO_%": ["CaO_%", "CaO", "CaO %", "CaO_Pct"],
    "MgO_%": ["MgO_%", "MgO", "MgO %", "MgO_Pct"],
    "LOI_%": ["LOI_%", "LOI", "LOI %", "LOI_Pct"],
    "Moisture_%": ["Moisture_%", "Moisture_Pct", "Moisture", "Moisture %", "Moisture_Pct_%"],
    "Tech_Min_kg_t": ["Tech_Min_kg_t", "Tech_Min", "Tech Min", "Technical_Min", "Technical_Min_kg_t"],
    "Tech_Max_kg_t": ["Tech_Max_kg_t", "Tech_Max", "Tech Max", "Technical_Max", "Technical_Max_kg_t"],
    "Available_Stock_t": ["Available_Stock_t", "Available_Tonnes", "Available Stock", "Stock_t", "Stock", "Availability_t"],
    "Price_Rs_t": ["Price_Rs_t", "Price", "Price Rs/t", "Price_Rs_per_t", "Cost_Rs_t"],
    "Fines_%": ["Fines_%", "Fines", "% Fines", "Fines %", "Fines_Pct", "Fines_Pct_%"],
    "CV_kcal_kg": ["CV_kcal_kg", "CV", "Calorific_Value", "CV (kcal/kg)"],
    "FC_Pct": ["FC_Pct", "FC", "Fixed_Carbon", "Fixed Carbon %", "FC %"],
}


def _normalize_excel_header(value):
    import unicodedata
    s = unicodedata.normalize("NFKC", str(value)).strip()
    s = s.replace("₂", "2").replace("₃", "3").replace("₄", "4").replace("₀", "0").replace("₁", "1")
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def _resolve_master_columns(columns):
    normalized = {}
    for c in columns:
        key = _normalize_excel_header(c)
        normalized.setdefault(key, []).append(c)
    mapping = {}; missing = []; ambiguous = []
    for canonical, aliases in MASTER_COLUMN_ALIASES.items():
        selected = None
        for alias in aliases:
            key = _normalize_excel_header(alias)
            matches = normalized.get(key, [])
            if matches:
                if len(matches) > 1:
                    ambiguous.append(f"{canonical}: multiple identical headers {matches}")
                else:
                    selected = matches[0]
                break
        if selected is None:
            if canonical not in ("CV_kcal_kg", "FC_Pct"):
                if not any(_normalize_excel_header(a) in normalized for a in aliases):
                    missing.append(canonical)
        else:
            mapping[canonical] = selected
    if missing or ambiguous:
        return None, missing, ambiguous
    return mapping, [], []


def _find_matching_master_sheet(uploaded_bytes):
    sheets = pd.read_excel(BytesIO(uploaded_bytes), sheet_name=None)
    candidates = []
    for sheet_name, frame in sheets.items():
        mapping, missing, ambiguous = _resolve_master_columns(frame.columns)
        if mapping is not None:
            candidates.append((sheet_name, frame, mapping))
    if not candidates:
        details = []
        for sheet_name, frame in sheets.items():
            mapping, missing, ambiguous = _resolve_master_columns(frame.columns)
            found = [c for c in MASTER_COLUMN_ALIASES if c not in missing]
            details.append(f"'{sheet_name}': matched {len(found)}/{len(MASTER_COLUMN_ALIASES)} required columns")
        raise ValueError("No worksheet contains the required optimizer columns.\n" + "\n".join(details))
    if len(candidates) > 1:
        names = ", ".join(repr(c[0]) for c in candidates)
        raise ValueError(f"Multiple worksheets match the optimizer input columns: {names}.")
    return candidates[0]


def load_master_chemistry_excel(uploaded_file):
    """uploaded_file: dict {filename: bytes} (Streamlit: {file.name: file.getvalue()})"""
    if not uploaded_file:
        raise ValueError("No Excel file was uploaded.")
    file_name = next(iter(uploaded_file))
    uploaded_bytes = uploaded_file[file_name]
    sheet_name, raw, column_map = _find_matching_master_sheet(uploaded_bytes)
    rename_map = {actual: canonical for canonical, actual in column_map.items()}
    raw = raw.rename(columns=rename_map).copy()
    for optional_col in ["CV_kcal_kg", "FC_Pct"]:
        if optional_col not in raw.columns:
            raw[optional_col] = 0.0
    raw = raw[MASTER_REQUIRED_COLUMNS].copy()
    raw["Material"] = raw["Material"].astype(str).str.strip()
    raw = raw[raw["Material"].ne("") & raw["Material"].ne("nan")].copy()
    numeric_cols = [c for c in MASTER_REQUIRED_COLUMNS if c not in ["Material", "Group", "Material_Role"]]
    for c in numeric_cols:
        raw[c] = pd.to_numeric(raw[c], errors="coerce").fillna(0.0)
    raw["Group"] = raw["Group"].astype(str).str.strip()
    if (raw["Fines_%"] < 0).any() or (raw["Fines_%"] > 100).any():
        bad = raw.loc[(raw["Fines_%"] < 0) | (raw["Fines_%"] > 100), "Material"].tolist()
        raise ValueError(f"Fines % must be between 0 and 100 for: {bad}")
    raw["Material_Role"] = raw["Material_Role"].astype(str).str.strip()
    raw["Available_Tonnes"] = raw["Available_Stock_t"]
    raw["Tech_Min"] = raw["Tech_Min_kg_t"]; raw["Tech_Max"] = raw["Tech_Max_kg_t"]
    raw["Moisture_Pct"] = raw["Moisture_%"]
    raw["Fe"] = raw["Fe_%"]; raw["SiO2"] = raw["SiO2_%"]; raw["Al2O3"] = raw["Al2O3_%"]
    raw["CaO"] = raw["CaO_%"]; raw["MgO"] = raw["MgO_%"]; raw["LOI"] = raw["LOI_%"]; raw["Fines_Pct"] = raw["Fines_%"]
    if raw["Material"].duplicated().any():
        dupes = sorted(raw.loc[raw["Material"].duplicated(keep=False), "Material"].unique())
        raise ValueError(f"Duplicate Material names in master: {dupes}")
    for mat in raw.loc[raw["Group"].eq("Recycle"), "Material"]:
        fixed = float(raw.loc[raw["Material"].eq(mat), "Tech_Min"].iloc[0])
        raw.loc[raw["Material"].eq(mat), "Tech_Max"] = fixed
    df = raw.set_index("Material")
    keep = ["Group", "Material_Role", "Fe", "SiO2", "Al2O3", "CaO", "MgO", "LOI", "Tech_Min", "Tech_Max",
            "Available_Tonnes", "Price_Rs_t", "Moisture_Pct", "Fines_Pct", "CV_kcal_kg", "FC_Pct"]
    return df[keep]


# ============================================================================
# STREAMLIT-SPECIFIC ADDITIONS (not present in the Colab notebook)
# ============================================================================

# app.py imports this as opt.TARGETS. The Colab notebook only defined this
# dict as a local variable inside its "MAIN EXECUTION" cell.
TARGETS = {
    "Fe_min": 54.0, "SiO2_max": 5.8, "Al2O3_max": 4.5, "Al2O3_SiO2_max": 0.6,
    "Basicity_min": 1.9, "Basicity_max": 2.0, "MgO_min": 2.2, "MgO_max": 2.4,
    "CaO_min": 10.5, "CaO_max": 11.5,
}


def quality_table(achieved, targets, fe_lo=FE_LOWER, fe_hi=FE_UPPER):
    """Achieved-vs-target compliance table used by Recipe & Composition / Scenario Analysis."""
    rows = [
        ("Fe", achieved.get("Fe", 0.0), f"{fe_lo:.1f}-{fe_hi:.1f}", fe_lo <= achieved.get("Fe", 0.0) <= fe_hi),
        ("SiO2", achieved.get("SiO2", 0.0), f"<= {targets['SiO2_max']}", achieved.get("SiO2", 0.0) <= targets["SiO2_max"]),
        ("Al2O3", achieved.get("Al2O3", 0.0), f"<= {targets['Al2O3_max']}", achieved.get("Al2O3", 0.0) <= targets["Al2O3_max"]),
        ("Al2O3/SiO2", achieved.get("Al2O3/SiO2", 0.0), f"<= {targets['Al2O3_SiO2_max']}",
         achieved.get("Al2O3/SiO2", 0.0) <= targets["Al2O3_SiO2_max"]),
        ("Basicity", achieved.get("Basicity", 0.0), f"{targets['Basicity_min']}-{targets['Basicity_max']}",
         targets["Basicity_min"] <= achieved.get("Basicity", 0.0) <= targets["Basicity_max"]),
        ("MgO", achieved.get("MgO", 0.0), f"{targets['MgO_min']}-{targets['MgO_max']}",
         targets["MgO_min"] <= achieved.get("MgO", 0.0) <= targets["MgO_max"]),
        ("CaO", achieved.get("CaO", 0.0), f"{targets['CaO_min']}-{targets['CaO_max']}",
         targets["CaO_min"] <= achieved.get("CaO", 0.0) <= targets["CaO_max"]),
        ("B4", achieved.get("B4", 0.0), "1.8-2.2 (info)", True),
    ]
    return pd.DataFrame([{"KPI": k, "Achieved": v, "Target": t, "Status": "OK" if ok else "OUT OF RANGE"}
                          for k, v, t, ok in rows])


def solve_manual_scenario(df, production_tonnes, targets, baseline_blend, fixed, **kwargs):
    """Manual Burden Control 'practical scenario': pin the user-changed materials
    at their requested kg/t and let the LP re-optimize everything else within
    the normal model constraints (availability, chemistry, Tech Min/Max, coke/
    thermal limits, R/S targets retained where physically available; shortfalls compensated)."""
    scenario_df = df.copy(deep=True)

    # Pandas 2.x/3.x can reject scalar assignments when the imported master
    # contains integer/object/mixed dtypes. Manual Burden Control only needs
    # numeric technical and stock columns, so normalise those columns before
    # applying the user's fixed quantities. This is a data-type safeguard only;
    # it does not change any optimisation rule or constraint.
    for col in ["Tech_Min", "Tech_Max", "Available_Tonnes"]:
        if col not in scenario_df.columns:
            raise ValueError(f"Manual scenario requires column: {col}")
        scenario_df[col] = pd.to_numeric(scenario_df[col], errors="coerce").astype(float)

    if not scenario_df.index.is_unique:
        raise ValueError("Manual Burden Control requires unique Material names in the master data.")

    for m, qty in fixed.items():
        if m not in scenario_df.index:
            continue
        qty = max(0.0, float(qty))
        scenario_df.at[m, "Tech_Min"] = qty
        scenario_df.at[m, "Tech_Max"] = qty
        # Make sure the fixed quantity is never blocked by an inventory cap.
        needed_tonnes = qty / 1000.0 * float(production_tonnes)
        current_stock = float(scenario_df.at[m, "Available_Tonnes"])
        if current_stock < needed_tonnes:
            scenario_df.at[m, "Available_Tonnes"] = needed_tonnes * 1.05 + 1.0
    result = solve_blend_with_compensation(scenario_df, production_tonnes, targets,
                                            baseline_blend=baseline_blend, **kwargs)
    status, blend, cost, achieved, diagnostics, is_fallback = result
    return status, blend, cost, achieved, diagnostics, is_fallback


def what_if_analysis(df, targets, **kwargs):
    """Scenario Analysis 'material shortage' sweep: knock out each material
    one at a time (Available_Tonnes = 0) and re-solve, reporting the cost
    impact versus the current base case."""
    candidates = [m for m in df.index
                  if (m in _eligible_iron_ores(df)
                      or df.loc[m, "Group"] in ["Flux", "IOL_Fines_Mandate", "BF_Returns_Mandate", "Fuel"]
                      or m in ["IOL_Fines", "BF_Returns"])
                  and df.loc[m, "Available_Tonnes"] > 0]
    rows = []
    for mat in candidates:
        scenario_df = df.copy()
        scenario_df.loc[mat, "Available_Tonnes"] = 0.0
        result = solve_blend_with_compensation(scenario_df, 1000, targets, **kwargs)
        status, blend, cost, achieved, diagnostics, is_fallback = result
        if status == "No_Production":
            rows.append({"Missing Material": mat, "Status": "NO PRODUCTION", "Cost ₹/t": np.nan})
        elif blend is not None:
            rows.append({"Missing Material": mat,
                         "Status": "Feasible" if status == "Optimal" and not is_fallback else status,
                         "Cost ₹/t": round(cost, 2) if cost is not None else np.nan})
        else:
            rows.append({"Missing Material": mat, "Status": "INFEASIBLE", "Cost ₹/t": np.nan})
    return pd.DataFrame(rows)
