import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

import optimizer as opt

st.set_page_config(page_title="Sinter Burden Control", page_icon="⚙️", layout="wide", initial_sidebar_state="expanded")

# ----------------------------- CONSTANTS ---------------------------------
TARGETS = opt.TARGETS
CHEM_COLS = ["Fe", "SiO2", "Al2O3", "CaO", "MgO", "LOI", "Moisture_Pct"]
GROUPS = ["Iron_ore", "Flux", "Recycle", "Fuel"]
GROUP_LABEL = {"Iron_ore": "Iron Ore", "Flux": "Flux", "Recycle": "Recycle", "Fuel": "Fuel"}

# ----------------------------- STYLE -------------------------------------
st.markdown("""
<style>
:root { --bg:#071016; --panel:#0d1a21; --panel2:#111f27; --line:#28404d; --text:#edf5fa; --muted:#8ea6b4; --accent:#2f82b3; --good:#25c481; --warn:#f2b94b; --bad:#ff5555; }
html, body, [data-testid="stAppViewContainer"] { background:var(--bg); color:var(--text); }
[data-testid="stHeader"] { background:transparent; }
[data-testid="stSidebar"] { width:220px !important; min-width:220px !important; background:#09131a; border-right:1px solid var(--line); }
[data-testid="stSidebar"] > div:first-child { width:220px !important; }
.block-container { max-width:1450px; padding:1.2rem 1.4rem 2.5rem; }
.small { color:var(--muted); font-size:.72rem; }
.eyebrow { color:#73b5d7; font-size:.62rem; letter-spacing:.16em; font-weight:800; text-transform:uppercase; }
h1 { font-size:2rem !important; margin:.05rem 0 .15rem !important; letter-spacing:.01em; }
h2,h3 { letter-spacing:.01em; }
.panel { background:linear-gradient(180deg,#102029,#0d1a21); border:1px solid var(--line); border-radius:8px; padding:.65rem .7rem; margin:.55rem 0; }
.panel-title { color:#8fd0ef; font-size:.67rem; font-weight:900; letter-spacing:.1em; text-transform:uppercase; margin-bottom:.35rem; }
.notice { background:#12242e; border:1px solid #31566a; border-radius:7px; padding:.6rem .75rem; color:#cce6f3; font-size:.76rem; }
.notice-w { border-color:#795b18; background:#241e0e; }
.notice-r { border-color:#7a2323; background:#241010; }
.hero { background:#0d1a21; border:1px solid var(--line); border-radius:8px; padding:.7rem .85rem; margin:.5rem 0 .7rem; }
.kpi { background:#101d24; border:1px solid var(--line); border-radius:8px; padding:.65rem .7rem; min-height:82px; }
.kpi-label { font-size:.6rem; color:#7da4b7; letter-spacing:.11em; font-weight:900; text-transform:uppercase; }
.kpi-value { font-size:1.2rem; font-weight:900; margin-top:.22rem; }
.kpi-sub { font-size:.62rem; color:#6f8998; margin-top:.12rem; }
.kpi-g { border-left:3px solid var(--good); } .kpi-r { border-left:3px solid var(--bad); } .kpi-a { border-left:3px solid var(--warn); } .kpi-s { border-left:3px solid #4ca8df; }
.nav-title { color:#7397a9; font-size:.58rem; font-weight:900; letter-spacing:.16em; margin:.75rem 0 .3rem; }
.sidebar-brand { font-weight:900; font-size:.86rem; letter-spacing:.03em; }
div.stButton > button { border-radius:6px; border:1px solid #2b4c5c; background:#101f27; color:#edf5fa; font-size:.72rem; min-height:2rem; }
div.stButton > button:hover { border-color:#4e94bb; color:white; }
button[kind="primary"] { background:#2c78a5 !important; border-color:#3e91c1 !important; }
[data-testid="stDataEditor"] { border:1px solid var(--line); border-radius:7px; overflow:hidden; }
[data-testid="stDataEditor"] [role="gridcell"], [data-testid="stDataFrame"] [role="gridcell"] { font-size:11px !important; }
[data-testid="stFileUploader"] { background:#111a21; border-radius:7px; padding:.25rem; }
[data-testid="stMetric"] { background:#101d24; border:1px solid var(--line); border-radius:8px; padding:.45rem; }
.footer { color:#526d7b; font-size:.58rem; text-align:right; margin-top:1rem; }
[data-testid="stTabs"] [data-baseweb="tab-list"] { gap:.25rem; border-bottom:1px solid var(--line); }
[data-testid="stTabs"] [data-baseweb="tab"] { background:#0d1a21; border:1px solid var(--line); border-bottom:none; border-radius:7px 7px 0 0; color:var(--muted); font-size:.74rem; font-weight:800; letter-spacing:.03em; padding:.5rem 1rem; }
[data-testid="stTabs"] [aria-selected="true"] { background:#12242e; color:var(--text); border-color:var(--accent); }
[data-testid="stTabs"] [data-baseweb="tab-panel"] { padding-top:.7rem; }
</style>
""", unsafe_allow_html=True)

# ----------------------------- STATE -------------------------------------
def initial_df():
    df = opt.get_default_chemistry().copy()
    df = opt._ensure_material_role(df)
    df["Material_Role"] = np.where(df["Group"].eq("Iron_ore"), "Primary_Iron_Ore", "Other")
    return df

def _backend_default(name, fallback):
    return float(getattr(opt, name, fallback))

if "master_df" not in st.session_state:
    st.session_state.master_df = initial_df()
    st.session_state.source = "Built-in Master Chemistry"
    st.session_state.production = 1100.0
    st.session_state.available = {m: (float(st.session_state.master_df.loc[m, "Available_Tonnes"]) > 0)
                                   for m in st.session_state.master_df.index}
    st.session_state.result = None
    st.session_state.manual_base = None
    st.session_state.whatif = None
    st.session_state.manual_scenario_result = None
    st.session_state.runs = 0
    st.session_state.changed = False
    st.session_state.changed_source = ""
    st.session_state.nav = "Dashboard"

if "om_cost" not in st.session_state: st.session_state.om_cost = _backend_default("DEFAULT_OM_COST_RS_T", 750.0)
if "coke_cv" not in st.session_state: st.session_state.coke_cv = _backend_default("DEFAULT_COKE_CV_KCAL_KG", 6800.0)
if "coke_fc" not in st.session_state: st.session_state.coke_fc = _backend_default("DEFAULT_COKE_FC_PCT", 71.35)
if "latent_heat" not in st.session_state: st.session_state.latent_heat = _backend_default("DEFAULT_HEAT_LATENT_MOISTURE", 540.0)
if "calcination_heat" not in st.session_state: st.session_state.calcination_heat = _backend_default("DEFAULT_HEAT_CALCINATION_PER_LOI_KG", 420.0)
if "melting_heat" not in st.session_state: st.session_state.melting_heat = _backend_default("DEFAULT_HEAT_MELTING_PER_KG_SINTER", 60.0)
if "loss_fraction" not in st.session_state: st.session_state.loss_fraction = _backend_default("DEFAULT_HEAT_LOSS_FRACTION", 0.12)
if "firing_ratio_max" not in st.session_state: st.session_state.firing_ratio_max = _backend_default("DEFAULT_FIRING_RATIO_MAX", 1.10)
if "coke_min_rate" not in st.session_state: st.session_state.coke_min_rate = _backend_default("DEFAULT_COKE_MIN_KG_T", 55.0)
if "coke_max_rate" not in st.session_state: st.session_state.coke_max_rate = _backend_default("DEFAULT_COKE_MAX_KG_T", 85.0)
if "feo_min" not in st.session_state: st.session_state.feo_min = _backend_default("DEFAULT_FEO_MIN_PCT", 8.5)
if "feo_target" not in st.session_state: st.session_state.feo_target = _backend_default("DEFAULT_FEO_TARGET_PCT", 9.2)
if "feo_max" not in st.session_state: st.session_state.feo_max = _backend_default("DEFAULT_FEO_MAX_PCT", 10.0)
if "manual_coke_override" not in st.session_state: st.session_state.manual_coke_override = False
if "manual_coke_rate" not in st.session_state: st.session_state.manual_coke_rate = 65.0
if "rs_iol_pct" not in st.session_state: st.session_state.rs_iol_pct = _backend_default("IOL_FINES_NOMINAL_PCT", 0.08) * 100
if "rs_bfr_pct" not in st.session_state: st.session_state.rs_bfr_pct = _backend_default("BF_RETURNS_NOMINAL_PCT", 0.17) * 100
if "rs_total_pct" not in st.session_state: st.session_state.rs_total_pct = _backend_default("TOTAL_RS_NOMINAL_PCT", 0.25) * 100
if "fuel_ash_settings" not in st.session_state:
    st.session_state.fuel_ash_settings = getattr(opt, "DEFAULT_FUEL_ASH_SETTINGS", {
        "KSL_COKE": {"Ash_AR_Pct": 14.0, "Ash_SiO2_Pct": 52.5, "Ash_Al2O3_Pct": 25.0, "Ash_CaO_Pct": 5.5, "Ash_MgO_Pct": 3.0, "Ash_Fe2O3_Pct": 10.0},
        "LOCAL_COKE": {"Ash_AR_Pct": 14.0, "Ash_SiO2_Pct": 52.5, "Ash_Al2O3_Pct": 25.0, "Ash_CaO_Pct": 5.5, "Ash_MgO_Pct": 3.0, "Ash_Fe2O3_Pct": 10.0},
    })
if "feed_rate_t_h" not in st.session_state: st.session_state.feed_rate_t_h = 70.0
if "productivity_target" not in st.session_state: st.session_state.productivity_target = 1.0
if "productivity_metrics" not in st.session_state: st.session_state.productivity_metrics = None
if "fines_specific_consumption" not in st.session_state: st.session_state.fines_specific_consumption = {}

# ----------------------------- CHANGE TRACKING ----------------------------
def mark_changed(source):
    """Flip the RERUN REQUIRED indicator and record which page/section caused it.
    Never call this from Manual Burden Control — that page is the sandboxed
    what-if scenario and never dirties the real optimizer baseline."""
    st.session_state.changed = True
    st.session_state.changed_source = source

def status_label():
    if st.session_state.runs == 0:
        return "NOT RUN", "s"
    if st.session_state.changed:
        return f"RERUN REQUIRED — Changed in: {st.session_state.changed_source}", "r"
    return "UP TO DATE", "g"

# ----------------------------- HELPERS ------------------------------------
def active_df():
    df = st.session_state.master_df.copy()
    for m in df.index:
        if not st.session_state.available.get(m, True):
            df.loc[m, "Available_Tonnes"] = 0.0
    return df

def _current_fuel_ash_settings():
    return {k: {kk: float(vv) for kk, vv in v.items()} for k, v in st.session_state.fuel_ash_settings.items()}

def _rs_split_valid():
    return abs((float(st.session_state.rs_iol_pct) + float(st.session_state.rs_bfr_pct)) - float(st.session_state.rs_total_pct)) <= 0.01

def _solver_kwargs():
    return dict(
        iol_nominal=float(st.session_state.rs_iol_pct) / 100.0,
        bf_nominal=float(st.session_state.rs_bfr_pct) / 100.0,
        latent_heat=st.session_state.latent_heat,
        calcination_heat=st.session_state.calcination_heat,
        melting_heat=st.session_state.melting_heat,
        loss_fraction=st.session_state.loss_fraction,
        firing_ratio_max=st.session_state.firing_ratio_max,
        coke_min_rate=st.session_state.coke_min_rate,
        coke_max_rate=st.session_state.coke_max_rate,
        feo_min=st.session_state.feo_min,
        feo_target=st.session_state.feo_target,
        feo_max=st.session_state.feo_max,
        manual_override=st.session_state.manual_coke_override,
        manual_coke_rate=st.session_state.manual_coke_rate,
        fuel_ash_settings=_current_fuel_ash_settings(),
    )

def run_optimizer():
    if not _rs_split_valid():
        raise ValueError(f"IOL Fines {st.session_state.rs_iol_pct:.2f}% + BF Returns {st.session_state.rs_bfr_pct:.2f}% must equal Total R/S {st.session_state.rs_total_pct:.2f}%.")
    df = active_df()
    status, blend, cost, achieved, diagnostics, is_fallback = opt.solve_blend_with_compensation(
        df, float(st.session_state.production), TARGETS, baseline_blend=None, **_solver_kwargs()
    )
    st.session_state.result = {"status": status, "blend": blend, "cost": cost, "achieved": achieved,
                                "diagnostics": diagnostics, "fallback": is_fallback, "df": df.copy()}
    st.session_state.manual_base = blend.copy() if blend else None
    st.session_state.manual_scenario_result = None
    st.session_state.runs += 1
    st.session_state.changed = False
    st.session_state.changed_source = ""
    st.session_state.whatif = None
    if blend:
        total_burden = sum(blend.values())
        st.session_state.fines_specific_consumption = {m: (blend.get(m, 0.0) / total_burden * 100.0 if total_burden else 0.0)
                                                         for m in df.index}
        try:
            st.session_state.productivity_metrics = opt.calculate_productivity_metrics(
                total_burden, st.session_state.feed_rate_t_h, st.session_state.productivity_target)
        except Exception:
            st.session_state.productivity_metrics = None

def quality_ok(a):
    if not a: return False
    return all([
        opt.FE_LOWER <= a["Fe"] <= opt.FE_UPPER, a["SiO2"] <= TARGETS["SiO2_max"], a["Al2O3"] <= TARGETS["Al2O3_max"],
        a["Al2O3/SiO2"] <= TARGETS["Al2O3_SiO2_max"], TARGETS["Basicity_min"] <= a["Basicity"] <= TARGETS["Basicity_max"],
        TARGETS["MgO_min"] <= a["MgO"] <= TARGETS["MgO_max"], TARGETS["CaO_min"] <= a["CaO"] <= TARGETS["CaO_max"],
    ])

def kpi(label, value, sub="", kind="s"):
    return f'<div class="kpi kpi-{kind}"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div><div class="kpi-sub">{sub}</div></div>'

def chemistry_status(value, lower=None, upper=None):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "r"
    if lower is not None and v < float(lower): return "r"
    if upper is not None and v > float(upper): return "r"
    return "g"

def quality_cards(ach):
    items = [
        ("Fe", ach.get("Fe", np.nan), f"{opt.FE_LOWER:.1f}–{opt.FE_UPPER:.1f}", "%", chemistry_status(ach.get("Fe"), opt.FE_LOWER, opt.FE_UPPER)),
        ("SiO₂", ach.get("SiO2", np.nan), f"≤ {TARGETS['SiO2_max']}", "%", chemistry_status(ach.get("SiO2"), None, TARGETS["SiO2_max"])),
        ("Al₂O₃", ach.get("Al2O3", np.nan), f"≤ {TARGETS['Al2O3_max']}", "%", chemistry_status(ach.get("Al2O3"), None, TARGETS["Al2O3_max"])),
        ("Al₂O₃/SiO₂", ach.get("Al2O3/SiO2", np.nan), f"≤ {TARGETS['Al2O3_SiO2_max']}", "", chemistry_status(ach.get("Al2O3/SiO2"), None, TARGETS["Al2O3_SiO2_max"])),
        ("Basicity", ach.get("Basicity", np.nan), f"{TARGETS['Basicity_min']}–{TARGETS['Basicity_max']}", "", chemistry_status(ach.get("Basicity"), TARGETS["Basicity_min"], TARGETS["Basicity_max"])),
        ("MgO", ach.get("MgO", np.nan), f"{TARGETS['MgO_min']}–{TARGETS['MgO_max']}", "%", chemistry_status(ach.get("MgO"), TARGETS["MgO_min"], TARGETS["MgO_max"])),
        ("CaO", ach.get("CaO", np.nan), f"{TARGETS['CaO_min']}–{TARGETS['CaO_max']}", "%", chemistry_status(ach.get("CaO"), TARGETS["CaO_min"], TARGETS["CaO_max"])),
        ("B4", ach.get("B4", np.nan), "1.8–2.2 info", "%", "g"),
    ]
    cols = st.columns(8)
    for c, (lab, val, tgt, unit, status) in zip(cols, items):
        c.markdown(kpi(lab, f"{val:.3f}{unit}", tgt, status), unsafe_allow_html=True)

def page_header(title, subtitle):
    st.markdown('<div class="eyebrow">HOSPET ALLOY STEEL PLANT</div>', unsafe_allow_html=True)
    st.markdown(f"<h1>{title}</h1><div class='small'>{subtitle}</div>", unsafe_allow_html=True)

def aligned_result_table(blend, df, include_total=True):
    mats = [str(m) for m in df.index]
    total = float(sum(float(blend.get(m, 0.0)) for m in mats))
    total_cost = float(sum(float(blend.get(m, 0.0)) * float(df.loc[m, "Price_Rs_t"]) / 1000 for m in mats))
    rows = []
    for m in mats:
        q = float(blend.get(m, 0.0))
        cost = q * float(df.loc[m, "Price_Rs_t"]) / 1000
        rows.append({"Material": m, "Group": GROUP_LABEL.get(df.loc[m, "Group"], df.loc[m, "Group"]),
                     "kg/t": q, "% Burden": q / total * 100 if total else 0.0,
                     "Cost ₹/t": cost, "% Cost": cost / total_cost * 100 if total_cost else 0.0})
    if include_total:
        rows.append({"Material": "TOTAL", "Group": "", "kg/t": total, "% Burden": 100.0 if total else 0.0,
                     "Cost ₹/t": total_cost, "% Cost": 100.0 if total_cost else 0.0})
    return pd.DataFrame(rows)

def result_table(blend, df, include_zero=False):
    rows = []
    if not blend:
        return pd.DataFrame(columns=["Material", "Group", "kg/t", "% Burden", "Cost ₹/t", "% Cost"])
    total = sum(float(v) for v in blend.values())
    total_cost = sum(float(q) * float(df.loc[m, "Price_Rs_t"]) / 1000 for m, q in blend.items())
    for m in df.index:
        q = float(blend.get(m, 0))
        if not include_zero and q <= 1e-8: continue
        cost = q * float(df.loc[m, "Price_Rs_t"]) / 1000
        rows.append({"Material": m, "Group": GROUP_LABEL.get(df.loc[m, "Group"], df.loc[m, "Group"]),
                     "kg/t": q, "% Burden": (q / total * 100 if total else 0), "Cost ₹/t": cost,
                     "% Cost": (cost / total_cost * 100 if total_cost else 0)})
    rows.append({"Material": "TOTAL", "Group": "", "kg/t": total, "% Burden": 100.0, "Cost ₹/t": total_cost, "% Cost": 100.0})
    return pd.DataFrame(rows)

def coke_diagnostic(blend, df):
    if not blend: return None
    try:
        return opt.compute_coke_heat_balance_diagnostic(
            blend, df, 1000, latent_heat=st.session_state.latent_heat, calcination_heat=st.session_state.calcination_heat,
            melting_heat=st.session_state.melting_heat, loss_fraction=st.session_state.loss_fraction,
            feo_min=st.session_state.feo_min, feo_target=st.session_state.feo_target, feo_max=st.session_state.feo_max)
    except Exception:
        return None

# ----------------------------- SETUP GATE ---------------------------------
def setup_missing_items():
    missing = []
    if float(st.session_state.rs_total_pct) <= 0:
        missing.append("Total Return Sinter % (Inputs → Return Sinter)")
    if not _rs_split_valid():
        missing.append("Return Sinter split — IOL Fines % + BF Returns % must equal Total R/S %")
    df = st.session_state.master_df
    for fuel in ["KSL_COKE", "LOCAL_COKE"]:
        if fuel in df.index:
            if not st.session_state.available.get(fuel, False):
                missing.append(f"{fuel.replace('_',' ').title()} — mark available (Inputs → Coke & Fuel)")
            elif float(df.loc[fuel, "Price_Rs_t"]) <= 0:
                missing.append(f"{fuel.replace('_',' ').title()} — enter a price (Inputs → Coke & Fuel)")
    return missing

# ----------------------------- SIDEBAR -------------------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-brand">HOSPET STEELS LIMITED</div><div class="small">Kalyani Steels × Mukand • Hospet</div>', unsafe_allow_html=True)
    st.markdown("---")
    nav_groups = [
        ("WORKSPACE", ["Dashboard", "Inputs"]),
        ("OPERATIONS", ["RM Stock & Materials", "Recipe & Composition", "Manual Burden Control"]),
        ("ANALYSIS", ["Scenario Analysis", "Plant Run Validation", "Productivity", "Wet Specific Consumption"]),
        ("REPORTING", ["Reports"]),
        ("SYSTEM", ["Upload & Settings"]),
    ]
    for head, items in nav_groups:
        st.markdown(f'<div class="nav-title">{head}</div>', unsafe_allow_html=True)
        for item in items:
            if st.button(item, key="nav_" + item, use_container_width=True,
                         type="primary" if st.session_state.nav == item else "secondary"):
                st.session_state.nav = item; st.rerun()
    st.markdown("---")
    label, _ = status_label()
    st.markdown(f'<div class="small"><b>DATA</b><br>{st.session_state.source}<br>{len(st.session_state.master_df)} materials<br><br>'
                f'<b>OPTIMIZER</b><br>{label}<br><br><b>MODEL</b><br>v37.0</div>', unsafe_allow_html=True)

# ----------------------------- DASHBOARD -----------------------------------
def dashboard():
    page_header("SINTER BURDEN CONTROL", "Cost optimization • quality assurance • raw material decision support")
    st.markdown('<div class="hero"><div class="panel-title">DATA CONTROL CENTER</div>'
                 '<div class="small">Upload one Master Chemistry Excel. Once activated, the raw material table below shows immediately with editable availability. Return Sinter, coke and fuel inputs are entered under Inputs before the optimizer can run for the first time.</div></div>',
                 unsafe_allow_html=True)

    up1, up2 = st.columns([1.7, 1])
    with up1:
        f = st.file_uploader("MASTER EXCEL • XLSX", type=["xlsx"], key="dash_master")
        if f is not None:
            try:
                newdf = opt.load_master_chemistry_excel({f.name: f.getvalue()})
                newdf = opt._ensure_material_role(newdf)
                if st.button("ACTIVATE MASTER", type="primary", key="activate_dash"):
                    st.session_state.master_df = newdf
                    st.session_state.source = f.name
                    st.session_state.available = {m: (float(newdf.loc[m, "Available_Tonnes"]) > 0) for m in newdf.index}
                    st.session_state.result = None
                    st.session_state.changed = False
                    st.session_state.runs = 0
                    st.session_state.manual_scenario_result = None
                    st.rerun()
                st.markdown(f'<div class="small">Detected {len(newdf)} materials in the uploaded workbook.</div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(str(e))
    with up2:
        st.markdown(f'<div class="notice"><b>ACTIVE MASTER</b><br>{st.session_state.source}<br>{len(st.session_state.master_df)} materials</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1, 1.6, 1])
    with c1:
        st.session_state.production = st.number_input("Production (t)", min_value=1.0, value=float(st.session_state.production), step=10.0, key="prod")

    missing = setup_missing_items() if st.session_state.runs == 0 else []
    can_run = st.session_state.runs > 0 or len(missing) == 0

    if st.session_state.runs == 0:
        st.markdown('<div class="panel"><div class="panel-title">SETUP CHECKLIST</div>'
                     '<div class="small">Complete these once, under the Inputs page, before the first optimizer run.</div></div>', unsafe_allow_html=True)
        if missing:
            st.markdown('<div class="notice notice-w"><b>Still needed:</b><br>' + "<br>".join(f"• {m}" for m in missing) + '</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="notice"><b>✅ All required inputs are set.</b> You can run the optimizer.</div>', unsafe_allow_html=True)

    with c2:
        if st.button("🚀 RUN OPTIMIZER", type="primary", use_container_width=True, disabled=not can_run):
            try:
                with st.spinner("Optimizing v37.0…"):
                    run_optimizer()
                st.rerun()
            except Exception as e:
                st.error(str(e))
    with c3:
        label, kind = status_label()
        st.markdown(f'<div class="notice {"notice-r" if kind=="r" else ("notice-w" if kind=="a" else "")}" style="text-align:center"><b>RUN #{st.session_state.runs}</b><br>{label}</div>', unsafe_allow_html=True)

    r = st.session_state.result

    # ---------------- KPI ROW 1 — exactly 5 KPIs ----------------
    if r and r["blend"]:
        bd = aligned_result_table(r["blend"], r["df"])
        rm_cost = float(bd.iloc[-1]["Cost ₹/t"])
        total_sinter_cost = rm_cost + float(st.session_state.om_cost)
        ok = quality_ok(r["achieved"])
        pm = st.session_state.productivity_metrics
        if pm and "Target_Achieved" in pm:
            prod_val = "TARGET MET" if pm["Target_Achieved"] else "BELOW TARGET"
            prod_sub = f"{pm['Achievement_Pct']:.1f}% of target"
            prod_kind = "g" if pm["Target_Achieved"] else "r"
        else:
            prod_val = "PENDING"; prod_sub = "Set feed rate on Productivity page"; prod_kind = "s"
        cols = st.columns(5)
        cards = [
            ("RAW MATERIAL COST", f"₹{rm_cost:,.2f}/t", "Optimised sinter RM cost", "s"),
            ("O&M COST", f"₹{st.session_state.om_cost:,.2f}/t", "Set under Inputs", "a"),
            ("TOTAL SINTER COST", f"₹{total_sinter_cost:,.2f}/t", "RM + O&M", "s"),
            ("QUALITY", "PASS" if ok else "REVIEW", "Overall chemistry achievement", "g" if ok else "r"),
            ("PRODUCTIVITY TARGET REACHED", prod_val, prod_sub, prod_kind),
        ]
        for c, (l, v, s, k) in zip(cols, cards):
            c.markdown(kpi(l, v, s, k), unsafe_allow_html=True)
    else:
        cols = st.columns(5)
        cards = [
            ("RAW MATERIAL COST", "—", "Run optimizer", "s"),
            ("O&M COST", f"₹{st.session_state.om_cost:,.2f}/t", "Set under Inputs", "a"),
            ("TOTAL SINTER COST", "—", "Run optimizer", "s"),
            ("QUALITY", "READY", "Awaiting run", "s"),
            ("PRODUCTIVITY TARGET REACHED", "PENDING", "Run optimizer first", "s"),
        ]
        for c, (l, v, s, k) in zip(cols, cards):
            c.markdown(kpi(l, v, s, k), unsafe_allow_html=True)

    # ---------------- KPI ROW 2 — CHEMISTRY ACHIEVEMENT ----------------
    st.markdown('<div class="panel"><div class="panel-title">CHEMISTRY CONSTRAINTS / ACHIEVED</div>'
                 '<div class="small">Green = within the defined target range. Red = outside the target range.</div></div>', unsafe_allow_html=True)
    if r and r["achieved"]:
        quality_cards(r["achieved"])
    else:
        st.info("Chemistry achievement will appear after optimization.")

    # ---------------- DRY BURDEN & COST TABLE ----------------
    st.markdown('<div class="panel"><div class="panel-title">DRY BURDEN & COST COMPOSITION</div>'
                 '<div class="small">RM Cost is editable and feeds the optimizer — changing it flags RERUN REQUIRED.</div></div>', unsafe_allow_html=True)
    if r and r["blend"]:
        df = r["df"]
        bd = aligned_result_table(r["blend"], df, include_total=True)
        view = bd.copy()
        view.insert(1, "RM Cost ₹/t", view["Material"].map(lambda m: float(df.loc[m, "Price_Rs_t"]) if m in df.index else np.nan))
        view = view.rename(columns={"kg/t": "kg/t", "% Burden": "% of Total Burden", "Cost ₹/t": "Cost/t", "% Cost": "% Cost"})
        view = view[["Material", "RM Cost ₹/t", "kg/t", "% of Total Burden", "Cost/t", "% Cost"]]
        edited = st.data_editor(
            view, hide_index=True, use_container_width=True, height=max(320, 34 * len(view) + 45),
            key="dashboard_dry_burden_editor",
            disabled=["Material", "kg/t", "% of Total Burden", "Cost/t", "% Cost"],
            column_config={
                "RM Cost ₹/t": st.column_config.NumberColumn("RM Cost ₹/t", min_value=0, step=1, format="%.0f"),
                "kg/t": st.column_config.NumberColumn("kg/t", format="%.2f"),
                "% of Total Burden": st.column_config.NumberColumn("% of Total Burden", format="%.2f"),
                "Cost/t": st.column_config.NumberColumn("Cost/t", format="₹ %.2f"),
                "% Cost": st.column_config.NumberColumn("% Cost", format="%.2f"),
            },
        )
        if not edited["RM Cost ₹/t"].equals(view["RM Cost ₹/t"]):
            for _, row in edited.iterrows():
                m = row["Material"]
                if m in st.session_state.master_df.index and pd.notna(row["RM Cost ₹/t"]):
                    st.session_state.master_df.loc[m, "Price_Rs_t"] = float(row["RM Cost ₹/t"])
            mark_changed("Dashboard → Dry Burden & Cost (RM Cost)")
            st.rerun()
    else:
        st.info("Run the optimizer to populate the dry burden and cost table.")

    # ---------------- RAW MATERIAL MASTER TABLE (always visible) ----------------
    st.markdown('<div class="panel"><div class="panel-title">RAW MATERIAL INPUTS — FULL WIDTH</div>'
                 '<div class="small">Chemistry, moisture, price, stock, Tech Min/Max and availability. Shown as soon as a master is activated; editable at any time.</div></div>', unsafe_allow_html=True)
    material_count = len(st.session_state.master_df.index)
    full_table_height = max(220, 34 * material_count + 52)
    inp = st.session_state.master_df.reset_index()[["Material", "Group"] + CHEM_COLS + ["Price_Rs_t", "Available_Tonnes", "Tech_Min", "Tech_Max"]].copy()
    inp.rename(columns={"SiO2": "SiO₂", "Al2O3": "Al₂O₃", "Moisture_Pct": "Moisture %", "Price_Rs_t": "Price ₹/t",
                        "Available_Tonnes": "RM Stock t", "Tech_Min": "Tech Min", "Tech_Max": "Tech Max"}, inplace=True)
    inp["Availability"] = [st.session_state.available.get(m, False) for m in inp.Material]
    ed = st.data_editor(
        inp, key="merged_master_editor", hide_index=True, use_container_width=True, height=full_table_height,
        disabled=["Group"],
        column_config={
            "Material": st.column_config.TextColumn("Raw Material", width="medium"),
            "Group": st.column_config.TextColumn("Group", width="small"),
            "Fe": st.column_config.NumberColumn("Fe %", width="small", format="%.2f"),
            "SiO₂": st.column_config.NumberColumn("SiO₂ %", width="small", format="%.2f"),
            "Al₂O₃": st.column_config.NumberColumn("Al₂O₃ %", width="small", format="%.2f"),
            "CaO": st.column_config.NumberColumn("CaO %", width="small", format="%.2f"),
            "MgO": st.column_config.NumberColumn("MgO %", width="small", format="%.2f"),
            "LOI": st.column_config.NumberColumn("LOI %", width="small", format="%.2f"),
            "Moisture %": st.column_config.NumberColumn("Moisture %", width="small", format="%.2f"),
            "Price ₹/t": st.column_config.NumberColumn("Price ₹/t", width="small", min_value=0, step=1, format="%.0f"),
            "RM Stock t": st.column_config.NumberColumn("RM Stock t", width="small", min_value=0, step=100, format="%.0f"),
            "Tech Min": st.column_config.NumberColumn("Tech Min", width="small", format="%.0f"),
            "Tech Max": st.column_config.NumberColumn("Tech Max", width="small", min_value=0, step=1, format="%.0f"),
            "Availability": st.column_config.CheckboxColumn("Use / Available"),
        }
    )
    if not ed.equals(inp):
        for _, row in ed.iterrows():
            m = row["Material"]
            for src, dst in [("Fe", "Fe"), ("SiO₂", "SiO2"), ("Al₂O₃", "Al2O3"), ("CaO", "CaO"), ("MgO", "MgO"),
                              ("LOI", "LOI"), ("Moisture %", "Moisture_Pct"), ("Tech Min", "Tech_Min"), ("Tech Max", "Tech_Max"),
                              ("Price ₹/t", "Price_Rs_t"), ("RM Stock t", "Available_Tonnes")]:
                st.session_state.master_df.loc[m, dst] = float(row[src])
            st.session_state.available[m] = bool(row["Availability"])
        mark_changed("Dashboard → Raw Material Inputs")
        st.rerun()

# ----------------------------- INPUTS PAGE ----------------------------------
def inputs_page():
    page_header("Inputs", "Return Sinter split, coke & fuel, fuel-ash chemistry, and O&M / thermal parameters — all feed the optimizer.")
    tabs = st.tabs(["RETURN SINTER", "COKE & FUEL", "FUEL ASH CHEMISTRY", "O&M & THERMAL"])

    with tabs[0]:
        st.markdown('<div class="small">All percentages are of the total burden. Coke is included in the burden denominator.</div>', unsafe_allow_html=True)
        rs1, rs2, rs3, rs4 = st.columns(4)
        with rs1:
            new_total = st.number_input("Total Return Sinter (%)", min_value=0.0, max_value=100.0, value=float(st.session_state.rs_total_pct), step=0.5, key="rs_total_input")
        with rs2:
            new_iol = st.number_input("IOL Fines (%)", min_value=0.0, max_value=100.0, value=float(st.session_state.rs_iol_pct), step=0.5, key="rs_iol_input")
        with rs3:
            new_bfr = st.number_input("BF Returns (%)", min_value=0.0, max_value=100.0, value=float(st.session_state.rs_bfr_pct), step=0.5, key="rs_bfr_input")
        if (new_total, new_iol, new_bfr) != (st.session_state.rs_total_pct, st.session_state.rs_iol_pct, st.session_state.rs_bfr_pct):
            st.session_state.rs_total_pct, st.session_state.rs_iol_pct, st.session_state.rs_bfr_pct = new_total, new_iol, new_bfr
            mark_changed("Inputs → Return Sinter")
        with rs4:
            rs_ok = _rs_split_valid()
            st.markdown(kpi("R/S SPLIT", "VALID" if rs_ok else "CHECK", f"IOL + BF = {st.session_state.rs_total_pct:.2f}%", "g" if rs_ok else "r"), unsafe_allow_html=True)

    with tabs[1]:
        st.markdown('<div class="small">KSL Coke and LOCAL Coke are the two fuel sources. Mark each available and set its price/stock; the optimizer balances the split for you.</div>', unsafe_allow_html=True)
        df = st.session_state.master_df
        fuels = [m for m in df.index if df.loc[m, "Group"] == "Fuel"]
        for fuel in fuels:
            fc1, fc2, fc3, fc4 = st.columns(4)
            avail = fc1.checkbox(f"{fuel} available", value=st.session_state.available.get(fuel, False), key=f"avail_{fuel}")
            price = fc2.number_input(f"{fuel} Price ₹/t", min_value=0.0, value=float(df.loc[fuel, "Price_Rs_t"]), step=50.0, key=f"price_{fuel}")
            stock = fc3.number_input(f"{fuel} Stock t", min_value=0.0, value=float(df.loc[fuel, "Available_Tonnes"]), step=100.0, key=f"stock_{fuel}")
            techmax = fc4.number_input(f"{fuel} Tech Max kg/t", min_value=0.0, value=float(df.loc[fuel, "Tech_Max"]), step=1.0, key=f"techmax_{fuel}")
            if (avail != st.session_state.available.get(fuel, False) or price != float(df.loc[fuel, "Price_Rs_t"])
                    or stock != float(df.loc[fuel, "Available_Tonnes"]) or techmax != float(df.loc[fuel, "Tech_Max"])):
                st.session_state.available[fuel] = avail
                st.session_state.master_df.loc[fuel, "Price_Rs_t"] = price
                st.session_state.master_df.loc[fuel, "Available_Tonnes"] = stock
                st.session_state.master_df.loc[fuel, "Tech_Max"] = techmax
                mark_changed("Inputs → Coke & Fuel")

        st.markdown('<div class="panel"><div class="panel-title">COKE PRACTICAL LIMITS</div></div>', unsafe_allow_html=True)
        q1, q2, q3, q4 = st.columns(4)
        new_cv = q1.number_input("Coke CV (kcal/kg)", min_value=1000.0, value=float(st.session_state.coke_cv), step=50.0)
        new_fc = q2.number_input("Fixed Carbon (%)", min_value=1.0, max_value=100.0, value=float(st.session_state.coke_fc), step=.1)
        new_min = q3.number_input("Coke Min (kg/t)", min_value=0.0, value=float(st.session_state.coke_min_rate), step=1.0)
        new_max = q4.number_input("Coke Max (kg/t)", min_value=0.0, value=float(st.session_state.coke_max_rate), step=1.0)
        if (new_cv, new_fc, new_min, new_max) != (st.session_state.coke_cv, st.session_state.coke_fc, st.session_state.coke_min_rate, st.session_state.coke_max_rate):
            st.session_state.coke_cv, st.session_state.coke_fc = new_cv, new_fc
            st.session_state.coke_min_rate, st.session_state.coke_max_rate = new_min, new_max
            mark_changed("Inputs → Coke & Fuel")
        new_override = st.checkbox("Manual Coke Override (fix total fuel rate)", value=bool(st.session_state.manual_coke_override))
        if new_override != st.session_state.manual_coke_override:
            st.session_state.manual_coke_override = new_override
            mark_changed("Inputs → Coke & Fuel")
        if st.session_state.manual_coke_override:
            new_rate = st.number_input("Fixed Coke Rate (kg/t)", min_value=float(st.session_state.coke_min_rate),
                                        max_value=float(st.session_state.coke_max_rate), value=float(st.session_state.manual_coke_rate), step=.5)
            if new_rate != st.session_state.manual_coke_rate:
                st.session_state.manual_coke_rate = new_rate
                mark_changed("Inputs → Coke & Fuel")

    with tabs[2]:
        st.markdown('<div class="small">Ash is entered on as-received basis. The backend converts retained fuel ash to the dry-burden calculation.</div>', unsafe_allow_html=True)
        ash_rows = []
        for fuel, vals in st.session_state.fuel_ash_settings.items():
            ash_rows.append({"Fuel": fuel, "Ash % AR": float(vals.get("Ash_AR_Pct", 0)), "SiO₂ % in ash": float(vals.get("Ash_SiO2_Pct", 0)),
                             "Al₂O₃ % in ash": float(vals.get("Ash_Al2O3_Pct", 0)), "CaO % in ash": float(vals.get("Ash_CaO_Pct", 0)),
                             "MgO % in ash": float(vals.get("Ash_MgO_Pct", 0)), "Fe₂O₃ % in ash": float(vals.get("Ash_Fe2O3_Pct", 0))})
        ash_df = pd.DataFrame(ash_rows)
        ash_ed = st.data_editor(ash_df, hide_index=True, use_container_width=True, key="inputs_fuel_ash", column_config={
            "Fuel": st.column_config.TextColumn("Fuel", disabled=True),
            "Ash % AR": st.column_config.NumberColumn("Ash % AR", min_value=0.0, max_value=100.0, step=0.1, format="%.2f"),
            "SiO₂ % in ash": st.column_config.NumberColumn("SiO₂ % in ash", min_value=0.0, max_value=100.0, step=0.1, format="%.2f"),
            "Al₂O₃ % in ash": st.column_config.NumberColumn("Al₂O₃ % in ash", min_value=0.0, max_value=100.0, step=0.1, format="%.2f"),
            "CaO % in ash": st.column_config.NumberColumn("CaO % in ash", min_value=0.0, max_value=100.0, step=0.1, format="%.2f"),
            "MgO % in ash": st.column_config.NumberColumn("MgO % in ash", min_value=0.0, max_value=100.0, step=0.1, format="%.2f"),
            "Fe₂O₃ % in ash": st.column_config.NumberColumn("Fe₂O₃ % in ash", min_value=0.0, max_value=100.0, step=0.1, format="%.2f"),
        })
        if not ash_ed.equals(ash_df):
            for _, row in ash_ed.iterrows():
                fuel = str(row["Fuel"])
                st.session_state.fuel_ash_settings[fuel] = {
                    "Ash_AR_Pct": float(row["Ash % AR"]), "Ash_SiO2_Pct": float(row["SiO₂ % in ash"]),
                    "Ash_Al2O3_Pct": float(row["Al₂O₃ % in ash"]), "Ash_CaO_Pct": float(row["CaO % in ash"]),
                    "Ash_MgO_Pct": float(row["MgO % in ash"]), "Ash_Fe2O3_Pct": float(row["Fe₂O₃ % in ash"])}
            mark_changed("Inputs → Fuel Ash Chemistry")

    with tabs[3]:
        c1, c2 = st.columns([1, 2])
        with c1:
            new_om = st.number_input("O&M Cost ₹/t", min_value=0.0, value=float(st.session_state.om_cost), step=50.0, key="om_cost_input")
            if new_om != st.session_state.om_cost:
                st.session_state.om_cost = new_om  # O&M is post-hoc cost, not an LP input — no rerun needed
        with c2:
            st.caption("O&M is added to total sinter cost; it does not change the raw-material optimizer objective, so it updates immediately without requiring a rerun.")
        st.markdown('<div class="panel"><div class="panel-title">THERMAL PARAMETERS</div><div class="small">These feed the heat-balance / FeO model. Changing them requires a rerun.</div></div>', unsafe_allow_html=True)
        q5, q6, q7 = st.columns(3)
        new_latent = q5.number_input("Latent Heat", value=float(st.session_state.latent_heat), step=10.0)
        new_calc = q6.number_input("Calcination Heat", value=float(st.session_state.calcination_heat), step=10.0)
        new_melt = q7.number_input("Melting Heat", value=float(st.session_state.melting_heat), step=5.0)
        q8, q9 = st.columns(2)
        new_loss = q8.number_input("Heat Loss Fraction (0-1)", min_value=0.0, max_value=0.9, value=float(st.session_state.loss_fraction), step=0.01)
        new_firing = q9.number_input("Firing Ratio Max", value=float(st.session_state.firing_ratio_max), step=0.01)
        q10, q11, q12 = st.columns(3)
        new_feo_min = q10.number_input("FeO Min (%)", value=float(st.session_state.feo_min), step=.1)
        new_feo_target = q11.number_input("FeO Target (%)", value=float(st.session_state.feo_target), step=.1)
        new_feo_max = q12.number_input("FeO Max (%)", value=float(st.session_state.feo_max), step=.1)
        thermal_vals = (new_latent, new_calc, new_melt, new_loss, new_firing, new_feo_min, new_feo_target, new_feo_max)
        thermal_old = (st.session_state.latent_heat, st.session_state.calcination_heat, st.session_state.melting_heat,
                       st.session_state.loss_fraction, st.session_state.firing_ratio_max, st.session_state.feo_min,
                       st.session_state.feo_target, st.session_state.feo_max)
        if thermal_vals != thermal_old:
            (st.session_state.latent_heat, st.session_state.calcination_heat, st.session_state.melting_heat,
             st.session_state.loss_fraction, st.session_state.firing_ratio_max, st.session_state.feo_min,
             st.session_state.feo_target, st.session_state.feo_max) = thermal_vals
            mark_changed("Inputs → O&M & Thermal")

        r = st.session_state.result
        if r and r.get("blend"):
            diag = coke_diagnostic(r["blend"], r["df"])
            if diag:
                st.markdown('<div class="panel"><div class="panel-title">LIVE DIAGNOSTICS</div></div>', unsafe_allow_html=True)
                x, y, z = st.columns(3)
                x.metric("Predicted FeO", f"{diag.get('FeO_Estimate_Pct', np.nan):.2f}%")
                y.metric("Thermal Surplus", f"{diag.get('Thermal_Surplus_kcal', np.nan):,.0f}")
                z.metric("Firing Ratio", f"{diag.get('Firing_Ratio', np.nan):.3f}")
                st.caption(str(diag.get("Controller_Suggestion", "")))

# ----------------------------- RM STOCK & MATERIALS -------------------------
def rm_stock_materials():
    page_header("RM Stock & Materials", "Daily material availability, price, stock and technical maximums — all from the same master workbook.")
    st.markdown('<div class="notice">Availability, price, RM stock and Tech Max are editable here. Chemistry is maintained on the Dashboard master table.</div>', unsafe_allow_html=True)
    inp = st.session_state.master_df.reset_index()[["Material", "Group", "Price_Rs_t", "Available_Tonnes", "Tech_Max"]].copy()
    inp.rename(columns={"Price_Rs_t": "Price ₹/t", "Available_Tonnes": "RM Stock t", "Tech_Max": "Tech Max t/d"}, inplace=True)
    inp["Availability / Include"] = [st.session_state.available.get(m, True) for m in inp.Material]
    ed = st.data_editor(inp, key="rm_editor", hide_index=True, use_container_width=True,
                         height=max(250, 38 * len(inp) + 45), disabled=["Material", "Group"])
    if not ed.equals(inp):
        for _, row in ed.iterrows():
            m = row.Material
            st.session_state.master_df.loc[m, "Price_Rs_t"] = float(row["Price ₹/t"])
            st.session_state.master_df.loc[m, "Available_Tonnes"] = float(row["RM Stock t"])
            st.session_state.master_df.loc[m, "Tech_Max"] = float(row["Tech Max t/d"])
            st.session_state.available[m] = bool(row["Availability / Include"])
        mark_changed("RM Stock & Materials")
        st.rerun()

# ----------------------------- RECIPE & COMPOSITION --------------------------
def composition_content(r, kind):
    df = r["df"]; blend = r["blend"]
    total = sum(float(v) for v in blend.values())
    cost = sum(float(blend[m]) * float(df.loc[m, "Price_Rs_t"]) / 1000 for m in blend)
    vals = ({GROUP_LABEL[g]: sum(float(blend[m]) for m in blend if df.loc[m, "Group"] == g) for g in GROUPS} if kind == "burden"
            else {GROUP_LABEL[g]: sum(float(blend[m]) * float(df.loc[m, "Price_Rs_t"]) / 1000 for m in blend if df.loc[m, "Group"] == g) for g in GROUPS})
    center = total if kind == "burden" else cost
    rows = [{"Group": g, "Value": v, ("% of Burden" if kind == "burden" else "% of Cost"): (v / center * 100 if center else 0)} for g, v in vals.items()]
    st.markdown('<div class="panel"><div class="panel-title">GROUP SUMMARY</div></div>', unsafe_allow_html=True)
    st.table(pd.DataFrame(rows).round(3))
    st.markdown('<div class="panel"><div class="panel-title">MATERIAL BREAKDOWN</div></div>', unsafe_allow_html=True)
    st.table(result_table(blend, df).round(3))

def recipe_content(r):
    bd = aligned_result_table(r["blend"], r["df"]); ach = r["achieved"]
    cost = float(bd.iloc[-1]["Cost ₹/t"]); total = float(bd.iloc[-1]["kg/t"])
    c = st.columns(4)
    for col, (l, v, s, k) in zip(c, [("TOTAL COST", f"₹{cost:,.2f}/t", "Optimized", "s"),
                                       ("BURDEN", f"{total:,.2f} kg/t", "Optimized", "g"),
                                       ("Fe", f"{ach['Fe']:.3f}%", f"{opt.FE_LOWER:.1f}–{opt.FE_UPPER:.1f}", "a"),
                                       ("QUALITY", "PASS" if quality_ok(ach) else "REVIEW", "All mandatory targets", "g" if quality_ok(ach) else "r")]):
        col.markdown(kpi(l, v, s, k), unsafe_allow_html=True)
    st.markdown('<div class="panel"><div class="panel-title">QUALITY</div></div>', unsafe_allow_html=True)
    quality_cards(ach)
    st.markdown('<div class="panel"><div class="panel-title">RECIPE</div></div>', unsafe_allow_html=True)
    st.table(bd.round(3))

def recipe_composition():
    page_header("Recipe & Composition", "Optimized recipe and burden/cost composition — one page, one source of truth.")
    r = st.session_state.result
    if not r or not r["blend"]:
        st.info("Run the optimizer first."); return
    tabs = st.tabs(["RECIPE", "BURDEN COMPOSITION", "COST COMPOSITION"])
    with tabs[0]: recipe_content(r)
    with tabs[1]: composition_content(r, "burden")
    with tabs[2]: composition_content(r, "cost")

# ----------------------------- MANUAL BURDEN CONTROL --------------------------
def manual():
    page_header("Manual Burden Control", "Practical scenario analysis — the optimized recipe remains the frozen theoretical baseline. Changes here never require the main optimizer to be rerun.")
    r = st.session_state.result
    if not r or not r.get("blend"):
        st.info("Run the optimizer first."); return

    df = r["df"]
    base = dict(st.session_state.get("manual_base") or r["blend"])
    st.session_state.manual_base = base

    st.markdown('<div class="notice"><b>THEORETICAL BASELINE → PRACTICAL SCENARIO</b><br>Change one or more raw materials. The optimizer re-optimizes everything else while preserving availability, chemistry, Tech Min/Max, coke/thermal limits and the IOL/BF mandates.</div>', unsafe_allow_html=True)

    mode = st.radio("Adjustment mode", ["kg/t", "%"], horizontal=True, key="manual_mode")
    rows = []
    for m in df.index:
        b = float(base.get(m, 0.0))
        if b <= 1e-9 and float(r["blend"].get(m, 0.0)) <= 1e-9: continue
        total = sum(float(v) for v in base.values())
        opct = b / total * 100 if total else 0.0
        locked = str(df.loc[m, "Group"]) in {"IOL_Fines_Mandate", "BF_Returns_Mandate"}
        rows.append({"Raw Material": m, "Optimized kg/t": b, "Optimized %": opct, "Change": 0.0, "Locked": locked})
    base_table = pd.DataFrame(rows)
    ed = st.data_editor(
        base_table, hide_index=True, use_container_width=True, key="manual_scenario_editor",
        disabled=["Raw Material", "Optimized kg/t", "Optimized %", "Locked"],
        column_config={
            "Optimized kg/t": st.column_config.NumberColumn("Optimized kg/t", format="%.2f"),
            "Optimized %": st.column_config.NumberColumn("Optimized %", format="%.2f%%"),
            "Change": st.column_config.NumberColumn("Change kg/t" if mode == "kg/t" else "Change %", step=0.5, format="%+.2f" if mode == "kg/t" else "%+.2f%%"),
            "Locked": st.column_config.CheckboxColumn("Locked"),
        }, height=max(180, 38 * len(base_table) + 45))

    # Only materials that the user actually changed are fixed in the practical
    # scenario. Unchanged materials must remain free so the optimiser can
    # re-optimise them around the practical constraint. Previously every
    # unlocked material was being added to `fixed`, which effectively froze the
    # entire baseline recipe and could return no practical result.
    fixed = {}
    for _, row in ed.iterrows():
        m = row["Raw Material"]
        if bool(row["Locked"]):
            continue
        b = float(row["Optimized kg/t"]); ch = float(row["Change"] or 0.0)
        if abs(ch) <= 1e-12:
            continue
        q = b + ch if mode == "kg/t" else b * (1 + ch / 100.0)
        fixed[m] = max(0.0, q)

    if st.button("🔄 RECALCULATE PRACTICAL SCENARIO", type="primary", use_container_width=True, key="manual_reopt"):
        with st.spinner("Re-optimizing around the practical constraints…"):
            status, blend, cost, achieved, diagnostics, is_fallback = opt.solve_manual_scenario(
                df, float(st.session_state.production), TARGETS, base, fixed, **_solver_kwargs())
            st.session_state.manual_scenario_result = {"status": status, "blend": blend, "cost": cost,
                                                          "achieved": achieved, "diagnostics": diagnostics, "df": df.copy()}
        st.rerun()

    practical = st.session_state.get("manual_scenario_result")
    if practical and practical.get("blend"):
        pbd = aligned_result_table(practical["blend"], df, include_total=True)
        base_cost = sum(float(base[m]) * float(df.loc[m, "Price_Rs_t"]) / 1000 for m in base if m in df.index)
        practical_cost = float(pbd.iloc[-1]["Cost ₹/t"])
        base_total = sum(float(v) for v in base.values()); practical_total = sum(float(v) for v in practical["blend"].values())
        a, b, c, d = st.columns(4)
        a.markdown(kpi("BASELINE COST", f"₹{base_cost:,.2f}/t", "Theoretical", "s"), unsafe_allow_html=True)
        b.markdown(kpi("PRACTICAL COST", f"₹{practical_cost:,.2f}/t", f"Δ ₹{practical_cost-base_cost:+,.2f}/t", "a"), unsafe_allow_html=True)
        c.markdown(kpi("BASELINE BURDEN", f"{base_total:,.2f} kg/t", "Theoretical", "g"), unsafe_allow_html=True)
        d.markdown(kpi("PRACTICAL BURDEN", f"{practical_total:,.2f} kg/t", f"Δ {practical_total-base_total:+,.2f}", "g"), unsafe_allow_html=True)

        compare = []
        for m in df.index:
            ov = float(base.get(m, 0)); pv = float(practical["blend"].get(m, 0))
            if ov == 0 and pv == 0: continue
            compare.append({"Raw Material": m, "Optimized kg/t": ov, "Optimized %": ov/base_total*100 if base_total else 0,
                            "Practical kg/t": pv, "Practical %": pv/practical_total*100 if practical_total else 0,
                            "Change kg/t": pv-ov, "Change %": (pv-ov)/ov*100 if ov else np.nan})
        comp = pd.DataFrame(compare)
        comp.loc[len(comp)] = {"Raw Material": "TOTAL", "Optimized kg/t": base_total, "Optimized %": 100.0,
                               "Practical kg/t": practical_total, "Practical %": 100.0,
                               "Change kg/t": practical_total-base_total, "Change %": (practical_total-base_total)/base_total*100 if base_total else 0}
        st.subheader("Theoretical vs Practical")
        st.table(comp.round(3))
        ach = practical["achieved"]
        st.subheader("Practical chemistry / compliance")
        quality_cards(ach)
        st.table(opt.quality_table(ach, TARGETS).round(3))
    else:
        st.info("Enter a change and press RE-CALCULATE PRACTICAL SCENARIO. The original optimized recipe remains unchanged.")

# ----------------------------- SCENARIO ANALYSIS ------------------------------
def whatif_content():
    r = st.session_state.result
    if not r or not r.get("blend"):
        st.info("Run the optimizer first."); return
    if st.button("▶ RUN MATERIAL SHORTAGE SCENARIOS", type="primary", key="run_whatif"):
        with st.spinner("Evaluating scenarios…"):
            base_cost = float(r.get("cost") or 0)
            scenarios = opt.what_if_analysis(active_df(), TARGETS, **_solver_kwargs())
            if "Cost ₹/t" in scenarios.columns:
                scenarios["Cost Impact ₹/t"] = scenarios["Cost ₹/t"].apply(lambda x: round(float(x) - base_cost, 2) if pd.notna(x) else np.nan)
            st.session_state.whatif = scenarios
    if st.session_state.whatif is not None:
        st.table(st.session_state.whatif.round(3))
    else:
        st.info("Run the scenario analysis.")

def bottleneck_content():
    r = st.session_state.result
    if not r or not r["achieved"]:
        st.info("Run optimizer first."); return
    st.table(opt.quality_table(r["achieved"], TARGETS).round(3))

def scenario_analysis():
    page_header("Scenario Analysis", "Material shortage stress-testing and quality-constraint pressure, side by side.")
    tabs = st.tabs(["MATERIAL SHORTAGE", "CONSTRAINT PRESSURE"])
    with tabs[0]:
        st.caption("Test one-at-a-time material unavailability against the current model.")
        whatif_content()
    with tabs[1]:
        st.caption("Identify the constraints closest to their limits.")
        bottleneck_content()

# ----------------------------- PLANT RUN VALIDATION ----------------------------
def plant_run_validation():
    page_header("Plant Run Validation", "Validate the same backend calculation engine against a fixed actual plant recipe and plant laboratory results.")
    df = active_df().copy()
    if df.empty:
        st.warning("Load a valid master chemistry file first."); return
    st.markdown('<div class="notice">The actual plant recipe is FIXED. The optimizer is NOT allowed to change it. Moisture is excluded from sinter chemistry; actual dry burden and actual finished sinter are retained exactly as entered.</div>', unsafe_allow_html=True)
    mats = list(df.index)
    st.markdown('<div class="panel"><div class="panel-title">STEP 1 — ACTUAL PLANT RECIPE</div></div>', unsafe_allow_html=True)
    default_recipe = pd.DataFrame({"Material": mats, "Actual kg/t sinter": [
        float(st.session_state.result["blend"].get(m, 0.0)) if st.session_state.result and st.session_state.result.get("blend") else 0.0 for m in mats]})
    recipe = st.data_editor(default_recipe, key="validation_recipe", hide_index=True, use_container_width=True,
                             column_config={"Material": st.column_config.TextColumn("Material", disabled=True),
                                           "Actual kg/t sinter": st.column_config.NumberColumn("Actual kg/t sinter", min_value=0.0, format="%.2f")})
    finished = st.number_input("Finished sinter produced (kg)", min_value=0.01, value=1000.0, step=1.0, key="validation_finished_sinter")
    st.markdown('<div class="panel"><div class="panel-title">STEP 2 — ACTUAL PLANT-RUN MATERIAL CHEMISTRY</div><div class="small">Edit the chemistry for this specific plant run.</div></div>', unsafe_allow_html=True)
    chem_view = df[["Fe", "SiO2", "Al2O3", "CaO", "MgO", "LOI", "Moisture_Pct"]].reset_index().rename(columns={"SiO2": "SiO₂", "Al2O3": "Al₂O₃", "Moisture_Pct": "Moisture %"})
    chem_edit = st.data_editor(chem_view, key="validation_chemistry", hide_index=True, use_container_width=True,
                                column_config={c: st.column_config.NumberColumn(c, min_value=0.0, format="%.2f") for c in ["Fe", "SiO₂", "Al₂O₃", "CaO", "MgO", "LOI", "Moisture %"]})
    st.markdown('<div class="panel"><div class="panel-title">STEP 3 — ACTUAL PLANT LAB RESULT</div></div>', unsafe_allow_html=True)
    lab_cols = st.columns(6); lab = {}
    for col, key in zip(lab_cols, ["Fe", "SiO2", "Al2O3", "CaO", "MgO", "Basicity"]):
        lab[key] = col.number_input(key, min_value=0.0, value=54.0 if key == "Fe" else 0.0, step=0.01, key="lab_" + key)
    if st.button("🔬 RUN PLANT VALIDATION", type="primary", use_container_width=True):
        try:
            actual_blend = {str(row["Material"]): float(row["Actual kg/t sinter"]) for _, row in recipe.iterrows() if float(row["Actual kg/t sinter"]) > 0}
            if not actual_blend: raise ValueError("Enter the actual plant recipe.")
            if any(m not in df.index for m in actual_blend): raise ValueError("Plant recipe contains a material missing from the active master.")
            run_df = df.copy()
            cv = chem_edit.copy().set_index("Material").rename(columns={"SiO₂": "SiO2", "Al₂O₃": "Al2O3", "Moisture %": "Moisture_Pct"})
            for m in run_df.index:
                if m in cv.index:
                    for c in ["Fe", "SiO2", "Al2O3", "CaO", "MgO", "LOI", "Moisture_Pct"]:
                        run_df.loc[m, c] = float(cv.loc[m, c])
            if any(float(lab[k]) <= 0 for k in lab): raise ValueError("Enter all six plant laboratory results before running validation.")
            predicted = opt.compute_validation_chemistry(actual_blend, run_df, float(finished), fuel_ash_settings=_current_fuel_ash_settings())
            compare = []
            for key in ["Fe", "SiO2", "Al2O3", "CaO", "MgO", "Basicity"]:
                model = float(predicted.get(key, 0)); plant = float(lab[key]); err = model - plant; ae = abs(err)
                status = "🟢 EXCELLENT" if ae <= 0.20 else ("🟡 ACCEPTABLE" if ae <= 0.50 else "🔴 SIGNIFICANT")
                compare.append({"Parameter": key, "Model": model, "Plant Lab": plant, "Model - Plant": err, "Absolute Error": ae, "Status": status})
            comp = pd.DataFrame(compare)
            max_err = float(comp["Absolute Error"].max())
            verdict = "🟢 EXCELLENT" if max_err <= 0.20 else ("🟡 ACCEPTABLE" if max_err <= 0.50 else "🔴 SIGNIFICANT DEVIATION")
            total_burden = sum(actual_blend.values())
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Actual dry burden", f"{total_burden:,.2f} kg/t")
            c2.metric("Finished sinter", f"{finished:,.2f} kg")
            c3.metric("Plant yield", f"{predicted.get('Actual_yield_pct', 0):.2f}%")
            c4.metric("Validation verdict", verdict)
            st.dataframe(comp, hide_index=True, use_container_width=True,
                         column_config={c: st.column_config.NumberColumn(c, format="%.2f") for c in ["Model", "Plant Lab", "Model - Plant", "Absolute Error"]})
        except Exception as e:
            st.error(f"Validation error: {e}")

# ----------------------------- PRODUCTIVITY ------------------------------------
def productivity_page():
    page_header("Productivity", "Sinter yield / productivity KPIs and the fines-contribution breakdown of the current burden.")
    r = st.session_state.result
    if not r or not r.get("blend"):
        st.info("Run the optimizer first."); return

    c1, c2 = st.columns(2)
    with c1:
        new_feed = st.number_input("Feed Rate (t/h)", min_value=0.1, value=float(st.session_state.feed_rate_t_h), step=0.5)
    with c2:
        new_target = st.number_input("Productivity Target", min_value=0.01, value=float(st.session_state.productivity_target), step=0.01)
    if new_feed != st.session_state.feed_rate_t_h or new_target != st.session_state.productivity_target:
        st.session_state.feed_rate_t_h = new_feed
        st.session_state.productivity_target = new_target

    total_burden = sum(float(v) for v in r["blend"].values())
    try:
        metrics = opt.calculate_productivity_metrics(total_burden, st.session_state.feed_rate_t_h, st.session_state.productivity_target)
        st.session_state.productivity_metrics = metrics
    except Exception as e:
        st.error(str(e)); return

    cols = st.columns(2)
    cols[0].markdown(kpi("PRODUCTIVITY TARGET", f"{metrics['Productivity_Target']:.4f}", "User-set target", "s"), unsafe_allow_html=True)
    achieved_kind = "g" if metrics["Target_Achieved"] else "r"
    cols[1].markdown(kpi("PRODUCTIVITY ACHIEVED", f"{metrics['Final_Productivity']:.4f}",
                          f"{metrics['Achievement_Pct']:.1f}% of target — {'MET' if metrics['Target_Achieved'] else 'NOT MET'}", achieved_kind), unsafe_allow_html=True)

    with st.expander("How this is calculated"):
        st.markdown(f"""
- Sinter yield = 1000 / total dry burden = **{metrics['Sinter_Yield_Pct']:.2f}%**
- Gross sinter = yield × feed rate = **{metrics['Gross_Sinter_t_h']:.2f} t/h**
- Net sinter = gross − 20% of gross = **{metrics['Net_Sinter_t_h']:.2f} t/h**
- Base productivity = gross / net = **{metrics['Base_Productivity']:.4f}**
- Final productivity = base − 1% of base = **{metrics['Final_Productivity']:.4f}**
""")

    st.markdown('<div class="panel"><div class="panel-title">FINES CONTRIBUTION TABLE</div>'
                 '<div class="small">Specific consumption % starts from the optimized burden and is editable here as a what-if — it does not change the optimizer result.</div></div>', unsafe_allow_html=True)

    df = r["df"]
    if not st.session_state.fines_specific_consumption:
        st.session_state.fines_specific_consumption = {m: (r["blend"].get(m, 0.0) / total_burden * 100.0 if total_burden else 0.0) for m in df.index}

    rows = []
    for m in df.index:
        fines_pct = float(df.loc[m, "Fines_Pct"]) if "Fines_Pct" in df.columns else 0.0
        spec_pct = float(st.session_state.fines_specific_consumption.get(m, 0.0))
        rows.append({"Raw Material": m, "% Fines": fines_pct, "Specific Consumption %": spec_pct})
    table = pd.DataFrame(rows)
    edited = st.data_editor(table, hide_index=True, use_container_width=True, key="fines_table_editor",
                             disabled=["Raw Material", "% Fines"],
                             column_config={"% Fines": st.column_config.NumberColumn("% Fines", format="%.2f"),
                                           "Specific Consumption %": st.column_config.NumberColumn("Specific Consumption % in Optimised Burden", min_value=0.0, format="%.2f")})
    if not edited["Specific Consumption %"].equals(table["Specific Consumption %"]):
        for _, row in edited.iterrows():
            st.session_state.fines_specific_consumption[row["Raw Material"]] = float(row["Specific Consumption %"])
        st.rerun()

    edited["Fines × Consumption"] = edited["% Fines"] * edited["Specific Consumption %"] / 100.0
    total_spec = float(edited["Specific Consumption %"].sum())
    total_fines = float(edited["Fines × Consumption"].sum())
    display_table = edited.copy()
    display_table.loc[len(display_table)] = {"Raw Material": "TOTAL", "% Fines": np.nan, "Specific Consumption %": total_spec, "Fines × Consumption": total_fines}
    st.table(display_table.round(3))
    if abs(total_spec - 100.0) > 0.5:
        st.warning(f"Specific-consumption total is {total_spec:.2f}%, not 100%. Adjust the values above if the burden split is meant to total 100%.")
    st.markdown(f'<div class="notice"><b>TOTAL WEIGHTED FINES = {total_fines:.3f}%</b></div>', unsafe_allow_html=True)
    if st.button("↩ RESET TO OPTIMISED BURDEN", key="reset_fines"):
        st.session_state.fines_specific_consumption = {m: (r["blend"].get(m, 0.0) / total_burden * 100.0 if total_burden else 0.0) for m in df.index}
        st.rerun()

# ----------------------------- WET SPECIFIC CONSUMPTION ------------------------
def wet_specific_consumption():
    page_header("Wet Specific Consumption", "As-received (wet) burden and cost composition, alongside the dry basis for comparison.")
    r = st.session_state.result
    if not r or not r.get("blend"):
        st.info("Run the optimizer first."); return
    df = r["df"]; blend = r["blend"]

    dry_table, rm_cost_dry, total_dry = opt.compute_dry_cost_table(blend, df, st.session_state.om_cost)
    wet_table, rm_cost_wet, total_wet = opt.compute_wet_cost_table(blend, df, st.session_state.om_cost)

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi("DRY BURDEN", f"{sum(blend.values()):,.2f} kg/t", "Optimizer / chemistry basis", "s"), unsafe_allow_html=True)
    wet_total = float(wet_table.loc["TOTAL", "Wet (As-Received) kg"])
    c2.markdown(kpi("WET BURDEN", f"{wet_total:,.2f} kg/t", "As-received basis", "g"), unsafe_allow_html=True)
    c3.markdown(kpi("DRY TOTAL COST", f"₹{total_dry:,.2f}/t", "RM + O&M", "a"), unsafe_allow_html=True)
    c4.markdown(kpi("WET TOTAL COST", f"₹{total_wet:,.2f}/t", "RM + O&M", "a"), unsafe_allow_html=True)

    a, b = st.columns(2, gap="small")
    with a:
        st.markdown('<div class="panel"><div class="panel-title">DRY BASIS</div></div>', unsafe_allow_html=True)
        st.table(dry_table.round(2))
    with b:
        st.markdown('<div class="panel"><div class="panel-title">WET / AS-RECEIVED BASIS</div></div>', unsafe_allow_html=True)
        st.table(wet_table.round(2))

# ----------------------------- REPORTS ------------------------------------
def reports():
    page_header("Reports & Export", "Export the latest optimized recipe with reconciled burden and cost contributions as an Excel workbook.")
    r = st.session_state.result
    if not r or not r["blend"]:
        st.info("Run optimizer first."); return
    df = r["df"]; blend = r["blend"]
    bd = aligned_result_table(blend, df)
    st.table(bd.round(3))

    dry_table, rm_cost_dry, total_dry = opt.compute_dry_cost_table(blend, df, st.session_state.om_cost)
    master_view = st.session_state.master_df.reset_index()
    chem_view = pd.DataFrame([opt.compute_achieved(blend, df, 1000, _current_fuel_ash_settings())])
    inputs_view = pd.DataFrame([{
        "Total Return Sinter %": st.session_state.rs_total_pct, "IOL Fines %": st.session_state.rs_iol_pct,
        "BF Returns %": st.session_state.rs_bfr_pct, "O&M Cost ₹/t": st.session_state.om_cost,
        "Coke CV": st.session_state.coke_cv, "Coke FC %": st.session_state.coke_fc,
    }])
    summary_view = pd.DataFrame([{
        "Run #": st.session_state.runs, "Optimizer Status": r["status"], "RM Cost ₹/t": rm_cost_dry,
        "O&M ₹/t": st.session_state.om_cost, "Total Cost ₹/t": rm_cost_dry + st.session_state.om_cost,
        "Quality": "PASS" if quality_ok(r["achieved"]) else "REVIEW",
    }])

    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        summary_view.to_excel(writer, sheet_name="Summary", index=False)
        bd.to_excel(writer, sheet_name="Dry Burden Cost", index=False)
        master_view.to_excel(writer, sheet_name="Raw Material Master", index=False)
        chem_view.to_excel(writer, sheet_name="Chemistry", index=False)
        inputs_view.to_excel(writer, sheet_name="Inputs", index=False)
        if st.session_state.whatif is not None:
            st.session_state.whatif.to_excel(writer, sheet_name="Scenario Analysis", index=False)
    st.download_button("⬇ DOWNLOAD OPTIMIZATION REPORT (.xlsx)", buf.getvalue(), "sinter_optimization_report.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)

# ----------------------------- UPLOAD & SETTINGS -------------------------------
def settings():
    page_header("Upload & Settings", "Single master workbook management.")
    f = st.file_uploader("UPLOAD MASTER CHEMISTRY EXCEL", type=["xlsx"], key="settings_master")
    if f:
        try:
            df = opt.load_master_chemistry_excel({f.name: f.getvalue()})
            df = opt._ensure_material_role(df)
            st.success(f"Validated: {len(df)} materials")
            if st.button("ACTIVATE MASTER", type="primary"):
                st.session_state.master_df = df
                st.session_state.source = f.name
                st.session_state.available = {m: (float(df.loc[m, "Available_Tonnes"]) > 0) for m in df.index}
                st.session_state.result = None; st.session_state.runs = 0
                st.session_state.manual_scenario_result = None
                st.rerun()
        except Exception as e:
            st.error(str(e))
    if st.button("↺ RESTORE BUILT-IN MASTER", use_container_width=True):
        st.session_state.master_df = initial_df()
        st.session_state.source = "Built-in Master Chemistry"
        st.session_state.available = {m: True for m in st.session_state.master_df.index}
        st.session_state.result = None; st.session_state.runs = 0
        st.rerun()

# ----------------------------- ROUTING -----------------------------------
pages = {
    "Dashboard": dashboard,
    "Inputs": inputs_page,
    "RM Stock & Materials": rm_stock_materials,
    "Recipe & Composition": recipe_composition,
    "Manual Burden Control": manual,
    "Scenario Analysis": scenario_analysis,
    "Plant Run Validation": plant_run_validation,
    "Productivity": productivity_page,
    "Wet Specific Consumption": wet_specific_consumption,
    "Reports": reports,
    "Upload & Settings": settings,
}
pages[st.session_state.nav]()
st.markdown('<div class="footer">Sinter Burden Control • Hospet Alloy Steel Plant • Production decision-support interface</div>', unsafe_allow_html=True)
