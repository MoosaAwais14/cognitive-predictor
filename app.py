import streamlit as st
import pandas as pd
import joblib


st.set_page_config(
    page_title="Cognitive Risk Predictor",
    layout="wide"
)

st.markdown(
    """
    <style>
    .group-chip {
        padding: 0.35rem 0.7rem;
        border-radius: 999px;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 0.35rem;
        color: #111827;
    }
    .demo-chip { background-color: #76aade; }
    .z-chip { background-color: #61d461; }
    .raw-chip { background-color: #e8e848; }

    .feature-label {
        padding: 0.3rem 0.55rem;
        border-radius: 0.6rem;
        font-size: 0.92rem;
        font-weight: 600;
        margin-bottom: 0.2rem;
        display: inline-block;
        color: #111827;
    }
    .feature-demo { background-color: #76aade; }
    .feature-z { background-color: #61d461; }
    .feature-raw { background-color: #e8e848; }

    .small-note {
        color: #4b5563;
        font-size: 0.92rem;
    }

    /* ── Sidebar restyle ─────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background-color: #101827;
    }
    section[data-testid="stSidebar"] * {
        color: #e2e8f0;
    }
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] .stMarkdown p {
        color: #e2e8f0;
    }
    section[data-testid="stSidebar"] hr {
        border-color: #263248;
    }

    .sidebar-eyebrow {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #7c8aa5 !important;
        margin: 1.1rem 0 0.4rem 0;
    }

    /* Feature-set dropdown (selectbox) */
    section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
        background-color: #1c2537;
        border: 1px solid #324364;
        border-radius: 0.65rem;
        box-shadow: none;
        transition: border-color 0.15s ease;
    }
    section[data-testid="stSidebar"] div[data-baseweb="select"] > div:hover {
        border-color: #3b82f6;
    }
    section[data-testid="stSidebar"] div[data-baseweb="select"] span {
        color: #e2e8f0 !important;
        font-weight: 600;
    }
    ul[data-baseweb="menu"] {
        background-color: #1c2537 !important;
        border: 1px solid #324364 !important;
    }
    ul[data-baseweb="menu"] li {
        color: #e2e8f0 !important;
    }
    ul[data-baseweb="menu"] li:hover {
        background-color: #2563eb !important;
    }

    /* Toggle switches (feature group inclusion) */
    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
        font-size: 0.9rem;
        font-weight: 500;
        color: #cbd5e1 !important;
    }
    section[data-testid="stSidebar"] button[role="switch"][aria-checked="true"] {
        background-color: #3b82f6 !important;
    }

    /* Model info card */
    .model-info-card {
        margin-top: 1.4rem;
        padding: 0.9rem 1rem;
        background: #1c2537;
        border: 1px solid #263248;
        border-radius: 0.75rem;
        font-size: 0.85rem;
        color: #cbd5e1 !important;
        line-height: 1.9;
    }
    .model-info-card strong {
        color: #f8fafc !important;
        float: right;
    }
    .model-info-row {
        display: flex;
        justify-content: space-between;
        color: #94a3b8;
    }

    /* Categorical dropdowns rendered in the main patient-feature grid */
    div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
        border-radius: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)


demographics = [
    "age7c", "gender1", "race1c", "educ1", "phx_income7", "curjob7"
]

raw_scores = [
    "craftursmb7c", "craftdremb7c", "dgtformb7c", "dgtbckmb7c",
    "dsymscrmb7c", "trailamb7c", "trailbmb7c_adjusted", "udsverfcmb7c",
    "vegmb7c", "animalsmb7c", "avlt_delayed_totalmb7c",
    "avlt_t1_totalmb7c", "avlt_t6_totalmb7c", "avlt_total_correctmb7c",
    "wrat5mb7c", "mocatotsmb7c", "craftdvrmb7c", "craftvrsmb7c",
    "udsbentcmb7c", "udsbentdmb7c", "avlt_lotmb7c", "avlt_listb_totalmb7c",
    "trailb_ceiling", "casisummb7c"
]

z_scores = [
    "memory_immed_domainmb7c", "memory_delay_domainmb7c",
    "lang_semantic_domainmb7c", "phonemic_domainmb7c",
    "attn_process_domainmb7c", "executive_domainmb7c",
    "visuo_domainmb7c", "lang_phonemic_domainmb7c"
]

reduced_z_scores = [
    "executive_domainmb7c",
    "memory_delay_domainmb7c",
    "trailbmb7c_adjusted",
    "casisummb7c",
    "dsymscrmb7c",
]

pretty_names = {
    "age7c": "Age",
    "gender1": "Gender",
    "race1c": "Race",
    "educ1": "Education",
    "phx_income7": "Income",
    "curjob7": "Current Job",
    "trailbmb7c_adjusted": "Trail B Adjusted",
    "memory_delay_domainmb7c": "Memory Delay Z-score",
    "executive_domainmb7c": "Executive Z-score",
    "mocatotsmb7c": "MoCA Total",
    "casisummb7c": "CASI Sum",
    "dsymscrmb7c": "Digit Symbol Score",
}

# Numeric-code -> human-readable label, for the categorical demographic
# fields. Age is left as a plain numeric input since it isn't categorical.
demographic_value_maps = {
    "gender1": {0: "Female", 1: "Male"},
    "race1c": {1: "White", 2: "Chinese", 3: "Black", 4: "Hispanic"},
    "educ1": {
        0: "No schooling",
        1: "Grades 1–8",
        2: "Grades 9–11",
        3: "High school/GED",
        4: "Some college",
        5: "Tech school",
        6: "Associate degree",
        7: "Bachelor’s",
        8: "Graduate/Professional",
    },
    "phx_income7": {
        1: "<$5k", 2: "$5k–7.9k", 3: "$8k–11.9k", 4: "$12k–15.9k",
        5: "$16k–19.9k", 6: "$20k–24.9k", 7: "$25k–29.9k", 8: "$30k–34.9k",
        9: "$35k–39.9k", 10: "$40k–49.9k", 11: "$50k–74.9k",
        12: "$75k–99.9k", 13: "$100k–124.9k", 14: "$125k–149.9k", 15: "≥$150k",
    },
    "curjob7": {
        1: "Homemaker", 2: "Employed FT", 3: "Employed PT", 4: "On leave",
        5: "Temp away", 6: "Unemployed <6mo", 7: "Unemployed >6mo",
        8: "Retired - Not Working", 9: "Retired - Working", 10: "Volunteering",
    },
}


REQUIRED_ARTIFACTS = ["model", "threshold", "features", "feature_means"]


@st.cache_resource
def load_artifacts(mode_key: str):
    """Load every artifact for a given model key.

    Raises FileNotFoundError / EOFError / UnpicklingError etc. rather than
    swallowing them, so the caller can show the user exactly what went
    wrong instead of the app silently rendering a blank page.
    """
    model = joblib.load(f"model_{mode_key}.pkl")
    threshold = joblib.load(f"threshold_{mode_key}.pkl")
    features = joblib.load(f"features_{mode_key}.pkl")
    means = joblib.load(f"feature_means_{mode_key}.pkl")

    # AUC is optional: older model exports may not have it yet, so we
    # don't want a missing auc_*.pkl file to take down the whole app.
    try:
        auc = joblib.load(f"auc_{mode_key}.pkl")
    except FileNotFoundError:
        auc = None

    return model, threshold, features, means, auc


def get_group(feature: str) -> str:
    if feature in demographics:
        return "demo"
    if feature in z_scores or feature in reduced_z_scores:
        return "z"
    return "raw"


def render_feature_input(feature: str, mean_value: float, key: str):
    group = get_group(feature)

    if group == "demo":
        css_class = "feature-demo"
    elif group == "z":
        css_class = "feature-z"
    else:
        css_class = "feature-raw"

    label = pretty_names.get(feature, feature)

    st.markdown(
        f'<div class="feature-label {css_class}">{label}</div>',
        unsafe_allow_html=True
    )

    if feature in demographic_value_maps:
        value_map = demographic_value_maps[feature]
        options = sorted(value_map.keys())
        # Default to whichever category is closest to the training-set mean.
        default_code = min(options, key=lambda code: abs(code - mean_value))
        selected_code = st.selectbox(
            label,
            options=options,
            index=options.index(default_code),
            format_func=lambda code: value_map[code],
            key=key,
            label_visibility="collapsed",
        )
        return float(selected_code)

    return st.number_input(
        label,
        value=float(mean_value),
        key=key,
        label_visibility="collapsed"
    )


def render_features(feature_list, means, ncols=3, prefix="main"):
    values = {}
    cols = st.columns(ncols)

    for i, feature in enumerate(feature_list):
        with cols[i % ncols]:
            values[feature] = render_feature_input(
                feature,
                means[feature],
                key=f"{prefix}_{feature}"
            )
    return values


st.title("Cognitive Risk Predictor")
st.markdown(
    '<div class="small-note">Fields are pre-filled with training-set averages. '
    'Blue = demographics, yellow = raw scores, green = z-scores.</div>',
    unsafe_allow_html=True
)

chip_col1, chip_col2, chip_col3 = st.columns([1, 1, 1])
with chip_col1:
    st.markdown('<div class="group-chip demo-chip">Demographics</div>', unsafe_allow_html=True)
with chip_col2:
    st.markdown('<div class="group-chip raw-chip">Raw Scores</div>', unsafe_allow_html=True)
with chip_col3:
    st.markdown('<div class="group-chip z-chip">Z-scores</div>', unsafe_allow_html=True)


st.sidebar.header("Model Settings")

st.sidebar.markdown('<div class="sidebar-eyebrow">Feature set</div>', unsafe_allow_html=True)
feature_mode = st.sidebar.selectbox(
    "Feature set",
    options=["Full", "Reduced"],
    index=0,
    label_visibility="collapsed"
)

# Defaults so these names always exist regardless of which branch runs below.
mode_key = None
model = threshold = features = feature_means = model_auc = None
active_features = []
ncols = 4
load_error = None

if feature_mode == "Full":
    mode_key = "full"
    ncols = 4
    try:
        model, threshold, features, feature_means, model_auc = load_artifacts(mode_key)
        active_features = list(features)
    except Exception as exc:
        load_error = exc

else:
    st.sidebar.markdown('<div class="sidebar-eyebrow">Include feature groups</div>', unsafe_allow_html=True)
    include_z = st.sidebar.toggle("Z-scores (reduced)", value=True)
    include_raw = st.sidebar.toggle("Raw scores", value=False)
    include_demo = st.sidebar.toggle("Demographics", value=False)

    mode_key = "reduced_demo" if include_demo else "reduced_nodemo"

    try:
        model, threshold, features, feature_means, model_auc = load_artifacts(mode_key)

        active_features = []
        if include_z:
            active_features += [f for f in reduced_z_scores if f in features]
        if include_raw:
            active_features += [f for f in raw_scores if f in features]
        if include_demo:
            active_features += [f for f in demographics if f in features]

        active_features = list(dict.fromkeys(active_features))
        ncols = max(2, min(4, len(active_features) // 3 + 1))
    except Exception as exc:
        load_error = exc

if load_error is not None:
    missing_files = [
        f"{name}_{mode_key}.pkl" for name in REQUIRED_ARTIFACTS
    ]
    st.error(
        f"Couldn't load the **{mode_key}** model artifacts, so there's nothing to "
        f"render for this feature set.\n\n"
        f"**Error:** `{load_error}`\n\n"
        f"This almost always means one of these files is missing from the app's "
        f"working directory:\n\n"
        + "\n".join(f"- `{f}`" for f in missing_files)
        + "\n\nRun `model.py` (it trains three variants: `full`, `reduced_demo`, "
        "`reduced_nodemo`) and make sure all of the resulting `.pkl` files are "
        "deployed alongside `app.py`."
    )
    st.stop()

st.sidebar.markdown(
    f"""
    <div class="model-info-card">
        <div class="model-info-row">Loaded model <strong>{mode_key}</strong></div>
        <div class="model-info-row">Features shown <strong>{len(active_features)}</strong></div>
        <div class="model-info-row">Decision threshold <strong>{threshold:.3f}</strong></div>
        <div class="model-info-row">Model AUC (test) <strong>{f"{model_auc:.3f}" if model_auc is not None else "N/A"}</strong></div>
    </div>
    """,
    unsafe_allow_html=True
)
if model_auc is None:
    st.sidebar.caption("Re-run model.py to generate auc_*.pkl and see the test-set AUC here.")

st.subheader("Patient Features")

if not active_features:
    st.warning("Select at least one feature group from the sidebar.")
else:
    inputs_shown = render_features(active_features, feature_means, ncols=ncols, prefix=mode_key)

    inputs = {f: float(feature_means[f]) for f in features}
    inputs.update(inputs_shown)

    predict_col1, predict_col2 = st.columns([1, 3])
    with predict_col1:
        predict_clicked = st.button("Predict", use_container_width=True)

    if predict_clicked:
        input_df = pd.DataFrame([inputs])[features]
        input_df = input_df.fillna(pd.Series(feature_means))

        prob = model.predict_proba(input_df)[0, 1]
        pred = int(prob >= threshold)

        st.subheader("Results")

        out1, out2, out3, out4 = st.columns(4)
        out1.metric("Risk Probability", f"{prob:.3f}")
        out2.metric("Threshold", f"{threshold:.3f}")
        out3.metric("Decision", "High Risk" if pred == 1 else "Low Risk")
        out4.metric("Model AUC (test)", f"{model_auc:.3f}" if model_auc is not None else "N/A")

        if pred == 1:
            st.error("High Risk (Cognitive Impairment Likely)")
        else:
            st.success("Low Risk")