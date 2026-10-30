"""Behind the Box Office Lab: classroom dashboard for the case "Behind the Box Office: Decoding Movie Magic"
(UVA-S-0410), day two. Built for a full MBA section at once.

Everything is precomputed by build_data.py into data/ (cleaned films, prediction models, studio market share), so a
click only filters a ~1,000-row table or reads a stored model. Nothing is fitted while students use the app. Charts
are Vega-Lite (drawn in the student's browser), so a full class clicking at once costs the server very little.

On the Hugging Face Space, data/ is not shipped with the code: it is pulled at startup from the private dataset named
in DATA_REPO using the HF_TOKEN secret, and APP_PASSWORD (also a secret) gates the app. Locally, data/ is read as is
and there is no password unless APP_PASSWORD is set.
"""
import hmac
import inspect
import json
import math
import os
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

st.set_page_config(page_title="Behind the Box Office Lab", page_icon="🎬", layout="wide",
                   initial_sidebar_state="collapsed")

# --- Design system ---------------------------------------------------------------------------------------------
BLUE, INK, GRAY, MUTED = "#3582F0", "#111827", "#9ca3af", "#6b7280"
# Categorical slots in fixed order (brand blue first; checked for color-vision-deficiency separation)
CAT = ["#3582F0", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
FONT = "Inter, 'Helvetica Neue', Helvetica, Arial, sans-serif"
GENRE_ORDER = ["Action", "Comedy", "Drama", "Adventure", "Crime", "Biography", "Animation", "Horror"]
GENRE_COLOR = dict(zip(GENRE_ORDER, CAT)) | {"Other": GRAY}     # color follows the entity, never its rank
EMOTIONS = ["Joy", "Surprise", "Fear", "Sadness", "Anger", "Disgust", "Neutral"]
EMOTION_COLOR = dict(zip(EMOTIONS, CAT))
STUDIO_ORDER = ["Universal", "Disney", "Warner Bros.", "Sony", "Paramount", "Fox", "Lionsgate"]
# Each studio keeps one brand-inspired color in every chart (checked for color-vision-deficiency separation on the
# chart panel). Paramount is white, so its marks get a charcoal keyline and its text is charcoal.
STUDIO_COLOR = {"Universal": "#8e3aa8", "Disney": "#1d4ed8", "Warner Bros.": "#7a5216", "Sony": "#7d8899",
                "Paramount": "#ffffff", "Fox": "#c2185b", "Lionsgate": "#e8a200"}
KEYLINE = "#374151"


def studio_text(stu):
    return KEYLINE if stu == "Paramount" else STUDIO_COLOR.get(stu, INK)


def studio_expr(field):
    """Vega expression giving each studio's text color, e.g. for axis labels (field 'value') or direct labels."""
    expr = f"'{KEYLINE}'"
    for stu in reversed(STUDIO_ORDER):
        expr = f"datum['{field}'] == '{stu}' ? '{studio_text(stu)}' : {expr}"
    return alt.ExprRef(expr=expr)
ACTS = {1: "Act 1 · Exposition", 2: "Act 2 · Rising action", 3: "Act 3 · Climax", 4: "Act 4 · Falling action",
        5: "Act 5 · Resolution"}
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
SEASON_ORDER = ["Winter", "Spring", "Summer", "Fall", "Holiday"]
HIT_MULTIPLE = 2.5

# One long page: section title -> anchor. Order here is the order on the page and in the menus.
NAV = {"Studio Playbooks": "studios", "Trends & Franchise Fatigue": "trends", "Script DNA": "scripts",
       "Film Economics": "economics", "Pitch a Film": "pitch", "Wall Street & Parent Companies": "parents",
       "Predict the Box Office": "predict", "Movie Database": "database"}
NAV_NUM = {t: str(i) for i, t in enumerate(NAV, start=1)}


# --- Definitions on hover ---------------------------------------------------------------------------------------
# Every stat tile and table column gets a hover definition looked up from its label, so a new measure only needs a
# rule here. Rules are checked in order (first match wins), so specific phrases come before general ones.
DEFINE_RULES = [
    ("films in this dataset", "How many of the studio's films are in this dataset (films whose scripts were "
                              "available), not its full slate."),
    ("number of films", "How many films in this dataset fall in the group."),
    ("films with stock data", "Films whose parent company was publicly traded at release, so a stock reaction "
                              "can be measured."),
    ("hit rate", f"Share of films grossing at least {HIT_MULTIPLE}× their production budget worldwide, roughly the "
                 f"level where a film turns a profit after marketing and the theaters' cut."),
    ("profitable", "Share of films whose estimated studio profit (from the deal assumptions in Film Economics) is "
                   "above zero."),
    ("break-even", "The worldwide box office, as a multiple of the production budget, a film needs before the "
                   "studio's share covers the budget and the marketing."),
    ("chance of a profit", "Share of the model's past prediction misses that would still leave this film above "
                           "break-even."),
    ("10th–90th", "The range from a bad case (only 1 film in 10 did worse) to a good case (only 1 in 10 did better)."),
    ("profit", "Estimated studio profit: the studio's share of ticket sales, minus production budget and marketing, "
               "plus the later windows if included. Uses the deal assumptions in Film Economics."),
    ("return", "Worldwide box office ÷ production budget (the gross multiple). 1× = the gross only matched the "
               "budget; roughly 2.5–3× is needed to profit once theaters and marketing take their share."),
    ("week-2 drop", "% of the first week's US gross lost in the second week. Higher = more front-loaded, often a "
                    "sign of weak word of mouth or a fan-driven opening."),
    ("week 1 share", "Share of the total US gross earned in the first 7 days: how front-loaded the run was."),
    ("predicted", "The model's median prediction of worldwide box office for these choices."),
    ("likely range", "The middle 50% of outcomes: half of comparable past films landed inside this range."),
    ("needed", "Worldwide gross needed to cover budget and marketing over the full release cycle."),
    ("opening theaters", "Number of US theaters showing the film on opening weekend. 600+ counts as a wide release."),
    ("theaters", "Number of US theaters showing the film on opening weekend."),
    ("opening", "US and Canada box office from the first Friday–Sunday of release (nominal dollars)."),
    ("worldwide", "Total box office: US and Canada plus international, in nominal dollars (not adjusted for "
                  "inflation)."),
    ("domestic", "US and Canada box office, in nominal dollars."),
    ("international", "Box office outside the US and Canada, in nominal dollars."),
    ("budget", "Production budget: what it cost to make the film, excluding marketing (nominal dollars, as "
               "reported by The Numbers or TMDB)."),
    ("imdb", "Average IMDb user rating, 1–10."),
    ("metascore", "Metacritic's weighted average of professional critics' reviews, 0–100."),
    ("follow-up", "A sequel, prequel, reboot, remake or shared-universe spin-off of an earlier film. Adaptations of "
                  "TV shows, games, comics or books count as originals."),
    ("original", "A film that isn't a sequel, prequel, reboot, remake or spin-off of an earlier film."),
    ("weeks in the top 10", "Weeks the film ranked in the US weekly top 10."),
    ("weeks in release", "Weeks the film played in US theaters."),
    ("years as #1", "Years the studio had the largest share of the US box office."),
    ("led us share", "Years (1995–2023) Universal had the largest share of the US box office, all releases."),
    ("explained by the data", "Share of the differences in worldwide box office between films that a model can "
                              "predict on films it never saw (cross-validated R²)."),
    ("stock reaction", "The parent company's stock return beyond the overall market's, from release day to 5 trading "
                       "days later (cumulative abnormal return)."),
    ("5-day reaction", "The parent company's stock return beyond the overall market's, from release day to 5 trading "
                       "days later (cumulative abnormal return)."),
    ("followed by a gain", "Share of releases after which the parent's stock beat the market over the next 5 days."),
    ("tone", "How upbeat executives sounded on the parent's quarterly earnings call, scored by FinBERT: share of "
             "positive minus negative sentences (−1 to +1)."),
    ("% of parent", "The film's worldwide box office ÷ its parent company's total revenue for the fiscal year."),
    ("roa", "Return on assets: the parent company's net income ÷ total assets for the quarter."),
    ("margin", "Profit margin: the parent company's net income ÷ revenue."),
    ("leverage", "Debt ÷ total assets of the parent company."),
    ("r²", "Share of the variation in the outcome the model explains on films it never saw: 1 = perfect, 0 = no "
           "better than guessing the average."),
    ("p-value", "How surprising this result would be if the feature had no real relationship with the outcome. "
                "Below 0.10 is flagged as significant here."),
    ("importance", "How much the random forest's predictions rely on this feature (all features sum to 1)."),
    ("effect", "Change in the outcome for a one-step increase in the feature (one standard deviation for "
               "continuous features; yes vs. no for 0/1 features), holding the others constant."),
    ("vif", "Variance inflation factor: how much a feature duplicates the others. Above 10 = removed."),
    ("studio", "The major studio group that distributed the film (labels such as Focus or New Line count under "
               "their parent studio)."),
    ("genre", "The film's primary genre (TMDB). Genres with fewer than 20 films are grouped as Other."),
    ("rating", "MPAA rating: G, PG, PG-13 or R."),
    ("runtime", "Running time in minutes."),
    ("director", "The film's credited director."),
    ("lead cast", "The top-billed cast members."),
    ("year", "Year of US release."),
    ("films", "Number of films in the group (in this dataset)."),
]


def define(label):
    """Hover definition for a measure's label, or None. 'Median ...' labels say so."""
    low = str(label).lower()
    for key, text in DEFINE_RULES:
        if key in low:
            if low.startswith("median") or "· follow-up vs. original" in low:
                text = "Median = the middle film (half above, half below), which ignores a few huge outliers. " + text
            return text
    return None


def measure_help(names):
    """Help text for a 'Measure' picker: every option with its definition."""
    return "  \n".join(f"**{n}**: {define(n) or ''}" for n in names)


# Default every st.metric's hover help to the label's definition (an explicit help= still wins).
_metric = st.delta_generator.DeltaGenerator.metric


def _metric_with_help(self, label, value, *args, help=None, **kw):
    return _metric(self, label, value, *args, help=help or define(label), **kw)


st.delta_generator.DeltaGenerator.metric = _metric_with_help

st.markdown(f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="st-"], .stMarkdown, .stCaption, button, input, textarea {{ font-family: {FONT}; }}
h1 {{ font-weight: 700 !important; letter-spacing: -0.025em; color: {INK}; line-height: 1.15 !important; }}
h2, h3, h4 {{ font-weight: 650 !important; letter-spacing: -0.015em; color: {INK}; }}
.kicker {{ text-transform: uppercase; letter-spacing: .09em; font-size: .74rem; font-weight: 600; color: {BLUE};
           margin-bottom: -.35rem; }}
.lede {{ font-size: 1.05rem; color: #374151; max-width: 62rem; }}
.rule {{ border: 0; border-top: 1px solid #e5e7eb; margin: 1.6rem 0 1.2rem; }}
[data-testid="stMetric"] {{ background: #fff; border: 1px solid #e5e7eb; border-left: 3px solid {BLUE};
                            border-radius: 4px; padding: .7rem .9rem; }}
[data-testid="stMetricLabel"] p {{ text-transform: uppercase; letter-spacing: .05em; font-size: .7rem !important;
                                   color: {MUTED}; font-weight: 600; }}
[data-testid="stMetricValue"] {{ font-size: 1.35rem !important; font-weight: 650; color: {INK}; }}
[data-testid="stMetricLabel"] *, [data-testid="stMetricValue"] * {{ white-space: normal !important;
    overflow: visible !important; text-overflow: clip !important; line-height: 1.25; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 1.4rem; border-bottom: 1px solid #e5e7eb; }}
.stTabs [data-baseweb="tab"] p {{ font-weight: 600; font-size: .95rem; }}
.chapter {{ border-top: 2px solid {INK}; margin: 3.2rem 0 1.1rem; }}
.navbar {{ display: flex; flex-wrap: wrap; gap: .45rem; margin: .6rem 0 1.4rem; padding: .5rem 0;
           background: #fff; border-bottom: 1px solid #e5e7eb; }}
/* Section pills stay pinned under Streamlit's header while scrolling; headings land below the bar when clicked */
div[data-testid="stElementContainer"]:has(.navbar), div.element-container:has(.navbar) {{
    position: sticky; top: 3.75rem; z-index: 990; }}
.jump {{ position: relative; }} .jump span {{ position: absolute; top: -11rem; }}
.navbar a {{ text-decoration: none; font-size: .82rem; font-weight: 600; color: {INK}; border: 1px solid #e5e7eb;
             border-radius: 999px; padding: .28rem .8rem; background: #fff; }}
.navbar a:hover {{ border-color: {BLUE}; color: {BLUE}; }}
.navbar a b {{ color: {BLUE}; margin-right: .3rem; }}
section[data-testid="stSidebar"] a {{ text-decoration: none; color: {INK}; }}
[data-testid="stIconMaterial"], .material-symbols-rounded, [class*="material-symbols"] {{
    font-family: 'Material Symbols Rounded' !important; }}
[data-testid="stCaptionContainer"] p, .stCaption p {{ color: {MUTED}; }}
.takeaway {{ border-left: 3px solid {BLUE}; background: #f5f9ff; padding: .75rem 1rem; margin: .4rem 0 1rem;
             color: {INK}; border-radius: 0 4px 4px 0; }}
.takeaway b {{ color: {BLUE}; }}
</style>""", unsafe_allow_html=True)

# st.dataframe takes width="stretch" from Streamlit 1.46; older versions (e.g. a local 1.45) need use_container_width
_NEW_WIDTH = tuple(int(x) for x in st.__version__.split(".")[:2]) >= (1, 46)


def _stretch(fn):
    return {"width": "stretch"} if _NEW_WIDTH and "width" in inspect.signature(fn).parameters \
        else {"use_container_width": True}


def show_table(df, column_config=None, **kw):
    """st.dataframe in house style, with each column header's hover help filled in from define()."""
    cfg = dict(column_config or {})
    cols = df.data.columns if hasattr(df, "data") else df.columns
    for c in cols:
        v = cfg.get(c)
        label = v if isinstance(v, str) else (v or {}).get("label") or str(c)
        text = define(label)
        if not text:
            continue
        if v is None or isinstance(v, str):
            cfg[c] = st.column_config.Column(label, help=text)
        elif not v.get("help"):
            cfg[c] = {**v, "help": text}
    st.dataframe(df, hide_index=True, column_config=cfg or None, **_stretch(st.dataframe), **kw)


PANEL, SOFT = "#eaeaf2", "#9aa3b2"     # seaborn "darkgrid": gray-blue panel, white grid; SOFT = comparison bars


def show_chart(chart, title=None, subtitle=None, panel=True):
    """House style for every chart, after seaborn's darkgrid: a soft gray-blue panel with white gridlines, no axis
    lines or ticks. The heading names the measure (subtitle); the insight (title) goes in an Insight box under the
    chart, so every chart ends with its answer. Heatmaps pass panel=False (white background, as in seaborn)."""
    if subtitle or title:
        chart = chart.properties(title=alt.TitleParams(text=subtitle or title, anchor="start"))
    grid = dict(grid=True, gridColor="white", gridWidth=1.2, domain=False, ticks=False)
    chart = (chart.configure(font=FONT, background="white", padding={"left": 4, "right": 18, "top": 6, "bottom": 4})
             .configure_view(stroke=None, fill=PANEL if panel else "white")
             .configure_title(fontSize=15, fontWeight=600, color=INK, subtitleColor=MUTED, subtitleFontSize=12,
                              subtitleFontWeight="normal", anchor="start", offset=14, subtitlePadding=5)
             .configure_axis(labelColor="#374151", titleColor=MUTED, labelFontSize=11, titleFontSize=11,
                             titleFontWeight="normal", labelPadding=6, domain=False, ticks=False)
             .configure_axisQuantitative(**grid)
             .configure_axisTemporal(**grid)
             .configure_axisDiscrete(grid=False)
             .configure_legend(labelFontSize=11, labelColor="#374151", symbolType="circle", symbolSize=80,
                               titleFontSize=11, titleColor=MUTED, columnPadding=14))
    st.altair_chart(chart, theme=None, **_stretch(st.altair_chart))
    if title and subtitle:
        takeaway(f"<b>Insight:</b> {title}")


def section(kicker, title, how):
    """Chapter opener on the one-page scroll. The anchor comes from NAV, so the menu links land here."""
    # The jump target sits above the heading so the pinned section bar doesn't cover it
    st.markdown(f"<div class='chapter'></div><div class='jump'><span id='{NAV[title]}'></span></div>"
                f"<div class='kicker'>{NAV_NUM[title]} · {kicker}</div>", unsafe_allow_html=True)
    st.subheader(title, anchor=False)
    st.markdown(how)


def exhibit(title):
    st.markdown(f"<hr class='rule'><h4 style='margin-bottom:.2rem'>{title}</h4>", unsafe_allow_html=True)


def takeaway(html):
    st.markdown(f"<div class='takeaway'>{html}</div>", unsafe_allow_html=True)


# --- Class password (only when APP_PASSWORD is set, i.e. on the Space) ------------------------------------------
def password_gate():
    expected = os.environ.get("APP_PASSWORD", "")
    if not expected or st.session_state.get("authed"):
        return
    st.markdown("<p class='kicker'>UVA-S-0410 · Day two</p>", unsafe_allow_html=True)
    st.title("Behind the Box Office Lab")
    with st.form("login"):
        pw = st.text_input("Class password", type="password")
        if st.form_submit_button("Enter"):
            if hmac.compare_digest(pw.encode(), expected.encode()):
                st.session_state["authed"] = True
                st.rerun()
            st.error("That password isn't right. Check the Canvas announcement.")
    st.stop()


password_gate()


# --- Data (loaded once per server, shared read-only by every session) -----------------------------------------
def data_dir():
    """Local data/ if it is there; otherwise download it from the private HF dataset (once per server)."""
    if (DATA / "films.json").exists():
        return DATA
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(repo_id=os.environ["DATA_REPO"], repo_type="dataset",
                                  token=os.environ.get("HF_TOKEN"), allow_patterns=["*.json"]))


@st.cache_resource(show_spinner="Loading the film data…")
def load():
    src = data_dir()
    films = pd.DataFrame(json.loads((src / "films.json").read_text(encoding="utf-8")))
    for c in ["budget", "opening", "domestic", "international", "worldwide", "week1", "week2"]:
        films[c + "_m"] = films[c] / 1e6
    films["label"] = films["title"] + " (" + films["year"].astype(str) + ")"
    films["kind"] = np.where(films["follow_up"] == 1, "Follow-up", "Original")
    films["hit"] = np.where(films["multiple"].notna(), (films["multiple"] >= HIT_MULTIPLE).astype(float), np.nan)
    start = films["year"] // 5 * 5
    films["period"] = start.astype(str) + "–" + ((start + 4) % 100).astype(str).str.zfill(2)
    films["decade"] = (films["year"] // 10 * 10).astype(str) + "s"
    models = json.loads((src / "models.json").read_text(encoding="utf-8"))
    ms_path = src / "market_share.json"
    share = pd.DataFrame(json.loads(ms_path.read_text(encoding="utf-8"))) if ms_path.exists() else None
    if share is not None:
        share["Gross per film"] = share["Gross"] / share["Films"]
        share["Top film share"] = share["Top film gross"] / share["Gross"]
    pitch = json.loads((src / "pitch.json").read_text(encoding="utf-8"))
    parents = pd.DataFrame(json.loads((src / "parents.json").read_text(encoding="utf-8")))
    return films, models, share, pitch, parents


FILMS, MODELS, SHARE, PITCH, PARENTS = load()
TITLE_BY_ID = dict(zip(FILMS["id"], FILMS["label"]))
STUDIOS = [s for s in STUDIO_ORDER if s in set(FILMS["studio"])]
GENRES = [g for g in GENRE_ORDER if g in set(FILMS["genre"])] + ["Other"]


# --- Small helpers ---------------------------------------------------------------------------------------------
def money(v):
    if v is None or pd.isna(v):
        return "—"
    sign = "−" if v < 0 else ""
    v = abs(v)
    return f"{sign}${v / 1e9:,.2f}B" if v >= 1e9 else f"{sign}${v / 1e6:,.1f}M"


def fmt(v, spec="{:,.2f}", missing="—"):
    return missing if v is None or pd.isna(v) else spec.format(v)


def median(s):
    s = pd.to_numeric(s, errors="coerce").dropna()
    return float(s.median()) if len(s) else np.nan


def welch_p(a, b):
    """Two-sided Welch t-test p-value with a normal approximation (fine for groups of 30+)."""
    a, b = pd.Series(a).dropna(), pd.Series(b).dropna()
    if len(a) < 5 or len(b) < 5:
        return np.nan
    se = math.sqrt(a.var() / len(a) + b.var() / len(b))
    return math.erfc(abs(a.mean() - b.mean()) / se / math.sqrt(2)) if se > 0 else np.nan


def mean_ci(s):
    s = pd.Series(s).dropna()
    if len(s) < 2:
        return np.nan, np.nan, np.nan, len(s)
    m, h = s.mean(), 1.96 * s.std() / math.sqrt(len(s))
    return m, m - h, m + h, len(s)


def quartile_groups(df, col="multiple"):
    """Top and bottom 25% of `col` within df (a relative definition of hit and miss)."""
    s = df[col].dropna()
    if len(s) < 8:
        return df.iloc[0:0], df.iloc[0:0]
    lo, hi = s.quantile(0.25), s.quantile(0.75)
    return df[df[col] >= hi], df[df[col] <= lo]


def act_axis():
    return alt.X("Act:O", title=None, axis=alt.Axis(labelExpr="'Act ' + datum.value", labelAngle=0))


def line_points(size=60):
    return alt.OverlayMarkDef(size=size, filled=True, stroke="white", strokeWidth=1.5)


CSV_COLS = {"title": "Title", "year": "Year", "release": "Release date", "studio": "Studio", "distributor": "Distributor",
            "genre": "Genre", "genres_all": "All genres", "mpaa": "Rating", "runtime": "Runtime (min)",
            "kind": "Original / follow-up", "director": "Director", "cast": "Lead cast", "budget": "Budget ($)",
            "opening": "Opening weekend US ($)", "domestic": "Domestic ($)", "international": "International ($)",
            "worldwide": "Worldwide ($)", "multiple": "Return (x budget)", "theaters": "Opening theaters",
            "drop2": "Week-2 drop (%)", "weeks": "Weeks in release", "imdb": "IMDb", "metascore": "Metascore",
            "car5": "Parent 5-day stock reaction", "during_tone": "Parent call tone (release qtr)"}


def csv_bytes(v):
    return v[list(CSV_COLS)].rename(columns=CSV_COLS).to_csv(index=False).encode("utf-8")


def comps(genre, budget, follow_up, year, exclude=None, k=6):
    """Nearest films: same genre, then closest budget (log scale), same original/follow-up status and nearest year.
    Distance = |log budget ratio| + 0.5 if the follow-up status differs + 1 per 20 years apart."""
    pool = FILMS[(FILMS["genre"] == genre) & FILMS["budget"].notna() & FILMS["worldwide"].notna()]
    if exclude is not None:
        pool = pool[pool["id"] != exclude]
    dist = (np.log(pool["budget"] / budget).abs() + 0.5 * (pool["follow_up"] != follow_up)
            + (pool["year"] - year).abs() / 20)
    return pool.loc[dist.nsmallest(k).index]


def show_comps(c):
    show_table(c[["title", "year", "studio", "kind", "budget_m", "worldwide_m", "multiple", "imdb"]], column_config={
        "title": "Film", "year": st.column_config.NumberColumn("Year", format="%d"), "studio": "Studio",
        "kind": "Original / follow-up", "budget_m": st.column_config.NumberColumn("Budget ($M)", format="%.0f"),
        "worldwide_m": st.column_config.NumberColumn("Worldwide ($M)", format="%.0f"),
        "multiple": st.column_config.NumberColumn("Return (×)", format="%.2f×"),
        "imdb": st.column_config.NumberColumn("IMDb", format="%.1f")})


# --- Header (always visible above the tabs) ---------------------------------------------------------------------
st.markdown("<div class='kicker'>Media &amp; Entertainment Businesses · Session two data lab</div>", unsafe_allow_html=True)
st.title("Behind the Box Office: What Movies Should Universal Make Next?")
st.markdown("<p class='lede'>1,079 real films released 1984–2025, read four ways: the box office, the studio, "
            "the script and the parent company. Companion to <em>Behind the Box Office: Decoding Movie Magic</em> "
            "(UVA-S-0410).</p>", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def headline_numbers():
    w = FILMS[(FILMS["wide"] == 1) & (FILMS["year"] >= 2010)]
    fu, og = w[w["follow_up"] == 1], w[w["follow_up"] == 0]
    u = FILMS[FILMS["studio"] == "Universal"]
    lead = SHARE.loc[SHARE.groupby("Year")["Share"].idxmax()] if SHARE is not None else None
    base = MODELS.get("All studios|Worldwide box office|Metadata", {}).get("cv", {}).get("Random forest")
    script = MODELS.get("All studios|Worldwide box office|Metadata + Script emotions", {}).get("cv", {}).get("Random forest")
    return {
        "drop_fu": median(fu["drop2"]), "drop_og": median(og["drop2"]),
        "hit_all": FILMS["hit"].mean() * 100, "hit_u": u["hit"].mean() * 100,
        "u_years": int((lead["Studio"] == "Universal").sum()) if lead is not None else None,
        "u_years_recent": int(((lead["Studio"] == "Universal") & (lead["Year"] >= 2014)).sum()) if lead is not None else None,
        "r2_base": float(np.mean(base)) if base else np.nan, "r2_script": float(np.mean(script)) if script else np.nan,
    }


H = headline_numbers()
cols = st.columns(4)
cols[0].metric("Week-2 drop, 2010s+ wide releases", f"{H['drop_fu']:.0f}% vs. {H['drop_og']:.0f}%",
               help="Median % of week-1 gross lost in week 2: follow-ups vs. originals.")
cols[1].metric(f"Hit rate (≥{HIT_MULTIPLE}× budget)", f"{H['hit_u']:.0f}% Universal",
               help=f"Share of films grossing at least {HIT_MULTIPLE}× their budget worldwide. All studios: {H['hit_all']:.0f}%.")
if H["u_years"] is not None:
    cols[2].metric("Years Universal led US share", f"{H['u_years']} of 29",
                   help=f"1995–2023, full slates. {H['u_years_recent']} of those since 2014.")
cols[3].metric("Box office explained by the data", f"{H['r2_base'] * 100:.0f}%",
               help=f"Random forest, cross-validated, all films, worldwide gross from metadata alone. Adding script "
                    f"emotions: {H['r2_script'] * 100:.0f}%.")

st.info(
    "**What this data can tell you.** Every number here comes from films that actually played, so it shows what has "
    "**gone together** with bigger grosses and better returns. It supports claims like *\"follow-ups lose more of "
    "their audience in week two\"*. It does **not** prove that *making* a follow-up causes the drop: studios choose "
    "which films get sequels, budgets and summer dates, and those choices travel together. Treat each finding as "
    "evidence for your argument, not as a forecast."
)
c1, c2 = st.columns(2)
c1.markdown("**The Problem**  \n"
            "Donna Langley has to decide what Universal Pictures should make next, and for whom, while franchise "
            "fatigue spreads, budgets and marketing costs rise, windows shrink, and Peacock needs films too. Hitting "
            "the two-to-three-times gross multiple that makes a film profitable is getting harder, and gut feel about "
            "what audiences want is a risky basis for a $200 million bet.")
c2.markdown("**The Solution**  \n"
            "This lab puts four decades of real films in one place, at four levels: the **film** (box office and "
            "theatrical run), the **studio** (playbooks and market share), the **script** (the emotional arc of the "
            "dialogue, measured with NLP across five acts) and the **parent company** (how Wall Street and the "
            "quarter's financials responded). Use it to build the evidence behind your Bottom Line Up Front.")
st.markdown("<div class='navbar'>" + "".join(f"<a href='#{a}'><b>{NAV_NUM[t]}</b>{t}</a>" for t, a in NAV.items())
            + "</div>", unsafe_allow_html=True)
st.caption("Hover any number, column header or ⓘ for its definition. Section menu and full glossary: the › arrow at "
           "the top left.")


# --- 1. Movie Database --------------------------------------------------------------------------------------------
@st.fragment
def database_section():
    section("Films", "Movie Database",
            "**How it works:** filter the films, sort any column by clicking its header, then open a film's profile "
            "below to see its run, its script's emotional arc and how its parent company was doing.")
    c1, c2, c3 = st.columns(3)
    query = c1.text_input("Search title, director or cast", placeholder="e.g. Jurassic, Spielberg, Tom Hanks")
    studios = c2.multiselect("Studio", STUDIOS, placeholder="All studios")
    genres = c3.multiselect("Genre", GENRES, placeholder="All genres")
    c4, c5, c6 = st.columns(3)
    years = c4.slider("Release year", int(FILMS["year"].min()), int(FILMS["year"].max()),
                      (int(FILMS["year"].min()), int(FILMS["year"].max())))
    ratings = c5.multiselect("Rating", ["G", "PG", "PG-13", "R", "Unrated/unknown"], placeholder="All ratings")
    kind = c6.radio("Original or follow-up", ["All", "Original", "Follow-up"], horizontal=True,
                    help="Follow-up = a sequel, prequel, reboot, remake or shared-universe spin-off of an earlier film. "
                         "Adaptations of TV shows, games, comics or books count as originals.")
    v = FILMS
    if query:
        q = query.lower()
        v = v[v["title"].str.lower().str.contains(q, regex=False)
              | v["director"].fillna("").str.lower().str.contains(q, regex=False)
              | v["cast"].fillna("").str.lower().str.contains(q, regex=False)]
    if studios:
        v = v[v["studio"].isin(studios)]
    if genres:
        v = v[v["genre"].isin(genres)]
    if ratings:
        v = v[v["mpaa"].isin(ratings)]
    if kind != "All":
        v = v[v["kind"] == kind]
    v = v[v["year"].between(*years)].sort_values("worldwide", ascending=False)

    show_table(v[["title", "year", "studio", "genre", "mpaa", "kind", "director", "cast", "budget_m", "opening_m",
                  "worldwide_m", "multiple", "drop2", "imdb", "metascore"]], height=360, column_config={
        "title": st.column_config.TextColumn("Title", width="medium"),
        "year": st.column_config.NumberColumn("Year", format="%d"),
        "studio": "Studio", "genre": "Genre", "mpaa": "Rating", "kind": "Original / follow-up",
        "director": "Director", "cast": st.column_config.TextColumn("Lead cast", width="medium"),
        "budget_m": st.column_config.NumberColumn("Budget ($M)", format="%.1f"),
        "opening_m": st.column_config.NumberColumn("Opening wknd ($M)", format="%.1f"),
        "worldwide_m": st.column_config.NumberColumn("Worldwide ($M)", format="%.1f"),
        "multiple": st.column_config.NumberColumn("Return (× budget)", format="%.2f×"),
        "drop2": st.column_config.NumberColumn("Week-2 drop (%)", format="%.0f%%"),
        "imdb": st.column_config.NumberColumn("IMDb", format="%.1f"),
        "metascore": st.column_config.NumberColumn("Metascore", format="%d"),
    })
    st.caption(f"*{len(v):,} of {len(FILMS):,} films match. A negative week-2 drop means the gross grew (usually a "
               f"platform release expanding). Sources: The Numbers, Box Office Mojo, TMDB.*")
    st.download_button(f"⬇ Download these {len(v):,} films (CSV)", csv_bytes(v), "box_office_lab_films.csv",
                       "text/csv", on_click="ignore")

    exhibit("Film profile")
    if v.empty:
        st.info("No films match these filters.")
        return
    labels = list(v["label"])
    default = labels.index("Jurassic Park (1993)") if "Jurassic Park (1993)" in labels else 0
    pick = st.selectbox("Open a film (from the filtered list above)", labels, index=default)
    render_profile(v[v["label"] == pick].iloc[0])


def render_profile(r):
    st.markdown(f"### {r['title']} ({r['year']})")
    bits = [r["studio"], r["genre"], f"Rated {r['mpaa']}" if r["mpaa"] != "Unrated/unknown" else "Unrated/unknown",
            f"{int(r['runtime'])} min" if pd.notna(r["runtime"]) else None, r["kind"],
            pd.to_datetime(r["release"]).strftime("released %b %d, %Y") if r["release"] else None]
    st.caption(" · ".join(b for b in bits if b))
    if isinstance(r["logline"], str):
        st.markdown(f"*{r['logline']}*")
    st.markdown(f"**Director:** {r['director'] or '—'}  \n**Cast:** {r['cast'] or '—'}")

    peers = FILMS[FILMS["genre"] == r["genre"]]
    cols = st.columns(6)
    cols[0].metric("Production budget", money(r["budget"]))
    cols[1].metric("Opening weekend (US)", money(r["opening"]))
    cols[2].metric("Domestic", money(r["domestic"]))
    cols[3].metric("Worldwide", money(r["worldwide"]))
    cols[4].metric("Return on budget", fmt(r["multiple"], "{:.2f}×"),
                   help=f"Worldwide ÷ budget. {r['genre']} median: {fmt(median(peers['multiple']), '{:.2f}×')}.")
    cols[5].metric("Week-2 drop", fmt(r["drop2"], "{:.0f}%"),
                   help=f"% of week-1 gross lost in week 2. {r['genre']} median (wide releases): "
                        f"{fmt(median(peers.loc[peers['wide'] == 1, 'drop2']), '{:.0f}%')}.")
    cols = st.columns(6)
    cols[0].metric("IMDb rating", fmt(r["imdb"], "{:.1f}"))
    cols[1].metric("Metascore", fmt(r["metascore"], "{:.0f}"))
    cols[2].metric("Opening theaters", fmt(r["theaters"], "{:,.0f}"))
    cols[3].metric("Weeks in release", fmt(r["weeks"], "{:.0f}"))
    cols[4].metric("Weeks in the top 10", fmt(r["top10_weeks"], "{:.0f}"))
    cols[5].metric("Week 1 share of domestic", fmt(r["week1_share"] * 100 if pd.notna(r["week1_share"]) else np.nan, "{:.0f}%"),
                   help="How front-loaded the run was: the share of the domestic total earned in the first 7 days.")

    left, right = st.columns(2)
    with left:
        st.markdown(f"**This film vs. the typical {r['genre']} film**")
        rows = [("Production budget ($M)", r["budget_m"], median(peers["budget_m"]), "{:,.1f}"),
                ("Worldwide ($M)", r["worldwide_m"], median(peers["worldwide_m"]), "{:,.1f}"),
                ("Return on budget (×)", r["multiple"], median(peers["multiple"]), "{:.2f}"),
                ("Week-2 drop (%)", r["drop2"], median(peers.loc[peers["wide"] == 1, "drop2"]), "{:.0f}"),
                ("Week 1 share of domestic (%)", r["week1_share"] * 100, median(peers["week1_share"]) * 100, "{:.0f}"),
                ("IMDb rating", r["imdb"], median(peers["imdb"]), "{:.1f}")]
        show_table(pd.DataFrame([(a, fmt(b, s), fmt(c, s)) for a, b, c, s in rows],
                                columns=["Measure", "This film", f"{r['genre']} median"]))
        st.caption(f"*Medians across the {len(peers)} {r['genre']} films in the data (week-2 drop: wide releases only).*")

        st.markdown("**The parent company around the release**")
        cols = st.columns(3)
        na = "N/A" if isinstance(r.get("wrong_parent_note"), str) else "—"
        cols[0].metric("Stock reaction, 5 days", fmt(r["car5"] * 100 if pd.notna(r["car5"]) else np.nan, "{:+.1f}%", na))
        cols[1].metric("Earnings-call tone", fmt(r["during_tone"], "{:+.2f}", na))
        cols[2].metric("Parent ROA (quarter)", fmt(r["during_roa"] * 100 if pd.notna(r["during_roa"]) else np.nan, "{:.2f}%"))
        if isinstance(r.get("parent"), str):
            share = r.get("film_share_parent")
            st.caption(f"Parent at release: **{r['parent']}**"
                       + (f" · annual revenue {money(r['parent_rev_fy'] * 1e6)}; this film's worldwide gross = "
                          f"{share * 100:.2f}% of it" if pd.notna(share) else ""))
        if na == "N/A":
            st.caption(f"*{r['wrong_parent_note']}*")
        if pd.isna(r["during_roa"]):
            st.caption("*Quarterly financials and earnings-call tone aren't available for this film; they cover about a "
                       "third of the films, mostly releases by publicly traded parents since the mid-2000s.*")
    with right:
        picked = st.multiselect("Emotions to show", EMOTIONS, default=["Joy", "Fear", "Sadness", "Anger"],
                                key=f"arc_{r['id']}", max_selections=4)
        if picked:
            recs = []
            for e in picked:
                for a in range(1, 6):
                    recs.append({"Emotion": e, "Act": a, "Series": "This film", "Share": r[f"{e}_{a}"] * 100})
                    recs.append({"Emotion": e, "Act": a, "Series": f"{r['genre']} average",
                                 "Share": peers[f"{e}_{a}"].mean() * 100})
            d = pd.DataFrame(recs)
            mine = d[d["Series"] == "This film"].copy()
            mine["gap"] = mine["Share"].values - d[d["Series"] != "This film"]["Share"].values
            top = mine.loc[mine["gap"].abs().idxmax()]
            title = (f"Most distinctive beat: {top['gap']:+.1f} pts of {top['Emotion'].lower()} in Act {int(top['Act'])} "
                     f"vs. the typical {r['genre'].lower()} film")
            chart = alt.Chart(d).mark_line(point=line_points(), strokeWidth=2.25).encode(
                x=act_axis(), y=alt.Y("Share:Q", title="% of the act's dialogue"),
                color=alt.Color("Emotion:N", scale=alt.Scale(domain=picked, range=[EMOTION_COLOR[e] for e in picked]),
                                legend=alt.Legend(orient="top", title=None)),
                strokeDash=alt.StrokeDash("Series:N", scale=alt.Scale(domain=["This film", f"{r['genre']} average"],
                                                                      range=[[1, 0], [5, 4]]),
                                          legend=alt.Legend(orient="top", title=None)),
                tooltip=["Emotion", "Series", "Act", alt.Tooltip("Share:Q", format=".1f", title="% of dialogue")])
            show_chart(chart.properties(height=330), title, "Emotional arc of the script, % of each act's dialogue")
            st.caption("*Solid = this film; dashed = the average film in its genre. Each point is the share of that act's "
                       "dialogue an NLP emotion model reads as that emotion. Acts follow the classic five-part "
                       "structure: exposition, rising action, climax, falling action, resolution.*")

    if pd.notna(r["budget"]):
        st.markdown(f"**Comparable films:** the {r['genre'].lower()} films closest to this one in budget, "
                    f"original vs. follow-up and release year")
        show_comps(comps(r["genre"], r["budget"], r["follow_up"], r["year"], exclude=r["id"]))


# --- 2. Studio Playbooks --------------------------------------------------------------------------------------------
YEARLY = {  # measure: (column, number format, axis title, multiplier)
    "Market share (%)": ("Share", ".1f", "Share of US box office (%)", 100),
    "Total US gross ($M)": ("Gross", ",.0f", "US box office ($M, nominal)", 1e-6),
    "Total US gross, inflation-adjusted ($M)": ("Adj gross", ",.0f", "US box office ($M, today's ticket prices)", 1e-6),
    "Films in release": ("Films", ".0f", "Films in release", 1),
    "Gross per film ($M)": ("Gross per film", ",.0f", "US gross per film in release ($M)", 1e-6),
    "Top film's share of the studio's year (%)": ("Top film share", ".0f", "Top film's share of the studio's gross (%)", 100),
}
YEARLY_NOTES = {
    "Market share (%)": "Share of all US box office that year.",
    "Total US gross ($M)": "Domestic box office of everything the studio had in release, in nominal dollars.",
    "Total US gross, inflation-adjusted ($M)": "Domestic box office restated at current ticket prices, so decades "
                                               "compare fairly.",
    "Films in release": "How many films the studio had in theaters; a shrinking slate means fewer, bigger bets.",
    "Gross per film ($M)": "Total US gross ÷ films in release: is the studio getting more out of each film?",
    "Top film's share of the studio's year (%)": "How much of the studio's year rode on its single biggest film: high "
                                                  "= tentpole-dependent; low = a broad slate.",
}
H2H = {  # measure: (column, median or rate, number format)
    "Median production budget ($M)": ("budget_m", "median", ".0f"),
    "Median worldwide box office ($M)": ("worldwide_m", "median", ".0f"),
    "Median return on budget (×)": ("multiple", "median", ".2f"),
    f"Hit rate (% at ≥{HIT_MULTIPLE}× budget)": ("hit", "rate", ".0f"),
    "Follow-ups (% of films)": ("follow_up", "rate", ".0f"),
    "Median week-2 drop (%, wide releases)": ("drop2", "median", ".0f"),
    "Median IMDb rating": ("imdb", "median", ".1f"),
}


# Labels, producers and co-financiers with enough films here for a deep dive (credited on 13+ films). Miramax (8),
# Blumhouse (8), Illumination (4) and A24 (0) have too few films with analyzable scripts.
COMPANIES = {  # name: (search pattern in the credits, business model, home studio or main partners)
    "Focus Features": ("focus features", "Label", "Universal's specialty label"),
    "Working Title Films": ("working title", "Producer", "Universal-owned producer"),
    "Amblin Entertainment": ("amblin", "Producer", "Spielberg's company; Universal and DreamWorks partner"),
    "Imagine Entertainment": ("imagine ent", "Producer", "Ron Howard and Brian Grazer; long-time Universal partner"),
    "DreamWorks": ("dreamworks", "Producer", "Distributed via Paramount, Disney and Universal over time"),
    "Relativity Media": ("relativity", "Co-financier", "Slate co-financier for Universal and Sony"),
    "Legendary Pictures": ("legendary", "Co-financier", "Co-financed Warner Bros. films, then Universal's"),
    "New Line Cinema": ("new line", "Label", "Warner Bros. label"),
    "Village Roadshow Pictures": ("village roadshow", "Co-financier", "Warner Bros. slate co-financier"),
    "Silver Pictures": ("silver pictures", "Producer", "Joel Silver; Warner Bros."),
    "Castle Rock Entertainment": ("castle rock", "Producer", "Warner Bros."),
    "Heyday Films": ("heyday", "Producer", "Warner Bros. (Harry Potter)"),
    "Dune Entertainment": ("dune ent", "Co-financier", "Fox slate co-financier"),
    "Regency Enterprises": ("regency", "Co-financier", "Fox partner"),
    "Searchlight Pictures": ("searchlight", "Label", "Fox's specialty label (Disney since 2019)"),
    "Scott Free Productions": ("scott free", "Producer", "Ridley and Tony Scott"),
    "Touchstone Pictures": ("touchstone", "Label", "Disney's adult-audience label"),
    "Hollywood Pictures": ("hollywood pictures", "Label", "Disney label"),
    "Jerry Bruckheimer Films": ("bruckheimer", "Producer", "Disney partner"),
    "Pixar": ("pixar", "Animation & IP", "Disney-owned since 2006"),
    "Marvel": ("marvel", "Animation & IP", "Disney-owned since 2009; earlier licensed to Sony and Fox"),
    "Spyglass": ("spyglass", "Co-financier", "Co-financed for Disney, Sony and others"),
    "Columbia Pictures": ("columbia pictures", "Label", "Sony's flagship label"),
    "TriStar Pictures": ("tristar", "Label", "Sony label"),
    "Screen Gems": ("screen gems", "Label", "Sony's genre label"),
    "Summit Entertainment": ("summit ent", "Label", "Lionsgate since 2012 (Twilight)"),
}
COMPANY_TYPES = ["Label", "Producer", "Co-financier", "Animation & IP"]
TYPE_COLOR = dict(zip(COMPANY_TYPES, [CAT[0], CAT[2], CAT[1], CAT[3]]))


@st.cache_resource(show_spinner=False)
def company_members():
    credits = FILMS["companies"].fillna("").str.lower()
    return {k: FILMS.index[credits.str.contains(pat, regex=False)] for k, (pat, _, _) in COMPANIES.items()}


def h2h_value(d, col, how):
    if col == "drop2":
        d = d[d["wide"] == 1]
    return median(d[col]) if how == "median" else d[col].mean() * 100


def studio_lines(d, x, y, pick, tooltip, point_size=60):
    """One line per studio in its own color, labeled at its last point (no legend to decode). Universal is drawn
    heavier; Paramount's white line sits on a charcoal keyline."""
    color = alt.Color("Studio:N", scale=alt.Scale(domain=pick, range=[STUDIO_COLOR[s] for s in pick]), legend=None)
    halo = alt.Chart(d).transform_filter("datum.Studio == 'Paramount'").encode(x=x, y=y)
    main = alt.Chart(d).mark_line(point=line_points(point_size)).encode(
        x=x, y=y, color=color, tooltip=tooltip,
        strokeWidth=alt.condition("datum.Studio == 'Universal'", alt.value(3.2), alt.value(1.9)))
    last = d.sort_values(x.shorthand.split(":")[0]).groupby("Studio").tail(1)
    labels = alt.Chart(last).mark_text(align="left", dx=8, fontSize=11, fontWeight=600,
                                       color=studio_expr("Studio")).encode(x=x, y=y, text="Studio:N")
    return alt.layer(halo.mark_line(color=KEYLINE, strokeWidth=4.4),
                     halo.mark_circle(color=KEYLINE, size=point_size + 45, opacity=1), main, labels)


def heatmap(d, x, y, x_order, y_order, legend_title, fmt_str):
    base = alt.Chart(d).encode(x=alt.X(f"{x}:N", sort=x_order, title=None, axis=alt.Axis(labelAngle=0, orient="top")),
                               y=alt.Y(f"{y}:N", sort=y_order, title=None,
                                       axis=alt.Axis(labelColor=studio_expr("value"), labelFontWeight=600)))
    rect = base.mark_rect(stroke="white", strokeWidth=2, cornerRadius=2).encode(
        color=alt.Color("Value:Q", title=legend_title, scale=alt.Scale(range=["#eef4fe", BLUE]),
                        legend=alt.Legend(orient="right", gradientLength=160)),
        tooltip=[y, x, alt.Tooltip("Value:Q", format=fmt_str, title=legend_title), alt.Tooltip("n:Q", title="Films")])
    cut = d["Value"].quantile(0.7) if len(d) else 0
    text = base.mark_text(fontSize=11).encode(
        text=alt.Text("Value:Q", format=fmt_str),
        color=alt.condition(f"datum.Value > {cut}", alt.value("white"), alt.value(INK)))
    return (rect + text).properties(height=36 * len(y_order) + 24)


@st.fragment
def studio_section():
    section("Studios", "Studio Playbooks",
            "**How it works:** compare the studios year by year on their full slates, then head to head on the films "
            "in this dataset: what each one makes, when it releases, and how it pays off. Each studio keeps its own "
            "color throughout, with Universal in NBCU purple.")

    if SHARE is not None:
        exhibit("Studio vs. studio, year by year (full slates, 1995–2023)")
        avail = [s for s in STUDIO_ORDER if s in set(SHARE["Studio"])]
        c1, c2, c3 = st.columns([2, 3, 1])
        yname = c1.selectbox("Measure", list(YEARLY), key="yearly_metric",
                             help="  \n".join(f"**{k}**: {v}" for k, v in YEARLY_NOTES.items()))
        pick = c2.multiselect("Studios", avail, default=avail, key="yearly_studios")
        smooth = c3.toggle("3-year average", value=False, key="yearly_smooth",
                           help="Single years swing with one franchise release; smoothing shows the underlying trend.")
        if pick:
            col, f, ytitle, mult = YEARLY[yname]
            s = SHARE[SHARE["Studio"].isin(pick)].sort_values("Year").copy()
            s["Value"] = s[col] * mult
            if smooth:
                s["Value"] = s.groupby("Studio")["Value"].transform(lambda x: x.rolling(3, min_periods=1, center=True).mean())
            last_year = int(s["Year"].max())
            avg = s[s["Year"] >= last_year - 4].groupby("Studio")["Value"].mean().sort_values(ascending=False)
            title = f"{avg.index[0]} leads on {yname.split(' (')[0].lower()} over {last_year - 4}–{last_year} " \
                    f"({format(avg.iloc[0], f)})"
            if "Universal" in avg.index and avg.index[0] != "Universal":
                title += f"; Universal ranks #{list(avg.index).index('Universal') + 1} ({format(avg['Universal'], f)})"
            chart = studio_lines(
                s, alt.X("Year:Q", axis=alt.Axis(format="d", tickCount=8), title=None,
                         scale=alt.Scale(domain=[1995, last_year + 3])),
                alt.Y("Value:Q", title=ytitle), pick, point_size=28,
                tooltip=["Studio", alt.Tooltip("Year:Q", format="d"), alt.Tooltip("Value:Q", format=f, title=yname),
                         alt.Tooltip("Films:Q", title="Films in release"), alt.Tooltip("Top film:N", title="Top film that year")])
            show_chart(chart.properties(height=360), title,
                       f"{ytitle}{', 3-year rolling average' if smooth else ''}")
            st.caption(f"*{YEARLY_NOTES[yname]} Every release by each studio that year, not just the films in this "
                       f"dataset. Hover a point for that year's top film. Source: The Numbers, via the case's Excel "
                       f"file (Fox and Lionsgate not included).*")

            lead = SHARE.loc[SHARE.groupby("Year")["Share"].idxmax(), ["Year", "Studio", "Share", "Top film"]]
            wins = lead["Studio"].value_counts()
            cols = st.columns(len(wins))
            for c, (stu, n) in zip(cols, wins.items()):
                c.metric(f"{stu}: years as #1", f"{n}",
                         help="Years on top of US box office share: " +
                              ", ".join(str(y) for y in lead.loc[lead["Studio"] == stu, "Year"]))
            st.caption("*Years (1995–2023) each studio held the largest share of US box office. Hover for the years.*")

    exhibit("Head-to-head: how have the studios' films changed?")
    c1, c2, c3 = st.columns([2, 3, 1])
    hname = c1.selectbox("Measure", list(H2H), key="h2h_metric", help=measure_help(H2H))
    pick = c2.multiselect("Studios (up to 4)", STUDIOS, default=["Universal", "Disney", "Warner Bros."],
                          max_selections=4, key="h2h_studios")
    per = c3.radio("Group years by", ["5 years", "Decade"], key="h2h_period")
    if pick:
        col, how, f = H2H[hname]
        g = "period" if per == "5 years" else "decade"
        recs = []
        for (p, stu), d in FILMS[FILMS["studio"].isin(pick)].groupby([g, "studio"]):
            n = int(d[col].notna().sum()) if col != "drop2" else int(d.loc[d["wide"] == 1, col].notna().sum())
            if n >= 3:
                recs.append({"Period": p, "Studio": stu, "Value": h2h_value(d, col, how), "Films": n})
        agg = pd.DataFrame(recs)
        overall = pd.Series({stu: h2h_value(FILMS[FILMS["studio"] == stu], col, how) for stu in pick})
        best = overall.idxmax()
        title = f"{best} is highest on {hname.split(' (')[0].lower()} across all years ({format(overall[best], f)})"
        if "Universal" in pick and best != "Universal":
            title += f"; Universal {format(overall['Universal'], f)}"
        periods = sorted(agg["Period"].unique())
        chart = studio_lines(agg, alt.X("Period:N", sort=periods, title=None, axis=alt.Axis(labelAngle=0),
                                        scale=alt.Scale(paddingOuter=0.35)),
                             alt.Y("Value:Q", title=hname), pick,
                             tooltip=["Studio", "Period", alt.Tooltip("Value:Q", format=f, title=hname), "Films"])
        show_chart(chart.properties(height=320), title, f"{hname}, by {'5-year period' if per == '5 years' else 'decade'}")
        st.caption("*Each point summarizes that studio's films from the period in this dataset; points from fewer than 3 "
                   "films are hidden (hover for counts). There are a few films per studio per year, so compare broad "
                   "directions, not small gaps.*")


    exhibit("Beyond the majors: labels, producers and co-financiers")
    st.markdown("Majors rarely work alone. **Labels** are brands inside a major aimed at a different audience (Focus "
                "at Universal, New Line at Warner). **Producers** develop and make films under a studio deal (Amblin, "
                "Imagine and Working Title with Universal). **Co-financiers** put up part of a slate's budget for a share "
                "of its returns (Village Roadshow with Warner, Dune with Fox). Which model has paid off?")
    members = company_members()
    c1, c2 = st.columns([2, 3])
    cname = c1.selectbox("Measure", list(H2H), index=2, key="co_metric", help=measure_help(H2H))
    picks = c2.multiselect("Companies (up to 12)", list(COMPANIES), max_selections=12, key="co_pick",
                           default=["Focus Features", "Working Title Films", "Amblin Entertainment", "Imagine Entertainment",
                                    "New Line Cinema", "Village Roadshow Pictures", "Legendary Pictures",
                                    "Dune Entertainment", "Searchlight Pictures", "Pixar"])
    if picks:
        col, how, f = H2H[cname]
        d = pd.DataFrame([{"Company": k, "Type": COMPANIES[k][1], "Home": COMPANIES[k][2], "Films": len(members[k]),
                           "Value": h2h_value(FILMS.loc[members[k]], col, how)} for k in picks]).dropna(subset=["Value"])
        d = d.sort_values("Value", ascending=False)
        allv = h2h_value(FILMS, col, how)
        measure = cname.split(" (")[0].replace("Median ", "").lower()
        title = (f"{d.iloc[0]['Company']} leads on {measure} ({format(d.iloc[0]['Value'], f)}) vs. "
                 f"{format(allv, f)} for all films")
        base = alt.Chart(d).encode(y=alt.Y("Company:N", sort=list(d["Company"]), title=None))
        bars = base.mark_bar(stroke="white", strokeWidth=1, cornerRadiusEnd=3, height={"band": 0.7}).encode(
            x=alt.X("Value:Q", title=cname),
            color=alt.Color("Type:N", scale=alt.Scale(domain=COMPANY_TYPES, range=[TYPE_COLOR[t] for t in COMPANY_TYPES]),
                            legend=alt.Legend(orient="top", title=None)),
            tooltip=["Company", "Type", "Home", alt.Tooltip("Value:Q", format=f, title=cname), "Films"])
        text = base.mark_text(align="left", dx=5, fontSize=11, color=INK).encode(
            x="Value:Q", text=alt.Text("Value:Q", format=f))
        ref = alt.Chart(pd.DataFrame({"x": [allv], "label": ["All films"]}))
        rule = ref.mark_rule(color=INK, strokeDash=[5, 4], strokeWidth=1.5).encode(x="x:Q")
        rlab = ref.mark_text(align="left", dx=4, dy=-4, fontSize=11, color=INK, baseline="bottom").encode(
            x="x:Q", y=alt.value(0), text="label:N")
        show_chart((bars + text + rule + rlab).properties(height=34 * len(d) + 30), title,
                   f"{cname}, films credited to each company, all years; dashed line = all films")
        st.caption("*Credits come from each film's production companies, so a film can count for several (a Universal film "
                   "produced by Amblin and Imagine counts for both). Each company's films are the ones in this dataset, "
                   "not its full output. Hover a bar for its home studio and film count. Miramax, Blumhouse and "
                   "Illumination have too few films here (4–8) for a fair comparison.*")

        st.markdown("**Deep dive**")
        one = st.selectbox("Company", list(COMPANIES), index=0, key="co_one",
                           format_func=lambda k: f"{k} ({len(members[k])} films)")
        dd = FILMS.loc[members[one]]
        st.caption(f"{COMPANIES[one][1]} · {COMPANIES[one][2]} · {len(dd)} films in this dataset, "
                   f"{int(dd['year'].min())}–{int(dd['year'].max())}")
        cols = st.columns(5)
        cards = [("Median budget ($M)", "Median production budget ($M)"),
                 ("Median worldwide ($M)", "Median worldwide box office ($M)"),
                 ("Median return (×)", "Median return on budget (×)"),
                 ("Hit rate (%)", f"Hit rate (% at ≥{HIT_MULTIPLE}× budget)"),
                 ("Follow-ups (%)", "Follow-ups (% of films)")]
        for c, (lab, key) in zip(cols, cards):
            mc, mh, mf = H2H[key]
            v, a = h2h_value(dd, mc, mh), h2h_value(FILMS, mc, mh)
            c.metric(lab, fmt(v, "{:" + mf + "}"),
                     delta=None if pd.isna(v) else f"{format(v - a, '+' + mf)} vs. all films",
                     delta_color="off" if mc in ("budget_m", "follow_up") else "normal")
        left, right = st.columns(2)
        with left:
            dist = pd.DataFrame({"Studio": STUDIOS, "Films": [int((dd["studio"] == x).sum()) for x in STUDIOS]})
            dist = dist[dist["Films"] > 0].sort_values("Films", ascending=False)
            top = dist.iloc[0]
            chart = alt.Chart(dist).mark_bar(cornerRadiusEnd=3, strokeWidth=1.2, height={"band": 0.7}).encode(
                y=alt.Y("Studio:N", sort=list(dist["Studio"]), title=None,
                        axis=alt.Axis(labelColor=studio_expr("value"), labelFontWeight=600)),
                x=alt.X("Films:Q", title="Films", axis=alt.Axis(tickMinStep=1)),
                color=alt.Color("Studio:N", scale=alt.Scale(domain=list(dist["Studio"]),
                                                            range=[STUDIO_COLOR[x] for x in dist["Studio"]]), legend=None),
                stroke=alt.condition("datum.Studio == 'Paramount'", alt.value(KEYLINE), alt.value("white")),
                tooltip=["Studio", "Films"])
            show_chart(chart.properties(height=34 * len(dist) + 20),
                       f"{top['Studio']} released {int(top['Films'])} of its {len(dd)} films",
                       "Which major distributed the company's films")
        with right:
            mix = pd.DataFrame({"Genre": GENRES,
                                "This company": [(dd["genre"] == g).mean() * 100 for g in GENRES],
                                "All films": [(FILMS["genre"] == g).mean() * 100 for g in GENRES]})
            mix["Gap"] = mix["This company"] - mix["All films"]
            over = mix.loc[mix["Gap"].idxmax()]
            long = mix.melt(id_vars="Genre", value_vars=["This company", "All films"], var_name="Series",
                            value_name="Share")
            long["Series"] = long["Series"].replace({"This company": one})
            chart = alt.Chart(long).mark_bar(stroke="white", strokeWidth=1, cornerRadiusEnd=3).encode(
                y=alt.Y("Genre:N", sort=GENRES, title=None), yOffset=alt.YOffset("Series:N", sort=[one, "All films"]),
                x=alt.X("Share:Q", title="% of films"),
                color=alt.Color("Series:N", scale=alt.Scale(domain=[one, "All films"], range=[BLUE, SOFT]),
                                legend=alt.Legend(orient="top", title=None)),
                tooltip=["Series", "Genre", alt.Tooltip("Share:Q", format=".0f", title="% of films")])
            show_chart(chart.properties(height=40 * len(GENRES)),
                       f"Over-indexes on {over['Genre'].lower()}: {over['This company']:.0f}% of its films vs. "
                       f"{over['All films']:.0f}% overall", "Genre mix vs. all films in the dataset")
        show_table(dd.sort_values("worldwide", ascending=False)[
            ["title", "year", "studio", "genre", "kind", "budget_m", "worldwide_m", "multiple", "imdb"]],
            height=300, column_config={
                "title": "Film", "year": st.column_config.NumberColumn("Year", format="%d"), "studio": "Distributor",
                "genre": "Genre", "kind": "Original / follow-up",
                "budget_m": st.column_config.NumberColumn("Budget ($M)", format="%.0f"),
                "worldwide_m": st.column_config.NumberColumn("Worldwide ($M)", format="%.0f"),
                "multiple": st.column_config.NumberColumn("Return (×)", format="%.2f"),
                "imdb": st.column_config.NumberColumn("IMDb", format="%.1f")})

    exhibit("Studio scorecard")
    rows = []
    for stu in STUDIOS:
        d = FILMS[FILMS["studio"] == stu]
        rows.append({"Studio": stu, "Films": len(d), "Median budget ($M)": median(d["budget_m"]),
                     "Median worldwide ($M)": median(d["worldwide_m"]), "Median return (×)": median(d["multiple"]),
                     f"Hit rate (≥{HIT_MULTIPLE}×)": d["hit"].mean() * 100, "Follow-ups (%)": d["follow_up"].mean() * 100,
                     "Median week-2 drop (%)": median(d.loc[d["wide"] == 1, "drop2"]), "Median IMDb": median(d["imdb"])})
    card = pd.DataFrame(rows).style.format({
        "Median budget ($M)": "{:.0f}", "Median worldwide ($M)": "{:.0f}", "Median return (×)": "{:.2f}",
        f"Hit rate (≥{HIT_MULTIPLE}×)": "{:.0f}%", "Follow-ups (%)": "{:.0f}%", "Median week-2 drop (%)": "{:.0f}%",
        "Median IMDb": "{:.1f}"}).map(lambda v: f"color: {studio_text(v)}; font-weight: 600", subset=["Studio"])
    show_table(card)
    st.caption("*These are the studio's films **in this dataset** (films whose scripts were available), not its full "
               "slate, so read them as a sample.*")

    exhibit("What each studio makes")
    metric = st.radio("Color the grid by", ["Share of the studio's films", "Median return on budget"], horizontal=True,
                      key="studio_genre_metric")
    is_share = metric.startswith("Share")
    recs = []
    for stu in STUDIOS:
        d = FILMS[FILMS["studio"] == stu]
        for g in GENRES:
            dg = d[d["genre"] == g]
            if is_share:
                recs.append({"Studio": stu, "Genre": g, "Value": len(dg) / len(d) * 100, "n": len(dg)})
            elif len(dg) >= 3:
                recs.append({"Studio": stu, "Genre": g, "Value": median(dg["multiple"]), "n": len(dg)})
    d = pd.DataFrame(recs)
    if is_share:
        u = d[d["Studio"] == "Universal"].sort_values("Value", ascending=False).iloc[0]
        title = f"Universal's biggest bet is {u['Genre'].lower()}: {u['Value']:.0f}% of its films"
    else:
        top = d.sort_values("Value", ascending=False).iloc[0]
        title = f"Best-paying cell: {top['Studio']} {top['Genre'].lower()}, {top['Value']:.1f}× budget (median)"
    show_chart(heatmap(d, "Genre", "Studio", GENRES, STUDIOS, "% of films" if is_share else "Median return (×)",
                       ".0f" if is_share else ".1f"), title,
               "Share of each studio's films, by genre" if is_share else "Median worldwide ÷ budget, by studio and genre",
               panel=False)
    st.caption("*Read across a row: where does this studio place its bets?* " +
               ("*Each row sums to 100%.*" if is_share else
                "*Blank = fewer than 3 films. Compare the playbook (share) with the payoff (return): a studio that makes "
                "few films in a high-return genre may be leaving money on the table.*"))

    exhibit("When each studio releases")
    recs = []
    for stu in STUDIOS:
        d = FILMS[FILMS["studio"] == stu]
        for i, m in enumerate(MONTHS, start=1):
            n = int((d["month"] == i).sum())
            recs.append({"Studio": stu, "Month": m, "Value": n / len(d) * 100, "n": n})
    d = pd.DataFrame(recs)
    u = d[d["Studio"] == "Universal"].sort_values("Value", ascending=False).iloc[0]
    show_chart(heatmap(d, "Month", "Studio", MONTHS, STUDIOS, "% of films", ".0f"),
               f"Universal releases most often in {u['Month']} ({u['Value']:.0f}% of its films)",
               "Share of each studio's films released in each month (US)", panel=False)
    st.caption("*Summer (May–August) and the holidays (November–December) are the tentpole windows; January, February "
               "and September are traditionally the quiet months.*")


# --- 3. Trends & Franchise Fatigue -------------------------------------------------------------------------------
METRICS = {
    "Median worldwide box office ($M)": ("worldwide_m", "median", ".0f"),
    "Median opening weekend ($M)": ("opening_m", "median", ".0f"),
    "Median return on budget (×)": ("multiple", "median", ".2f"),
    f"Hit rate (% at ≥{HIT_MULTIPLE}× budget)": ("hit", "rate", ".0f"),
    "Median week-2 drop (%, wide releases)": ("drop2", "median", ".0f"),
    "Median production budget ($M)": ("budget_m", "median", ".0f"),
    "Median IMDb rating": ("imdb", "median", ".1f"),
    "Number of films": ("id", "count", ".0f"),
}
GROUPS = {"Release period (5 years)": "period", "Season": "season", "Month": "month_name", "Genre": "genre",
          "Rating": "mpaa", "Runtime": "runtime_band", "Original vs. follow-up": "kind"}


@st.fragment
def trends_section():
    section("Trends", "Trends & Franchise Fatigue",
            "**How it works:** pick a measure and a way to slice the films. Medians are used instead of averages so a "
            "single billion-dollar outlier can't drag a whole group.")
    c1, c2, c3, c4 = st.columns(4)
    mname = c1.selectbox("Measure", list(METRICS), help=measure_help(METRICS))
    gname = c2.selectbox("Slice by", list(GROUPS))
    stu = c3.selectbox("Studio", ["All studios"] + STUDIOS, key="trend_studio")
    kind = c4.selectbox("Films", ["All films", "Originals only", "Follow-ups only"], key="trend_kind")
    col, how, f = METRICS[mname]
    d = FILMS.copy()
    d["month_name"] = d["month"].map(lambda m: MONTHS[int(m) - 1] if pd.notna(m) else None)
    d["runtime_band"] = pd.cut(d["runtime"], [0, 90, 105, 120, 135, 999],
                               labels=["Under 90 min", "90–104", "105–119", "120–134", "135+"]).astype(str)
    if stu != "All studios":
        d = d[d["studio"] == stu]
    if kind != "All films":
        d = d[d["kind"] == ("Original" if kind == "Originals only" else "Follow-up")]
    if col == "drop2":
        d = d[d["wide"] == 1]
    g = GROUPS[gname]
    order = {"season": SEASON_ORDER, "month_name": MONTHS, "genre": GENRES, "mpaa": ["G", "PG", "PG-13", "R", "Unrated/unknown"],
             "runtime_band": ["Under 90 min", "90–104", "105–119", "120–134", "135+"],
             "kind": ["Original", "Follow-up"]}.get(g, sorted(d[g].dropna().unique()))
    grp = d.groupby(g)
    out = grp[col].median() if how == "median" else (grp[col].mean() * 100 if how == "rate" else grp[col].count())
    agg = pd.DataFrame({"Group": out.index, "Value": out.values, "Films": grp[col].count().reindex(out.index).values})
    agg = agg[agg["Group"].isin(order) & (agg["Films"] >= 3)]
    if agg.empty:
        st.info("Too few films for this combination. Broaden the studio or film filter.")
    else:
        hi, lo = agg.loc[agg["Value"].idxmax()], agg.loc[agg["Value"].idxmin()]
        measure = mname.replace("Median ", "").split(" (")[0].lower()
        title = f"{hi['Group']} leads on {measure} ({format(hi['Value'], f)}); {lo['Group']} trails ({format(lo['Value'], f)})"
        agg["Top"] = agg["Group"] == hi["Group"]
        bars = alt.Chart(agg).mark_bar(stroke=KEYLINE if stu == "Paramount" else "white", strokeWidth=1,
                                       cornerRadiusEnd=3, color=STUDIO_COLOR.get(stu, BLUE)).encode(
            x=alt.X("Group:N", sort=order, title=None, axis=alt.Axis(labelAngle=0 if len(agg) <= 9 else -40)),
            y=alt.Y("Value:Q", title=mname),
            opacity=alt.condition("datum.Top", alt.value(1), alt.value(0.55)),
            tooltip=[alt.Tooltip("Group:N", title=gname), alt.Tooltip("Value:Q", format=f, title=mname), "Films"])
        labels = bars.mark_text(dy=-7, fontSize=11, color=INK).encode(text=alt.Text("Value:Q", format=f))
        show_chart((bars + labels).properties(height=330), title,
                   f"{mname} by {gname.lower()} · {stu.lower() if stu == 'All studios' else stu} · {kind.lower()}")
        st.caption(f"*{len(d):,} films. Groups with fewer than 3 films are hidden; hover a bar for its film count. Dollar "
                   f"figures are nominal, so later periods look bigger partly because tickets cost more; return on "
                   f"budget and hit rate are ratios and compare cleanly across decades.*")

    exhibit("Franchise fatigue: do follow-ups burn out faster?")
    st.markdown("The case's Table 1 shows Marvel films losing two-thirds of their audience in week two, against an "
                "industry average near 40% (and about 16% in the 1980s, 22% in the 1990s). Here is the same question "
                "across every wide release in the data.")
    w = FILMS[FILMS["wide"] == 1]
    dec = w.groupby(["decade", "kind"])["drop2"].agg(["median", "count"]).reset_index()
    dec = dec[dec["count"] >= 5]
    both = dec.pivot(index="decade", columns="kind", values="median").dropna()
    latest = both.index.max()
    title = (f"Follow-ups lose {both.loc[latest, 'Follow-up']:.0f}% of their audience in week two vs. "
             f"{both.loc[latest, 'Original']:.0f}% for originals ({latest}); in the {both.index.min()}: "
             f"{both.iloc[0]['Follow-up']:.0f}% vs. {both.iloc[0]['Original']:.0f}%")
    kinds = ["Original", "Follow-up"]
    bars = alt.Chart(dec).mark_bar(stroke="white", strokeWidth=0.8, cornerRadiusEnd=3).encode(
        x=alt.X("decade:N", title=None, axis=alt.Axis(labelAngle=0)),
        xOffset=alt.XOffset("kind:N", sort=kinds),
        y=alt.Y("median:Q", title="Median % of week-1 gross lost in week 2"),
        color=alt.Color("kind:N", sort=kinds, scale=alt.Scale(domain=kinds, range=[SOFT, BLUE]),
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=[alt.Tooltip("decade:N", title="Decade"), alt.Tooltip("kind:N", title="Film"),
                 alt.Tooltip("median:Q", format=".0f", title="Median drop (%)"), alt.Tooltip("count:Q", title="Films")])
    labels = bars.mark_text(dy=-7, fontSize=11, color=INK).encode(text=alt.Text("median:Q", format=".0f"))
    show_chart((bars + labels).properties(height=310), title, "Median week-1 to week-2 drop, wide releases, by decade")
    st.caption("*Wide releases only (600+ opening theaters), decades with at least 5 films per group. The drop is week 1 "
               "to week 2 (7-day grosses), a little gentler than the case's weekend-to-weekend figures. Fans who waited "
               "for the sequel show up opening weekend, leaving fewer new viewers for week two.*")

    o, fu = FILMS[FILMS["follow_up"] == 0], FILMS[FILMS["follow_up"] == 1]
    cols = st.columns(4)
    cols[0].metric("Median worldwide · follow-up vs. original",
                   f"{money(median(fu['worldwide']))} vs. {money(median(o['worldwide']))}")
    cols[1].metric("Median return · follow-up vs. original",
                   f"{median(fu['multiple']):.2f}× vs. {median(o['multiple']):.2f}×")
    cols[2].metric(f"Hit rate (≥{HIT_MULTIPLE}×) · follow-up vs. original",
                   f"{fu['hit'].mean() * 100:.0f}% vs. {o['hit'].mean() * 100:.0f}%")
    cols[3].metric("Median IMDb · follow-up vs. original", f"{median(fu['imdb']):.1f} vs. {median(o['imdb']):.1f}")
    takeaway(f"<b>So what:</b> a franchise buys a bigger, safer opening and a shorter run. Descriptive only: studios "
             f"greenlight sequels to their biggest hits, so follow-ups start from proven demand ({len(fu)} follow-ups "
             f"vs. {len(o)} originals).")

    st.markdown("**The steepest week-2 drops among follow-ups since 2000**")
    t = fu[(fu["wide"] == 1) & (fu["year"] >= 2000)].nlargest(12, "drop2")
    show_table(t[["title", "year", "studio", "week1_m", "week2_m", "drop2", "multiple"]], column_config={
        "title": "Film", "year": st.column_config.NumberColumn("Year", format="%d"), "studio": "Studio",
        "week1_m": st.column_config.NumberColumn("Week 1 ($M)", format="%.1f"),
        "week2_m": st.column_config.NumberColumn("Week 2 ($M)", format="%.1f"),
        "drop2": st.column_config.NumberColumn("Drop (%)", format="%.0f%%"),
        "multiple": st.column_config.NumberColumn("Return (× budget)", format="%.2f×")})
    st.caption("*Compare with the case's Table 1. A steep drop doesn't have to mean a flop: check the return column.*")


# --- 4. Script DNA ---------------------------------------------------------------------------------------------------
EMOTION_DEFS = {
    "Joy": "happiness, humor, warmth and relief", "Surprise": "shock, astonishment and reveals",
    "Fear": "threat, danger, dread and suspense", "Sadness": "loss, grief, vulnerability and regret",
    "Anger": "conflict, confrontation and hostility", "Disgust": "revulsion and moral violation",
    "Neutral": "plain, informational dialogue with no strong emotion",
}


def arc_long(d, emotion):
    return pd.DataFrame([{"Act": a, "Share": d[f"{emotion}_{a}"].mean() * 100} for a in range(1, 6)])


@st.fragment
def script_section():
    section("Scripts", "Script DNA",
            "**How it works:** each screenplay was split into five acts, and a transformer-based NLP emotion model read "
            "every line of dialogue and scored it on seven emotions (joy, surprise, fear, sadness, anger, disgust, "
            "neutral). A film's value for an act is the share of that act's dialogue carrying each emotion, so a "
            "script becomes an **emotional arc** you can compare across genres, hits and misses.")
    with st.expander("What each emotion captures"):
        st.markdown("\n".join(f"- **{e}**: {d}" for e, d in EMOTION_DEFS.items()))

    exhibit("Emotional arcs by genre")
    c1, c2 = st.columns([1, 2])
    emo = c1.selectbox("Emotion", EMOTIONS, index=EMOTIONS.index("Fear"), key="dna_emo")
    gs = c2.multiselect("Genres (up to 4)", GENRE_ORDER, default=["Action", "Comedy", "Drama", "Horror"],
                        max_selections=4, key="dna_genres")
    recs = []
    for g in gs:
        a = arc_long(FILMS[FILMS["genre"] == g], emo)
        a["Series"] = g
        recs.append(a)
    allf = arc_long(FILMS, emo)
    allf["Series"] = "All films"
    d = pd.concat(recs + [allf])
    peak = d[d["Series"] != "All films"].sort_values("Share", ascending=False).iloc[0] if recs else d.iloc[0]
    title = (f"{peak['Series']} carries the most {emo.lower()}, peaking in Act {int(peak['Act'])} "
             f"({peak['Share']:.1f}% of dialogue)")
    dom = gs + ["All films"]
    chart = alt.Chart(d).mark_line(point=line_points(), strokeWidth=2.25).encode(
        x=act_axis(), y=alt.Y("Share:Q", title=f"% of dialogue read as {emo.lower()}"),
        color=alt.Color("Series:N", scale=alt.Scale(domain=dom, range=[GENRE_COLOR[g] for g in gs] + [GRAY]),
                        legend=alt.Legend(orient="top", title=None)),
        strokeDash=alt.condition("datum.Series == 'All films'", alt.value([5, 4]), alt.value([1, 0])),
        tooltip=["Series", "Act", alt.Tooltip("Share:Q", format=".1f", title="% of dialogue")])
    show_chart(chart.properties(height=330), title, f"Average share of each act's dialogue read as {emo.lower()}")
    st.caption(f"*{emo} = {EMOTION_DEFS[emo]}. Dashed gray = all films. Genres have signature shapes; a new film that "
               f"breaks its genre's shape is taking a creative risk.*")

    exhibit("Hits vs. misses: where do their scripts differ?")
    c1, c2, c3 = st.columns(3)
    scope = c1.selectbox("Films", ["All films", "Universal only"] + [f"Genre: {g}" for g in GENRE_ORDER], key="dna_scope")
    emo2 = c2.selectbox("Emotion to plot", EMOTIONS, index=0, key="dna_emo2")
    d = FILMS if scope == "All films" else (FILMS[FILMS["studio"] == "Universal"] if scope == "Universal only"
                                            else FILMS[FILMS["genre"] == scope.split(": ")[1]])
    hits, misses = quartile_groups(d)
    c3.metric("Films compared", f"{len(hits)} hits · {len(misses)} misses",
              help="Hits = top 25% of return on budget within the films selected; misses = bottom 25%.")
    if len(hits) < 10:
        st.info("Too few films with a return on budget in this selection. Choose a broader one.")
        return
    a, b = arc_long(hits, emo2), arc_long(misses, emo2)
    gap = a["Share"] - b["Share"]
    k = int(gap.abs().idxmax())
    title = (f"Hits carry {'more' if gap[k] > 0 else 'less'} {emo2.lower()} than misses, most in Act {k + 1} "
             f"({gap[k]:+.1f} pts)")
    a["Group"], b["Group"] = "Hits (top 25% return)", "Misses (bottom 25% return)"
    groups = ["Hits (top 25% return)", "Misses (bottom 25% return)"]
    chart = alt.Chart(pd.concat([a, b])).mark_line(point=line_points(), strokeWidth=2.25).encode(
        x=act_axis(), y=alt.Y("Share:Q", title=f"% of dialogue read as {emo2.lower()}"),
        color=alt.Color("Group:N", scale=alt.Scale(domain=groups, range=[BLUE, INK]),
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=["Group", "Act", alt.Tooltip("Share:Q", format=".1f", title="% of dialogue")])
    show_chart(chart.properties(height=290), title, f"{scope}: {emo2.lower()} across the five acts, hits vs. misses")

    rows = []
    for e in EMOTIONS:
        for act in range(1, 6):
            c = f"{e}_{act}"
            rows.append({"Emotion · act": f"{e} · {ACTS[act]}", "Hits": hits[c].mean() * 100,
                         "Misses": misses[c].mean() * 100,
                         "Difference": (hits[c].mean() - misses[c].mean()) * 100, "P-value": welch_p(hits[c], misses[c])})
    for c, lab in [("emo_intensity", "Emotional intensity (index)"), ("emo_swing", "Emotional swing across acts (index)"),
                   ("emo_novelty", "Emotional novelty vs. earlier films (index)"),
                   ("narr_novelty", "Story novelty vs. earlier films (index)")]:
        rows.append({"Emotion · act": lab, "Hits": hits[c].mean(), "Misses": misses[c].mean(),
                     "Difference": hits[c].mean() - misses[c].mean(), "P-value": welch_p(hits[c], misses[c])})
    diff = pd.DataFrame(rows).sort_values("P-value").head(10)
    diff["Direction"] = np.where(diff["Difference"] > 0, "More in hits", "More in misses")
    st.markdown("**The ten biggest script differences between hits and misses**")
    show_table(diff[["Emotion · act", "Direction", "Hits", "Misses", "Difference", "P-value"]],
               column_config={"Hits": st.column_config.NumberColumn("Hits (% of dialogue)", format="%.2f"),
                              "Misses": st.column_config.NumberColumn("Misses (% of dialogue)", format="%.2f"),
                              "Difference": st.column_config.NumberColumn("Difference (pts)", format="%+.2f"),
                              "P-value": st.column_config.NumberColumn(format="%.3f")})
    st.caption("*Ranked by p-value (Welch t-test). Rows marked (index) are scores, not percentages. With 39 comparisons, "
               "about 4 will reach p < 0.10 by pure chance, so trust a difference more when it is far below 0.10, shows "
               "up in neighboring acts, or makes story sense. Descriptive only: hits also differ in budget, genre and "
               "stars, which the Predict tab holds constant.*")

    exhibit("Does a fresh script pay off?")
    c1, c2 = st.columns([1, 3])
    nov = c1.radio("Novelty measure", ["Emotional novelty", "Story novelty"], key="dna_nov",
                   help="How far a film's emotional profile (or its story language) sits from earlier films in the data.")
    col = "emo_novelty" if nov == "Emotional novelty" else "narr_novelty"
    q = d[[col, "multiple", "hit"]].dropna()
    if len(q) >= 25:
        names = ["1 · Most familiar", "2", "3", "4", "5 · Most novel"]
        q = q.assign(Quintile=pd.qcut(q[col].rank(method="first"), 5, labels=names).astype(str))
        agg = q.groupby("Quintile").agg(Return=("multiple", "median"), Hit=("hit", "mean"), Films=("hit", "size")).reset_index()
        agg["Hit"] *= 100
        first, last = agg.set_index("Quintile").loc[names[0], "Return"], agg.set_index("Quintile").loc[names[-1], "Return"]
        title = f"The most novel fifth returns {last:.2f}× budget vs. {first:.2f}× for the most familiar"
        bars = alt.Chart(agg).mark_bar(color=BLUE, cornerRadiusEnd=3).encode(
            x=alt.X("Quintile:N", sort=names, title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Return:Q", title="Median return on budget (×)"),
            tooltip=["Quintile", alt.Tooltip("Return:Q", format=".2f", title="Median return (×)"),
                     alt.Tooltip("Hit:Q", format=".0f", title=f"Hit rate (%, ≥{HIT_MULTIPLE}×)"), "Films"])
        with c2:
            show_chart((bars + bars.mark_text(dy=-7, fontSize=11, color=INK).encode(
                text=alt.Text("Return:Q", format=".2f"))).properties(height=270), title,
                f"{scope}: median return on budget by {nov.lower()} (five equal groups)")
        st.caption("*A rising pattern means audiences rewarded freshness; a falling one means the familiar template paid. "
                   "Uses the films selected in 'Hits vs. misses' above.*")

    exhibit("Movie Emotions finder")
    st.markdown("Pick the emotions you want in a new film and see which existing films best match them. Useful for "
                "finding comps for a pitch.")
    c1, c2 = st.columns([2, 1])
    want = c1.multiselect("Emotions", EMOTIONS, default=["Joy", "Surprise"], key="finder_emo")
    where = c2.selectbox("Where in the film", ["Whole film"] + list(ACTS.values()), key="finder_act")
    if want:
        acts = range(1, 6) if where == "Whole film" else [list(ACTS.values()).index(where) + 1]
        r = FILMS[["title", "year", "studio", "genre", "worldwide_m", "multiple"]].copy()
        for e in want:
            r[e] = FILMS[[f"{e}_{a}" for a in acts]].mean(axis=1) * 100
        r["Match score"] = r[want].mean(axis=1)
        show_table(r.sort_values("Match score", ascending=False).head(25)[
            ["title", "year", "studio", "genre", "Match score"] + want + ["worldwide_m", "multiple"]],
            height=320, column_config={"title": "Film", "year": st.column_config.NumberColumn("Year", format="%d"),
                                       "worldwide_m": st.column_config.NumberColumn("Worldwide ($M)", format="%.0f"),
                                       "multiple": st.column_config.NumberColumn("Return (×)", format="%.2f"),
                                       "Match score": st.column_config.NumberColumn(format="%.1f"),
                                       **{e: st.column_config.NumberColumn(f"{e} (%)", format="%.1f") for e in want}})
        st.caption("*Match score = average % of dialogue across the chosen emotions. Descriptive only: a high score says "
                   "what a film feels like, not whether it worked; check its return.*")


# --- 5. Wall Street Reaction --------------------------------------------------------------------------------------
def ci_bars(d, order, colors, ytitle):
    base = alt.Chart(d).encode(x=alt.X("Group:N", sort=order, title=None, axis=alt.Axis(labelAngle=0)))
    bars = base.mark_bar(stroke="white", strokeWidth=0.8, cornerRadiusEnd=3, size=46).encode(
        y=alt.Y("Mean:Q", title=ytitle),
        color=alt.Color("Group:N", scale=alt.Scale(domain=list(colors), range=list(colors.values())), legend=None),
        stroke=alt.condition("datum.Group == 'Paramount'", alt.value(KEYLINE), alt.value("white")),
        strokeWidth=alt.value(1.2),
        tooltip=["Group", alt.Tooltip("Mean:Q", format="+.2f", title="Average (%)"),
                 alt.Tooltip("Low:Q", format="+.2f", title="95% CI low"),
                 alt.Tooltip("High:Q", format="+.2f", title="95% CI high"), "Films"])
    whisk = base.mark_rule(color="#4b5563", strokeWidth=1.5).encode(y="Low:Q", y2="High:Q")
    zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#9ca3af").encode(y="y:Q")
    return (bars + whisk + zero).properties(height=290)


PARENT_METRICS = {  # label: (parents.json column, unit, how it's computed)
    "Revenue": ("Revenue", "$B per year", "total revenue over the fiscal year"),
    "Profit margin": ("Margin", "%", "net income ÷ revenue"),
    "ROA": ("ROA", "% per quarter", "net income ÷ total assets"),
    "Leverage": ("Leverage", "%", "debt ÷ total assets"),
}
PARENT_FILM_COL = {"Revenue": "margin", "Profit margin": "margin", "ROA": "roa", "Leverage": "leverage"}


@st.fragment
def wallstreet_section():
    section("Parent companies", "Wall Street & Parent Companies",
            "**How it works:** a studio is a planet orbiting a much bigger parent (Universal belongs to Comcast). Here we "
            "ask whether investors noticed a film. The **stock reaction** is the parent's return beyond the overall "
            "market's from release day to five trading days later (a cumulative abnormal return). The **earnings-call "
            "tone** is how upbeat executives sounded on the quarter's call, scored by FinBERT, a finance-tuned "
            "language model. Paramount 2006–2019 and Universal before 2011 are N/A: the source data measured them on "
            "CBS and Comcast, which didn't own those studios then (Viacom and GE did).")
    c1, c2 = st.columns([1, 3])
    stu = c1.selectbox("Studio", ["All studios"] + STUDIOS, key="ws_studio")
    d = FILMS if stu == "All studios" else FILMS[FILMS["studio"] == stu]
    car = d["car5"].dropna()
    with c2:
        cols = st.columns(3)
        cols[0].metric("Films with stock data", f"{len(car):,}")
        cols[1].metric("Average 5-day reaction", f"{car.mean() * 100:+.2f}%")
        cols[2].metric("Releases followed by a gain", f"{(car > 0).mean() * 100:.0f}%")

    exhibit("Did investors reward hits and punish misses?")
    hits, misses = quartile_groups(d)
    middle = d[~d.index.isin(hits.index) & ~d.index.isin(misses.index) & d["multiple"].notna()]
    recs = []
    for name, g in [("Misses (bottom 25% return)", misses), ("Middle 50%", middle), ("Hits (top 25% return)", hits)]:
        m, lo, hi, n = mean_ci(g["car5"] * 100)
        recs.append({"Group": name, "Mean": m, "Low": lo, "High": hi, "Films": n})
    r = pd.DataFrame(recs).set_index("Group")
    title = (f"Parent stocks moved {r.loc['Hits (top 25% return)', 'Mean']:+.2f}% after hits vs. "
             f"{r.loc['Misses (bottom 25% return)', 'Mean']:+.2f}% after misses")
    show_chart(ci_bars(r.reset_index(), list(r.index), {"Misses (bottom 25% return)": INK, "Middle 50%": SOFT,
                                                        "Hits (top 25% return)": BLUE}, "Average 5-day stock reaction (%)"),
               title, f"{stu}: parent's abnormal stock return, release day to +5 trading days, with 95% intervals")
    st.caption("*Whiskers = 95% confidence interval; a whisker crossing zero means the reaction can't be told apart from "
               "none. Investors only see the opening weekend in this window, and one film is small next to a "
               "conglomerate's broadband, parks and TV businesses, so expect small, noisy effects. Source: CRSP.*")

    exhibit("Stock reaction by studio")
    recs = []
    for s in STUDIOS:
        m, lo, hi, n = mean_ci(FILMS.loc[FILMS["studio"] == s, "car5"] * 100)
        recs.append({"Group": s, "Mean": m, "Low": lo, "High": hi, "Films": n})
    r = pd.DataFrame(recs)
    best = r.loc[r["Mean"].idxmax()]
    u = r.set_index("Group").loc["Universal", "Mean"]
    show_chart(ci_bars(r, STUDIOS, {s: STUDIO_COLOR[s] for s in STUDIOS},
                       "Average 5-day stock reaction (%)"),
               f"{best['Group']} releases drew the warmest stock reaction ({best['Mean']:+.2f}%); Universal {u:+.2f}%",
               "Average 5-day abnormal return of the parent's stock after each release, all films")
    st.caption("*All of each studio's releases, whatever their box office. Read the whiskers before the bars.*")

    exhibit("Did executives sound different after a hit?")
    recs = []
    for name, g in [("Hits (top 25% return)", hits), ("Misses (bottom 25% return)", misses)]:
        for t, lab in [("prior", "Quarter before release"), ("during", "Release quarter"), ("post", "Quarter after")]:
            s = g[f"{t}_tone"].dropna()
            recs.append({"Group": name, "Quarter": lab, "Tone": s.mean() if len(s) else np.nan, "Films": len(s)})
    tone = pd.DataFrame(recs)
    if tone["Films"].min() >= 10:
        qorder = ["Quarter before release", "Release quarter", "Quarter after"]
        groups = ["Hits (top 25% return)", "Misses (bottom 25% return)"]
        t = tone.set_index(["Group", "Quarter"])["Tone"]
        dh = t[(groups[0], qorder[2])] - t[(groups[0], qorder[0])]
        dm = t[(groups[1], qorder[2])] - t[(groups[1], qorder[0])]
        title = f"From the quarter before to the quarter after release, call tone moved {dh:+.3f} around hits vs. {dm:+.3f} around misses"
        chart = alt.Chart(tone).mark_line(point=line_points(70), strokeWidth=2.25).encode(
            x=alt.X("Quarter:N", sort=qorder, title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Tone:Q", title="Average earnings-call tone (positive − negative)", scale=alt.Scale(zero=False)),
            color=alt.Color("Group:N", scale=alt.Scale(domain=groups, range=[BLUE, INK]), legend=alt.Legend(orient="top", title=None)),
            tooltip=["Group", "Quarter", alt.Tooltip("Tone:Q", format=".3f"), "Films"])
        show_chart(chart.properties(height=290), title, f"{stu}: parent's earnings-call tone (FinBERT) around the release")
        st.caption(f"*Films with call transcripts ({int(tone['Films'].max())} or fewer per point). Calls cover the whole "
                   f"company, so a film can only move the tone if it's big relative to the parent. Compare the slopes, "
                   f"not the levels.*")
    else:
        st.info("Too few films with earnings-call transcripts for this studio to compare hits and misses.")

    exhibit("How healthy were the parent companies?")
    metric = st.radio("Measure", list(PARENT_METRICS), horizontal=True, key="ws_metric")
    col, unit, how = PARENT_METRICS[metric]
    if metric == "Revenue":   # calendar years with all four quarters from the same owner, summed
        pa = (PARENTS.dropna(subset=["Revenue"]).groupby(["Studio", "Year", "Parent"])["Revenue"]
              .agg(["sum", "count"]).reset_index())
        pa = pa[pa["count"] == 4].assign(Value=lambda x: x["sum"] / 1e3)
    else:
        pa = PARENTS.groupby(["Studio", "Year"], as_index=False).agg(Value=(col, "mean"), Parent=("Parent", "last"))
        pa["Value"] = pa["Value"] * 100
    pick = [x for x in STUDIOS if x in set(pa["Studio"])]
    recent = pa[pa["Year"] >= pa["Year"].max() - 4].groupby("Studio")["Value"].mean()
    lead = recent.idxmin() if metric == "Leverage" else recent.idxmax()
    best = ("lowest leverage" if metric == "Leverage" else "largest revenue" if metric == "Revenue"
            else f"highest {metric.lower() if metric != 'ROA' else 'ROA'}")
    lo, hi = pa["Value"].quantile([0.04, 0.96])
    pad = (hi - lo) * 0.15
    pct = "" if metric == "Revenue" else "%"
    money_unit = "$" if metric == "Revenue" else ""
    show_chart(studio_lines(pa, alt.X("Year:Q", title=None, axis=alt.Axis(format="d")),
                            alt.Y("Value:Q", title=f"{metric} ({unit})",
                                  scale=alt.Scale(domain=[float(max(lo - pad, 0) if metric == "Revenue" else lo - pad),
                                                          float(hi + pad)], clamp=True)), pick,
                            ["Studio", "Parent", alt.Tooltip("Year:Q", format="d", title="Fiscal year"),
                             alt.Tooltip("Value:Q", format=".2f", title=f"{metric} ({unit})")])
               .properties(height=330),
               f"Last five years: {lead}'s parent had the {best} ({money_unit}{recent[lead]:.1f}{pct}"
               f"{'B' if metric == 'Revenue' else ''})"
               + ("" if lead == "Universal" else
                  f"; Universal's {money_unit}{recent.get('Universal', np.nan):.1f}{pct}{'B' if metric == 'Revenue' else ''}"),
               f"Parent company's {metric if metric == 'ROA' else metric.lower()} by year "
               f"({'sum of the four quarters' if metric == 'Revenue' else 'average of its quarters'}): {how}")
    st.caption("*Each line is the company that owned the studio at the time (hover for its name): GE, then Comcast, "
               "for Universal; Time Warner, AT&T, then Warner Bros. Discovery for Warner; Viacom, then Paramount Global "
               "for Paramount; 21st Century Fox; Disney; Lionsgate; Sony Group. So film is one business among many: "
               "cable, parks, TV networks, streaming, wireless, electronics. A few extreme years (2008 write-downs, "
               "one-off gains) are pinned to the chart's edge so the others stay readable; hover for the true value. "
               "Sources: SEC 10-Q/10-K filings from 2008; earlier years and Sony from Compustat (ratios only, so no "
               "revenue line for them).*")

    exhibit("Did the parent's results change after a hit?")
    if metric == "Revenue":
        metric, unit = "Profit margin", "%"
    recs = []
    for name, g in [("Hits (top 25% return)", hits), ("Misses (bottom 25% return)", misses)]:
        for t, lab in [("prior", "Quarter before release"), ("during", "Release quarter"), ("post", "Quarter after")]:
            v = g[f"{t}_{PARENT_FILM_COL[metric]}"].dropna() * 100
            recs.append({"Group": name, "Quarter": lab, "Value": v.mean() if len(v) else np.nan, "Films": len(v)})
    pq = pd.DataFrame(recs)
    if pq["Films"].min() >= 10:
        qorder = ["Quarter before release", "Release quarter", "Quarter after"]
        groups = ["Hits (top 25% return)", "Misses (bottom 25% return)"]
        tq = pq.set_index(["Group", "Quarter"])["Value"]
        dh = tq[(groups[0], qorder[2])] - tq[(groups[0], qorder[0])]
        dm = tq[(groups[1], qorder[2])] - tq[(groups[1], qorder[0])]
        chart = alt.Chart(pq).mark_line(point=line_points(70), strokeWidth=2.25).encode(
            x=alt.X("Quarter:N", sort=qorder, title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Value:Q", title=f"Average parent {metric if metric == 'ROA' else metric.lower()} ({unit})",
                    scale=alt.Scale(zero=False)),
            color=alt.Color("Group:N", scale=alt.Scale(domain=groups, range=[BLUE, INK]),
                            legend=alt.Legend(orient="top", title=None)),
            tooltip=["Group", "Quarter", alt.Tooltip("Value:Q", format=".2f"), "Films"])
        show_chart(chart.properties(height=290),
                   f"From the quarter before to the quarter after, parent {metric if metric == 'ROA' else metric.lower()} "
                   f"moved {dh:+.2f} pts around hits vs. {dm:+.2f} pts around misses",
                   f"{stu}: parent's {metric if metric == 'ROA' else metric.lower()} in the quarters around each release")
        st.caption("*Same hits and misses as above; switch the measure with the buttons above the previous chart. Compare "
                   "the slopes, not the levels: hits and misses come from different parents and years. A flat line is "
                   "the usual result, because one film rarely moves a conglomerate's quarter.*")
    else:
        st.info("Too few films with parent financials for this studio to compare hits and misses.")

    exhibit("How big is one film next to its parent?")
    fs = d.dropna(subset=["film_share_parent"]).copy()
    fs["Share"] = fs["film_share_parent"] * 100
    if len(fs) >= 10:
        med = fs["Share"].median()
        topf = fs.loc[fs["Share"].idxmax()]
        by = fs.groupby("studio")["Share"].median().reindex(STUDIOS).dropna()
        bars = alt.Chart(by.reset_index().rename(columns={"studio": "Studio", "Share": "Median"})).mark_bar(
            cornerRadiusEnd=3, stroke="white", strokeWidth=0.8).encode(
            x=alt.X("Studio:N", sort=list(by.index), title=None,
                    axis=alt.Axis(labelAngle=0, labelColor=studio_expr("value"), labelFontWeight=600)),
            y=alt.Y("Median:Q", title="Film's worldwide gross as % of parent's annual revenue"),
            color=alt.Color("Studio:N", scale=alt.Scale(domain=list(by.index),
                                                        range=[STUDIO_COLOR[x] for x in by.index]), legend=None),
            stroke=alt.condition("datum.Studio == 'Paramount'", alt.value(KEYLINE), alt.value("white")),
            tooltip=["Studio", alt.Tooltip("Median:Q", format=".2f", title="Median % of parent revenue")])
        show_chart(bars.properties(height=260),
                   f"The typical film grosses {med:.2f}% of its parent's annual revenue; the biggest here, "
                   f"{topf['title']}, {topf['Share']:.1f}%",
                   f"{stu}: worldwide box office ÷ the parent's revenue in the release fiscal year, median by studio")
        st.caption("*Box office isn't revenue (theaters keep about half), so the film's real weight is smaller still. "
                   "This is why one film rarely moves a conglomerate's stock or its quarter, and why Lionsgate, the "
                   "only near-pure film company here, is the place to look for film-driven swings. SEC filings, "
                   "fiscal years with all four quarters reported.*")
    else:
        st.info("Too few films with parent revenue for this selection.")

    exhibit("Biggest stock reactions")
    cols = st.columns(2)
    t = d.dropna(subset=["car5"])
    for col, (title, part) in zip(cols, [("Largest gains", t.nlargest(8, "car5")), ("Largest drops", t.nsmallest(8, "car5"))]):
        with col:
            st.markdown(f"**{title}**")
            p = part[["title", "year", "studio", "opening_m", "multiple", "car5"]].assign(car5=lambda x: x["car5"] * 100)
            show_table(p, column_config={"title": "Film", "year": st.column_config.NumberColumn("Year", format="%d"),
                                         "studio": "Studio",
                                         "opening_m": st.column_config.NumberColumn("Opening ($M)", format="%.1f"),
                                         "multiple": st.column_config.NumberColumn("Return (×)", format="%.2f"),
                                         "car5": st.column_config.NumberColumn("5-day reaction", format="%+.1f%%")})
    st.caption("*Extreme reactions usually reflect other news that week (earnings, mergers, the market) as much as the "
               "film. Use them as prompts for a story you can check, not as proof.*")


# --- 6. Predict the Box Office ------------------------------------------------------------------------------------
OUTCOMES = ["Worldwide box office", "Opening weekend (US)", "Return on budget (worldwide ÷ budget)"]
LAYERS = ["Metadata", "Metadata + Script emotions", "Metadata + Script emotions + Parent company"]
LAYER_SHORT = {"Metadata": "1 · Metadata", "Metadata + Script emotions": "2 · + Script emotions",
               "Metadata + Script emotions + Parent company": "3 · + Parent company"}
MODEL_NAMES = ["Linear regression", "Decision tree", "Random forest"]
SCOPES = ["All studios", "Universal only"] + [f"Genre: {g}" for g in GENRE_ORDER]

BUSINESS = {  # feature: (definition, if positive, if negative, lever type)
    "Budget (log)": ("Production budget (log scale, so equal steps are equal % increases).",
                     "→ Bigger budgets go with bigger grosses; check whether they also pay back (Return on budget).",
                     "→ Bigger budgets haven't paid back proportionally here; consider more mid-budget bets.", "Business lever"),
    "Opening theaters (log)": ("Number of US theaters on opening weekend (log scale).",
                               "→ A wider release goes with bigger grosses; distribution reach matters.",
                               "→ Going wider hasn't helped here; a platform release may fit better.", "Business lever"),
    "Runtime (minutes)": ("Running time.", "→ Longer films have done better here, often a sign of event films.",
                          "→ Shorter films have done better here; tighten the edit.", "Creative lever"),
    "Release year": ("Year of release. Absorbs inflation and market growth.",
                     "→ Context: later films gross more in nominal dollars.", "→ Context: returns have fallen over time.",
                     "Context (not a lever)"),
    "Follow-up film": ("Sequel, prequel, reboot, remake or spin-off of an earlier film (vs. an original).",
                       "→ Franchise films have done better here, all else equal; existing IP still pays.",
                       "→ Follow-ups have done worse here, all else equal: a sign of franchise fatigue.", "Business lever"),
    "Star power": ("Box-office track record of the lead cast before this film.",
                   "→ Proven stars go with better results; the talent premium may be worth it.",
                   "→ Star salaries haven't paid back here; consider rising talent.", "Business lever"),
    "Emotional intensity": ("How strongly emotional the dialogue is overall, as opposed to neutral.",
                            "→ Keep the dialogue emotionally charged throughout.",
                            "→ Dial back; relentless intensity goes with weaker results here.", "Creative lever"),
    "Emotional swing": ("How much the emotional mix shifts from act to act.",
                        "→ Design a clear arc: let the emotional mix change act to act.",
                        "→ Keep the tone more consistent across acts.", "Creative lever"),
    "Emotional novelty vs. peers": ("How far the film's emotional profile sits from earlier films.",
                                    "→ Differentiate: an emotional profile unlike recent films goes with better results.",
                                    "→ Audiences here reward the familiar emotional template.", "Creative lever"),
    "Story novelty vs. peers": ("How far the film's story language sits from earlier films.",
                                "→ Fresh stories go with better results here; new IP can pay.",
                                "→ Familiar stories go with better results here.", "Creative lever"),
}
EMOTION_TIPS = {
    "Joy": ("→ Give this act moments of celebration, humor or relief.",
            "→ Be careful with light-hearted beats in this act."),
    "Surprise": ("→ Place twists or reveals in this act.", "→ Limit twists in this act; keep the line clear."),
    "Fear": ("→ Raise the stakes and suspense in this act: threats, danger, ticking clocks.",
             "→ Ease off fear and dread in this act."),
    "Sadness": ("→ Let characters be vulnerable here; loss and grief build empathy.",
                "→ Keep sadness in check in this act."),
    "Anger": ("→ Use conflict and confrontation in this act: arguments, rivalries, showdowns.",
              "→ Use anger sparingly in this act; constant conflict can alienate."),
    "Disgust": ("→ Use moral violation or revulsion here to sharpen villains and stakes.",
                "→ Pull back on disgust and transgression in this act."),
    "Neutral": ("→ Leave room for exposition and breathing space in this act.",
                "→ Cut flat, purely informational dialogue in this act; keep it charged."),
}


def describe_feature(f):
    """(definition, positive tip, negative tip, lever type) for any model feature."""
    if f in BUSINESS:
        return BUSINESS[f]
    if " · Act " in f:
        e, a = f.split(" · Act ")
        return (f"Share of {ACTS[int(a)].split(' · ')[1].lower()} (act {a}) dialogue read as {e.lower()} "
                f"({EMOTION_DEFS[e]}).", *EMOTION_TIPS[e], "Creative lever")
    if f.startswith("Rated "):
        return (f"{f} (vs. rated R).", "→ This rating has done better than R here; a wider audience pays.",
                "→ This rating has done worse than R here.", "Business lever")
    if f.endswith(" release"):
        return (f"{f} (vs. a January–February release).", "→ This window has paid off; claim the date early.",
                "→ This window has done worse; avoid it or counter-program.", "Business lever")
    if f.startswith("Genre: "):
        return (f"{f} (vs. Drama).", "→ This genre has outperformed drama, all else equal.",
                "→ This genre has underperformed drama, all else equal.", "Business lever")
    if f.startswith("Studio: "):
        return (f"Released by {f[8:]} (vs. Warner Bros.).", "→ Context: this studio's films have done better.",
                "→ Context: this studio's films have done worse.", "Context (not a lever)")
    if f.startswith("Parent "):
        return (f"{f}.", "→ Context: a stronger parent going in goes with better results (marketing muscle, "
                         "distribution).", "→ Context: this parent-company measure goes with weaker results.",
                "Context (not a lever)")
    return (f, "→ Positively associated.", "→ Negatively associated.", "Context (not a lever)")


@st.fragment
def predict_section():
    section("Models", "Predict the Box Office",
            "**How it works:** choose which films, which outcome and how much data the models may use. Three models "
            "learn from a random 80% of the films and are scored on the 20% they never saw: a **linear regression** "
            "(readable coefficients), a **decision tree** (splits the films into smaller and smaller groups) and a "
            "**random forest** (hundreds of trees averaged, which handles more complexity). The question isn't only "
            "*which model wins*, but **how predictable a film's success is at all**, and what each extra layer of data "
            "adds.")
    c1, c2 = st.columns(2)
    scope = c1.selectbox("Films", SCOPES, key="pred_scope")
    outcome = c2.selectbox("Outcome to predict", OUTCOMES, key="pred_outcome")
    layer = st.radio("Data the models can use", LAYERS, horizontal=True, key="pred_layer",
                     help="Metadata = known before release: budget, opening theaters, runtime, year, rating, season, "
                          "follow-up or original, star power, genre, studio. Script emotions = the 7 emotions × 5 acts "
                          "plus intensity, swing and novelty. Parent company = the parent's ROA, margin, leverage, "
                          "earnings-call tone and stock momentum in the quarter before release.")
    out_l = outcome.split(" (")[0].lower()

    exhibit("Does more data help?")
    recs = []
    for li in LAYERS:
        e = MODELS.get(f"{scope}|{outcome}|{li}", {})
        for mname in MODEL_NAMES:
            if "cv" in e:
                recs.append({"Data": LAYER_SHORT[li], "Model": mname, "R²": float(np.mean(e["cv"][mname])),
                             "Films": e["films"]})
    if recs:
        d = pd.DataFrame(recs)
        best = d.loc[d["R²"].idxmax()]
        title = f"Best: {best['Model'].lower()} with {best['Data'][4:].lstrip('+ ').lower()} explains " \
                f"{max(best['R²'], 0) * 100:.0f}% of the variation in {out_l}"
        rf = d[d["Model"] == "Random forest"].set_index("Data")["R²"]
        if LAYER_SHORT[LAYERS[0]] in rf and LAYER_SHORT[LAYERS[1]] in rf:
            title += f"; script emotions change that by {rf[LAYER_SHORT[LAYERS[1]]] - rf[LAYER_SHORT[LAYERS[0]]]:+.2f}"
        bars = alt.Chart(d).mark_bar(stroke="white", strokeWidth=0.8, cornerRadiusEnd=3).encode(
            x=alt.X("Data:N", sort=[LAYER_SHORT[l] for l in LAYERS], title=None, axis=alt.Axis(labelAngle=0)),
            xOffset=alt.XOffset("Model:N", sort=MODEL_NAMES),
            y=alt.Y("R²:Q", title="Cross-validated R²"),
            color=alt.Color("Model:N", sort=MODEL_NAMES, scale=alt.Scale(domain=MODEL_NAMES, range=CAT[:3]),
                            legend=alt.Legend(orient="top", title=None)),
            tooltip=["Data", "Model", alt.Tooltip("R²:Q", format=".2f"), "Films"])
        show_chart((bars + bars.mark_text(dy=-7, fontSize=10, color=INK).encode(text=alt.Text("R²:Q", format=".2f")))
                   .properties(height=290), title,
                   f"{scope}, {out_l}: share of variation explained on unseen films (1 = perfect, 0 = no better than the average)")
        full = MODELS.get(f"{scope}|{outcome}|{LAYERS[2]}", {})
        same = full.get("same_films_without_parent") or {}
        note = ""
        if same.get("cv") and "cv" in full:
            rf_with, rf_without = np.mean(full["cv"]["Random forest"]), np.mean(same["cv"]["Random forest"])
            note = (f" **Careful with layer 3:** parent financials exist for only {full['films']} of these films, so it's "
                    f"a different sample. On those same films *without* parent data the random forest scores "
                    f"{rf_without:.2f}, vs. {rf_with:.2f} with it: the parent layer's real contribution is "
                    f"{rf_with - rf_without:+.2f}.")
        st.caption("*Each bar is the average R² over 5-fold cross-validation (five different 80/20 splits), steadier than "
                   "any single split. If a layer doesn't raise the bars, that data isn't helping the models predict, even "
                   "when it describes the films well.*" + note)
    entry = MODELS.get(f"{scope}|{outcome}|{layer}", {})
    if "results" not in entry:
        st.warning(f"Only {entry.get('films', 0)} films in '{scope}' have this outcome and these data (minimum 50), "
                   f"too few for stable models. Choose more films or fewer data layers.")
        return

    exhibit(f"Model benchmark: {layer}")
    cols = st.columns(3)
    for col, m in zip(cols, MODEL_NAMES):
        col.metric(f"{m} · R² (test)", fmt(entry["benchmarks"][m]),
                   help=f"Cross-validated R²: {np.mean(entry['cv'][m]):.2f} (5 folds: "
                        f"{', '.join(f'{v:.2f}' for v in entry['cv'][m])}).")
    st.caption(f"*Scored on the {entry['test']} test films the models never saw ({entry['train']} training films). The "
               f"linear regression uses the {entry['parameters'] - 1} screened features shown below; the tree and the "
               f"forest may use every feature in the data layer. Hover a tile for its cross-validated R².*")
    if entry["train"] / entry["parameters"] < 10:
        st.warning(f"Small sample: {entry['train']} training films for {entry['parameters']} regression parameters. "
                   f"Expect unstable coefficients; trust the cross-validated R² over the single test split.")

    dollars = outcome != OUTCOMES[2]
    mpick = st.radio("Diagnostic plots for", MODEL_NAMES, index=2, horizontal=True, key="pred_diag")
    actual = np.exp(np.array(entry["test_actual"]))
    pred = np.exp(np.array(entry["test_pred"][mpick]))
    div = 1e6 if dollars else 1
    dd = pd.DataFrame({"Film": [TITLE_BY_ID.get(i, i) for i in entry["test_films"]],
                       "Actual": actual / div, "Predicted": pred / div, "Ratio": actual / pred})
    within2 = float(((dd["Ratio"] <= 2) & (dd["Ratio"] >= 0.5)).mean() * 100)
    miss = float(np.exp(np.median(np.abs(np.log(dd["Ratio"])))))
    axis_t = " ($M, log scale)" if dollars else " (×, log scale)"
    left, right = st.columns(2)
    with left:
        lo, hi = float(min(dd["Actual"].min(), dd["Predicted"].min())), float(max(dd["Actual"].max(), dd["Predicted"].max()))
        pts = alt.Chart(dd).mark_circle(size=64, color=BLUE, opacity=0.8, stroke="white", strokeWidth=1).encode(
            x=alt.X("Predicted:Q", scale=alt.Scale(type="log"), title="Predicted" + axis_t),
            y=alt.Y("Actual:Q", scale=alt.Scale(type="log"), title="Actual" + axis_t),
            tooltip=["Film", alt.Tooltip("Actual:Q", format=",.1f"), alt.Tooltip("Predicted:Q", format=",.1f")])
        diag = alt.Chart(pd.DataFrame({"x": [lo, hi], "y": [lo, hi]})).mark_line(color="#9ca3af", strokeDash=[5, 4]).encode(x="x:Q", y="y:Q")
        show_chart((diag + pts).properties(height=330),
                   f"{within2:.0f}% of unseen films land within 2× of the prediction",
                   f"Actual vs. predicted {out_l}, {mpick.lower()}, {entry['test']} test films")
        st.caption("*On the dashed line = predicted exactly; above it = the film beat its prediction. Hover for titles.*")
    with right:
        res = alt.Chart(dd).mark_circle(size=64, color=BLUE, opacity=0.8, stroke="white", strokeWidth=1).encode(
            x=alt.X("Predicted:Q", scale=alt.Scale(type="log"), title="Predicted" + axis_t),
            y=alt.Y("Ratio:Q", scale=alt.Scale(type="log"), title="Actual ÷ predicted (log scale)"),
            tooltip=["Film", alt.Tooltip("Ratio:Q", format=".2f", title="Actual ÷ predicted")])
        one = alt.Chart(pd.DataFrame({"y": [1]})).mark_rule(color="#9ca3af", strokeDash=[5, 4]).encode(y="y:Q")
        show_chart((one + res).properties(height=330), f"The typical miss is a factor of {miss:.1f} either way",
                   f"Residuals: actual ÷ predicted, {mpick.lower()}")
        st.caption("*1 = exact; 2 = the film earned twice its prediction; 0.5 = half. The spread is the uncertainty left "
                   "for creative judgment, marketing and luck.*")

    rows = pd.DataFrame(entry["results"])
    top = rows.sort_values("Importance", ascending=False).iloc[0]
    exhibit(f"What drives {out_l}?")
    render_importance(rows, f"{top['Feature']} is the strongest predictor of {out_l} "
                            f"({top['Effect %']:+.0f}% per step, holding the rest constant)",
                      f"{scope}, {layer.lower()}: random forest importance, signed by the regression's direction")
    with st.expander(f"🔍 How {len(rows)} of {len(entry['candidates'])} screened features were kept"):
        st.markdown(f"A random forest ranked every feature in the data layer by how much it helps predict the outcome on "
                    f"the training films; the top **{len(entry['candidates'])}** were kept: "
                    + ", ".join(entry["candidates"]) + ". Features that mostly duplicate each other were then removed "
                    "one at a time (highest variance inflation factor first, until all are below 10), so the regression "
                    "isn't measuring the same signal twice.")
        if entry["vif_dropped"]:
            show_table(pd.DataFrame(entry["vif_dropped"]).round({"VIF at removal": 1}))
        else:
            st.markdown("No feature needed to be dropped.")

    exhibit("Regression results")
    table = pd.DataFrame({
        "Feature": rows["Feature"],
        "Step": np.where(rows["Scaled"], "+1 standard deviation", "Yes vs. no"),
        "Effect on outcome (%)": rows["Effect %"], "95% CI low (%)": rows["CI low %"], "95% CI high (%)": rows["CI high %"],
        "P-value": rows["P-value"]})
    if "Confirmed" in rows:
        table["Held-out effect (%)"] = rows["Held-out effect %"]
        table["Confirmed"] = np.where(rows["Confirmed"], "✓", "")
    show_table(table, column_config={c: st.column_config.NumberColumn(format="%+.1f") for c in
                                     ["Effect on outcome (%)", "95% CI low (%)", "95% CI high (%)", "Held-out effect (%)"]}
               | {"P-value": st.column_config.NumberColumn(format="%.3f")})
    st.caption(f"*Linear regression on the {entry['train']} training films, robust (HC3) standard errors. The outcome is "
               f"logged, so each effect is the **% change in {out_l}** for the step shown: one standard deviation more "
               f"of a continuous feature (a typical difference between films), or having the yes/no feature. Baselines: "
               f"rated R, a January–February release, a drama, Warner Bros. P-value < 0.10 = unlikely to be pure "
               f"chance. Held-out columns refit the same model on the {entry['test']} test films (which played no part "
               f"in choosing the features); ✓ = significant in both, same direction. Training R² = "
               f"{entry['r2_train_ols']:.2f}.*")
    render_recommendations(rows, entry["heldout_ok"], out_l)


def render_importance(rows, title, subtitle):
    d = rows.copy()
    d["Signed"] = np.sign(d["Coef"]) * d["Importance"]
    d["Direction"] = np.where(d["Coef"] >= 0, "Positive association", "Negative association")
    d = d.sort_values("Importance", ascending=False)
    lim = float(d["Importance"].max()) * 1.3
    base = alt.Chart(d).encode(y=alt.Y("Feature:N", sort=list(d["Feature"]), title=None))
    bars = base.mark_bar(stroke="white", strokeWidth=0.8, cornerRadiusEnd=3, height={"band": 0.68}).encode(
        x=alt.X("Signed:Q", scale=alt.Scale(domain=[-lim, lim]),
                title="← Negative association   |   Random forest importance   |   Positive association →",
                axis=alt.Axis(labelExpr="format(abs(datum.value), '.2f')")),
        color=alt.Color("Direction:N", scale=alt.Scale(domain=["Positive association", "Negative association"],
                                                       range=[BLUE, INK]), legend=alt.Legend(orient="top", title=None)),
        tooltip=["Feature", alt.Tooltip("Importance:Q", format=".3f"),
                 alt.Tooltip("Effect %:Q", format="+.1f", title="Effect on outcome (%)")])
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#9ca3af").encode(x="x:Q")
    pos = base.transform_filter("datum.Signed >= 0").mark_text(align="left", dx=4, fontSize=11, color=INK).encode(
        x="Signed:Q", text=alt.Text("Importance:Q", format=".3f"))
    neg = base.transform_filter("datum.Signed < 0").mark_text(align="right", dx=-4, fontSize=11, color=INK).encode(
        x="Signed:Q", text=alt.Text("Importance:Q", format=".3f"))
    show_chart((bars + zero + pos + neg).properties(height=max(170, 32 * len(d))), title, subtitle)
    st.caption("*Right (blue) = more of this goes with a bigger outcome; left (black) = smaller. Longer bar = stronger "
               "predictor. Direction comes from the regression coefficient, holding the other features constant.*")


def render_recommendations(rows, heldout_ok, out_l):
    exhibit("Recommendations")
    st.markdown(
        "Features significant at p < 0.10, split by direction, each tagged as a **creative lever** (the script), a "
        "**business lever** (budget, release, casting, rating, IP) or **context** (things nobody at the studio "
        "controls). Use them to back up your slate, release and creative choices.\n\n"
        "*Focus on direction, not size. About 1 in 10 features passes p < 0.10 by luck, so each one is rechecked on the "
        "held-out test films: **✓ Confirmed** means it held up there too; the rest are tentative.*"
    )
    sig = rows[rows["P-value"] < 0.10]
    if sig.empty:
        st.info("No feature is significant at p < 0.10 here. The honest conclusion: in these films, nothing in this "
                "data layer reliably separates bigger and smaller results once the rest is held constant.")
        return

    def badge(r):
        if not heldout_ok:
            return "<em>Held-out confirmation not available (too few test films): treat as tentative.</em>"
        return ("<strong>✓ Confirmed on held-out films.</strong>" if r.get("Confirmed")
                else "<em>Not confirmed on held-out films: treat as tentative.</em>")

    for title, part, color in [("Positive Relationships: Consider Emphasizing", sig[sig["Coef"] > 0], BLUE),
                               ("Negative Relationships: Use with Caution or Avoid", sig[sig["Coef"] < 0], "inherit")]:
        if part.empty:
            continue
        st.markdown(f"<h5 style='color:{color};'>{title}</h5>", unsafe_allow_html=True)
        for _, r in part.iterrows():
            definition, pos_tip, neg_tip, lever = describe_feature(r["Feature"])
            direction = "positively" if r["Coef"] > 0 else "negatively"
            st.markdown(
                f"<p style='color:{color};'>● <strong>{r['Feature']}</strong> ({r['Effect %']:+.0f}% {out_l}) "
                f"[{lever}]: {direction} associated with {out_l}. {badge(r)}"
                f"<br><em>Definition:</em> {definition}<br>{pos_tip if r['Coef'] > 0 else neg_tip}</p>",
                unsafe_allow_html=True)



# --- 4-5. Film Economics and Pitch a Film ----------------------------------------------------------------------------
# One fragment for both sections, so the deal assumptions set in Film Economics also price the pitch.
ASSUME = {  # key: (label, min, max, default, step, help)
    "dom": ("Studio's share of US ticket sales", 35, 65, 50, 1,
            "Film rentals: the share of the domestic box office the theaters pass back to the distributor. Typically "
            "45–55%, higher for blockbusters in their opening weeks."),
    "intl": ("Studio's share of international tickets", 25, 55, 40, 1,
             "Lower than at home because local distributors and taxes take a cut."),
    "pa": ("Marketing (P&A) as % of production budget", 0, 120, 60, 5,
           "Prints and advertising: the worldwide marketing spend. Wide releases often spend 50–100% of the production "
           "budget. Not in the data, so you set it."),
    "home": ("Home windows (PVOD, digital, disc) as % of US gross", 0, 50, 20, 5,
             "Studio's net revenue from premium and regular video-on-demand, digital purchase and disc, as a share of "
             "the domestic box office. Studios keep about 75% of a $20–30 PVOD rental."),
    "stream": ("Streaming / pay-TV license as % of worldwide gross", 0, 40, 15, 5,
               "What a streamer or pay-TV window (e.g. Peacock, a pay-1 deal) is worth, as a share of worldwide box "
               "office. Rough by nature: set 0 to look at theaters alone."),
}
TIERS = [(0, 50e6, "Under $50M"), (50e6, 150e6, "$50–150M"), (150e6, np.inf, "$150M+")]


def economics(d, a):
    """Studio P&L per film under assumptions a (fractions). International falls back to worldwide − domestic."""
    intl = d["international"].where(d["international"].notna(), d["worldwide"] - d["domestic"])
    e = pd.DataFrame(index=d.index)
    e["Rentals"] = d["domestic"] * a["dom"] + intl * a["intl"]
    e["Budget"] = d["budget"]
    e["P&A"] = d["budget"] * a["pa"]
    e["Theatrical profit"] = e["Rentals"] - e["Budget"] - e["P&A"]
    e["Home windows"] = d["domestic"] * a["home"]
    e["Streaming"] = d["worldwide"] * a["stream"]
    e["Full-cycle profit"] = e["Theatrical profit"] + e["Home windows"] + e["Streaming"]
    return e


def breakeven(a, dom_share):
    """Worldwide gross ÷ budget a film needs to break even, theatrically and over the full cycle."""
    k_theat = a["dom"] * dom_share + a["intl"] * (1 - dom_share)
    k_full = k_theat + a["home"] * dom_share + a["stream"]
    return (1 + a["pa"]) / k_theat, (1 + a["pa"]) / k_full


def tier_of(b):
    for lo, hi, name in TIERS:
        if lo <= b < hi:
            return name
    return None


def waterfall(rows):
    """rows: [(label, amount, kind)], kind in start/up/down/total. Bars float from the running total."""
    recs, run = [], 0.0
    for label, v, kind in rows:
        if kind in ("start", "total"):
            lo, hi = 0.0, v
            run = v
        else:
            lo, hi = run, run + v
            run += v
        recs.append({"Step": label, "Low": min(lo, hi), "High": max(lo, hi), "Amount": v,
                     "Kind": kind if kind in ("start", "total") else ("up" if v >= 0 else "down")})
    d = pd.DataFrame(recs)
    order = list(d["Step"])
    colors = {"start": SOFT, "up": BLUE, "down": INK, "total": CAT[2]}
    base = alt.Chart(d).encode(x=alt.X("Step:N", sort=order, title=None, axis=alt.Axis(labelAngle=0, labelLimit=110)))
    bars = base.mark_bar(size=44, cornerRadius=2).encode(
        y=alt.Y("Low:Q", title="$M"), y2="High:Q",
        color=alt.Color("Kind:N", scale=alt.Scale(domain=list(colors), range=list(colors.values())), legend=None),
        tooltip=["Step", alt.Tooltip("Amount:Q", format="+,.1f", title="$M")])
    text = base.mark_text(dy=-8, fontSize=11, color=INK).encode(
        y="High:Q", text=alt.Text("Amount:Q", format=",.0f"))
    zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#6b7280").encode(y="y:Q")
    return (bars + text + zero).properties(height=320)


@st.fragment
def economics_and_pitch():
    section("Money", "Film Economics",
            "**How it works:** a film's box office isn't the studio's revenue. Theaters keep roughly half of ticket "
            "sales; the studio's share (*film rentals*) has to pay back the production budget **and** the marketing "
            "(*prints and advertising*, P&A). Then come the later *windows*: premium video-on-demand (PVOD), digital and "
            "disc, then streaming and pay TV. The data has budgets and grosses; you set the deal terms below, and every "
            "number in this section and in *Pitch a Film* updates.")
    with st.container(border=True):
        st.markdown("**Your deal assumptions**")
        cols = st.columns(5)
        a = {}
        for col, (k, (label, lo, hi, dflt, step, hlp)) in zip(cols, ASSUME.items()):
            a[k] = col.slider(label, lo, hi, dflt, step, format="%d%%", help=hlp, key=f"eco_{k}") / 100

    d = FILMS[FILMS["budget"].notna() & FILMS["worldwide"].notna() & FILMS["domestic"].notna()].copy()
    e = economics(d, a)
    d = d.join(e)
    dom_share = float((d["domestic"] / d["worldwide"]).median())
    be_t, be_f = breakeven(a, dom_share)
    cols = st.columns(4)
    cols[0].metric("Break-even, theaters only", f"{be_t:.1f}× budget",
                   help=f"Worldwide gross ÷ budget needed to cover budget + P&A from film rentals, for a film earning "
                        f"{dom_share * 100:.0f}% of its gross in the US (the median here).")
    cols[1].metric("Break-even, full cycle", f"{be_f:.1f}× budget", help="Adding the home and streaming windows.")
    cols[2].metric("Films profitable in theaters alone", f"{(d['Theatrical profit'] > 0).mean() * 100:.0f}%")
    cols[3].metric("Films profitable over the full cycle", f"{(d['Full-cycle profit'] > 0).mean() * 100:.0f}%")
    takeaway(f"Under these terms a film needs about <b>{be_t:.1f}× its budget</b> at the worldwide box office to break "
             f"even in theaters, or <b>{be_f:.1f}×</b> counting the later windows. That's the case's "
             f"\"two-to-three-times gross multiple\" made explicit.")

    exhibit("From box office to profit: one film")
    labels = list(d.sort_values("worldwide", ascending=False)["label"])
    default = labels.index("Jurassic World (2015)") if "Jurassic World (2015)" in labels else 0
    pick = st.selectbox("Film", labels, index=default, key="eco_film")
    r = d[d["label"] == pick].iloc[0]
    m = 1e6
    rows = [("Worldwide gross", r["worldwide"] / m, "start"),
            ("Theaters' share", -(r["worldwide"] - r["Rentals"]) / m, "down"),
            ("Production budget", -r["Budget"] / m, "down"), ("Marketing (P&A)", -r["P&A"] / m, "down"),
            ("Home windows", r["Home windows"] / m, "up"), ("Streaming / pay TV", r["Streaming"] / m, "up"),
            ("Studio profit", r["Full-cycle profit"] / m, "total")]
    verdict = "made" if r["Full-cycle profit"] >= 0 else "lost"
    show_chart(waterfall(rows),
               f"{r['title']} {verdict} an estimated {money(abs(r['Full-cycle profit']))} for the studio "
               f"({money(r['Theatrical profit'])} from theaters alone)",
               f"Estimated studio P&L, $M, under your assumptions ({r['multiple']:.2f}× budget at the box office)")
    st.caption("*Gray = what the box office took in; black = what's paid out or kept by others; blue = later windows; "
               "green = what's left for the studio. An estimate: real deals add distribution fees, talent "
               "participations (gross points), co-financier shares and overhead, which all shrink the studio's share.*")

    exhibit("Who makes money: theaters alone vs. the full cycle")
    recs = []
    for stu in STUDIOS:
        g = d[d["studio"] == stu]
        recs += [{"Studio": stu, "Window": "Theaters only", "Share": (g["Theatrical profit"] > 0).mean() * 100,
                  "Films": len(g)},
                 {"Studio": stu, "Window": "Full cycle", "Share": (g["Full-cycle profit"] > 0).mean() * 100,
                  "Films": len(g)}]
    w = pd.DataFrame(recs)
    gain = w.pivot(index="Studio", columns="Window", values="Share")
    gain = (gain["Full cycle"] - gain["Theaters only"]).sort_values(ascending=False)
    wins = ["Theaters only", "Full cycle"]
    bars = alt.Chart(w).mark_bar(cornerRadiusEnd=3, stroke="white", strokeWidth=0.8).encode(
        x=alt.X("Studio:N", sort=STUDIOS, title=None, axis=alt.Axis(labelAngle=0, labelColor=studio_expr("value"),
                                                                     labelFontWeight=600)),
        xOffset=alt.XOffset("Window:N", sort=wins),
        y=alt.Y("Share:Q", title="% of films profitable", scale=alt.Scale(domain=[0, 100])),
        color=alt.Color("Window:N", sort=wins, scale=alt.Scale(domain=wins, range=[SOFT, BLUE]),
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=["Studio", "Window", alt.Tooltip("Share:Q", format=".0f", title="% profitable"), "Films"])
    show_chart(bars.properties(height=300),
               f"The later windows rescue the most films at {gain.index[0]} (+{gain.iloc[0]:.0f} pts); "
               f"Universal +{gain.get('Universal', np.nan):.0f} pts",
               "Share of each studio's films with a positive estimated profit, under your assumptions")
    st.caption("*Same assumptions for every studio, so differences come from what each studio made and how it "
               "performed. In reality Disney and Universal own their streamers (Disney+, Peacock), which changes who "
               "captures the streaming value.*")

    exhibit("Big bets vs. many small bets")
    d["Tier"] = d["budget"].map(tier_of)
    tiers = [t[2] for t in TIERS]
    recs = []
    for t in tiers:
        for kind in ["Original", "Follow-up"]:
            g = d[(d["Tier"] == t) & (d["kind"] == kind)]["Full-cycle profit"] / 1e6
            if len(g) >= 8:
                recs.append({"Tier": t, "Kind": kind, "P10": g.quantile(0.1), "P50": g.median(),
                             "P90": g.quantile(0.9), "Win": (g > 0).mean() * 100, "Films": len(g)})
    rt = pd.DataFrame(recs)
    kinds = ["Original", "Follow-up"]
    base = alt.Chart(rt).encode(y=alt.Y("Tier:N", sort=tiers, title=None),
                                yOffset=alt.YOffset("Kind:N", sort=kinds),
                                color=alt.Color("Kind:N", sort=kinds, scale=alt.Scale(domain=kinds, range=[BLUE, CAT[1]]),
                                                legend=alt.Legend(orient="top", title=None)))
    tip = ["Tier", "Kind", alt.Tooltip("P10:Q", format=",.0f", title="Bad case (10th pct) $M"),
           alt.Tooltip("P50:Q", format=",.0f", title="Typical (median) $M"),
           alt.Tooltip("P90:Q", format=",.0f", title="Good case (90th pct) $M"),
           alt.Tooltip("Win:Q", format=".0f", title="% profitable"), "Films"]
    rng = base.mark_rule(strokeWidth=5, opacity=0.45).encode(x=alt.X("P10:Q", title="Estimated full-cycle profit ($M)"),
                                                             x2="P90:Q", tooltip=tip)
    mid = base.mark_point(filled=True, size=140, stroke="white", strokeWidth=1.5, opacity=1).encode(x="P50:Q", tooltip=tip)
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#6b7280", strokeDash=[4, 3]).encode(x="x:Q")
    if len(rt):
        big = rt[rt["Tier"] == "$150M+"]
        small = rt[rt["Tier"] == "Under $50M"]
        tmed = d.groupby("Tier")["Full-cycle profit"].median()
        title = (f"$150M+ films risk the most and earn the most: median {money(tmed['$150M+'])} vs. "
                 f"{money(tmed['Under $50M'])} under $50M") if len(big) and len(small) else "Profit by budget tier"
        show_chart((zero + rng + mid).properties(height=260), title,
                   "Estimated full-cycle profit per film: dot = median, bar = 10th to 90th percentile")
        show_table(rt.assign(Range=rt.apply(lambda x: f"{money(x['P10'] * 1e6)} to {money(x['P90'] * 1e6)}", axis=1))
                   [["Tier", "Kind", "Films", "Win", "P50", "Range"]],
                   column_config={"Win": st.column_config.NumberColumn("% profitable", format="%.0f%%"),
                                  "P50": st.column_config.NumberColumn("Median profit ($M)", format="%.0f"),
                                  "Range": "10th–90th percentile"})
    st.caption("*Budget tiers in nominal dollars. Read the bars as risk: a long bar means outcomes swing widely, so a "
               "slate needs several of these films for the averages to show up. Tiers with fewer than 8 films are "
               "left out.*")

    exhibit("When to release: the calendar")
    cal = d[d["wide"] == 1].copy()
    cal["Month"] = cal["month"].map(lambda x: MONTHS[int(x) - 1] if pd.notna(x) else None)
    g = cal.groupby(["genre", "Month"])["multiple"].agg(["median", "size"]).reset_index()
    g = g[g["size"] >= 4].rename(columns={"genre": "Genre", "median": "Value", "size": "n"})
    rows_order = [x for x in GENRES if x in set(g["Genre"])]
    if len(g):
        top = g.loc[g[g["n"] >= 8]["Value"].idxmax()] if (g["n"] >= 8).any() else g.loc[g["Value"].idxmax()]
        show_chart(heatmap(g, "Month", "Genre", MONTHS, rows_order, "Median return (×)", ".1f"),
                   f"Best slot with 8+ films: {top['Genre'].lower()} in {top['Month']} "
                   f"(median {top['Value']:.1f}× budget, {int(top['n'])} films)",
                   "Median worldwide gross ÷ budget by genre and release month, wide releases", panel=False)
    st.caption("*Blank cells have fewer than 4 films. Small cells are noisy: hover for the film count before you "
               "build a release strategy on one square.*")

    pitch_section(a)


def pitch_predict(budget, genre, follow_up, rating, season, year):
    """Median predicted worldwide gross from the Pitch model (log-linear), in dollars."""
    pr = PITCH["params"]
    z = pr["const"] + pr["Budget (log)"] * math.log(budget) + pr["Follow-up film"] * follow_up \
        + pr["Release year"] * (year - 2015)
    z += pr.get(f"Genre: {genre}", 0) + pr.get(f"Rated {rating}", 0) + pr.get(f"{season} release", 0)
    return math.exp(z)


def pitch_section(a):
    section("Decide", "Pitch a Film",
            "**How it works:** describe a film you'd greenlight. A regression fit on "
            f"{PITCH['films']} wide releases since {PITCH['since']} predicts its worldwide gross from the choices a "
            "studio makes before shooting (budget, genre, original or follow-up, rating, release season, year). Real "
            "films missed that prediction by a known spread, so you get a **range**, not a point, and the deal terms "
            "from *Film Economics* turn the range into a **chance of making money**.")
    c1, c2, c3 = st.columns(3)
    genre = c1.selectbox("Genre", [g for g in GENRES], index=GENRES.index("Horror") if "Horror" in GENRES else 0,
                         key="pitch_genre")
    budget = c2.select_slider("Production budget", options=[5, 10, 15, 20, 30, 40, 50, 60, 75, 90, 100, 125, 150, 175,
                                                             200, 250, 300], value=30, format_func=lambda x: f"${x}M",
                              key="pitch_budget") * 1e6
    kind = c3.radio("Original or follow-up", ["Original", "Follow-up"], horizontal=True, key="pitch_kind")
    c4, c5, c6 = st.columns(3)
    rating = c4.selectbox("Rating", ["G", "PG", "PG-13", "R"], index=3, key="pitch_rating")
    season = c5.selectbox("Release window", SEASON_ORDER, index=3, key="pitch_season",
                          help="Winter = Jan–Feb, Spring = Mar–Apr, Summer = May–Aug, Fall = Sep–Oct, Holiday = Nov–Dec.")
    year = c6.slider("Release year", 2000, 2027, 2026, key="pitch_year",
                     help="The model has a gentle upward trend in nominal grosses; years after 2025 extend it.")
    fu = 1 if kind == "Follow-up" else 0
    mid = pitch_predict(budget, genre, fu, rating, season, year)
    q = PITCH["resid_q"]
    lo80, lo50, hi50, hi80 = (mid * math.exp(q[k]) for k in ("p10", "p25", "p75", "p90"))
    peers = FILMS[(FILMS["genre"] == genre) & FILMS["domestic"].notna() & FILMS["worldwide"].notna()]
    ds = float((peers["domestic"] / peers["worldwide"]).median()) if len(peers) else 0.5
    be_t, be_f = breakeven(a, ds)
    need = be_f * budget
    resid = np.array(PITCH["resid"])
    p_win = float((mid * np.exp(resid) >= need).mean() * 100)
    k_full = (1 + a["pa"]) / be_f
    profit = lambda gross: gross * k_full - budget * (1 + a["pa"])

    cols = st.columns(4)
    cols[0].metric("Predicted worldwide gross", money(mid), help="The model's median prediction.")
    short = lambda v: f"${v / 1e9:.1f}B" if v >= 1e9 else f"${v / 1e6:.0f}M"
    cols[1].metric("Likely range (middle 50%)", f"{short(lo50)}–{short(hi50)}",
                   help=f"80% range: {money(lo80)} – {money(hi80)}.")
    cols[2].metric("Needed to break even", money(need), help=f"{be_f:.1f}× budget over the full cycle "
                                                              f"({ds * 100:.0f}% domestic, the {genre.lower()} median).")
    cols[3].metric("Chance of a profit", f"{p_win:.0f}%",
                   help="Share of the model's past misses that would still leave this film above break-even.")
    takeaway(f"A {money(budget)} {kind.lower()} {genre.lower()} film rated {rating}, released in {season.lower()} "
             f"{year}: typical outcome <b>{money(mid)}</b> worldwide and an estimated studio profit of "
             f"<b>{money(profit(mid))}</b>, with a <b>{p_win:.0f}% chance</b> of making money under your deal terms.")

    sc = pd.DataFrame([{"Case": "Bad case (1 in 10 do worse)", "Gross": lo80}, {"Case": "Weak (1 in 4)", "Gross": lo50},
                       {"Case": "Typical", "Gross": mid}, {"Case": "Strong (1 in 4)", "Gross": hi50},
                       {"Case": "Great case (1 in 10 do better)", "Gross": hi80}])
    sc["Profit"] = sc["Gross"].map(profit) / 1e6
    order = list(sc["Case"])
    base = alt.Chart(sc).encode(x=alt.X("Case:N", sort=order, title=None, axis=alt.Axis(labelAngle=0, labelLimit=150)))
    bars = base.mark_bar(size=50, cornerRadiusEnd=3).encode(
        y=alt.Y("Profit:Q", title="Estimated studio profit ($M)"),
        color=alt.condition("datum.Profit >= 0", alt.value(BLUE), alt.value(INK)),
        tooltip=["Case", alt.Tooltip("Gross:Q", format="$,.0f", title="Worldwide gross"),
                 alt.Tooltip("Profit:Q", format="+,.0f", title="Profit ($M)")])
    lab = base.mark_text(fontSize=11, color=INK, dy=-8).encode(y=alt.Y("Profit:Q"),
                                                               text=alt.Text("Profit:Q", format="+,.0f"))
    zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#6b7280").encode(y="y:Q")
    show_chart((bars + lab + zero).properties(height=300),
               f"{p_win:.0f}% chance of profit: loses money in the weak case" if profit(lo50) < 0
               else f"{p_win:.0f}% chance of profit: still profitable in the weak case",
               "Estimated full-cycle studio profit across the range of outcomes, $M")
    st.caption(f"*Ranges come from how far real films landed from this model's predictions on films it didn't see "
               f"(cross-validated R² {PITCH['r2_cv']:.2f}: the model explains about {PITCH['r2_cv'] * 100:.0f}% of the "
               f"variation; the rest is execution, marketing, competition and luck). Nominal dollars. An association "
               f"model, not a causal one: summer slots and sequels go to films studios already believe in. And the films "
               f"in this dataset are ones whose scripts were available, which tilts toward well-known films, so "
               f"read the chances as optimistic.*")

    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Comparable films** (same genre, closest budget, status and year)")
        show_comps(comps(genre, budget, fu, year))
    with right:
        st.markdown("**What moves the forecast**")
        pr = PITCH["params"]
        lines = [("Doubling the budget", (2 ** pr["Budget (log)"] - 1) * 100),
                 ("Follow-up instead of original", (math.exp(pr["Follow-up film"]) - 1) * 100),
                 ("Summer instead of winter", (math.exp(pr["Summer release"]) - 1) * 100),
                 ("PG-13 instead of R", (math.exp(pr["Rated PG-13"]) - 1) * 100),
                 (f"{genre} instead of drama", (math.exp(pr.get(f"Genre: {genre}", 0)) - 1) * 100)]
        show_table(pd.DataFrame(lines, columns=["Change", "Effect on predicted gross"]), column_config={
            "Effect on predicted gross": st.column_config.NumberColumn(
                format="%+.0f%%", help="% change in the predicted worldwide gross from this one change, all else equal.")})
        st.caption("*Holding the other choices fixed. Doubling the budget raises the predicted gross by less than "
                   "double, which is why bigger budgets need a bigger hit to pay back.*")


# --- Sidebar: menu + glossary ---------------------------------------------------------------------------------------
GLOSSARY = {
    "Return (× budget)": "Worldwide box office ÷ production budget. Also called the gross multiple.",
    "Film rentals": "The studio's share of ticket sales after theaters take theirs: about half in the US, less abroad.",
    "P&A": "Prints and advertising: the marketing spend to launch a film, on top of the production budget.",
    "Break-even multiple": "The return (× budget) a film needs before rentals cover budget plus marketing.",
    "Windows": "The release sequence after theaters: PVOD, digital and disc, then pay TV and streaming, then free TV.",
    "PVOD": "Premium video-on-demand: renting a new film at home for $20–30, weeks after it opens.",
    "Follow-up": "A sequel, prequel, reboot, remake or spin-off of an earlier film. Adaptations count as originals.",
    "Week-2 drop": "% of the first week's gross lost in the second week. Higher = more front-loaded.",
    "Abnormal return": "A stock's return minus the overall market's, so market-wide moves don't count.",
    "Earnings-call tone": "How upbeat executives sounded on the quarterly call, scored by FinBERT (−1 to +1).",
    "ROA": "Return on assets: net income ÷ total assets.",
    "Profit margin": "Net income ÷ revenue.",
    "Leverage": "Debt ÷ total assets.",
    "R²": "Share of the variation a model explains: 1 = perfect, 0 = no better than guessing the average.",
    "Cross-validation": "Scoring a model on films it didn't learn from, several times over, for an honest grade.",
    "p-value": "How surprising a result would be if there were no real relationship. Below 0.10 = flagged here.",
    "95% interval": "The range that would contain the true average 95% of the time. Crossing zero = can't rule out none.",
}
for term, key in [("Return (× budget)", "return"), ("Follow-up", "follow-up"), ("Week-2 drop", "week-2 drop"),
                  ("Earnings-call tone", "tone"), ("ROA", "roa"), ("Profit margin", "margin"),
                  ("Leverage", "leverage"), ("R²", "r²"), ("p-value", "p-value")]:
    GLOSSARY[term] = define(key)
with st.sidebar:
    st.markdown("### Sections")
    st.markdown("\n".join(f"{NAV_NUM[t]}. [{t}](#{a})" for t, a in NAV.items()))
    st.markdown("### Glossary")
    for term, text in GLOSSARY.items():
        st.markdown(f"**{term}**  \n<span style='color:{MUTED};font-size:.85rem'>{text}</span>",
                    unsafe_allow_html=True)

# --- Page: one long scroll, in NAV order -----------------------------------------------------------------------------
studio_section()
trends_section()
script_section()
economics_and_pitch()
wallstreet_section()
predict_section()
database_section()
