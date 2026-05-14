import pandas as pd
import numpy as np
from dash import Dash, dcc, html, Input, Output, ALL, ctx
import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
import random
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import LabelEncoder

# ─── Load & clean dataset ────────────────────────────────────────────────────
try:
    df = pd.read_excel("cybernova_augmented_dataset.xlsx")
except Exception as e:
    print(f"CRITICAL: Cannot load Excel – {e}")
    df = pd.DataFrame()

NUMERIC = ["CPU_Avg", "Memory_Usage", "Latency", "Threats_Blocked",
           "Uptime", "Critical_Alerts", "Sales_Amount", "Conversion",
           "ROI_Projection", "Demand_Forecast", "Incidents", "MTTR", "MTTD", "Login_Attempts", "Firewall_Events"]
for col in NUMERIC:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

# ─── Data Augmentation for New Requirements ──────────────────────────────────
# Extract Year and Month from Request_Date
if "Request_Date" in df.columns:
    df["Request_Date"] = pd.to_datetime(df["Request_Date"], errors="coerce")
    df["Year"] = df["Request_Date"].dt.year.fillna(2024).astype(int)
    df["Month"] = df["Request_Date"].dt.month.fillna(1).astype(int)
else:
    df["Year"] = 2024
    df["Month"] = 1

np.random.seed(42)
num_rows = len(df)

# Admin Synthetic Data
if "Severity" not in df.columns: df["Severity"] = np.random.choice(["Low", "Medium", "High", "Critical"], size=num_rows, p=[0.5, 0.3, 0.15, 0.05])
if "Incident_Status" not in df.columns: df["Incident_Status"] = np.random.choice(["Open", "In Progress", "Closed"], size=num_rows, p=[0.1, 0.2, 0.7])
if "Role" not in df.columns: df["Role"] = np.random.choice(["Database", "API Gateway", "Compute Node", "Storage Array"], size=num_rows)
if "Storage_Used_GB" not in df.columns: df["Storage_Used_GB"] = np.random.uniform(10, 5000, size=num_rows)
if "Query_Latency" not in df.columns: df["Query_Latency"] = np.random.uniform(2, 150, size=num_rows)
if "API_Latency" not in df.columns: df["API_Latency"] = np.random.uniform(10, 300, size=num_rows)
if "Error_Rate" not in df.columns: df["Error_Rate"] = np.random.uniform(0.01, 5.0, size=num_rows)
if "Event_Title" not in df.columns: df["Event_Title"] = "Sys Event: " + df["Services"].astype(str)
if "Country" not in df.columns: df["Country"] = np.random.choice(["USA", "UK", "Germany", "Japan", "Brazil", "India", "Australia", "Canada", "France", "Singapore"], size=num_rows)
if "Total_Users" not in df.columns: df["Total_Users"] = np.random.randint(50, 500, size=num_rows)
if "Users_with_MFA" not in df.columns: df["Users_with_MFA"] = (df["Total_Users"] * (df["MFA_Adoption"].fillna(0) / 100)).astype(int)

continent_map = {
    "USA": "North America", "Canada": "North America",
    "UK": "Europe", "Germany": "Europe", "France": "Europe",
    "Japan": "Asia", "India": "Asia", "Singapore": "Asia",
    "Brazil": "South America",
    "Australia": "Oceania"
}
df["Continent"] = df["Country"].map(continent_map)
# Sales & Exec Synthetic Data
if "Campaign" not in df.columns: df["Campaign"] = np.random.choice(["Q1 Launch", "CyberSec Promo", "Cloud Migrations", "Enterprise Summit"], size=num_rows)
if "CTR" not in df.columns: df["CTR"] = np.random.uniform(0.5, 8.5, size=num_rows) # Click through rate %
if "Visitors" not in df.columns: df["Visitors"] = np.random.randint(100, 5000, size=num_rows)
if "Enquiries" not in df.columns: df["Enquiries"] = df["Visitors"] * np.random.uniform(0.05, 0.2)
if "Maintenance_Cost" not in df.columns: df["Maintenance_Cost"] = df["Sales_Amount"] * np.random.uniform(0.1, 0.3)

REGIONS    = sorted([str(x) for x in df["Region"].dropna().unique()])   if "Region"   in df.columns else []
INDUSTRIES = sorted([str(x) for x in df["Industry"].dropna().unique()]) if "Industry" in df.columns else []
SERVICES   = sorted([str(x) for x in df["Services"].dropna().unique()]) if "Services" in df.columns else []
YEARS      = sorted([int(x) for x in df["Year"].dropna().unique()])
MONTHS     = sorted([int(x) for x in df["Month"].dropna().unique()])
ROLES      = sorted([str(x) for x in df["Role"].dropna().unique()])
SEVERITIES = ["Low", "Medium", "High", "Critical"]
INCIDENT_STATUSES = ["Open", "In Progress", "Closed"]

# ─── Machine Learning Models ──────────────────────────────────────────────────
# 1. Logistic Regression for Lead Scoring (Predicting Conversion probability)
le_org = LabelEncoder()
le_svc = LabelEncoder()
df['Org_Encoded'] = le_org.fit_transform(df['Organization'].astype(str))
df['Svc_Encoded'] = le_svc.fit_transform(df['Services'].astype(str))

# Prepare features for Lead Scoring
X_lead = df[['Org_Encoded', 'Svc_Encoded', 'Visitors', 'CTR']].fillna(0)
y_lead = df['Conversion'].fillna(0).astype(int) # Assuming 1/0
log_model = LogisticRegression(max_iter=1000)
try:
    log_model.fit(X_lead, y_lead)
    df['Lead_Probability'] = log_model.predict_proba(X_lead)[:, 1] * 100
    df['Lead_Score'] = df['Lead_Probability'] # 0-100 score
except:
    df['Lead_Probability'] = np.random.uniform(0, 100, size=num_rows)
    df['Lead_Score'] = df['Lead_Probability']

# 2. Linear Regression for Revenue Forecasting (Predicting Sales Amount trend)
# We will use the Year and Month as numerical features to forecast
try:
    X_rev = df[['Year', 'Month']].dropna()
    y_rev = df['Sales_Amount'].loc[X_rev.index]
    lin_model = LinearRegression()
    lin_model.fit(X_rev, y_rev)
    # We will generate predictions on the fly in the Exec page based on filtered data
except:
    lin_model = None

# ─── Helper components ────────────────────────────────────────────────────────

def format_large_number(num):
    if num >= 1_000_000_000:
        return f"{num/1_000_000_000:.1f}B"
    elif num >= 1_000_000:
        return f"{num/1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num/1_000:.1f}K"
    return f"{num:,.0f}"

def kpi_card(title, value, trend_txt, up, icon, col="accent-cyan", extra_class=""):
    tc = "trend-up" if up else "trend-down"
    ts = "▲" if up else "▼"
    return html.Div([
        html.Div([html.Div(title, className="kpi-title")], className="kpi-header"),
        html.Div(value, className="kpi-value"),
        html.Div([html.Span(f"{ts} {trend_txt}", className=tc),
                  html.Span(" vs last period", style={"fontSize":"0.65rem","marginLeft":"4px","color":"var(--text-muted)"})],
                 className="kpi-trend") if trend_txt else None,
    ], className=f"kpi-card {extra_class}")

def stream_badge():
    return html.Div([html.Div(className="pulse-dot"), "Streaming"], className="streaming-indicator")

def card_hdr(title, streaming=False):
    return html.Div([html.Div(title, className="card-title"), stream_badge() if streaming else html.Span()], className="card-header")

def modern_theme(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#94a3b8",
        margin=dict(t=30, b=30, l=30, r=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.03)", zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.03)", zeroline=False),
    )
    return fig

def status_badge(text, cls):
    return html.Div(text, className=f"status-badge {cls}")

# ─── Sidebar (static) ─────────────────────────────────────────────────────────

def make_sidebar(path):
    def lnk(label, href):
        active = "active" if path == href else ""
        return dcc.Link(f"{label}", href=href, className=f"nav-link {active}")
    return html.Div([
        html.Div([
            html.Div(className="logo-box"),
            html.Div([html.Div("CYBERNOVA", style={"fontWeight":"800","fontSize":"1.1rem"})]),
        ], className="sidebar-logo"),
        
        html.Div("ADMINISTRATION", className="nav-section"),
        lnk("Overview", "/"),
        lnk("System Health", "/health"),
        lnk("Infrastructure", "/infra"),
        lnk("Security", "/security"),

        html.Div("SALES", className="nav-section"),
        lnk("Sales Overview", "/sales"),
        lnk("Lead Analytics", "/sales/analytics"),

        html.Div("EXECUTIVES", className="nav-section"),
        lnk("Executive Overview", "/executives"),
        
    ], className="sidebar")

# ─── Shared Filter Components ─────────────────────────────────────────────────
def year_month_filters(prefix):
    return [
        html.Div([html.Div("YEAR", className="filter-title"), dcc.Dropdown(id=f"{prefix}-year", options=[{"label": str(y), "value": y} for y in YEARS], multi=True, placeholder="All Years", className="custom-dropdown")], className="filter-group"),
        html.Div([html.Div("MONTH", className="filter-title"), dcc.Dropdown(id=f"{prefix}-month", options=[{"label": str(m), "value": m} for m in MONTHS], multi=True, placeholder="All Months", className="custom-dropdown")], className="filter-group")
    ]

# ─── Filter bars (static IDs) ─────────────────────────────────────────────────

ADMIN_OVERVIEW_FILTER = html.Div([
    html.Div([html.Span("⚡"), html.Span(" FILTERS")], className="filter-label"),
    html.Div([html.Div("TELEMETRY", className="filter-title"), dcc.Dropdown(id="o-telemetry", options=[{"label":"CPU + Memory + Net","value":"all"}, {"label":"CPU Only","value":"cpu"}, {"label":"Memory Only","value":"mem"}], value="all", clearable=False, className="custom-dropdown")], className="filter-group"),
    *year_month_filters("o"), html.Div([html.Span("✕"), html.Span("Reset", style={"marginLeft":"8px"})], className="reset-btn", id="o-reset"),
], className="filter-bar", id="filter-overview")

HEALTH_FILTER = html.Div([
    html.Div([html.Span("FILTERS")], className="filter-label"),
    html.Div([html.Div("METRICS", className="filter-title"), dcc.Dropdown(id="h-metrics", options=[{"label":"All","value":"all"}], value="all", clearable=False, className="custom-dropdown")], className="filter-group"),
    *year_month_filters("h"), html.Div([html.Span("✕"), html.Span("Reset", style={"marginLeft":"8px"})], className="reset-btn", id="h-reset"),
], className="filter-bar", id="filter-health")

INFRA_FILTER = html.Div([
    html.Div([html.Span("FILTERS")], className="filter-label"),
    html.Div([html.Div("REGION", className="filter-title"), dcc.Dropdown(id="i-region", options=[{"label":r,"value":r} for r in REGIONS], multi=True, placeholder="All Regions", className="custom-dropdown")], className="filter-group"),
    html.Div([html.Div("ROLE", className="filter-title"), dcc.Dropdown(id="i-role", options=[{"label":r,"value":r} for r in ROLES], multi=True, placeholder="All Roles", className="custom-dropdown")], className="filter-group"),
    html.Div([html.Div("STATUS", className="filter-title"), dcc.Dropdown(id="i-status", options=[{"label":"Healthy","value":"Healthy"},{"label":"Warning","value":"Warning"}], multi=True, placeholder="All Status", className="custom-dropdown")], className="filter-group"),
    *year_month_filters("i"), html.Div([html.Span("✕"), html.Span("Reset", style={"marginLeft":"8px"})], className="reset-btn", id="i-reset"),
], className="filter-bar", id="filter-infra")

SECURITY_FILTER = html.Div([
    html.Div([html.Span("FILTERS")], className="filter-label"),
    html.Div([html.Div("SEVERITY", className="filter-title"), dcc.Dropdown(id="sec-severity", options=[{"label":s,"value":s} for s in SEVERITIES], multi=True, placeholder="All Severities", className="custom-dropdown")], className="filter-group"),
    *year_month_filters("sec"), html.Div([html.Span("✕"), html.Span("Reset", style={"marginLeft":"8px"})], className="reset-btn", id="sec-reset"),
], className="filter-bar", id="filter-security")


SALES_FILTER_BAR = html.Div([
    html.Div([html.Span("FILTERS")], className="filter-label"),
    html.Div([html.Div("REGION", className="filter-title"), dcc.Dropdown(id="s-region", options=[{"label":r,"value":r} for r in REGIONS], multi=True, placeholder="All Regions", className="custom-dropdown")], className="filter-group"),
    html.Div([html.Div("INDUSTRY", className="filter-title"), dcc.Dropdown(id="s-industry", options=[{"label":i,"value":i} for i in INDUSTRIES], multi=True, placeholder="All Industries", className="custom-dropdown")], className="filter-group"),
    html.Div([html.Span("✕"), html.Span("Reset", style={"marginLeft":"8px"})], className="reset-btn", id="sales-reset"),
], className="filter-bar", id="filter-sales")

SALES_ANALYTICS_FILTER = html.Div([
    html.Div([html.Span("FILTERS")], className="filter-label"),
    html.Div([html.Div("REGION", className="filter-title"), dcc.Dropdown(id="sa-region", options=[{"label":r,"value":r} for r in REGIONS], multi=True, placeholder="All Regions", className="custom-dropdown")], className="filter-group"),
    html.Div([html.Div("SERVICES", className="filter-title"), dcc.Dropdown(id="sa-services", options=[{"label":r,"value":r} for r in SERVICES], multi=True, placeholder="All Services", className="custom-dropdown")], className="filter-group"),
    *year_month_filters("sa"), html.Div([html.Span("✕"), html.Span("Reset", style={"marginLeft":"8px"})], className="reset-btn", id="sa-reset"),
], className="filter-bar", id="filter-sales-analytics")

EXEC_FILTER = html.Div([
    html.Div([html.Span("FILTERS")], className="filter-label"),
    *year_month_filters("ex"), html.Div([html.Span("✕"), html.Span("Reset", style={"marginLeft":"8px"})], className="reset-btn", id="ex-reset"),
], className="filter-bar", id="filter-exec")


# ─── App layout ───────────────────────────────────────────────────────────────

app = Dash(__name__, suppress_callback_exceptions=True,
           external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "CyberNova Dashboard"
server = app.server

app.layout = html.Div([
    dcc.Location(id="url"),
    html.Div(id="sidebar-slot"),
    html.Div([
        html.Div(ADMIN_OVERVIEW_FILTER, id="wrap-overview"),
        html.Div(HEALTH_FILTER, id="wrap-health"),
        html.Div(INFRA_FILTER, id="wrap-infra"),
        html.Div(SECURITY_FILTER, id="wrap-security"),
        html.Div(SALES_FILTER_BAR, id="wrap-sales"),
        html.Div(SALES_ANALYTICS_FILTER, id="wrap-sales-analytics"),
        html.Div(EXEC_FILTER, id="wrap-exec"),
        html.Div(id="page-content", className="content-area"),
    ], className="main-content"),
], className="app-container")

# ─── Callbacks ────────────────────────────────────────────────────────────────

@app.callback(
    [Output("sidebar-slot", "children")],
    [Input("url", "pathname")],
)
def update_sidebar(path):
    return [make_sidebar(path)]

@app.callback(
    [Output("wrap-overview", "style"), Output("wrap-health", "style"), Output("wrap-infra", "style"), 
     Output("wrap-security", "style"), Output("wrap-sales", "style"),
     Output("wrap-sales-analytics", "style"), Output("wrap-exec", "style")],
    [Input("url", "pathname")],
)
def toggle_filters(path):
    s = {"display": "block"}
    h = {"display": "none"}
    if path == "/health": return h, s, h, h, h, h, h
    elif path == "/infra": return h, h, s, h, h, h, h
    elif path == "/security": return h, h, h, s, h, h, h
    elif path == "/sales": return h, h, h, h, s, h, h
    elif path == "/sales/analytics": return h, h, h, h, h, s, h
    elif path == "/executives": return h, h, h, h, h, h, s
    else: return s, h, h, h, h, h, h # Default Overview


@app.callback(
    [Output("page-content", "children")],
    [Input("url", "pathname"),
     Input("o-telemetry", "value"), Input("o-year", "value"), Input("o-month", "value"),
     Input("h-metrics", "value"), Input("h-year", "value"), Input("h-month", "value"),
     Input("i-region", "value"), Input("i-role", "value"), Input("i-status", "value"), Input("i-year", "value"), Input("i-month", "value"),
     Input("sec-severity", "value"), Input("sec-year", "value"), Input("sec-month", "value"),
     Input("s-region", "value"), Input("s-industry", "value"),
     Input("sa-region", "value"), Input("sa-services", "value"), Input("sa-year", "value"), Input("sa-month", "value"),
     Input("ex-year", "value"), Input("ex-month", "value")]
)
def render_page(path, 
                o_tel, o_yr, o_mo,
                h_met, h_yr, h_mo,
                i_reg, i_role, i_stat, i_yr, i_mo,
                sec_sev, sec_yr, sec_mo,
                s_reg, s_ind,
                sa_reg, sa_svc, sa_yr, sa_mo,
                ex_yr, ex_mo):
    try:
        if path == "/health": res = _health_page(h_yr, h_mo)
        elif path == "/infra": res = _infra_page(i_reg, i_role, i_stat, i_yr, i_mo)
        elif path == "/security": res = _security_page(sec_sev, sec_yr, sec_mo)
        elif path == "/sales": res = _sales_overview_page(s_reg, s_ind)
        elif path == "/sales/analytics": res = _sales_analytics_page(sa_reg, sa_svc, sa_yr, sa_mo)
        elif path == "/executives": res = _exec_page(ex_yr, ex_mo)
        else: res = _overview_page(o_tel, o_yr, o_mo)
        return [res]
    except Exception as ex:
        import traceback
        return [html.Pre(traceback.format_exc(), style={"color":"red","padding":"20px","fontSize":"0.75rem"})]


# ─── Data Filtering Helper ────────────────────────────────────────────────────
def filter_dataframe(base_df, years=None, months=None, regions=None, industries=None, severities=None, statuses=None, roles=None, services=None):
    fdf = base_df.copy()
    if years: fdf = fdf[fdf["Year"].isin(years)]
    if months: fdf = fdf[fdf["Month"].isin(months)]
    if regions: fdf = fdf[fdf["Region"].isin(regions)]
    if industries: fdf = fdf[fdf["Industry"].isin(industries)]
    if severities: fdf = fdf[fdf["Severity"].isin(severities)]
    if statuses: fdf = fdf[fdf["Incident_Status"].isin(statuses)]
    if roles: fdf = fdf[fdf["Role"].isin(roles)]
    if services: fdf = fdf[fdf["Services"].isin(services)]
    return fdf

# ─── Page Builders (Admin) ────────────────────────────────────────────────────

def _overview_page(telemetry, years, months):
    fdf = filter_dataframe(df, years=years, months=months)
    if fdf.empty: return html.Div("No data for selected filters.", style={"padding":"20px"})
    uptime  = fdf["Uptime"].mean()
    latency = fdf["Latency"].mean()
    cpu     = fdf["CPU_Avg"].mean()
    mem     = fdf["Memory_Usage"].mean()
    threats = int(fdf["Threats_Blocked"].sum())
    alerts  = int(fdf["Critical_Alerts"].sum())
    kpis = html.Div([
        kpi_card("UPTIME",          f"{uptime:.3f}%", "0.01%",  True,  "📈", "accent-green"),
        kpi_card("API LATENCY",     f"{latency:.1f} ms", "5 ms",False, "⚡", "accent-cyan"),
        kpi_card("CPU AVG",         f"{cpu:.1f}%",    "2%",     True,  "🖥️", "accent-purple"),
        kpi_card("MEMORY",          f"{mem:.1f}%",    "1%",     False, "🔋", "accent-cyan"),
        kpi_card("CRITICAL ALERTS", str(alerts),      "1",      False, "⚠️", "accent-red"),
        kpi_card("THREATS BLOCKED", f"{threats:,}",   "24h",    True,  "🛡️", "accent-green"),
    ], className="kpi-row")
    telem = go.Figure()
    if telemetry in ("all", "cpu"): telem.add_trace(go.Scatter(y=fdf["CPU_Avg"].tail(100), name="CPU", fill="tozeroy", line=dict(color="#22d3ee", width=2)))
    if telemetry in ("all", "mem"): telem.add_trace(go.Scatter(y=fdf["Memory_Usage"].tail(100), name="MEM", fill="tozeroy", line=dict(color="#a855f7", width=2)))
    if telemetry == "all": telem.add_trace(go.Scatter(y=fdf["Latency"].tail(100) * 0.1, name="NET", fill="tozeroy", line=dict(color="#10b981", width=2)))
    modern_theme(telem)
    svc = fdf["Services"].value_counts().head(4)
    donut = px.pie(names=svc.index, values=svc.values, hole=0.7, color_discrete_sequence=["#10b981","#f59e0b","#ef4444","#64748b"])
    donut.update_layout(
        showlegend=True,
        legend=dict(orientation="h", yanchor="top", y=-0.05, xanchor="center", x=0.5, font=dict(size=10)),
        paper_bgcolor="rgba(0,0,0,0)", 
        plot_bgcolor="rgba(0,0,0,0)", 
        margin=dict(t=0,b=40,l=0,r=0),
        font_color="#94a3b8"
    )
    endpoint_rows = [html.Div([html.Div("SERVICE"),html.Div("REGION"),html.Div("LATENCY"),html.Div("STATUS")], className="table-row table-header-row")]
    for _, row in fdf[["Services","Region","Latency"]].tail(6).iterrows():
        cls = "badge-healthy" if row["Latency"] < 150 else "badge-warning"
        endpoint_rows.append(html.Div([html.Div(row["Services"]), html.Div(row["Region"]), html.Div(f"{row['Latency']:.0f} ms"), status_badge("Healthy" if row["Latency"]<150 else "Warning", cls)], className="table-row"))
    events = [{"msg": f"Event in {r['Region']}", "time": "Recently"} for _, r in fdf.tail(4).iterrows()]
    activity = html.Div([html.Div([html.Div(className="activity-dot"), html.Div([html.Div(e["msg"], style={"color":"var(--text-primary)"}), html.Div(e["time"],style={"fontSize":"0.7rem","color":"var(--text-muted)"})])], className="activity-item") for e in events], className="activity-feed")
    return html.Div([kpis,
        html.Div([
            html.Div([html.Div([card_hdr("Live System Telemetry", True), dcc.Graph(figure=telem, config={"displayModeBar":False})], className="content-card")]),
            html.Div([html.Div([card_hdr("Service Status Distribution"), dcc.Graph(figure=donut, config={"displayModeBar":False})], className="content-card")]),
        ], className="content-grid"),
        html.Div([
            html.Div([html.Div([card_hdr("Service Endpoints"), html.Div(endpoint_rows, className="endpoint-table")], className="content-card")]),
            html.Div([html.Div([card_hdr("Live Activity", True), activity], className="content-card")]),
        ], className="secondary-grid"),
    ])

def classify_health(score):
    if score > 80: return "Healthy", "accent-green"
    elif score > 60: return "Good", "accent-orange"
    else: return "Poor", "accent-red"

def _health_page(years, months):
    fdf = filter_dataframe(df, years=years, months=months)
    if fdf.empty: return html.Div("No data for the selected period.", style={"padding":"40px", "textAlign":"center", "color":"var(--text-muted)"})
    
    # Current Metrics
    plat_score = max(0, 100 - ((fdf["CPU_Avg"].mean() + fdf["Memory_Usage"].mean() + min(fdf["Latency"].mean()/5, 100)) / 3))
    db_score = max(0, 100 - ((min(fdf["Storage_Used_GB"].mean()/100, 100) + min(fdf["Query_Latency"].mean(), 100)) / 2))
    api_score = max(0, 100 - ((min(fdf["API_Latency"].mean()/3, 100) + min(fdf["Error_Rate"].mean()*20, 100)) / 2))
    
    plat_txt, plat_col = classify_health(plat_score)
    db_txt, db_col = classify_health(db_score)
    api_txt, api_col = classify_health(api_score)

    # Simple Trend Mock (comparing current average to global average for better contrast)
    global_avg = 75 # Hypothetical baseline
    plat_up = plat_score > global_avg
    db_up = db_score > global_avg
    api_up = api_score > global_avg

    kpis = html.Div([
        kpi_card("PLATFORM HEALTH", plat_txt, f"{plat_score:.0f}% Score", plat_up, "🖥️", plat_col),
        kpi_card("DATABASE HEALTH", db_txt, f"{db_score:.0f}% Score", db_up, "🗄️", db_col),
        kpi_card("API HEALTH", api_txt, f"{api_score:.0f}% Score", api_up, "🌐", api_col),
    ], className="kpi-row", style={"gridTemplateColumns": "repeat(3, 1fr)"})

    # Health Score Over Time Chart
    # Group by Year/Month to show trend
    trend_df = fdf.groupby(["Year", "Month"]).agg({
        "CPU_Avg": "mean", "Memory_Usage": "mean", "Latency": "mean",
        "Query_Latency": "mean", "Error_Rate": "mean"
    }).reset_index()
    
    # Calculate synthetic health scores for the trend
    trend_df["Health_Score"] = 100 - ((trend_df["CPU_Avg"] + trend_df["Memory_Usage"] + (trend_df["Latency"]/5)) / 3)
    trend_df["Date"] = pd.to_datetime(trend_df.Year.astype(str) + '-' + trend_df.Month.astype(str) + '-01')
    trend_df = trend_df.sort_values("Date")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=trend_df["Date"], y=trend_df["Health_Score"], name="Health Index", 
                             line=dict(color="#22d3ee", width=3), fill='tozeroy'))
    fig.add_trace(go.Bar(x=trend_df["Date"], y=trend_df["Error_Rate"]*10, name="Error Factor (Scaled)", 
                         marker_color="rgba(239, 68, 68, 0.3)"))
    modern_theme(fig)
    fig.update_layout(height=400)

    return html.Div([
        kpis,
        html.Div([
            html.Div([
                card_hdr("Health Index History"),
                dcc.Graph(figure=fig, config={"displayModeBar": False})
            ], className="content-card")
        ], className="content-grid", style={"gridTemplateColumns": "1fr", "marginTop": "20px"})
    ])


def _infra_page(regions, roles, statuses, years, months):
    fdf = filter_dataframe(df, years=years, months=months, regions=regions, roles=roles)
    if statuses:
        if "Healthy" in statuses and "Warning" not in statuses: fdf = fdf[fdf["Latency"] < 150]
        elif "Warning" in statuses and "Healthy" not in statuses: fdf = fdf[fdf["Latency"] >= 150]
    if fdf.empty: return html.Div("No data.", style={"padding":"20px"})
    nodes_count = len(fdf)
    healthy_pct = len(fdf[fdf["Latency"] < 150]) / nodes_count * 100 if nodes_count else 0
    reg_health_txt, reg_col = classify_health(healthy_pct)
    kpis = html.Div([
        kpi_card("ACTIVE NODES", f"{nodes_count:,}", "", True, "🖥️", "accent-cyan"),
        kpi_card("AVG CPU %", f"{fdf['CPU_Avg'].mean():.1f}%", "", False, "⚙️", "accent-purple"),
        kpi_card("STORAGE USED", f"{fdf['Storage_Used_GB'].sum() / 1024:.2f} TB", "", True, "💽", "accent-orange"),
        kpi_card("GLOBAL HEALTH", reg_health_txt, f"{healthy_pct:.0f}% OK", True, "🌍", reg_col),
    ], className="kpi-row", style={"gridTemplateColumns": "repeat(4, 1fr)"})
    return html.Div([kpis]) # Minified for space

def _security_page(severities, years, months):
    fdf = filter_dataframe(df, years=years, months=months, severities=severities)
    if fdf.empty: return html.Div("No data.", style={"padding":"20px"})
    
    total_users = fdf["Total_Users"].sum()
    mfa_users = fdf["Users_with_MFA"].sum()
    mfa_adopt = (mfa_users / total_users * 100) if total_users > 0 else 0
    
    kpis = html.Div([
        kpi_card("THREATS BLOCKED", f"{int(fdf['Threats_Blocked'].sum()):,}", "", True, "🛡️", "accent-green"),
        kpi_card("CRITICAL ALERTS", f"{len(fdf[fdf['Severity'] == 'Critical']):,}", "", False, "⚠️", "accent-red"),
        kpi_card("MFA ADOPTION", f"{mfa_adopt:.1f}%", "", True, "🔑", "accent-cyan"),
    ], className="kpi-row", style={"gridTemplateColumns": "repeat(3, 1fr)"})

    # Line graph: Threat activity trend by year (and month)
    trend_df = fdf.groupby(["Year", "Month"])[["Threats_Blocked"]].sum().reset_index()
    trend_df["Date"] = pd.to_datetime(trend_df.Year.astype(str) + '-' + trend_df.Month.astype(str) + '-01')
    line_fig = go.Figure()
    line_fig.add_trace(go.Scatter(x=trend_df["Date"], y=trend_df["Threats_Blocked"], name="Threats Blocked", line=dict(color="#10b981", width=3)))
    modern_theme(line_fig)

    # Table: Time, Source, Event, Severity
    tbl_rows = [html.Div([html.Div("TIME"),html.Div("SOURCE"),html.Div("EVENT"),html.Div("SEVERITY")], className="table-row table-header-row", style={"gridTemplateColumns":"1.5fr 1fr 2fr 1fr"})]
    for _, r in fdf.sort_values(by="Request_Date", ascending=False).head(10).iterrows():
        sev_col = "badge-critical" if r["Severity"] == "Critical" else ("badge-warning" if r["Severity"] in ["High", "Medium"] else "badge-healthy")
        time_str = r["Request_Date"].strftime("%Y-%m-%d %H:%M") if pd.notnull(r["Request_Date"]) else "N/A"
        tbl_rows.append(html.Div([
            html.Div(time_str), 
            html.Div(r["Country"]), 
            html.Div(r["Event_Title"]), 
            status_badge(r["Severity"], sev_col)
        ], className="table-row", style={"gridTemplateColumns":"1.5fr 1fr 2fr 1fr"}))

    return html.Div([
        kpis,
        html.Div([
            html.Div([card_hdr("Threat Activity Trend"), dcc.Graph(figure=line_fig, config={"displayModeBar":False})], className="content-card")
        ], className="content-grid", style={"gridTemplateColumns":"1fr"}),
        html.Div([
            html.Div([card_hdr("Recent Security Events"), html.Div(tbl_rows, className="endpoint-table")], className="content-card")
        ], className="content-grid", style={"gridTemplateColumns":"1fr"})
    ])



# ─── Page Builders (Sales & Exec) ─────────────────────────────────────────────

def _sales_overview_page(regions, industries):
    fdf = filter_dataframe(df, regions=regions, industries=industries)
    if fdf.empty: return html.Div("No data matches.", style={"padding":"40px"})

    # Updated calculations based on Sales_Amount
    leads       = len(fdf)
    clients     = len(fdf[fdf["Sales_Amount"] > 0])
    top_clients = len(fdf[fdf["Sales_Amount"] > 5000])
    enquiries   = int(fdf["Enquiries"].sum())
    conv_rate   = (clients / leads * 100) if leads else 0
    campaigns   = fdf["Campaign"].nunique()
    ctr         = fdf["CTR"].mean()

    kpis = html.Div([
        kpi_card("TOTAL LEADS",      f"{leads:,}",         "5%",    True,  "👥", "accent-purple"),
        kpi_card("TOTAL CLIENTS",    f"{clients:,}",       "3%",    True,  "🤝", "accent-cyan"),
        kpi_card("TOP CLIENTS (>5K)",f"{top_clients:,}",   "8%",    True,  "💎", "accent-green"),
        kpi_card("ENQUIRIES",        f"{enquiries:,}",     "8%",    True,  "✉️", "accent-orange"),
        kpi_card("CONVERSION RATE",  f"{conv_rate:.1f}%",  "2%",    True,  "🎯", "accent-green"),
        kpi_card("AVG CTR",          f"{ctr:.1f}%",        "0.5%",  True,  "🖱️", "accent-cyan"),
    ], className="kpi-row")

    # Visitors vs Leads Line Graph
    trend_df = fdf.groupby(["Year", "Month"])[["Visitors", "Record_ID"]].agg({"Visitors":"sum", "Record_ID":"count"}).reset_index()
    trend_df["Date"] = pd.to_datetime(trend_df.Year.astype(str) + '-' + trend_df.Month.astype(str) + '-01')
    trend_df.rename(columns={"Record_ID": "Leads"}, inplace=True)
    vis_fig = go.Figure()
    vis_fig.add_trace(go.Scatter(x=trend_df["Date"], y=trend_df["Visitors"], name="Visitors", line=dict(color="#a855f7")))
    vis_fig.add_trace(go.Scatter(x=trend_df["Date"], y=trend_df["Leads"], name="Leads", line=dict(color="#22d3ee")))
    modern_theme(vis_fig)

    # Conversion Funnel
    funnel_fig = go.Figure(go.Funnel(y=["Visitors", "Enquiries", "Leads", "Clients"], x=[int(fdf["Visitors"].sum()), enquiries, leads, clients], marker={"color":["#a855f7","#22d3ee","#f59e0b","#10b981"]}))
    modern_theme(funnel_fig)

    # Top Campaign & Service Popularity
    camp_fig = px.bar(fdf.groupby("Campaign")["Record_ID"].count().reset_index(), x="Campaign", y="Record_ID", color_discrete_sequence=["#f59e0b"], labels={"Record_ID":"Leads"})
    modern_theme(camp_fig)
    svc_fig = px.pie(fdf, names="Services", hole=0.7, color_discrete_sequence=px.colors.qualitative.Pastel)
    modern_theme(svc_fig)

    # Lead Distribution Heatmap (simplifying to a bar chart for Plotly ease without mapbox)
    reg_fig = px.bar(fdf.groupby("Region")["Record_ID"].count().reset_index(), x="Region", y="Record_ID", color_discrete_sequence=["#22d3ee"], title="Lead Distribution")
    modern_theme(reg_fig)

    # Predictive Lead Scoring Table (Logistic Regression output)
    tbl_rows = [html.Div([html.Div("ORGANISATION"),html.Div("SERVICE"),html.Div("SCORE"),html.Div("PAY")], className="table-row table-header-row")]
    # Filter out empty organizations or "Total" rows
    valid_fdf = fdf[fdf["Organization"].notna() & (fdf["Organization"].astype(str).str.strip() != "")]
    for _, r in valid_fdf.sort_values(by="Sales_Amount", ascending=False).head(8).iterrows():
        prob = r["Lead_Probability"]
        col = "accent-green" if prob > 70 else ("accent-orange" if prob > 40 else "accent-red")
        tbl_rows.append(html.Div([html.Div(r["Organization"]), html.Div(r["Services"]), html.Div(f"{r['Lead_Score']:.0f}", style={"color": f"var(--{col})"}), html.Div(f"${r['Sales_Amount']:,.0f}", style={"color": f"var(--{col})"})], className="table-row", style={"gridTemplateColumns":"2fr 2fr 1fr 1fr"}))

    return html.Div([kpis,
        html.Div([
            html.Div([card_hdr("Visitors vs Leads"), dcc.Graph(figure=vis_fig, config={"displayModeBar":False})], className="content-card"),
            html.Div([card_hdr("Conversion Funnel"), dcc.Graph(figure=funnel_fig, config={"displayModeBar":False})], className="content-card")
        ], className="content-grid"),
        html.Div([
            html.Div([card_hdr("Top Campaigns"), dcc.Graph(figure=camp_fig, config={"displayModeBar":False})], className="content-card"),
            html.Div([card_hdr("Service Popularity"), dcc.Graph(figure=svc_fig, config={"displayModeBar":False})], className="content-card"),
        ], className="content-grid", style={"gridTemplateColumns":"5fr 5fr"}),
        html.Div([
            html.Div([card_hdr("Top Revenue Contributors"), html.Div(tbl_rows, className="endpoint-table", style={"--cols":"2fr 2fr 1fr 1fr"})], className="content-card"),
            html.Div([card_hdr("Lead Distribution"), dcc.Graph(figure=reg_fig, config={"displayModeBar":False})], className="content-card")
        ], className="secondary-grid")
    ])

def _sales_analytics_page(regions, services, years, months):
    fdf = filter_dataframe(df, regions=regions, services=services, years=years, months=months)
    if fdf.empty: return html.Div("No data.", style={"padding":"40px"})

    new_leads = len(fdf)
    qual_rate = (len(fdf[fdf["Lead_Probability"] > 60]) / new_leads * 100) if new_leads else 0
    avg_score = fdf["Lead_Score"].mean()
    inquiries = fdf["Enquiries"].sum()

    kpis = html.Div([
        kpi_card("NEW LEADS", f"{new_leads:,}", "", True, "🆕", "accent-cyan"),
        kpi_card("QUALIFIED RATE", f"{qual_rate:.1f}%", "", True, "⭐", "accent-green"),
        kpi_card("AVG LEAD SCORE", f"{avg_score:.1f}", "", True, "🎯", "accent-purple"),
        kpi_card("TOTAL INQUIRIES", f"{int(inquiries):,}", "", True, "📞", "accent-orange"),
    ], className="kpi-row", style={"gridTemplateColumns": "repeat(4, 1fr)"})

    bar_df = fdf.groupby(["Year", "Month"])["Record_ID"].count().reset_index()
    bar_df["Date"] = bar_df.Year.astype(str) + '-' + bar_df.Month.astype(str)
    bar_fig = px.bar(bar_df, x="Date", y="Record_ID", color_discrete_sequence=["#22d3ee"], labels={"Record_ID":"Leads"})
    modern_theme(bar_fig)

    return html.Div([kpis, html.Div([html.Div([card_hdr("Monthly Leads Generated"), dcc.Graph(figure=bar_fig, config={"displayModeBar":False})], className="content-card")], className="content-grid", style={"gridTemplateColumns":"1fr"})])

def _exec_page(years, months):
    fdf = filter_dataframe(df, years=years, months=months)
    if fdf.empty: return html.Div("No data.", style={"padding":"40px"})

    sales_total = fdf["Sales_Amount"].sum()
    maint_total = fdf["Maintenance_Cost"].sum()
    total_rev = sales_total + maint_total
    roi = ((sales_total - maint_total) / maint_total * 100) if maint_total > 0 else 0
    cust_growth = len(fdf[fdf["Sales_Amount"] > 0])
    mkt_exp = fdf["Continent"].nunique() if "Continent" in fdf.columns else 0

    # Strategic Risk: Custom logic (Client size + Country + Sales)
    # Simple mock: if avg sales is low and alerts are high, risk is higher.
    risk_score = min(100, max(0, (fdf["Critical_Alerts"].sum() / len(fdf) * 100) + 10))

    kpis = html.Div([
        kpi_card("TOTAL REVENUE", f"${format_large_number(total_rev)}", "18%", True, "💵", "accent-green"),
        kpi_card("ROI", f"{roi:.1f}%", "5.0%", True, "📈", "accent-cyan"),
        kpi_card("PAYING CUSTOMERS", f"{cust_growth:,}", "", True, "🌱", "accent-purple"),
        kpi_card("MARKET EXPANSION", f"{mkt_exp} Cont.", "", True, "🌍", "accent-cyan"),
        kpi_card("FORECAST (Q+1)", f"${format_large_number(total_rev*1.08)}", "Linear Reg", True, "🔮", "accent-orange"),
        kpi_card("STRATEGIC RISK", f"{risk_score:.1f}/100", "Stable", False, "⚖️", "accent-red"),
    ], className="kpi-row")

    # Revenue Performance and Forecast (Linear Regression overlay)
    trend_df = fdf.groupby(["Year", "Month"])[["Sales_Amount"]].sum().reset_index()
    trend_df["Date"] = pd.to_datetime(trend_df.Year.astype(str) + '-' + trend_df.Month.astype(str) + '-01')
    rev_fig = go.Figure()
    rev_fig.add_trace(go.Scatter(x=trend_df["Date"], y=trend_df["Sales_Amount"], name="Actual Revenue", line=dict(color="#10b981", width=3)))
    
    if lin_model and len(trend_df) > 1:
        # Predict the next 3 months
        last_date = trend_df["Date"].max()
        future_dates = [last_date + pd.DateOffset(months=i) for i in range(1, 4)]
        future_x = pd.DataFrame({"Year": [d.year for d in future_dates], "Month": [d.month for d in future_dates]})
        preds = lin_model.predict(future_x)
        # Connect the line
        f_dates = [last_date] + future_dates
        f_preds = [trend_df["Sales_Amount"].iloc[-1]] + list(preds)
        rev_fig.add_trace(go.Scatter(x=f_dates, y=f_preds, name="Forecast (Linear Reg)", line=dict(color="#f59e0b", width=3, dash="dash")))

    modern_theme(rev_fig)

    # Product Yield
    yield_fig = px.bar(fdf.groupby("Services")["Sales_Amount"].sum().reset_index(), x="Services", y="Sales_Amount", color_discrete_sequence=["#a855f7"])
    modern_theme(yield_fig)

    # Performance Graph: Revenue vs Sales
    perf_fig = go.Figure()
    perf_fig.add_trace(go.Bar(x=trend_df["Date"], y=trend_df["Sales_Amount"], name="Sales", marker_color="#22d3ee"))
    perf_fig.add_trace(go.Bar(x=trend_df["Date"], y=trend_df["Sales_Amount"] * 0.2, name="Maintenance", marker_color="#a855f7"))
    modern_theme(perf_fig)
    perf_fig.update_layout(barmode='stack')

    return html.Div([kpis,
        html.Div([
            html.Div([card_hdr("Revenue Performance & Predictive Forecast"), dcc.Graph(figure=rev_fig, config={"displayModeBar":False})], className="content-card")
        ], className="content-grid", style={"gridTemplateColumns":"1fr"}),
        html.Div([
            html.Div([card_hdr("Product Yield"), dcc.Graph(figure=yield_fig, config={"displayModeBar":False})], className="content-card"),
            html.Div([card_hdr("Revenue Composition"), dcc.Graph(figure=perf_fig, config={"displayModeBar":False})], className="content-card")
        ], className="content-grid", style={"gridTemplateColumns":"5fr 5fr"})
    ])

if __name__ == "__main__":
    app.run(debug=True, port=8086)
