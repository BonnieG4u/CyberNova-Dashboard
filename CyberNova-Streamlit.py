import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression, LinearRegression

# Setup Page Config
st.set_page_config(page_title="CyberNova Intelligence", layout="wide", initial_sidebar_state="expanded")

# Custom CSS
st.markdown("""
<style>
[data-testid="stMetricValue"] {
    font-size: 1.8rem;
    color: white;
}
.stMetric {
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 12px;
    padding: 1rem;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
}
</style>
""", unsafe_allow_html=True)

# Data Loading
@st.cache_data
def load_and_augment_data():
    try:
        df = pd.read_excel("cybernova_augmented_dataset.xlsx")
    except Exception as e:
        st.error(f"Cannot load Excel – {e}")
        return pd.DataFrame()
    
    NUMERIC = ["CPU_Avg", "Memory_Usage", "Latency", "Threats_Blocked",
               "Uptime", "Critical_Alerts", "Sales_Amount", "Conversion",
               "ROI_Projection", "Demand_Forecast", "Incidents", "MTTR", "MTTD", "Login_Attempts", "Firewall_Events"]
    for col in NUMERIC:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    if "Request_Date" in df.columns:
        df["Request_Date"] = pd.to_datetime(df["Request_Date"], errors="coerce")
        df["Year"] = df["Request_Date"].dt.year.fillna(2024).astype(int)
        df["Month"] = df["Request_Date"].dt.month.fillna(1).astype(int)
    else:
        df["Year"] = 2024
        df["Month"] = 1

    np.random.seed(42)
    num_rows = len(df)

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

    if "Campaign" not in df.columns: df["Campaign"] = np.random.choice(["Q1 Launch", "CyberSec Promo", "Cloud Migrations", "Enterprise Summit"], size=num_rows)
    if "CTR" not in df.columns: df["CTR"] = np.random.uniform(0.5, 8.5, size=num_rows)
    if "Visitors" not in df.columns: df["Visitors"] = np.random.randint(100, 5000, size=num_rows)
    if "Enquiries" not in df.columns: df["Enquiries"] = df["Visitors"] * np.random.uniform(0.05, 0.2)
    if "Maintenance_Cost" not in df.columns: df["Maintenance_Cost"] = df["Sales_Amount"] * np.random.uniform(0.1, 0.3)

    return df

@st.cache_resource
def train_models(df):
    X_lead = df[['Org_Encoded', 'Svc_Encoded', 'Visitors', 'CTR']].fillna(0) if 'Org_Encoded' in df.columns else pd.DataFrame(np.random.rand(len(df), 4))
    y_lead = df['Conversion'].fillna(0).astype(int)
    log_model = LogisticRegression(max_iter=1000)
    try:
        log_model.fit(X_lead, y_lead)
        df['Lead_Probability'] = log_model.predict_proba(X_lead)[:, 1] * 100
    except:
        df['Lead_Probability'] = np.random.uniform(0, 100, size=len(df))
    df['Lead_Score'] = df['Lead_Probability']

    try:
        X_rev = df[['Year', 'Month']].dropna()
        y_rev = df['Sales_Amount'].loc[X_rev.index]
        lin_model = LinearRegression()
        lin_model.fit(X_rev, y_rev)
    except:
        lin_model = None

    return df, log_model, lin_model

# Load Data
df = load_and_augment_data()
if not df.empty:
    df, log_model, lin_model = train_models(df)

# Global Variables
REGIONS = sorted([str(x) for x in df["Region"].dropna().unique()]) if "Region" in df.columns else []
INDUSTRIES = sorted([str(x) for x in df["Industry"].dropna().unique()]) if "Industry" in df.columns else []
SERVICES = sorted([str(x) for x in df["Services"].dropna().unique()]) if "Services" in df.columns else []
YEARS = sorted([int(x) for x in df["Year"].dropna().unique()])
MONTHS = sorted([int(x) for x in df["Month"].dropna().unique()])
ROLES = sorted([str(x) for x in df["Role"].dropna().unique()])
SEVERITIES = ["Low", "Medium", "High", "Critical"]

# Helpers
def modern_theme(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#94a3b8",
        margin=dict(t=30, b=30, l=30, r=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.03)", zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.03)", zeroline=False),
    )
    return fig

def format_large_number(num):
    if num >= 1_000_000_000: return f"{num/1_000_000_000:.1f}B"
    elif num >= 1_000_000: return f"{num/1_000_000:.1f}M"
    elif num >= 1_000: return f"{num/1_000:.1f}K"
    return f"{num:,.0f}"

# Sidebar
st.sidebar.title("CyberNova")
st.sidebar.markdown("### Navigation")
page = st.sidebar.radio("Go to", ["Admin Overview", "System Health", "Infrastructure", "Security", "Sales Overview", "Sales Analytics", "Executive"])

# Pages
if page == "Admin Overview":
    st.title("Admin Overview")
    roles = st.sidebar.multiselect("Role", ROLES, default=ROLES)
    fdf = df[df["Role"].isin(roles)] if roles else df
    
    if fdf.empty: st.warning("No data.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("OPEN INCIDENTS", fdf[fdf['Incident_Status']=='Open']['Record_ID'].count())
        c2.metric("CRITICAL ALERTS", fdf['Critical_Alerts'].sum())
        c3.metric("AVG CPU USAGE", f"{fdf['CPU_Avg'].mean():.1f}%")
        c4.metric("UPTIME", f"{fdf['Uptime'].mean():.2f}%")
        
        st.subheader("Global Service Distribution")
        donut = px.pie(fdf, names="Region", hole=0.65, color_discrete_sequence=px.colors.sequential.Teal)
        st.plotly_chart(modern_theme(donut), use_container_width=True)

elif page == "System Health":
    st.title("System Health")
    years = st.sidebar.multiselect("Years", YEARS, default=YEARS)
    months = st.sidebar.multiselect("Months", MONTHS, default=MONTHS)
    
    fdf = df.copy()
    if years: fdf = fdf[fdf["Year"].isin(years)]
    if months: fdf = fdf[fdf["Month"].isin(months)]
    
    plat_score = max(0, 100 - ((fdf["CPU_Avg"].mean() + fdf["Memory_Usage"].mean() + min(fdf["Latency"].mean()/5, 100)) / 3))
    db_score = max(0, 100 - ((min(fdf["Storage_Used_GB"].mean()/100, 100) + min(fdf["Query_Latency"].mean(), 100)) / 2))
    api_score = max(0, 100 - ((min(fdf["API_Latency"].mean()/3, 100) + min(fdf["Error_Rate"].mean()*20, 100)) / 2))
    
    c1, c2, c3 = st.columns(3)
    c1.metric("PLATFORM HEALTH", f"{plat_score:.0f}% Score")
    c2.metric("DATABASE HEALTH", f"{db_score:.0f}% Score")
    c3.metric("API HEALTH", f"{api_score:.0f}% Score")

elif page == "Infrastructure":
    st.title("Infrastructure")
    regions = st.sidebar.multiselect("Regions", REGIONS, default=REGIONS)
    years = st.sidebar.multiselect("Years", YEARS, default=YEARS)
    
    fdf = df.copy()
    if regions: fdf = fdf[fdf["Region"].isin(regions)]
    if years: fdf = fdf[fdf["Year"].isin(years)]
    
    c1, c2, c3, c4 = st.columns(4)
    nodes_count = len(fdf)
    healthy_pct = len(fdf[fdf["Latency"] < 150]) / nodes_count * 100 if nodes_count else 0
    c1.metric("ACTIVE NODES", f"{nodes_count:,}")
    c2.metric("AVG CPU %", f"{fdf['CPU_Avg'].mean():.1f}%")
    c3.metric("STORAGE USED", f"{fdf['Storage_Used_GB'].sum() / 1024:.2f} TB")
    c4.metric("GLOBAL HEALTH", f"{healthy_pct:.0f}% OK")

elif page == "Security":
    st.title("Security")
    severities = st.sidebar.multiselect("Severities", SEVERITIES, default=SEVERITIES)
    years = st.sidebar.multiselect("Years", YEARS, default=YEARS)
    
    fdf = df.copy()
    if severities: fdf = fdf[fdf["Severity"].isin(severities)]
    if years: fdf = fdf[fdf["Year"].isin(years)]
    
    total_users = fdf["Total_Users"].sum()
    mfa_users = fdf["Users_with_MFA"].sum()
    mfa_adopt = (mfa_users / total_users * 100) if total_users > 0 else 0
    
    c1, c2, c3 = st.columns(3)
    c1.metric("THREATS BLOCKED", f"{int(fdf['Threats_Blocked'].sum()):,}")
    c2.metric("CRITICAL ALERTS", f"{len(fdf[fdf['Severity'] == 'Critical']):,}")
    c3.metric("MFA ADOPTION", f"{mfa_adopt:.1f}%")
    
    st.subheader("Threat Activity Trend")
    trend_df = fdf.groupby(["Year", "Month"])[["Threats_Blocked"]].sum().reset_index()
    trend_df["Date"] = pd.to_datetime(trend_df.Year.astype(str) + '-' + trend_df.Month.astype(str) + '-01')
    line_fig = go.Figure()
    line_fig.add_trace(go.Scatter(x=trend_df["Date"], y=trend_df["Threats_Blocked"], name="Threats Blocked", line=dict(color="#10b981", width=3)))
    st.plotly_chart(modern_theme(line_fig), use_container_width=True)

    st.subheader("Recent Security Events")
    st.dataframe(fdf[["Request_Date", "Country", "Event_Title", "Severity"]].sort_values("Request_Date", ascending=False).head(10))

elif page == "Sales Overview":
    st.title("Sales Overview")
    regions = st.sidebar.multiselect("Regions", REGIONS, default=REGIONS)
    
    fdf = df.copy()
    if regions: fdf = fdf[fdf["Region"].isin(regions)]
    
    leads = len(fdf)
    clients = len(fdf[fdf["Conversion"] == 1]) if "Conversion" in fdf.columns else 0
    enquiries = int(fdf["Enquiries"].sum())
    conv_rate = (clients / leads * 100) if leads else 0
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("TOTAL LEADS", f"{leads:,}")
    c2.metric("TOTAL CLIENTS", f"{clients:,}")
    c3.metric("ENQUIRIES", f"{enquiries:,}")
    c4.metric("CONVERSION RATE", f"{conv_rate:.1f}%")
    
    st.subheader("Visitors vs Leads")
    trend_df = fdf.groupby(["Year", "Month"])[["Visitors", "Record_ID"]].agg({"Visitors":"sum", "Record_ID":"count"}).reset_index()
    trend_df["Date"] = pd.to_datetime(trend_df.Year.astype(str) + '-' + trend_df.Month.astype(str) + '-01')
    vis_fig = go.Figure()
    vis_fig.add_trace(go.Scatter(x=trend_df["Date"], y=trend_df["Visitors"], name="Visitors", line=dict(color="#a855f7")))
    vis_fig.add_trace(go.Scatter(x=trend_df["Date"], y=trend_df["Record_ID"], name="Leads", line=dict(color="#22d3ee")))
    st.plotly_chart(modern_theme(vis_fig), use_container_width=True)

elif page == "Sales Analytics":
    st.title("Sales Analytics")
    services = st.sidebar.multiselect("Services", SERVICES, default=SERVICES)
    years = st.sidebar.multiselect("Years", YEARS, default=YEARS)
    
    fdf = df.copy()
    if services: fdf = fdf[fdf["Services"].isin(services)]
    if years: fdf = fdf[fdf["Year"].isin(years)]
    
    new_leads = len(fdf)
    qual_rate = (len(fdf[fdf["Lead_Probability"] > 60]) / new_leads * 100) if new_leads else 0
    
    c1, c2, c3 = st.columns(3)
    c1.metric("NEW LEADS", f"{new_leads:,}")
    c2.metric("QUALIFIED RATE", f"{qual_rate:.1f}%")
    c3.metric("AVG LEAD SCORE", f"{fdf['Lead_Score'].mean():.1f}")
    
    st.subheader("Monthly Leads Generated")
    bar_df = fdf.groupby(["Year", "Month"])["Record_ID"].count().reset_index()
    bar_df["Date"] = bar_df.Year.astype(str) + '-' + bar_df.Month.astype(str)
    bar_fig = px.bar(bar_df, x="Date", y="Record_ID", color_discrete_sequence=["#22d3ee"])
    st.plotly_chart(modern_theme(bar_fig), use_container_width=True)

elif page == "Executive":
    st.title("Executive Dashboard")
    years = st.sidebar.multiselect("Years", YEARS, default=YEARS)
    
    fdf = df.copy()
    if years: fdf = fdf[fdf["Year"].isin(years)]
    
    sales_total = fdf["Sales_Amount"].sum()
    maint_total = fdf["Maintenance_Cost"].sum()
    total_rev = sales_total + maint_total
    roi = ((sales_total - maint_total) / maint_total * 100) if maint_total > 0 else 0
    cust_growth = len(fdf[fdf["Sales_Amount"] > 0])
    mkt_exp = fdf["Continent"].nunique() if "Continent" in fdf.columns else 0
    risk_score = min(100, max(0, (fdf["Critical_Alerts"].sum() / len(fdf) * 100) + 10))
    
    c1, c2, c3 = st.columns(3)
    c1.metric("TOTAL REVENUE", f"${format_large_number(total_rev)}")
    c2.metric("ROI", f"{roi:.1f}%")
    c3.metric("PAYING CUSTOMERS", f"{cust_growth:,}")
    
    c4, c5, c6 = st.columns(3)
    c4.metric("MARKET EXPANSION", f"{mkt_exp} Cont.")
    c5.metric("FORECAST (Q+1)", f"${format_large_number(total_rev*1.08)}")
    c6.metric("STRATEGIC RISK", f"{risk_score:.1f}/100")
    
    st.subheader("Revenue Performance & Predictive Forecast")
    trend_df = fdf.groupby(["Year", "Month"])[["Sales_Amount"]].sum().reset_index()
    trend_df["Date"] = pd.to_datetime(trend_df.Year.astype(str) + '-' + trend_df.Month.astype(str) + '-01')
    rev_fig = go.Figure()
    rev_fig.add_trace(go.Scatter(x=trend_df["Date"], y=trend_df["Sales_Amount"], name="Actual Revenue", line=dict(color="#10b981", width=3)))
    
    if lin_model and len(trend_df) > 1:
        last_date = trend_df["Date"].max()
        future_dates = [last_date + pd.DateOffset(months=i) for i in range(1, 4)]
        future_x = pd.DataFrame({"Year": [d.year for d in future_dates], "Month": [d.month for d in future_dates]})
        preds = lin_model.predict(future_x)
        f_dates = [last_date] + future_dates
        f_preds = [trend_df["Sales_Amount"].iloc[-1]] + list(preds)
        rev_fig.add_trace(go.Scatter(x=f_dates, y=f_preds, name="Forecast (Linear Reg)", line=dict(color="#f59e0b", width=3, dash="dash")))
        
    st.plotly_chart(modern_theme(rev_fig), use_container_width=True)
    
    c7, c8 = st.columns(2)
    with c7:
        st.subheader("Product Yield")
        yield_fig = px.bar(fdf.groupby("Services")["Sales_Amount"].sum().reset_index(), x="Services", y="Sales_Amount", color_discrete_sequence=["#a855f7"])
        st.plotly_chart(modern_theme(yield_fig), use_container_width=True)
    with c8:
        st.subheader("Revenue Composition")
        perf_fig = go.Figure()
        perf_fig.add_trace(go.Bar(x=trend_df["Date"], y=trend_df["Sales_Amount"], name="Sales", marker_color="#22d3ee"))
        perf_fig.add_trace(go.Bar(x=trend_df["Date"], y=trend_df["Sales_Amount"] * 0.2, name="Maintenance", marker_color="#a855f7"))
        perf_fig.update_layout(barmode='stack')
        st.plotly_chart(modern_theme(perf_fig), use_container_width=True)
