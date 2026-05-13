import dash
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
from sklearn.ensemble import IsolationForest
import numpy as np
from datetime import datetime
import random

# =========================
# 1. Data Layer & Processing
# =========================

CONTINENT_MAP = {
    'Namibia': 'Africa', 'Senegal': 'Africa', 'Sierra Leone': 'Africa', 'Liberia': 'Africa', 'Cote d\'Ivoire': 'Africa',
    'Botswana': 'Africa', 'Western Sahara': 'Africa', 'Guinea': 'Africa', 'Rwanda': 'Africa', 'Mauritania': 'Africa',
    'Eritrea': 'Africa', 'Mauritius': 'Africa', 'Argentina': 'South America', 'Bolivia': 'South America', 'Suriname': 'South America',
    'Macao': 'Asia', 'Palau': 'Oceania', 'French Polynesia': 'Oceania', 'New Caledonia': 'Oceania', 'Solomon Islands': 'Oceania',
    'Fiji': 'Oceania', 'Georgia': 'Asia', 'Israel': 'Asia', 'Tajikistan': 'Asia', 'Timor-Leste': 'Asia', 'Myanmar': 'Asia',
    'Sri Lanka': 'Asia', 'Lao People\'s Democratic Republic': 'Asia', 'Singapore': 'Asia', 'Azerbaijan': 'Asia', 'China': 'Asia',
    'Greece': 'Europe', 'San Marino': 'Europe', 'Montenegro': 'Europe', 'Slovakia (Slovak Republic)': 'Europe', 'Ireland': 'Europe',
    'Iceland': 'Europe', 'Bosnia and Herzegovina': 'Europe', 'Greenland': 'North America', 'Turks and Caicos Islands': 'North America',
    'Cuba': 'North America', 'Aruba': 'North America', 'Costa Rica': 'North America', 'Saint Martin': 'North America',
    'South Africa': 'Africa', 'Egypt': 'Africa', 'Nigeria': 'Africa', 'Kenya': 'Africa', 'United States': 'North America',
    'Canada': 'North America', 'Mexico': 'North America', 'Brazil': 'South America', 'Chile': 'South America',
    'United Kingdom': 'Europe', 'Germany': 'Europe', 'France': 'Europe', 'Italy': 'Europe', 'Spain': 'Europe',
    'India': 'Asia', 'Japan': 'Asia', 'South Korea': 'Asia', 'Australia': 'Oceania'
}

COUNTRY_ISO = {
    'United States': 'USA', 'Canada': 'CAN', 'United Kingdom': 'GBR', 'Germany': 'DEU', 'France': 'FRA', 
    'India': 'IND', 'China': 'CHN', 'Japan': 'JPN', 'Australia': 'AUS', 'Brazil': 'BRA', 'South Africa': 'ZAF',
    'Nigeria': 'NGA', 'Egypt': 'EGY', 'Mexico': 'MEX', 'Argentina': 'ARG', 'Germany': 'DEU'
}

try:
    df = pd.read_excel("cybernova_augmented_dataset.xlsx")
    
    # Pre-process Request_Date
    if 'Request_Date' in df.columns:
        df['Request_Date'] = pd.to_datetime(df['Request_Date'], errors='coerce')
        df['Year'] = df['Request_Date'].dt.year.astype(str)
        df['Month'] = df['Request_Date'].dt.month_name()
        df['Month_Num'] = df['Request_Date'].dt.month
    else:
        df['Year'] = "2025"
        df['Month'] = "January"
        df['Month_Num'] = 1

    # Mapping and Numeric Conversion
    numeric_cols = [
        'Uptime', 'CPU_Avg', 'Memory_Usage', 'Threats_Blocked', 
        'Critical_Alerts', 'MFA_Adoption', 'MTTR', 'MTTD', 
        'Closure_Rate', 'Latency', 'Sales_Amount'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Derived Status based on Latency thresholds
    # Healthy (<100ms), Warning (100-150ms), Critical (>150ms)
    def get_status(lat):
        if lat < 100: return 'Healthy'
        if lat < 150: return 'Warning'
        return 'Critical'
    
    df['Status'] = df['Latency'].apply(get_status)

    # Continent Mapping (using existing CONTINENT_MAP)
    df['Continent'] = df['Country'].map(CONTINENT_MAP).fillna('Other')
    df['ISO'] = df['Country'].map(COUNTRY_ISO).fillna('USA')

except Exception as e:
    import traceback
    print(f"Data Init Error: {e}")
    traceback.print_exc()
    df = pd.DataFrame()

# =========================
# 2. App Setup
# =========================
app = Dash(__name__, external_stylesheets=[dbc.icons.BOOTSTRAP], suppress_callback_exceptions=True)

CHART_THEME = dict(
    template='plotly_dark',
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#94a3b8', family='Inter, sans-serif'),
    margin=dict(l=10, r=10, t=30, b=10)
)

MAP_THEME = dict(
    **CHART_THEME,
    geo=dict(
        bgcolor='rgba(0,0,0,0)',
        lakecolor='#0f172a',
        showlakes=True,
        projection_type='natural earth',
        showland=True,
        landcolor='#1e293b',
        showocean=True,
        oceancolor='#020617',
        showframe=False,
        showcountries=True,
        countrycolor='#334155'
    )
)

# =========================
# 3. UI Components
# =========================

def create_sidebar():
    return html.Div(className='sidebar', children=[
        html.Div(className='sidebar-logo', children=[
            html.Div("CN", className='logo-icon'),
            html.Div(className='logo-text', children=[html.H5("CyberNova"), html.Small("INTELLIGENCE")])
        ]),
        html.Div("Operations", className='sidebar-section-title'),
        dcc.Link([html.I(className="bi bi-grid-fill"), "Ops Overview"], href='/', className='nav-link'),
        dcc.Link([html.I(className="bi bi-activity"), "System Health"], href='/health', className='nav-link'),
        dcc.Link([html.I(className="bi bi-diagram-3"), "Infrastructure"], href='/infra', className='nav-link'),
        dcc.Link([html.I(className="bi bi-shield-check"), "Security"], href='/security', className='nav-link'),
        dcc.Link([html.I(className="bi bi-exclamation-triangle"), "Incidents"], href='/incidents', className='nav-link'),
    ])

def create_header(title, subtitle):
    return html.Div(className='header-bar', children=[
        html.Div([html.H6(title, className="mb-0 text-white fw-bold"), html.Small(subtitle, className="text-muted smaller")], className="flex-grow-1"),
        html.Div(className='d-flex align-items-center gap-4', children=[
            html.I(className="bi bi-bell text-muted h5 mb-0 cursor-pointer"),
            html.Div(className='d-flex align-items-center gap-2 ps-3 border-start border-secondary', style={'borderColor': 'rgba(255,255,255,0.05)'}, children=[
                html.Div(className='pulse-dot'),
                html.Small("LIVE", className="text-success small fw-bold"),
                html.Small(id="live-clock", className="text-muted small fw-bold ms-2")
            ])
        ])
    ])

def create_kpi_card(title, value, subtext, icon, color_class):
    return html.Div(className='kpi-card', children=[
        html.Div(className='d-flex justify-content-between align-items-center', children=[
            html.Div([
                html.Div(title, className='kpi-title'),
                html.Div(value, className='kpi-value'),
                html.Div([
                    html.I(className="bi bi-caret-up-fill me-1" if "▲" in subtext or "days" in subtext or "24h" in subtext else "bi bi-caret-down-fill me-1"),
                    html.Span(subtext)
                ], className=f"kpi-subtext {'text-success' if '▲' in subtext or 'Optimal' in subtext or 'Stable' in subtext or 'days' in subtext or '24h' in subtext else 'text-danger'}")
            ]),
            html.Div(html.I(className=f"bi {icon}"), className=f"icon-box {color_class}")
        ])
    ])

# =========================
# 4. Main Layout
# =========================

app.layout = html.Div(className='app-container', children=[
    dcc.Location(id='url'),
    dcc.Interval(id='clock-interval', interval=1000, n_intervals=0),
    create_sidebar(),
    html.Div(className='main-content', children=[
        html.Div(id='header-container'),
        # Filters moved here to be part of initial layout
        html.Div(className='filter-container', children=[
            html.Div(className='filter-title', children=[html.I(className="bi bi-funnel"), "FILTERS"]),
            html.Div(className='filter-group', children=[
                html.Label("TELEMETRY", className='filter-label'),
                dcc.Dropdown(id='service-filter', multi=True)
            ]),
            html.Div(className='filter-group', children=[
                html.Label("YEAR", className='filter-label'),
                dcc.Dropdown(id='year-filter', multi=True)
            ]),
            html.Div(className='filter-group', children=[
                html.Label("MONTH", className='filter-label'),
                dcc.Dropdown(id='month-filter', multi=True)
            ]),
            html.Div(className='reset-btn', id='reset-filters', children=[html.I(className="bi bi-x-lg"), "Reset"])
        ]),
        html.Div(id='page-content', className='content-area')
    ])
])

# =========================
# 5. Callbacks
# =========================

@app.callback(
    Output('header-container', 'children'),
    [Input('url', 'pathname')]
)
def update_navigation(pathname):
    titles = {
        '/': ("Operational Intelligence Overview", "Global system health • Traffic telemetry • Oversight"),
        '/health': ("System Health Monitor", "Real-time performance metrics and service diagnostics"),
        '/infra': ("Infrastructure Topology", "Global node distribution and regional connectivity"),
        '/security': ("Security Intelligence", "Threat detection, firewall events, and MFA adoption"),
        '/incidents': ("Incident Management", "Response times, detect times, and resolution rates")
    }
    title, subtitle = titles.get(pathname, ("CyberNova Dash", "Intelligence Suite"))
    return create_header(title, subtitle)

@app.callback(
    [Output('service-filter', 'options'), Output('service-filter', 'value'),
     Output('year-filter', 'options'), Output('year-filter', 'value'),
     Output('month-filter', 'options'), Output('month-filter', 'value')],
    [Input('url', 'pathname')] # Trigger on load
)
def initialize_filters(pathname):
    if df.empty: return [[], [], [], [], [], []]
    
    services = sorted([s for s in df['Services'].unique() if pd.notna(s)])
    years = sorted(df['Year'].unique())
    months = sorted(df['Month'].unique(), key=lambda x: datetime.strptime(x, "%B"))
    
    service_opts = [{'label': i, 'value': i} for i in services]
    year_opts = [{'label': i, 'value': i} for i in years]
    month_opts = [{'label': i, 'value': i} for i in months]
    
    return service_opts, services, year_opts, years, month_opts, months

@app.callback(
    [Output('year-filter', 'value'), Output('service-filter', 'value'), Output('month-filter', 'value')],
    [Input('reset-filters', 'n_clicks')]
)
def reset_all_filters(n_clicks):
    if n_clicks:
        years = sorted(df['Year'].unique()) if not df.empty else []
        months = sorted(df['Month'].unique()) if not df.empty else []
        services = sorted([s for s in df['Services'].unique() if pd.notna(s)]) if not df.empty else []
        return years, services, months
    return dash.no_update

@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname'), 
     Input('service-filter', 'value'), 
     Input('year-filter', 'value'), 
     Input('month-filter', 'value')]
)
def render_page_content(pathname, services, years, months):
    print(f"Rendering page: {pathname}")
    print(f"Filters - Services: {services}, Years: {years}, Months: {months}")
    
    if df.empty: 
        print("ERROR: Global df is empty!")
        return html.Div("Dataset error: Global dataframe is empty.", className="text-danger p-5")

    # Protection against initial null values from dynamic layout
    if not services or not years or not months:
        print("Waiting for filters to initialize...")
        return html.Div("Initializing filters...", className="text-muted p-5 text-center")

    # Filter data
    try:
        dff = df[
            (df['Services'].isin(services)) & 
            (df['Year'].isin(years)) & 
            (df['Month'].isin(months))
        ]
        print(f"Filtered data shape: {dff.shape}")
    except Exception as e:
        print(f"Filtering error: {e}")
        return html.Div(f"Filter error: {e}", className="text-danger p-5")
    
    if dff.empty: 
        print("WARNING: Filtered dataframe is empty!")
        return html.Div("No data matches the selected filters.", className="text-muted p-5 text-center")

    # Page: Operational Overview
    if pathname == '/':
        avg_uptime = dff['Uptime'].mean()
        avg_latency = dff['Latency'].mean()
        avg_cpu = dff['CPU_Avg'].mean()
        avg_mem = dff['Memory_Usage'].mean()
        threats = int(dff['Threats_Blocked'].sum())
        anomalies = len(dff[dff['Status'] == 'Critical']) # Using Critical as a proxy for anomalies as requested

        map_fig = px.choropleth(
            dff.groupby(['ISO', 'Country']).size().reset_index(name='Clients'),
            locations="ISO", color="Clients", hover_name="Country",
            color_continuous_scale="Viridis"
        ).update_layout(**MAP_THEME)

        return html.Div([
            html.Div(className='kpi-grid mb-3', children=[
                create_kpi_card("UPTIME", f"{avg_uptime:.2f}%", "Average", "bi-activity", "icon-green"),
                create_kpi_card("API LATENCY", f"{avg_latency:.0f} ms", "Average", "bi-lightning-charge", "icon-cyan"),
                create_kpi_card("CPU AVG", f"{avg_cpu:.1f}%", "Usage", "bi-cpu", "icon-purple"),
                create_kpi_card("MEMORY", f"{avg_mem:.1f}%", "Usage", "bi-hdd-stack", "icon-cyan"),
                create_kpi_card("THREATS BLOCKED", f"{threats:,}", "Total Sum", "bi-shield-shaded", "icon-red"),
                create_kpi_card("ANOMALIES", str(anomalies), "Critical Count", "bi-exclamation-triangle", "icon-orange"),
            ]),
            dbc.Row([
                dbc.Col([
                    html.Div(className='table-card p-3', children=[
                        html.H6("System Performance Over Time", className="text-white mb-3 fw-bold"),
                        dcc.Graph(
                            figure=px.line(dff.sort_values('Request_Date'), x='Request_Date', y=['CPU_Avg', 'Memory_Usage', 'Latency'],
                                         color_discrete_sequence=["#a855f7", "#06b6d4", "#10b981"])
                                  .update_layout(**CHART_THEME),
                            style={"height": "350px"}
                        )
                    ])
                ], lg=8),
                dbc.Col([
                    html.Div(className='table-card p-3', children=[
                        html.H6("Service Status Distribution", className="text-white mb-3 fw-bold"),
                        dcc.Graph(
                            figure=px.pie(dff, names='Status', color='Status',
                                         color_discrete_map={'Healthy': '#10b981', 'Warning': '#f59e0b', 'Critical': '#ef4444'},
                                         hole=.7)
                                  .update_layout(**CHART_THEME),
                            style={"height": "350px"}
                        )
                    ])
                ], lg=4)
            ], className="g-3 mb-3"),
            dbc.Row([
                dbc.Col([
                    html.Div(className='table-card p-3', children=[
                        html.H6("Top 10 Services by Latency", className="text-white mb-3 fw-bold"),
                        dash_table.DataTable(
                            data=dff.sort_values('Latency', ascending=False).head(10).to_dict('records'),
                            columns=[{'name': i, 'id': i} for i in ['Services', 'Region', 'Status', 'Latency']],
                            style_table={'overflowX': 'auto'},
                            style_header={'backgroundColor': 'rgba(255,255,255,0.05)', 'color': 'white', 'fontWeight': 'bold'},
                            style_cell={'backgroundColor': 'transparent', 'color': '#94a3b8', 'border': '1px solid rgba(255,255,255,0.05)'}
                        )
                    ])
                ], lg=6),
                dbc.Col([
                    html.Div(className='table-card p-3', children=[
                        html.H6("Client Distribution by Region", className="text-white mb-3 fw-bold"),
                        dcc.Graph(figure=map_fig, style={"height": "300px"})
                    ])
                ], lg=6)
            ], className="g-3")
        ])

    # Page: System Health
    elif pathname == '/health':
        plat_health = (dff['CPU_Avg'].mean() + dff['Memory_Usage'].mean() + (200 - dff['Latency'].mean())/2) / 3
        db_health = (80 + random.uniform(-10, 10)) # Storage/Query Latency proxy
        api_health = (100 - (dff['Latency'].mean() / 2)) # Error Rate proxy
        
        def get_health_label(val):
            if val > 80: return "Healthy", "text-success"
            if val > 60: return "Good", "text-warning"
            return "Poor", "text-danger"

        return html.Div([
            html.Div(className='kpi-grid mb-3', children=[
                create_kpi_card("PLATFORM HEALTH", f"{plat_health:.1f}%", get_health_label(plat_health)[0], "bi-cpu", "icon-purple"),
                create_kpi_card("DATABASE HEALTH", f"{db_health:.1f}%", get_health_label(db_health)[0], "bi-database", "icon-cyan"),
                create_kpi_card("API HEALTH", f"{api_health:.1f}%", get_health_label(api_health)[0], "bi-hdd-network", "icon-green"),
            ])
        ])

    # Page: Infrastructure
    elif pathname == '/infra':
        node_count = dff['Node'].nunique()
        avg_cpu = dff['CPU_Avg'].mean()
        storage = dff['Memory_Usage'].mean() * 1.2 # Storage proxy
        
        # 8 Random Regions
        regions = dff['Region'].unique()
        random_regions = random.sample(list(regions), min(8, len(regions)))
        region_cards = []
        for reg in random_regions:
            reg_data = dff[dff['Region'] == reg]
            reg_health = reg_data['Status'].iloc[0]
            reg_cpu = reg_data['CPU_Avg'].mean()
            region_cards.append(dbc.Col([
                html.Div(className='table-card p-3', children=[
                    html.H6(reg, className="text-white mb-1 fw-bold"),
                    html.Small(f"Health: {reg_health}", className="text-success" if reg_health == 'Healthy' else "text-warning"),
                    html.Div(className='mt-2', children=[
                        html.Small(f"CPU: {reg_cpu:.1f}%", className="text-muted"),
                        html.Div(className='progress', style={'height': '4px', 'background': '#1e293b'}, children=[
                            html.Div(className='progress-bar', style={'width': f'{reg_cpu}%', 'background': 'var(--accent-purple)'})
                        ])
                    ])
                ])
            ], md=3))

        return html.Div([
            html.Div(className='kpi-grid mb-3', children=[
                create_kpi_card("NODES", str(node_count), "Total Count", "bi-diagram-3", "icon-cyan"),
                create_kpi_card("AVG CPU", f"{avg_cpu:.1f}%", "Regional Avg", "bi-cpu", "icon-purple"),
                create_kpi_card("STORAGE USED", f"{storage:.1f} TB", "Aggregated", "bi-hdd-stack", "icon-cyan"),
                create_kpi_card("REGION HEALTH", "Aggregated", "Stable", "bi-globe", "icon-green"),
            ]),
            html.H6("Regional Node Status", className="text-white mb-3 fw-bold px-2"),
            dbc.Row(region_cards, className="g-3")
        ])

    # Page: Security
    elif pathname == '/security':
        threats = int(dff['Threats_Blocked'].sum())
        critical = len(dff[dff['Critical_Alerts'] > 15])
        mfa = dff['MFA_Adoption'].mean()

        return html.Div([
            html.Div(className='kpi-grid mb-3', children=[
                create_kpi_card("THREATS BLOCKED", f"{threats:,}", "Total Sum", "bi-shield-shaded", "icon-red"),
                create_kpi_card("CRITICAL ALERTS", str(critical), "Count", "bi-exclamation-octagon", "icon-orange"),
                create_kpi_card("MFA ADOPTION", f"{mfa:.1f}%", "User Base", "bi-person-check", "icon-green"),
            ]),
            dbc.Row([
                dbc.Col([
                    html.Div(className='table-card p-3', children=[
                        html.H6("Threat Activity Trend", className="text-white mb-3 fw-bold"),
                        dcc.Graph(
                            figure=px.line(dff.groupby('Year')['Threats_Blocked'].sum().reset_index(), x='Year', y='Threats_Blocked')
                                  .update_layout(**CHART_THEME),
                            style={"height": "300px"}
                        )
                    ])
                ], lg=12)
            ], className="mb-3"),
            html.Div(className='table-card p-3', children=[
                html.H6("Security Event Log", className="text-white mb-3 fw-bold"),
                dash_table.DataTable(
                    data=dff.tail(10).to_dict('records'),
                    columns=[{'name': i, 'id': i} for i in ['Request_Date', 'IP_Address', 'Services', 'Critical_Alerts']],
                    style_table={'overflowX': 'auto'},
                    style_header={'backgroundColor': 'rgba(255,255,255,0.05)', 'color': 'white', 'fontWeight': 'bold'},
                    style_cell={'backgroundColor': 'transparent', 'color': '#94a3b8', 'border': '1px solid rgba(255,255,255,0.05)'}
                )
            ])
        ])

    # Page: Incidents
    elif pathname == '/incidents':
        open_incidents = int(dff['Incidents'].sum())
        mttr = dff['MTTR'].mean()
        mttd = dff['MTTD'].mean()
        closure = dff['Closure_Rate'].mean()

        return html.Div([
            html.Div(className='kpi-grid mb-3', children=[
                create_kpi_card("OPEN INCIDENTS", str(open_incidents), "Active Count", "bi-exclamation-triangle", "icon-orange"),
                create_kpi_card("MTTR", f"{mttr:.1f} hrs", "Mean Time to Resolve", "bi-clock-history", "icon-cyan"),
                create_kpi_card("MTTD", f"{mttd:.1f} hrs", "Mean Time to Detect", "bi-search", "icon-purple"),
                create_kpi_card("CLOSURE RATE", f"{closure:.1f}%", "Resolution Eff.", "bi-check-circle", "icon-green"),
            ]),
            html.Div(className='table-card p-3', children=[
                html.H6("Incident Management Log", className="text-white mb-3 fw-bold"),
                dash_table.DataTable(
                    data=dff.tail(10).to_dict('records'),
                    columns=[{'name': i, 'id': i} for i in ['Record_ID', 'Services', 'Critical_Alerts', 'Status']],
                    style_table={'overflowX': 'auto'},
                    style_header={'backgroundColor': 'rgba(255,255,255,0.05)', 'color': 'white', 'fontWeight': 'bold'},
                    style_cell={'backgroundColor': 'transparent', 'color': '#94a3b8', 'border': '1px solid rgba(255,255,255,0.05)'}
                )
            ])
        ])

    return html.Div([html.H3("Page Not Found", className="text-muted text-center py-5")])

@app.callback(Output("live-clock", "children"), Input("clock-interval", "n_intervals"))
def update_clock(n):
    return datetime.now().strftime("%a, %d %b - %H:%M:%S")

if __name__ == "__main__":
    app.run(debug=True, port=8453)
