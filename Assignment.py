import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, dash_table, State
from datetime import datetime

# 1. Continent Mapping Utility
CONTINENT_MAP = {
    'Namibia': 'Africa', 'Senegal': 'Africa', 'Sierra Leone': 'Africa', 'Liberia': 'Africa', 'Cote d\'Ivoire': 'Africa',
    'Botswana': 'Africa', 'Western Sahara': 'Africa', 'Guinea': 'Africa', 'Rwanda': 'Africa', 'Mauritania': 'Africa',
    'Eritrea': 'Africa', 'Mauritius': 'Africa', 'Argentina': 'South America', 'Bolivia': 'South America', 'Suriname': 'South America',
    'Macao': 'Asia', 'Palau': 'Oceania', 'French Polynesia': 'Oceania', 'New Caledonia': 'Oceania', 'Solomon Islands': 'Oceania',
    'Fiji': 'Oceania', 'Georgia': 'Asia', 'Israel': 'Asia', 'Tajikistan': 'Asia', 'Timor-Leste': 'Asia', 'Myanmar': 'Asia',
    'Sri Lanka': 'Asia', 'Lao People\'s Democratic Republic': 'Asia', 'Singapore': 'Asia', 'Azerbaijan': 'Asia', 'China': 'Asia',
    'Greece': 'Europe', 'San Marino': 'Europe', 'Montenegro': 'Europe', 'Slovakia (Slovak Republic)': 'Europe', 'Ireland': 'Europe',
    'Iceland': 'Europe', 'Bosnia and Herzegovina': 'Europe', 'Greenland': 'North America', 'Turks and Caicos Islands': 'North America',
    'Cuba': 'North America', 'Aruba': 'North America', 'Costa Rica': 'North America', 'Saint Martin': 'North America'
}

def get_continent(country):
    return CONTINENT_MAP.get(country, 'Other')

# 2. Load and Clean Dataset
try:
    df = pd.read_excel('cybernova_20000_dataset.xlsx')
    if not df.empty:
        df = df.dropna(subset=['Country']) # Focus on entries with country
        df['Continent'] = df['Country'].apply(get_continent)
        
        if 'Request_Date' in df.columns:
            df['Request_Date'] = pd.to_datetime(df['Request_Date'], errors='coerce')
        if 'Sales_Amount' in df.columns:
            df['Sales_Amount'] = pd.to_numeric(df['Sales_Amount'], errors='coerce').fillna(0)
            
        status_col = 'Status' if 'Status' in df.columns else 'Contract_Status' if 'Contract_Status' in df.columns else 'Conversion' if 'Conversion' in df.columns else None
        if status_col:
            df['Conversion_Flag'] = df[status_col].apply(lambda x: 1 if str(x).strip().lower() in ['confirmed contract', 'yes', 'true', '1'] else 0)
        else:
            df['Conversion_Flag'] = 0
except FileNotFoundError:
    print("Dataset not found. Please upload 'cybernova_20000_dataset.xlsx'")
    df = pd.DataFrame()

# App Initialization
app = Dash(__name__, suppress_callback_exceptions=True)

# Common chart layout settings for dark theme
chart_layout = dict(
    template='plotly_dark',
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    margin=dict(l=20, r=20, t=40, b=20),
    font=dict(color='#94a3b8', family='Inter, sans-serif')
)

# --- Layout Components ---

def create_sidebar():
    return html.Div(className='sidebar', children=[
        html.Div(className='sidebar-logo', children=[
            html.Div(className='logo-icon', children="CN"),
            html.Div(className='logo-text', children=[
                "CyberNova",
                html.Span("Intelligence")
            ])
        ]),
        
        html.Div(className='sidebar-section', children=[
            html.Div("Operations", className='sidebar-section-title'),
            dcc.Link([html.I(className="bi bi-grid"), "Ops Overview"], href='/', className='nav-link'),
            dcc.Link("Infrastructure", href='/infrastructure', className='nav-link'),
            dcc.Link("Security", href='/security', className='nav-link'),
            dcc.Link("Incidents", href='/incidents', className='nav-link'),
        ]),
        
        html.Div(className='sidebar-section', children=[
            html.Div("Business", className='sidebar-section-title'),
            dcc.Link("Sales & Marketing", href='/sales', className='nav-link'),
            dcc.Link("Executive", href='/executive', className='nav-link'),
        ]),

        html.Div(className='sidebar-section', children=[
            html.Div("Administration", className='sidebar-section-title'),
            dcc.Link("User Management", href='/users', className='nav-link'),
            dcc.Link("Settings", href='/settings', className='nav-link'),
        ]),
        
        # Bottom profile info (mini)
        html.Div(style={'marginTop': 'auto', 'padding': '20px'}, children=[
            html.Div(className='user-profile', children=[
                html.Div("NM", className='avatar'),
                html.Div(className='user-info', children=[
                    html.P("Naledi Mokwena", className='user-name'),
                    html.P("Administrator", className='user-role')
                ])
            ])
        ])
    ])

def create_header(title, subtitle):
    return html.Div(className='header-container', children=[
        html.Div(className='page-info', children=[
            html.H1(title, className='header-title'),
            html.P(subtitle, className='header-subtitle')
        ]),
        html.Div(className='header-right', children=[
            html.Div(className='live-indicator', children=[
                html.Div(className='pulse-dot'),
                html.Div("System Running Live", className='live-text'),
                html.Div(id='live-clock', className='live-time')
            ]),
            html.Div(className='user-profile', children=[
                html.Div("NM", className='avatar'),
                html.Div(className='user-info', children=[
                    html.P("Naledi Mokwena", className='user-name'),
                    html.P("Administrator", className='user-role')
                ])
            ])
        ])
    ])

app.layout = html.Div(id='app-container', children=[
    dcc.Location(id='url', refresh=False),
    dcc.Interval(id='clock-interval', interval=1000, n_intervals=0),
    create_sidebar(),
    html.Div(className='main-wrapper', children=[
        html.Div(id='header-wrapper'),
        html.Div(id='page-content', className='content-area')
    ])
])

# --- Helper Functions ---

def create_kpi_card(title, value):
    return html.Div(className='kpi-card', children=[
        html.H3(title, className='kpi-title'),
        html.P(value, className='kpi-value')
    ])

def create_monitor_card(name, location, type, cpu, status='Healthy'):
    status_class = 'status-badge status-healthy' if status == 'Healthy' else 'status-badge status-warning'
    return html.Div(className='monitor-card', children=[
        html.Div(className='monitor-header', children=[
            html.Span(name, className='monitor-name'),
            html.Span(status, className=status_class)
        ]),
        html.Div(className='monitor-stats', children=[
            html.Div(className='stat-row', children=[
                html.Span(location, className='stat-label'),
                html.Span(type, className='stat-label')
            ]),
            html.Div(className='stat-row', children=[
                html.Span(f"CPU {cpu}%", className='stat-label'),
            ]),
            html.Div(className='progress-bar-bg', children=[
                html.Div(className='progress-bar-fill', style={'width': f'{cpu}%'})
            ])
        ])
    ])

# --- Callbacks ---

@app.callback(
    [Output('page-content', 'children'),
     Output('header-wrapper', 'children')],
    [Input('url', 'pathname')]
)
def display_page(pathname):
    if df.empty:
        return html.Div("Dataset error.", style={'color': '#ef4444'}), ""

    if pathname == '/infrastructure':
        avg_cpu = df['CPU_Usage_%'].mean()
        avg_mem = df['Memory_Usage_%'].mean()
        regions = df['Country'].nunique()
        
        header = create_header("Infrastructure Analytics", "Compute fleet across regions")
        
        content = html.Div([
            html.Div(className='kpi-row', children=[
                create_kpi_card("Nodes Online", "42 / 44"),
                create_kpi_card("Avg CPU", f"{avg_cpu:.1f}%"),
                create_kpi_card("Storage Used", f"{avg_mem:.1f}%"),
                create_kpi_card("Regions", str(regions))
            ]),
            html.Div(className='monitor-grid', children=[
                create_monitor_card("edge-bw-01", "Gaborone", "Edge", 38),
                create_monitor_card("edge-za-01", "Cape Town", "Edge", 62),
                create_monitor_card("core-eu-01", "Frankfurt", "Core", 71, "Warning"),
                create_monitor_card("core-us-01", "Virginia", "Core", 44),
                create_monitor_card("ml-gpu-01", "Frankfurt", "GPU", 88, "Warning"),
                create_monitor_card("ml-gpu-02", "Virginia", "GPU", 24),
                create_monitor_card("db-primary", "Gaborone", "DB", 51),
                create_monitor_card("db-replica-eu", "Frankfurt", "DB", 33)
            ])
        ])
        return content, header

    elif pathname == '/security':
        tot_fw = df['Firewall_Events'].sum()
        failed_logins = df['Login_Attempts'].sum() # Simplified for demo
        
        header = create_header("Security Monitoring", "Threats • Failed logins • Anomalies")
        
        fig_threat = px.bar(x=[0, 4, 8, 12, 16, 20], y=[12, 8, 22, 45, 38, 18], 
                           title="Threat Activity (24h)", template='plotly_dark', color_discrete_sequence=['#ef4444'])
        fig_threat.update_layout(**chart_layout)
        
        content = html.Div([
            html.Div(className='kpi-row', children=[
                create_kpi_card("Threats Blocked", f"{tot_fw:,}"),
                create_kpi_card("Failed Logins", f"{failed_logins:,}"),
                create_kpi_card("Critical Alerts", "2"),
                create_kpi_card("MFA Adoption", "94%")
            ]),
            html.Div(className='table-card', children=[
                html.Div(className='table-header', children=[html.H3("Live Security Events", className='table-title')]),
                dash_table.DataTable(
                    data=[
                        {"Time": "12:42:08", "Source": "41.220.18.4", "Event": "Failed login", "Severity": "Warning"},
                        {"Time": "12:39:51", "Source": "102.89.5.221", "Event": "Brute force", "Severity": "Critical"},
                        {"Time": "12:31:14", "Source": "18.8.4.18", "Event": "Privilege escalation attempt", "Severity": "Critical"},
                        {"Time": "12:18:02", "Source": "172.58.21.9", "Event": "Geo anomaly", "Severity": "Warning"},
                    ],
                    columns=[{"name": i, "id": i} for i in ["Time", "Source", "Event", "Severity"]],
                    style_table={'backgroundColor': 'transparent'},
                    style_cell={'backgroundColor': 'transparent', 'color': '#94a3b8', 'border': 'none', 'padding': '15px'},
                    style_header={'backgroundColor': '#030712', 'color': '#f8fafc', 'fontWeight': 'bold'}
                )
            ])
        ])
        return content, header

    elif pathname == '/incidents':
        header = create_header("Incident Management", "MTTR • MTTD • Active incidents")
        content = html.Div([
            html.Div(className='kpi-row', children=[
                create_kpi_card("Open Incidents", "2"),
                create_kpi_card("MTTR", "38 min"),
                create_kpi_card("MTTD", "4.2 min"),
                create_kpi_card("Closure Rate", "96%")
            ]),
            html.Div(className='table-card', children=[
                html.Div(className='table-header', children=[html.H3("Active Incidents", className='table-title')]),
                dash_table.DataTable(
                    data=[
                        {"ID": "INC-2941", "Title": "Elevated billing-svc latency", "Severity": "Critical", "Status": "Investigating", "Opened": "12 min ago"},
                        {"ID": "INC-2938", "Title": "ML inference queue backlog", "Severity": "Warning", "Status": "Mitigated", "Opened": "2 h ago"},
                        {"ID": "INC-2935", "Title": "Replica lag eu-west-1", "Severity": "Warning", "Status": "Resolved", "Opened": "8 h ago"},
                        {"ID": "INC-2891", "Title": "Auth token rotation failure", "Severity": "Critical", "Status": "Resolved", "Opened": "1 d ago"},
                    ],
                    columns=[{"name": i, "id": i} for i in ["ID", "Title", "Severity", "Status", "Opened"]],
                    style_table={'backgroundColor': 'transparent'},
                    style_cell={'backgroundColor': 'transparent', 'color': '#94a3b8', 'border': 'none', 'padding': '15px'},
                    style_header={'backgroundColor': '#030712', 'color': '#f8fafc', 'fontWeight': 'bold'}
                )
            ])
        ])
        return content, header

    elif pathname == '/sales':
        header = create_header("Sales & Marketing", "Revenue and conversion by continent")
        sales_by_continent = df.groupby('Continent')['Sales_Amount'].sum().reset_index()
        fig_continent = px.bar(sales_by_continent, x='Continent', y='Sales_Amount', title="Revenue by Continent", template='plotly_dark')
        fig_continent.update_layout(**chart_layout)
        
        return html.Div([dcc.Graph(figure=fig_continent)]), header

    else:
        # Default Ops Overview
        header = create_header("Operations Overview", "Global infrastructure and service health")
        latency_by_continent = df.groupby('Continent')['Latency_ms'].mean().reset_index()
        fig_latency = px.pie(latency_by_continent, names='Continent', values='Latency_ms', title="Avg Latency by Continent", hole=0.5)
        fig_latency.update_layout(**chart_layout)
        
        return html.Div([dcc.Graph(figure=fig_latency)]), header

@app.callback(
    Output('live-clock', 'children'),
    [Input('clock-interval', 'n_intervals')]
)
def update_clock(n):
    now = datetime.now()
    return now.strftime("%a, %d %b - %H:%M:%S")

if __name__ == '__main__':
    app.run(debug=True)