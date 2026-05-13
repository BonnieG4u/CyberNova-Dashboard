import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output

df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/gapminderDataOnce.csv')

app = Dash(__name__)

app.layout = html.Div([
    dcc.Graph(id='life-exp-vs-gdp'),
    dcc.Dropdown(
        id='year-dropdown',
        options=[{'label': year, 'value': year} for year in df['year'].unique()],
        value=df['year'].max()
    )
])

@app.callback(
    Output('life-exp-vs-gdp', 'figure'),
    Input('year-dropdown', 'value')
)
def update_graph(selected_year):
    filtered_df = df[df['year'] == selected_year]
    fig = px.scatter(filtered_df, x='gdpPercap', y='lifeExp', size='pop', color='continent', hover_name='country')
    return fig

if __name__ == '__main__':
    app.run(debug=True)