#IMPORTACIONES
import sympy
import numpy as np
import scipy as sp
from scipy.integrate import odeint
import matplotlib.pyplot as plt

from dash import Dash, html, dcc, callback, Output, Input
import dash_bootstrap_components as dbc
import dash_latex as dl
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

import pandas as pd





# Función de tipo Monod
def f_monod(s, m, K):
    return (m * s) / (K + s)


# Función de tipo Haldane
def f_haldane(s, m, K, I):
    return (m * s) / (K + s + (s ** 2 / I))


#-----------------UNA ESPECIE------------------
# Valores para el tipo Monod
monod_values = pd.DataFrame(dict(
))

# Valores para el tipo Haldane
haldane_values = pd.DataFrame(dict(
))


#-----------------VARIAS ESPECIES-----------------
# Valores para el tipo Monod
monod_values1 = pd.DataFrame(dict(
))

monod_values2 = pd.DataFrame(dict(
))

# Valores para el tipo Haldane
haldane_values1 = pd.DataFrame(dict(
))

haldane_values2 = pd.DataFrame(dict(
))





# -------------------------------------------------DASH APP-------------------------------------------------
app = Dash(__name__, external_stylesheets=[dbc.themes.SPACELAB])
server = app.server
app.title = 'Quimiostato simulaciones'

encabezado = html.H4(
    "Simulaciones sobre modelos de Quimiostatos", className="bg-primary text-white p-2", style={'textAlign': 'center'}
)

#------------------------------------CONFIGURACIÓN UNA ESPECIE-----------------------------------
m_parametroM = html.Div(
    [dl.DashLatex(r"""$$ \mu_{max} $$ (Tasa de crecimiento máxima)"""), dbc.Input(id="m-M", type="number", value=1)],
    className="mt-2",
)

m_parametroH = html.Div(
    [dl.DashLatex(r"""$$ \mu_0 $$ (Tasa de crecimiento inicial)"""), dbc.Input(id="m-H", type="number", value=5)],
    className="mt-2",
)

K_parametroH = html.Div(
    [dl.DashLatex(r"""$$ K_s $$ (Constante de semi-saturación)"""), dbc.Input(id="K-H", type="number", value=0.5)],
    className="mt-2",
)

I_parametroH = html.Div(
    [dl.DashLatex(r"""$$ K_i $$ (Constante de inhibición)"""), dbc.Input(id="I-H", type="number", value=0.2)],
    className="mt-2",
)

D_parametroH = html.Div([
    dl.DashLatex(r"""$$ D $$ (Tasa de dilución)"""),
    dcc.Slider(
        id="D-sliderH",
        min=0,
        max=1.2,
        step=0.05,
        value=1,
    )],
    className="mt-2"
)

Sin_parametroH = html.Div([
    dl.DashLatex(r"""$$ S_{in} $$ (Concentración del recurso limitante en el alimento)"""),
    dcc.Slider(
        id="Sin-sliderH",
        min=0,
        max=3,
        step=0.1,
        value=2,
    )],
    className="mt-2"
)

D_parametroM = html.Div([
    dl.DashLatex(r"""$$ D $$ (Tasa de dilución)"""),
    dcc.Slider(
        id="D-sliderM",
        min=0,
        max=0.99,
        step=0.05,
        value=0.7,
    )],
    className="mt-2"
)

K_parametroM = html.Div(
    [dl.DashLatex(r"""$$ K_s $$ (Constante de semi-saturación)"""), dbc.Input(id="K-M", type="number", value=0.2)],
    className="mt-2",
)

Sin_parametroM = html.Div([
    dl.DashLatex(r"""$$ S_{in} $$ (Concentración del recurso limitante en el alimento)"""),
    dcc.Slider(
        id="Sin-sliderM",
        min=0,
        max=3,
        step=0.1,
        value=2,
    )],
    className="mt-2"
)

monod_configuracion = dbc.Card(
    dbc.CardBody([
        m_parametroM,
        K_parametroM,
        dcc.Graph(id='monod-graph', style={'marginTop': 5}, mathjax=True),

    ],
        className="bg-light",
    )
)

haldane_configuracion = dbc.Card(
    dbc.CardBody([
        m_parametroH,
        K_parametroH,
        I_parametroH,
        dcc.Graph(id='haldane-graph', style={'marginTop': 5}, mathjax=True),

    ],
        className="bg-light",
    )
)

monod_evolucion_configuracion = html.Div([
    D_parametroM,
    Sin_parametroM
])

haldane_evolucion_configuracion = html.Div([
    D_parametroH,
    Sin_parametroH
])

planoFaseMonod = html.Div([
    dcc.Graph(id='planoFaseMonod', style={'marginTop': 5}, mathjax=True),
])

regionesMonod = html.Div([
    dcc.Graph(id='regionesMonod', style={'marginTop': 5}, mathjax=True),
])

planoFaseHaldane = html.Div([
    dcc.Graph(id='planoFaseHaldane', style={'marginTop': 5}, mathjax=True),
])

regionesHaldane = html.Div([
    dcc.Graph(id='regionesHaldane', style={'marginTop': 5}, mathjax=True),
])

monod_content = dbc.Card(
    dbc.CardBody(
        [
            dl.DashLatex(
                r"""
                Función de consumo de tipo Monod: 
                 $$\mu(s) = { s \over K + s }.$$
                """,
            ),
            dbc.Row([dbc.Col(monod_configuracion, md=4),
                     dbc.Col([dcc.Graph(id='monod-time-evolution-graph'), monod_evolucion_configuracion], md=8)],
                    className="m-2"),
            dbc.Row([dbc.Col(regionesMonod, md=6), dbc.Col(planoFaseMonod, md=6)], className="m-2"),
        ]
    ),
    className="m-3",
)

haldane_content = dbc.Card(
    dbc.CardBody(
        [
            dl.DashLatex(
                r"""
                Función de consumo de tipo Haldane: $$\mu(s) = {(m * s) \over (K + s + \frac{s^2}{I})}.$$
                """,
            ),
            dbc.Row([dbc.Col(haldane_configuracion, md=4),
                     dbc.Col([dcc.Graph(id='haldane-time-evolution-graph'), haldane_evolucion_configuracion], md=8)],
                    className="m-2"),
            dbc.Row([dbc.Col(regionesHaldane, md=6), dbc.Col(planoFaseHaldane, md=6)], className="m-2"),
        ]
    ),
    className="m-3",
)

#------------------------------------CONFIGURACIÓN VARIAS ESPECIES-----------------------------------
m_parametroM1 = html.Div(
    [dl.DashLatex(r"""$$ \mu_{1_{max}} $$ (Tasa de crecimiento máxima)"""), dbc.Input(id="m-M1", type="number", value=1.2)],
    className="mt-2",
)

m_parametroH1 = html.Div(
    [dl.DashLatex(r"""$$ \mu_{1_0} $$ (Tasa de crecimiento inicial)"""), dbc.Input(id="m-H1", type="number", value=5)],
    className="mt-2",
)

m_parametroM2 = html.Div(
    [dl.DashLatex(r"""$$ \mu_{2_{max}} $$ (Tasa de crecimiento máxima)"""), dbc.Input(id="m-M2", type="number", value=1)],
    className="mt-2",
)

m_parametroH2 = html.Div(
    [dl.DashLatex(r"""$$ \mu_{2_0} $$ (Tasa de crecimiento inicial)"""), dbc.Input(id="m-H2", type="number", value=5)],
    className="mt-2",
)

K_parametroH1 = html.Div(
    [dl.DashLatex(r"""$$ K_{s_1} $$ (Constante de semi-saturación)"""), dbc.Input(id="K-H1", type="number", value=0.5)],
    className="mt-2",
)

I_parametroH1 = html.Div(
    [dl.DashLatex(r"""$$ K_{i_1} $$ (Constante de inhibición)"""), dbc.Input(id="I-H1", type="number", value=0.2)],
    className="mt-2",
)

D_parametroH1 = html.Div([
    dl.DashLatex(r"""$$ D $$ (Tasa de dilución)"""),
    dcc.Slider(
        id="D-sliderH1",
        min=0,
        max=1.19,
        step=0.05,
        value=1,
    )],
    className="mt-2"
)

Sin_parametroH1 = html.Div([
    dl.DashLatex(r"""$$ S_{in} $$ (Concentración del recurso limitante en el alimento)"""),
    dcc.Slider(
        id="Sin-sliderH1",
        min=0,
        max=3,
        step=0.1,
        value=2,
    )],
    className="mt-2"
)

K_parametroH2 = html.Div(
    [dl.DashLatex(r"""$$ K_{s_2} $$ (Constante de semi-saturación)"""), dbc.Input(id="K-H2", type="number", value=0.1)],
    className="mt-2",
)

I_parametroH2 = html.Div(
    [dl.DashLatex(r"""$$ K_{i_2} $$ (Constante de inhibición)"""), dbc.Input(id="I-H2", type="number", value=0.05)],
    className="mt-2",
)

D_parametroM1 = html.Div([
    dl.DashLatex(r"""$$ D $$ (Tasa de dilución)"""),
    dcc.Slider(
        id="D-sliderM1",
        min=0,
        max=0.99,
        step=0.05,
        value=0.7,
    )],
    className="mt-2"
)

K_parametroM1 = html.Div(
    [dl.DashLatex(r"""$$ K_{s_1} $$ (Constante de semi-saturación)"""), dbc.Input(id="K-M1", type="number", value=0.2)],
    className="mt-2",
)

K_parametroM2 = html.Div(
    [dl.DashLatex(r"""$$ K_{s_2} $$ (Constante de semi-saturación)"""), dbc.Input(id="K-M2", type="number", value=0.2)],
    className="mt-2",
)

Sin_parametroM1 = html.Div([
    dl.DashLatex(r"""$$ S_{in} $$ (Concentración del recurso limitante en el alimento)"""),
    dcc.Slider(
        id="Sin-sliderM1",
        min=0,
        max=3,
        step=0.1,
        value=2,
    )],
    className="mt-2"
)

monod_configuracion1 = dbc.Card(
    dbc.CardBody([
        dbc.Label("Parámetros primera especie"),
        m_parametroM1,
        K_parametroM1,
        html.Br(),
        html.Br(),
        dbc.Label("Parámetros segunda especie"),
        m_parametroM2,
        K_parametroM2,
        html.Br(),
        dcc.Graph(id='monod-graph1', style={'marginTop': 5}, mathjax=True),

    ],
        className="bg-light",
    )
)

haldane_configuracion1 = dbc.Card(
    dbc.CardBody([
        dbc.Label("Parámetros primera especie"),
        m_parametroH1,
        K_parametroH1,
        I_parametroH1,
        html.Br(),
        html.Br(),
        dbc.Label("Parámetros segunda especie"),
        m_parametroH2,
        K_parametroH2,
        I_parametroH2,
        html.Br(),
        dcc.Graph(id='haldane-graph1', style={'marginTop': 5}, mathjax=True),

    ],
        className="bg-light",
    )
)

monod_evolucion_configuracion1 = html.Div([
    D_parametroM1,
    Sin_parametroM1
])

haldane_evolucion_configuracion1 = html.Div([
    D_parametroH1,
    Sin_parametroH1
])

monod_content1 = dbc.Card(
    dbc.CardBody(
        [
            dl.DashLatex(
                r"""
                Función de consumo de tipo Monod: $$\mu(s) = { s \over K + s }.$$
                """,
            ),
            dbc.Row([dbc.Col(monod_configuracion1, md=4),
                     dbc.Col([dcc.Graph(id='monod-time-evolution-graph1'), monod_evolucion_configuracion1], md=8)],
                    className="m-2"),
        ]
    ),
    className="m-3",
)

haldane_content1 = dbc.Card(
    dbc.CardBody(
        [
            dl.DashLatex(
                r"""
                Función de consumo de tipo Haldane: $$\mu(s) = {(m * s) \over (K + s + \frac{s^2}{I})}.$$
                """,
            ),
            dbc.Row([dbc.Col(haldane_configuracion1, md=4),
                     dbc.Col([dcc.Graph(id='haldane-time-evolution-graph1'), haldane_evolucion_configuracion1], md=8)],
                    className="m-2"),
        ]
    ),
    className="m-3",
)

variasEspecies = dbc.Card(
    dbc.CardBody(
        [
            dl.DashLatex(
                r"""
                Modelo con varias especies.$$
                """,
            ),
            dbc.Row(monod_content1,
                    className="m-2"),
            dbc.Row(haldane_content1, className="m-2"),
        ]
    ),
    className="m-3",
)

#-------------------------------PESTAÑAS---------------------------------
tabsFunctions = dbc.Tabs(
    [
        dbc.Tab(monod_content, label="Función de consumo de tipo Monod"),
        dbc.Tab(haldane_content, label="Función de consumo de tipo Haldane"),
        dbc.Tab(variasEspecies, label="Varias Especies"),
    ],
    className="mt-5",
)
app.layout = html.Div(
    [encabezado, tabsFunctions]
)
















#---------------------------------------------------------FUNCIONES UNA ESPECIE------------------------------------------------------------------------------

#---------------------------------------------------------TIPO MONOD------------------------------------------------------------------------------
# Función gráfica para el tipo Monod con los parámentros m y K
@callback(
    Output('monod-graph', 'figure'),
    Input('m-M', 'value'),
    Input('K-M', 'value'),
    Input('D-sliderM', 'value'),
    Input('Sin-sliderM', 'value'),
)
def monod_consume(valuem, valueK, valueD, valueSin):
    #Parámetros m y K
    m = valuem
    K = valueK

    #Valor de lambda
    lam = (K * valueD) / (m - valueD)

    #Malla de s para la representación gráfica de la función Monod
    s = np.linspace(0, 2*valueSin if(lam<valueSin) else 2*lam, 100)
    #DataFrame de puntos de la función Monod para representar
    monod_values = pd.DataFrame(dict(
        x=s,
        y=f_monod(s, m, K)
    ))

    #Valor máximo de la función
    sup = pd.DataFrame(dict(
        s=s,
        x=m
    ))

    #Líneas discontinuas para los valores de D
    x = np.linspace(0, f_monod(lam, m, K), 100)
    s = np.linspace(0, lam, 100)
    lX = pd.DataFrame(dict(
        s=lam,
        x= x
    ))
    lY = pd.DataFrame(dict(
        s=s,
        x=f_monod(lam, m, K)
    ))

    #Líneas discontinuas para los valores de Sin
    x = np.linspace(0, f_monod(valueSin, m, K), 100)
    s = np.linspace(0, valueSin, 100)
    sX = pd.DataFrame(dict(
        s=valueSin,
        x=x
    ))
    sY = pd.DataFrame(dict(
        s=s,
        x=f_monod(valueSin, m, K)
    ))

    #Gráfica usando Plotly
    fig = go.Figure()
    #Función Monod
    fig.add_trace(go.Scatter(x=monod_values['x'], y=monod_values['y']))
    #Valor máximo de la función
    fig.add_trace(go.Scatter(x=sup['s'], y=sup['x'], mode='lines + text', line=dict(color='#4B4B4B', width=4,
                              dash='dash')))
    #Punto asociado a D
    fig.add_trace(go.Scatter(x=[lam],
                             y=[valueD],
                             text=[r'$(\lambda(D),D)$'],
                             textposition="bottom right",
                             mode='markers + text',
                             marker_size=8,
                             name='points',
                             marker_color='#f54242'))
    #Punto asociado a Sin
    fig.add_trace(go.Scatter(x=[valueSin],
                             y=[f_monod(valueSin, valuem, valueK)],
                             text=[r'$(S_{in},\mu(S_{in}))$'],
                             textposition="bottom right",
                             mode='markers + text',
                             marker_size=8,
                             name='points',
                             marker_color='#167B02'))
    #Líneas discontinuas para los valores D y Sin
    fig.add_trace(go.Scatter(x=lX['s'], y=lX['x'], mode='lines + text', line=dict(color='#f54242', width=2,
                              dash='dash')))
    fig.add_trace(go.Scatter(x=lY['s'], y=lY['x'], mode='lines + text', line=dict(color='#f54242', width=2,
                                                                                  dash='dash')))
    fig.add_trace(go.Scatter(x=sX['s'], y=sX['x'], mode='lines + text', line=dict(color='#167B02', width=2,
                                                                                  dash='dash')))
    fig.add_trace(go.Scatter(x=sY['s'], y=sY['x'], mode='lines + text', line=dict(color='#167B02', width=2,
                                                                                  dash='dash')))
    #Configuración ejes y fondo gráfica
    fig.update_layout(xaxis_title=r'$s$', yaxis_title=r'$\mu(s)$', )
    fig.update_layout(
        plot_bgcolor='white',
        showlegend=False
    )
    fig.update_xaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        zerolinecolor='lightgrey',
        gridcolor='lightgrey'
    )
    fig.update_yaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        zerolinecolor='lightgrey',
        gridcolor='lightgrey'
    )
    #Texto y anotaciones para valor máximo y puntos D y Sin
    anotaciones = [[r'$\mu_{max}$', s[10], valuem], [r'$D$', s[0], valueD], [r'$\mu(S_{in})$', s[0], f_monod(valueSin, m, K)], 
                    [r'$\lambda(D)$', lam, 0], [r'$S_{in}$', valueSin, 0]]
    for elem in anotaciones:
        fig.add_annotation(
            text=elem[0],
            xanchor='left',
            yanchor='bottom',
            x=elem[1],
            y=elem[2],
            font=dict(
                size=16 if(elem[2]==valuem) else 9,
                color="#000000"
            ),
            showarrow=False)
    
    return fig


# Función evolución de alimento y la población en función del tiempo, introduciendo parámetros D y Sin, para el tipo Monod
@callback(
    Output('monod-time-evolution-graph', 'figure'),
    Input('D-sliderM', 'value'),
    Input('Sin-sliderM', 'value'),
    Input('m-M', 'value'),
    Input('K-M', 'value'),
)
def monod_time_evolution(valueD, valueSin, valuem, valueK):
    #Condición inicial
    c0 = 0, 0.2

    #Mallado para el tiempo
    t = np.linspace(0, 50, 100)
    #Parámetros
    Sin = valueSin
    D = valueD
    m = valuem
    K = valueK

    #Modelización para el problema de una especie para el tipo Monod
    def model_monod(y, t):
        s, x = y
        ds = D * (Sin - s) - f_monod(s, m, K) * x
        dx = (f_monod(s, m, K) - D) * x
        dy = [ds, dx]

        return dy
    #Solución del sistema diferencial anterior
    sol = odeint(model_monod, c0, t)
    #Valores solución en un DataFrame
    values = pd.DataFrame(dict(
        t=t,
        s=sol[:, 0],
        x=sol[:, 1]
    ))

    #Gráfico evolución temporal en función de t para las concentraciones de s y x
    fig = px.line(values, x='t', y=['s', 'x'])
    fig.update_layout(
        plot_bgcolor='white'
    )
    #Configuración ejes y fondo del gráfico
    fig.update_xaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        zerolinecolor='lightgrey',
        gridcolor='lightgrey'
    )
    fig.update_yaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        zerolinecolor='lightgrey',
        gridcolor='lightgrey'
    )

    return fig


# Función hallar las regiones Monod
@callback(
    Output('regionesMonod', 'figure'),
    Input('D-sliderM', 'value'),
    Input('Sin-sliderM', 'value'),
    Input('m-M', 'value'),
    Input('K-M', 'value'),
)
def regionesMonod(valueD, valueSin, valuem, valueK):
    #Valor lambda
    lam = (valueK * valueD) / (valuem - valueD)
    #Intervalo valores de s
    s = np.linspace(0.01, valueSin if (lam < valueSin) else 2 * lam, 100)
    rangeEjex = valueSin if (lam < valueSin) else 2 * lam
    #Isoclina de s
    isoclinaS = pd.DataFrame(dict(
        s=s,
        x=(valueD * (valueSin - s)) / f_monod(s, valuem, valueK)
    ))
    #Isoclina de x
    x = np.linspace(0, 2 * valueSin, 100)
    isoclinaX = pd.DataFrame(dict(
        x=x,
        s=(valueK * valueD) / (valuem - valueD)
    ))
    #Límite de la gráfica
    limitX = pd.DataFrame(dict(
        x=x,
        s=rangeEjex
    ))

    #Gráfica de las regiones delimitadas por las isoclinas en el tipo Monod
    fig = go.Figure()
    #Regiones de las isoclinas
    fig.add_trace(go.Scatter(x=isoclinaX['s'], y=isoclinaX['x'], mode='none', fill='tozerox'))
    fig.add_trace(go.Scatter(x=limitX['s'], y=limitX['x'], mode='lines', fill='tonexty'))
    fig.add_trace(
        go.Scatter(x=isoclinaS['s'], y=isoclinaS['x'], mode='none', fill='tozeroy',
                    fillcolor='rgba(255,255,102, 0.5)'))
    #Gráficas isoclinas
    fig.add_trace(go.Scatter(x=isoclinaX['s'], y=isoclinaX['x'], mode='lines', line_color='rgba(0,0,102,0.8)'))
    fig.add_trace(go.Scatter(x=isoclinaS['s'], y=isoclinaS['x'], mode='lines', line_color='rgba(0,0,102,0.8)'))
    #Puntos de equilibrio
    equilibrios = [[valueSin, 0, r'$\Large{E_0=(S_{in},0)}$'], [(valueK * valueD) / (valuem - valueD), valueSin - (valueK * valueD) / (valuem - valueD), r'$\large{E_1=(s^*,x^*)}$']] if (lam < valueSin) else [[valueSin, 0, r'$\Large{E_0=(S_{in},0)}$']]
    for elem in equilibrios:
        fig.add_trace(go.Scatter(x=[elem[0]],
                                    y=[elem[1]],
                                    text=[elem[2]],
                                    textposition="top right",
                                    textfont_color ='black',
                                    mode='markers + text',
                                    marker_size=10,
                                    name='points',
                                    marker_color='rgba(255,51,51,1)'))
    #Configuración ejes y fondo gráfica
    fig.update_layout(
        width=700,
        height=700,
        plot_bgcolor='white',
        showlegend=False
    )
    fig.update_xaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        zerolinecolor='lightgrey',
        gridcolor='lightgrey',
        range=[0, rangeEjex],
        title=r"$s(t)$"
    )
    fig.update_yaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        zerolinecolor='lightgrey',
        gridcolor='lightgrey',
        range=[0, 2 * valueSin],
        title=r"$x(t)$"
    )
    #Anotaciones isoclinas
    isoclinas = [[r'$\Large{x = \frac{D(S_{in}-s)}{\mu(s)}}$', isoclinaS['s'].iloc[55], isoclinaS['x'].iloc[55]], 
                [r'$\Large{s = \lambda(D)}$', isoclinaX['s'].iloc[75], isoclinaX['x'].iloc[75]]]
    for elem in isoclinas:
        fig.add_annotation(
            text=elem[0],
            xanchor='left',
            yanchor='bottom',
            x=elem[1],
            y=elem[2],
            arrowhead=1,
            ax=20,
            ay=-30,
            font=dict(
                size=25,
                color="#000000"
            ),
            arrowcolor="#000000",
            showarrow=True)
    #Anotaciones regiones
    anotaciones = [[r'$\Large{R_1}$', lam + (valueSin-lam)/2, (valueSin+lam)/2 + (valueSin-lam),"#d94343"],[r'$\Large{R_2}$', 2*lam/3, 2*(valueSin+lam)/3 + (valueSin-lam), "#4364d9"],
                    [r'$\Large{R_3}$', lam/2, (valueSin-lam)/2, "#48992c" ], [r'$\Large{R_4}$', lam +(valueSin-lam)/4, (valueSin-lam)/2,"#fcc603"]] if (lam < valueSin) else [[r'$\Large{R_1}$', 3*lam/2, 4*valueSin/3,"#d94343"],[r'$\Large{R_2}$', lam/2, 4*valueSin/3, "#4364d9"],
                    [r'$\Large{R_3}$', lam/3, 2*valueSin/3, "#48992c" ]]
    for elem in anotaciones:
        fig.add_annotation(
            text=elem[0],
            xanchor='right',
            x=elem[1],
            y=elem[2],
            font=dict(
                size=16,
                color="#000000"
            ),
            bordercolor=elem[3],
            borderwidth=5,
            borderpad=4,
            bgcolor="#ffffff",
            opacity=0.8,
            showarrow=False)

    return fig
    


# Función hallar los planos de fase Monod
@callback(
    Output('planoFaseMonod', 'figure'),
    Input('D-sliderM', 'value'),
    Input('Sin-sliderM', 'value'),
    Input('m-M', 'value'),
    Input('K-M', 'value'),
)
def planoFaseMonod(valueD, valueSin, valuem, valueK):
    #Valor lambda
    lam = (valueK * valueD) / (valuem - valueD)

    #Intervalo valores de s
    s = np.linspace(0.01, valueSin if (lam < valueSin) else 2 * lam, 100)
    rangeEjex = valueSin if (lam < valueSin) else 2 * lam
    #Isoclina de s
    isoclinaS = pd.DataFrame(dict(
        s=s,
        x=(valueD * (valueSin - s)) / f_monod(s, valuem, valueK)
    ))
    #Isoclina de x
    x = np.linspace(0, 2 * valueSin, 100)
    isoclinaX = pd.DataFrame(dict(
        x=x,
        s=(valueK * valueD) / (valuem - valueD)
    ))
    #Límite de la gráfica
    limitX = pd.DataFrame(dict(
        x=x,
        s=rangeEjex
    ))
    #Flujo de direcciones
    s1, x1 = np.meshgrid(np.linspace(0.001, valueSin if (lam < valueSin) else 2 * lam, 20), np.linspace(0, 2 * valueSin, 30))
    ds = valueD * (valueSin - s1) - f_monod(s1, valuem, valueK) * x1
    dx = (f_monod(s1, valuem, valueK) - valueD) * x1
    #Plano de fases
    figFases = ff.create_quiver(s1, x1, ds, dx, scaleratio=0.4,
                                scale=0.15 ,
                                arrow_scale=.25,
                                name='quiver',
                                line_width=2,
                                line_color='rgba(0,0,102,1)')

    #Gráfica de las regiones delimitadas por las isoclinas en el tipo Monod
    fig = go.Figure()
    #Regiones de las isoclinas
    fig.add_trace(go.Scatter(x=isoclinaX['s'], y=isoclinaX['x'], mode='none', fill='tozerox'))
    fig.add_trace(go.Scatter(x=limitX['s'], y=limitX['x'], mode='lines', fill='tonexty'))
    fig.add_trace(
        go.Scatter(x=isoclinaS['s'], y=isoclinaS['x'], mode='none', fill='tozeroy',
                    fillcolor='rgba(255,255,102, 0.5)'))
    #Gráficas isoclinas
    fig.add_trace(go.Scatter(x=isoclinaX['s'], y=isoclinaX['x'], mode='lines', line_color='rgba(0,0,102,0.8)'))
    fig.add_trace(go.Scatter(x=isoclinaS['s'], y=isoclinaS['x'], mode='lines', line_color='rgba(0,0,102,0.8)'))
    #Añade plano de fases
    fig.add_traces(data=figFases.data)
    #Puntos de equilibrio
    equilibrios = [[valueSin, 0], [(valueK * valueD) / (valuem - valueD), valueSin - (valueK * valueD) / (valuem - valueD)]] if (lam < valueSin) else [[valueSin, 0]]
    for elem in equilibrios:
        fig.add_trace(go.Scatter(x=[elem[0]],
                                    y=[elem[1]],
                                    mode='markers',
                                    marker_size=10,
                                    name='points',
                                    marker_color='rgba(255,51,51,1)'))
    #Configuración ejes y fondo gráfica
    fig.update_layout(
        width=700,
        height=700,
        plot_bgcolor='white',
        showlegend=False
    )
    fig.update_xaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        zerolinecolor='lightgrey',
        gridcolor='lightgrey',
        range=[0, rangeEjex],
        title=r"$s(t)$"
    )
    fig.update_yaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        zerolinecolor='lightgrey',
        gridcolor='lightgrey',
        range=[0, 2 * valueSin],
        title=r"$x(t)$"
    )

   
    return fig




#---------------------------------------------------------HALDANE------------------------------------------------------------------------------
# Función gráfica para el tipo Haldane con los parámentros m, K y I
@callback(
    Output('haldane-graph', 'figure'),
    Input('m-H', 'value'),
    Input('K-H', 'value'),
    Input('I-H', 'value'),
    Input('D-sliderH', 'value'),
    Input('Sin-sliderH', 'value'),
)
def haldane_consume(valuem, valueK, valueI, valueD, valueSin):
    #Parámetros
    m = valuem
    I = valueI
    K = valueK
    #Valor máximo
    sm = np.sqrt(valueK*valueI)
    val_sm = f_haldane(sm, valuem, valueK, valueI)
    #Valores de lambda1 y lambda2
    lam1=((-(valuem-valueD)*valueI) + np.sqrt(((valuem-valueD)*valueI)**2 - 4*(valueD**2)*valueK*valueI))/(-2*valueD)
    lam2 = (-(valuem-valueD)*valueI-np.sqrt(((valuem-valueD)*valueI)**2 - 4*(valueD**2)*valueK*valueI))/(-2*valueD)
    #Intervalo y puntos de s
    s = np.linspace(0, 2*valueSin if(lam2 < valueSin) else 2 * lam2, 100)
    #Valores de la función para s
    haldane_values = pd.DataFrame(dict(
        x=s,
        y=f_haldane(s, m, K, I)
    ))
    #Discontinuas lambda1
    x = np.linspace(0, f_haldane(lam1, m, K, I), 100)
    s = np.linspace(0, lam1, 100)
    l1X = pd.DataFrame(dict(
        s=lam1,
        x=x
    ))
    l1Y = pd.DataFrame(dict(
        s=s,
        x=f_haldane(lam1, m, K, I)
    ))
    #Discontinuas lambda2
    x = np.linspace(0, f_haldane(lam2, m, K, I), 100)
    s = np.linspace(0, lam2, 100)
    l2X = pd.DataFrame(dict(
        s=lam2,
        x=x
    ))
    l2Y = pd.DataFrame(dict(
        s=s,
        x=f_haldane(lam2, m, K, I)
    ))
    #Discontinuas Sin
    x = np.linspace(0, f_haldane(valueSin, m, K, I), 100)
    s = np.linspace(0, valueSin, 100)
    sX = pd.DataFrame(dict(
        s=valueSin,
        x=x
    ))
    sY = pd.DataFrame(dict(
        s=s,
        x=f_haldane(valueSin, m, K, I)
    ))
    #Gráfica usando Plotly
    fig = px.line(haldane_values, x='x', y='y', range_y=[0, val_sm+0.2])
    #Puntos importantes de la gráfica lambda1, lambda2, Sin, y sm
    puntos = [[lam1, valueD, r'$(\lambda(D), D)$', "bottom right", '#f54242'], [lam2, valueD, r'$(\overline{\lambda}(D), D)$', "top right", '#f54242'], 
                [valueSin, f_haldane(valueSin, valuem, valueK, valueI), r'$(S_{in},\mu(S_{in}))$', "top right", '#167B02'], [sm, val_sm, r'$(s_m,\mu(s_m))$', "top right", '#4B4B4B']]
    for elem in puntos:
        fig.add_trace(go.Scatter(x=[elem[0]],
                                y=[elem[1]],
                                text=[elem[2]],
                                textposition=elem[3],
                                mode='markers + text',
                                marker_size=8,
                                name='points',
                                marker_color=elem[4]))
    #Líneas discontinuas para los puntos
    fig.add_trace(go.Scatter(x=l1X['s'], y=l1X['x'], mode='lines + text', line=dict(color='#f54242', width=2,
                                                                                  dash='dash')))
    fig.add_trace(go.Scatter(x=l1Y['s'], y=l1Y['x'], mode='lines + text', line=dict(color='#f54242', width=2,
                                                                                  dash='dash')))
    fig.add_trace(go.Scatter(x=l2X['s'], y=l2X['x'], mode='lines + text', line=dict(color='#f54242', width=2,
                                                                                    dash='dash')))
    fig.add_trace(go.Scatter(x=l2Y['s'], y=l2Y['x'], mode='lines + text', line=dict(color='#f54242', width=2,
                                                                                    dash='dash')))
    fig.add_trace(go.Scatter(x=sX['s'], y=sX['x'], mode='lines + text', line=dict(color='#167B02', width=2,
                                                                                  dash='dash')))
    fig.add_trace(go.Scatter(x=sY['s'], y=sY['x'], mode='lines + text', line=dict(color='#167B02', width=2,
                                                                                  dash='dash')))
    #Configuración ejes y fondo gráfica
    fig.update_layout(xaxis_title=r'$s$', yaxis_title=r'$\mu(s)$', plot_bgcolor='white',showlegend=False)
    fig.update_xaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        zerolinecolor='lightgrey',
        gridcolor='lightgrey',
    )
    fig.update_yaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        zerolinecolor='lightgrey',
        gridcolor='lightgrey'
    )
    #Anotaciones sobre la gráfica para valores D, lambda1, lambda2 y Sin
    anotaciones = [[r'$D$', s[0], f_haldane(lam1, m, K, I)],[r'$\mu(S_{in})$', s[0], f_haldane(valueSin, m, K, I)],[r'$\lambda(D)$',lam1,0 ],
                    [r'$\overline{\lambda}(D)$', lam2, 0],[r'$S_{in}$', valueSin, 0]]
    for elem in anotaciones:
        fig.add_annotation(
            text=elem[0],
            xanchor='left',
            yanchor='bottom',
            x=elem[1],
            y=elem[2],
            font=dict(
                size=9,
                color="#000000"
            ),
            showarrow=False)
    
    return fig


# Función evolución de alimento y la población en función del tiempo, introduciendo parámetros D y Sin, para el tipo Haldane
@callback(
    Output('haldane-time-evolution-graph', 'figure'),
    Input('D-sliderH', 'value'),
    Input('Sin-sliderH', 'value'),
    Input('m-H', 'value'),
    Input('K-H', 'value'),
    Input('I-H', 'value')
)
def haldane_time_evolution(valueD, valueSin, valuem, valueK, valueI):
    #Condición inicial
    c0 = 0, 0.2
    #Intervalo de tiempo e instantes
    t = np.linspace(0, 50, 100)
    #Parámetros
    Sin = valueSin
    D = valueD
    #Modelización del problema para una especie utilizando función de tipo Haldane
    def model_haldane(y, t):
        s, x = y
        ds = D * (Sin - s) - f_haldane(s, valuem, valueK, valueI) * x
        dx = (f_haldane(s, valuem, valueK, valueI) - D) * x
        dy = [ds, dx]

        return dy
    #Solución del sistema diferencial anterior
    sol = odeint(model_haldane, c0, t)
    #DataFrame con los valores solución del sistema
    values = pd.DataFrame(dict(
        t=t,
        s=sol[:, 0],
        x=sol[:, 1]
    ))
    #Gráfica para cada tiempo t de las concentraciones de s y x
    fig = px.line(values, x='t', y=['s', 'x'])
    #Configuración de ejes y del fondo de la gráfica
    fig.update_layout(
        plot_bgcolor='white'
    )
    fig.update_xaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        zerolinecolor='lightgrey',
        gridcolor='lightgrey'
    )
    fig.update_yaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        zerolinecolor='lightgrey',
        gridcolor='lightgrey'
    )

    return fig

# Función hallar las regiones Haldane
@callback(
    Output('regionesHaldane', 'figure'),
    Input('D-sliderH', 'value'),
    Input('Sin-sliderH', 'value'),
    Input('m-H', 'value'),
    Input('K-H', 'value'),
    Input('I-H', 'value'),
)
def regionesHaldane(valueD, valueSin, valuem, valueK, valueI):
    #Valores lambda1 y lambda2
    lam1 = ((-(valuem - valueD) * valueI) + np.sqrt(
        ((valuem - valueD) * valueI) ** 2 - 4 * (valueD ** 2) * valueK * valueI)) / (-2 * valueD)
    val_lam1 = f_haldane(lam1, valuem, valueK, valueI)
    lam2 = (-(valuem - valueD) * valueI - np.sqrt(
        ((valuem - valueD) * valueI) ** 2 - 4 * (valueD ** 2) * valueK * valueI)) / (-2 * valueD)
    val_lam2 = f_haldane(lam2, valuem, valueK, valueI)
    #Intervalo y valores de s
    s = np.linspace(0.01, 2*lam2 if (lam2 > valueSin) else 2*valueSin, 100)
    rangeEjex = lam2*2 if (lam2 > valueSin) else 2*valueSin+0.4
    #Isoclina de s
    isoclinaS = pd.DataFrame(dict(
        s=s,
        x=(valueD * (valueSin - s)) / f_haldane(s, valuem, valueK, valueI)
    ))
    #Isoclina de x asociada a lambda1
    x = np.linspace(0, 2*valueSin, 100)
    isoclinaX1 = pd.DataFrame(dict(
        x=x,
        s=lam1
    ))
    #Isoclina de x asociada a lambda2
    isoclinaX2 = pd.DataFrame(dict(
        x=x,
        s=lam2
    ))
    #Límite del gráfico
    limitX = pd.DataFrame(dict(
        x=x,
        s=rangeEjex
    ))

    #Gráfica de las regiones delimitadas por las isoclinas
    fig= go.Figure()
    fig.add_trace(
        go.Scatter(x=isoclinaX1['s'], y=isoclinaX1['x'], mode='none', fill='tozerox'))
    fig.add_trace(go.Scatter(x=isoclinaX2['s'], y=isoclinaX2['x'], mode='none', fill='tonexty'))
    fig.add_trace(go.Scatter(x=limitX['s'], y=limitX['x'], mode='lines', fill='tonexty', fillcolor='rgba(50, 168, 82, 0.5)'))
    fig.add_trace(go.Scatter(x=isoclinaS['s'], y=isoclinaS['x'], mode='none', fill='tozeroy', fillcolor='rgba(255,255,102, 0.7)'))
    fig.add_trace(go.Scatter(x=isoclinaX1['s'], y=isoclinaX1['x'], mode='lines', line_color='rgba(0,0,102,0.8)'))
    fig.add_trace(go.Scatter(x=isoclinaX2['s'], y=isoclinaX2['x'], mode='lines', line_color='rgba(0,0,102,0.8)'))
    fig.add_trace(go.Scatter(x=isoclinaS['s'], y=isoclinaS['x'], mode='lines', line_color='rgba(0,0,102,0.8)'))
    #Puntos de equilibrio
    equilibrios = [[valueSin, 0, r'$\large{E_0=(S_{in},0)}$']] if (lam1 > valueSin) else [[valueSin, 0, r'$\large{E_0=(S_{in},0)}$'], [lam1, (valueD * (valueSin - lam1)) / val_lam1, r'$\large{E_1=(s^*,x^*)}$']]
    for elem in equilibrios:
        fig.add_trace(go.Scatter(x=[elem[0]],
                                 y=[elem[1]],
                                 text=[elem[2]],
                                 textposition="top right",
                                 textfont_color="black",
                                 mode='markers + text',
                                 marker_size=10,
                                 name='points',
                                 marker_color='rgba(255,51,51,1)'))
    #Isoclinas
    isoclinas = [[r'$\Large{x = \frac{D(S_{in}-s)}{\mu(s)}}$', isoclinaS['s'].iloc[20], isoclinaS['x'].iloc[20]], [r'$\large{s = \lambda(D)}$', isoclinaX1['s'].iloc[75], isoclinaX1['x'].iloc[75]], 
                [r'$\large{s = \overline{\lambda}(D)}$', isoclinaX2['s'].iloc[70], isoclinaX2['x'].iloc[70]]]
    for elem in isoclinas:
        fig.add_annotation(
            text=elem[0],
            xanchor='left',
            yanchor='bottom',
            x=elem[1],
            y=elem[2],
            arrowhead=1,
            ax=20,
            ay=-30,
            font=dict(
                size=16,
                color="#000000"
            ),
            arrowcolor="#000000",
            showarrow=True)
    #Anotaciones texto regiones
    anotaciones = [[r'$\Large{R_1}$', lam2 + (lam2)/2, valueSin, "#48992c"],[r'$\Large{R_2}$', lam1 +(lam2-lam1)/2, valueSin,"#d94343"],[r'$\Large{R_3}$', 3*(lam1)/4, valueSin,"#4364d9"],[r'$\Large{R_4}$', 3*valueSin/4, 2*valueSin/12, "#fcc603"]]
    if (lam1 < valueSin and lam2 > valueSin): anotaciones=[[r'$\Large{R_1}$', lam2 + (lam2)/2, valueSin, "#48992c"],[r'$\Large{R_2}$', lam1 +(lam2-lam1)/2, valueSin,"#d94343"],[r'$\Large{R_3}$', lam1 - (lam1)/6, ((valueD * (valueSin - lam1)) / val_lam1) + (2*valueSin - ((valueD * (valueSin - lam1)) / val_lam1))/2,"#4364d9"],[r'$\Large{R_4}$', lam1 - lam1/4, rangeEjex/8, "#7db38b"],[r'$\Large{R_5}$', 3*valueSin/4, rangeEjex/12,"#fcc603"]]
    if (lam2 < valueSin): anotaciones=[[r'$\Large{R_1}$', lam2 + (2*valueSin-lam2)/2, valueSin, "#48992c"],[r'$\Large{R_2}$', lam1 + 3*(lam2-lam1)/4, ((valueD * (valueSin - lam1)) / val_lam1) + (2*valueSin - ((valueD * (valueSin - lam1)) / val_lam1))/2,"#d94343"],[r'$\Large{R_3}$', lam1 -(lam1)/10, ((valueD * (valueSin - lam1)) / val_lam1) + (2*valueSin - ((valueD * (valueSin - lam1)) / val_lam1))/2,"#4364d9"],[r'$\Large{R_4}$', lam1 - lam1/4, rangeEjex/8, "#7db38b"],[r'$\Large{R_5}$', lam2 - (lam2-lam1)/4, rangeEjex/10,"#fcc603"],[r'$\Large{R_6}$', 9*valueSin/10, rangeEjex/12,"#8cd934"]]
    for elem in anotaciones:
        fig.add_annotation(
            text=elem[0],
            xanchor='right',
            x=elem[1],
            y=elem[2],
            font=dict(
                size=16,
                color="#000000"
            ),
            bordercolor=elem[3],
            borderwidth=5,
            borderpad=4,
            bgcolor="#ffffff",
            opacity=0.8,
            showarrow=False)
    #Configuración de los ejes y del fondo de la gráfica
    fig.update_layout(
        width=700,
        height=700,
        plot_bgcolor='white',
        showlegend=False
    )
    fig.update_xaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        zerolinecolor='lightgrey',
        gridcolor='lightgrey',
        range=[0, rangeEjex],
        title=r"$s(t)$"
    )
    fig.update_yaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        zerolinecolor='lightgrey',
        gridcolor='lightgrey',
        range=[0, 2 * valueSin],
        title=r"$x(t)$"
    )


    return fig

@callback(
    Output('planoFaseHaldane', 'figure'),
    Input('D-sliderH', 'value'),
    Input('Sin-sliderH', 'value'),
    Input('m-H', 'value'),
    Input('K-H', 'value'),
    Input('I-H', 'value'),
)
def planoFaseHaldane(valueD, valueSin, valuem, valueK, valueI):
    #Valores lambda1 y lambda2
    lam1 = ((-(valuem - valueD) * valueI) + np.sqrt(
        ((valuem - valueD) * valueI) ** 2 - 4 * (valueD ** 2) * valueK * valueI)) / (-2 * valueD)
    val_lam1 = f_haldane(lam1, valuem, valueK, valueI)
    lam2 = (-(valuem - valueD) * valueI - np.sqrt(
        ((valuem - valueD) * valueI) ** 2 - 4 * (valueD ** 2) * valueK * valueI)) / (-2 * valueD)
    val_lam2 = f_haldane(lam2, valuem, valueK, valueI)

    #Intervalo y valores de s
    s = np.linspace(0.01, 2*lam2 if (lam2 > valueSin) else 2*valueSin, 100)
    rangeEjex = lam2*2 if (lam2 > valueSin) else 2*valueSin+0.4
    #Isoclina de s
    isoclinaS = pd.DataFrame(dict(
        s=s,
        x=(valueD * (valueSin - s)) / f_haldane(s, valuem, valueK, valueI)
    ))
    #Isoclina de x asociada a lambda1
    x = np.linspace(0, 2*valueSin, 100)
    isoclinaX1 = pd.DataFrame(dict(
        x=x,
        s=lam1
    ))
    #Isoclina de x asociada a lambda2
    isoclinaX2 = pd.DataFrame(dict(
        x=x,
        s=lam2
    ))
    #Límite del gráfico
    limitX = pd.DataFrame(dict(
        x=x,
        s=rangeEjex
    ))

    #Malla para el plano de fases
    s1, x1 = np.meshgrid(np.linspace(0.01, rangeEjex , 30), np.linspace(0, 2 * valueSin, 20))
    #Campo de velocidades
    ds = valueD * (valueSin - s1) - f_haldane(s1, valuem, valueK, valueI) * x1
    dx = (f_haldane(s1, valuem, valueK, valueI) - valueD) * x1
    #Digrama de fases
    figFases = ff.create_quiver(s1, x1, ds, dx, scaleratio=0.4*valueSin,
                                scale=0.05*valueSin,
                                arrow_scale=.25,
                                name='quiver',
                                line_width=2,
                                line_color='rgba(0,0,102,1)')
    #Gráfica de las regiones delimitadas por las isoclinas
    fig= go.Figure()
    fig.add_trace(
        go.Scatter(x=isoclinaX1['s'], y=isoclinaX1['x'], mode='none', fill='tozerox'))
    fig.add_trace(go.Scatter(x=isoclinaX2['s'], y=isoclinaX2['x'], mode='none', fill='tonexty'))
    fig.add_trace(go.Scatter(x=limitX['s'], y=limitX['x'], mode='lines', fill='tonexty', fillcolor='rgba(50, 168, 82, 0.5)'))
    fig.add_trace(go.Scatter(x=isoclinaS['s'], y=isoclinaS['x'], mode='none', fill='tozeroy', fillcolor='rgba(255,255,102, 0.7)'))
    fig.add_trace(go.Scatter(x=isoclinaX1['s'], y=isoclinaX1['x'], mode='lines', line_color='rgba(0,0,102,0.8)'))
    fig.add_trace(go.Scatter(x=isoclinaX2['s'], y=isoclinaX2['x'], mode='lines', line_color='rgba(0,0,102,0.8)'))
    fig.add_trace(go.Scatter(x=isoclinaS['s'], y=isoclinaS['x'], mode='lines', line_color='rgba(0,0,102,0.8)'))
    #Añade plano de fases
    fig.add_traces(data=figFases.data)
    #Puntos de equilibrio
    equilibrios = [[valueSin, 0, r'$\large{E_0=(S_{in},0)}$']] if (lam1 > valueSin) else [[valueSin, 0, r'$\large{E_0=(S_{in},0)}$'], [lam1, (valueD * (valueSin - lam1)) / val_lam1, r'$\large{E_1=(s^*,x^*)}$']]
    for elem in equilibrios:
        fig.add_trace(go.Scatter(x=[elem[0]],
                                 y=[elem[1]],
                                 mode='markers',
                                 marker_size=10,
                                 name='points',
                                 marker_color='rgba(255,51,51,1)'))
    #Configuración de los ejes y del fondo de la gráfica
    fig.update_layout(
        width=700,
        height=700,
        plot_bgcolor='white',
        showlegend=False
    )
    fig.update_xaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        zerolinecolor='lightgrey',
        gridcolor='lightgrey',
        range=[0, rangeEjex],
        title=r"$s(t)$"
    )
    fig.update_yaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        zerolinecolor='lightgrey',
        gridcolor='lightgrey',
        range=[0, 2 * valueSin],
        title=r"$x(t)$"
    )

    return fig





#---------------------------------------------------------MODELO VARIAS ESPECIES------------------------------------------------------------------------------

#---------------------------------------------------------TIPO MONOD------------------------------------------------------------------------------
# Función para añadir parámetros y visualizar la función de tipo Monod
@callback(
    Output('monod-graph1', 'figure'),
    Input('m-M1', 'value'),
    Input('K-M1', 'value'),
    Input('m-M2', 'value'),
    Input('K-M2', 'value'),
    Input('D-sliderM1', 'value'),
    Input('Sin-sliderM1', 'value'),
)
def monod_consume(valuem, valueK, valuem2, valueK2, valueD, valueSin):
    #Parámetros primera especie
    m1 = valuem
    K1 = valueK
    lam1 = (K1 * valueD) / (m1 - valueD)
    #Parámetros segunda especie
    m2 = valuem2
    K2 = valueK2
    lam2 = (K2 * valueD) / (m2 - valueD)
    #Intervalo de valores s
    s = np.linspace(0, 2*valueSin if(lam2<valueSin) else 2*lam2, 100)
    #Evaluación de los valores de s para las funciones de tipo Monod
    monod_values1 = pd.DataFrame(dict(
        x=s,
        y=f_monod(s, m1, K1)
    ))
    monod_values2 = pd.DataFrame(dict(
        x=s,
        y=f_monod(s, m2, K2)
    ))
    #Discontinuas asociadas a los valores de lambda1 y lambda2
    x = np.linspace(0, valueD, 100)
    s = np.linspace(0, max([lam1, lam2]), 100)
    l1X = pd.DataFrame(dict(
        s=lam1,
        x= x
    ))
    lY = pd.DataFrame(dict(
        s=s,
        x=valueD
    ))
    l2X = pd.DataFrame(dict(
        s=lam2,
        x=x
    ))
    #Discontinuas asociadas a los valores de Sin
    x = np.linspace(0, max([f_monod(valueSin, m1, K1),f_monod(valueSin, m2, K2)]), 100)
    s = np.linspace(0, valueSin, 100)
    sX = pd.DataFrame(dict(
        s=valueSin,
        x=x
    ))
    s1Y = pd.DataFrame(dict(
        s=s,
        x=f_monod(valueSin, m1, K1)
    ))
    x = np.linspace(0, f_monod(valueSin, m2, K2), 100)
    s2Y = pd.DataFrame(dict(
        s=s,
        x=f_monod(valueSin, m2, K2)
    ))
    #Gráficas de consumo utilizando Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=monod_values1['x'], y=monod_values1['y'], line=dict(color='#6B60F7')))
    fig.add_trace(go.Scatter(x=monod_values2['x'], y=monod_values2['y'], line=dict(color='#EA6345')))
    #Puntos importantes señalados en la gráfica
    puntos = [[lam1, valueD, '#FF2D2D'], [valueSin, f_monod(valueSin, valuem, valueK), '#167B02'], [lam2, valueD, '#FF2D2D'], [valueSin, f_monod(valueSin, valuem2, valueK2), '#167B02']]
    for elem in puntos:
        fig.add_trace(go.Scatter(x=[elem[0]],
                                y=[elem[1]],
                                mode='markers',
                                marker_size=8,
                                name='points',
                                marker_color=elem[2]))
    #Líneas discontinuas asociadas a los valores de lambda y Sin
    discontinuas = [[l1X['s'], l1X['x'], '#FF2D2D'], [l2X['s'], l2X['x'], '#FF2D2D'], [lY['s'], lY['x'], '#FF2D2D'], [sX['s'], sX['x'], '#167B02'], [s1Y['s'], s1Y['x'], '#167B02'], [s2Y['s'], s2Y['x'], '#167B02']]
    for elem in discontinuas:
        fig.add_trace(go.Scatter(x=elem[0], y=elem[1], mode='lines + text', line=dict(color=elem[2], width=1,
                              dash='dash')))
    #Configuración de ejes y fondo de gráfica
    fig.update_layout(xaxis_title=r'$s$', yaxis_title=r'$\mu(s)$', )
    fig.update_layout(
        plot_bgcolor='white',
        showlegend=False
    )
    fig.update_xaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        zerolinecolor='lightgrey',
        gridcolor='lightgrey'
    )
    fig.update_yaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        zerolinecolor='lightgrey',
        gridcolor='lightgrey'
    )
    #Anotaciones
    anotaciones = [[r'$D$', s[0], valueD], [r'$\mu_1(S_{in})$', s[0], f_monod(valueSin, m1, K1)], [r'$\mu_2(S_{in})$', s[0], f_monod(valueSin, m2, K2)], 
                    [r'$\lambda_1(D)$', lam1, 0], [r'$\lambda_2(D)$', lam2, 0], [r'$S_{in}$', valueSin,0]]
    for elem in anotaciones:
        fig.add_annotation(
            text=elem[0],
            xanchor='left',
            yanchor='bottom',
            x=elem[1],
            y=elem[2],
            font=dict(
                size=9,
                color="#000000"
            ),
            showarrow=False)
    
    return fig


# Función evolución de alimento y la población en función del tiempo, introduciendo parámetros D y Sin, para el tipo Monod
@callback(
    Output('monod-time-evolution-graph1', 'figure'),
    Input('D-sliderM1', 'value'),
    Input('Sin-sliderM1', 'value'),
    Input('m-M1', 'value'),
    Input('K-M1', 'value'),
    Input('m-M2', 'value'),
    Input('K-M2', 'value'),
)
def monod_time_evolution(valueD, valueSin, valuem, valueK, valuem2, valueK2):
    #Condiciones iniciales
    c0 = 0, 0.2, 0.2
    #Mallado de tiempo
    t = np.linspace(0, 50, 100)
    #Parámetros para las funciones
    Sin = valueSin
    D = valueD
    m = valuem
    K = valueK
    m2 = valuem2
    K2 = valueK2
    #Modelo diferencial Monod con varias especies
    def model_monod(y, t):
        s, x1, x2 = y
        ds = D * (Sin - s) - (f_monod(s, m, K) * x1 + f_monod(s, m2, K2) * x2)
        dx1 = (f_monod(s, m, K) - D) * x1
        dx2 = (f_monod(s, m2, K2) - D) * x2
        dy = [ds, dx1, dx2]

        return dy
    #Solución modelo anterior
    sol = odeint(model_monod, c0, t)
    #Valores solución en DataFrame
    values = pd.DataFrame(dict(
        t=t,
        s=sol[:, 0],
        x1=sol[:, 1],
        x2 = sol[:, 2]
    ))
    #Gráfica evolución de la especie x1, x2 y del sustrato
    fig = px.line(values, x='t', y=[ 'x1', 'x2', 's'])
    #Configuración de ejes y del fondo de la gráfica
    fig.update_layout(
        plot_bgcolor='white'
    )
    fig.update_xaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        zerolinecolor='lightgrey',
        gridcolor='lightgrey'
    )
    fig.update_yaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        zerolinecolor='lightgrey',
        gridcolor='lightgrey'
    )

    return fig


#---------------------------------------------------------HALDANE------------------------------------------------------------------------------
# Función para añadir parámetros y visualizar la función de tipo Haldane
@callback(
    Output('haldane-graph1', 'figure'),
    Input('m-H1', 'value'),
    Input('K-H1', 'value'),
    Input('I-H1', 'value'),
    Input('m-H2', 'value'),
    Input('K-H2', 'value'),
    Input('I-H2', 'value'),
    Input('D-sliderH1', 'value'),
    Input('Sin-sliderH1', 'value'),
)
def haldane_consume(valuem, valueK, valueI, valuem2, valueK2, valueI2, valueD, valueSin):
    #Parámetros primera especie
    m = valuem
    I = valueI
    K = valueK
    #Parámetros segunda especie
    m2 = valuem2
    I2 = valueI2
    K2 = valueK2
    #Valores máximos y lambda1
    sm1 = np.sqrt(valueK*valueI)
    val_sm1 = f_haldane(sm1, valuem, valueK, valueI)
    lam11=((-(valuem-valueD)*valueI) + np.sqrt(((valuem-valueD)*valueI)**2 - 4*(valueD**2)*valueK*valueI))/(-2*valueD)
    lam12 = (-(valuem-valueD)*valueI-np.sqrt(((valuem-valueD)*valueI)**2 - 4*(valueD**2)*valueK*valueI))/(-2*valueD)
    #Valores máximos y lambda2
    sm2 = np.sqrt(K2 * I2)
    val_sm2 = f_haldane(sm2, valuem2, valueK2, valueI2)
    lam21 = ((-(valuem2 - valueD) * valueI2) + np.sqrt(
        ((valuem2 - valueD) * valueI2) ** 2 - 4 * (valueD ** 2) * valueK2 * valueI2)) / (-2 * valueD)
    lam22 = (-(valuem2 - valueD) * valueI2 - np.sqrt(
        ((valuem2 - valueD) * valueI2) ** 2 - 4 * (valueD ** 2) * valueK2 * valueI2)) / (-2 * valueD)
    #Intervalo para los valores de s
    s = np.linspace(0, 2*valueSin if (max([lam22,lam21]) < valueSin) else 2 * max([lam22,lam21]), 100)
    #Evaluación de los valores de s para la funciones de consumo de tipo Haldane
    haldane_values1 = pd.DataFrame(dict(
        x=s,
        y=f_haldane(s, m, K, I)
    ))
    haldane_values2 = pd.DataFrame(dict(
        x=s,
        y=f_haldane(s, m2, K2, I2)
    ))
    #Discontinuas asociadas a los valores de lambda
    x = np.linspace(0, f_haldane(lam11, m, K, I), 100)
    l11X = pd.DataFrame(dict(
        s=lam11,
        x=x
    ))
    x = np.linspace(0, f_haldane(lam12, m, K, I), 100)
    l12X = pd.DataFrame(dict(
        s=lam12,
        x=x
    ))
    x = np.linspace(0, f_haldane(lam21, m2, K2, I2), 100)
    l21X = pd.DataFrame(dict(
        s=lam21,
        x=x
    ))
    x = np.linspace(0, f_haldane(lam22, m2, K2, I2), 100)
    l22X = pd.DataFrame(dict(
        s=lam22,
        x=x
    ))
    s = np.linspace(0, max([lam12, lam22]), 100)
    lY = pd.DataFrame(dict(
        s=s,
        x=valueD
    ))
    #Discontinuas para valores de Sin
    x = np.linspace(0, f_haldane(valueSin, m, K, I), 100)
    s = np.linspace(0, valueSin, 100)
    sX = pd.DataFrame(dict(
        s=valueSin,
        x=x
    ))
    s1Y = pd.DataFrame(dict(
        s=s,
        x=f_haldane(valueSin, m, K, I)
    ))
    x = np.linspace(0, f_haldane(valueSin, m2, K2, I2), 100)
    s2Y = pd.DataFrame(dict(
        s=s,
        x=f_haldane(valueSin, m2, K2, I2)
    ))

    #Gráfica funciones de consumo de tipo Haldane
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=haldane_values1['x'], y=haldane_values1['y'], line=dict(color='#6B60F7')))
    fig.add_trace(go.Scatter(x=haldane_values2['x'], y=haldane_values2['y'], line=dict(color='#EA6345')))
    #Puntos importantes sobre la gráfica
    puntos=[[lam11, valueD, '#f54242'], [lam12, valueD, '#f54242'], [lam21, valueD, '#f54242'], [lam21, valueD, '#f54242'], 
            [valueSin, f_haldane(valueSin, valuem, valueK, valueI), '#167B02'], [valueSin, f_haldane(valueSin, valuem2, valueK2, valueI2), '#167B02']]
    for elem in puntos:
        fig.add_trace(go.Scatter(x=[elem[0]],
                                y=[elem[1]],
                                mode='markers',
                                marker_size=8,
                                name='points',
                                marker_color=elem[2]))
    #Discontinuas asociadas a lambdas y Sin
    discontinuas = [[l11X['s'], l11X['x'], '#f54242'], [l12X['s'], l12X['x'], '#f54242'], [l22X['s'], l22X['x'], '#f54242'], [l21X['s'], l21X['x'], '#f54242'], 
                    [lY['s'], lY['x'], '#f54242'], [sX['s'], sX['x'], '#167B02'], [s1Y['s'], s1Y['x'], '#167B02'], [s2Y['s'], s2Y['x'], '#167B02']]
    for elem in discontinuas:
        fig.add_trace(go.Scatter(x=elem[0], y=elem[1], mode='lines + text', line=dict(color=elem[2], width=1,
                                                                                  dash='dash')))
    #Configuración ejes y fondo de gráfica
    fig.update_layout(xaxis_title=r'$s$', yaxis_title=r'$\mu(s)$',
        plot_bgcolor='white',
        showlegend=False
    )
    fig.update_xaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        zerolinecolor='lightgrey',
        gridcolor='lightgrey',
    )
    fig.update_yaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        zerolinecolor='lightgrey',
        gridcolor='lightgrey'
    )
    #Anotaciones
    anotaciones=[[r'$D$', s[0], valueD], [r'$\mu_1(S_{in})$', s[0], f_haldane(valueSin, m, K, I)], [r'$\mu_2(S_{in})$', s[0], f_haldane(valueSin, m2, K2, I2)], 
                [r'$\lambda_1(D)$', lam11, 0], [r'$\overline{\lambda_1(D)}$', lam12, 0], [r'$\lambda_2(D)$', lam21, 0], [r'$\overline{\lambda_2(D)}$', lam22, 0], [r'$S_{in}$', valueSin, 0]]
    for elem in anotaciones:
        fig.add_annotation(
            text=elem[0],
            xanchor='left',
            yanchor='bottom',
            x=elem[1],
            y=elem[2],
            font=dict(
                size=9,
                color="#000000"
            ),
            showarrow=False)
    
    return fig


# Función evolución de alimento y la población en función del tiempo, introduciendo parámetros D y Sin, para el tipo Haldane
@callback(
    Output('haldane-time-evolution-graph1', 'figure'),
    Input('D-sliderH1', 'value'),
    Input('Sin-sliderH1', 'value'),
    Input('m-H1', 'value'),
    Input('K-H1', 'value'),
    Input('I-H1', 'value'),
    Input('m-H2', 'value'),
    Input('K-H2', 'value'),
    Input('I-H2', 'value')
)
def haldane_time_evolution(valueD, valueSin, valuem, valueK, valueI, valuem2, valueK2, valueI2):
    #Condiciones iniciales
    c0 = 0, 0.2, 0.2
    #Malla de tiempo 
    t = np.linspace(0, 50, 100)
    #Parámetros
    Sin = valueSin
    D = valueD
    #Modelo varias especies con el tipo Haldane
    def model_haldane(y, t):
        s, x1, x2 = y
        ds = D * (Sin - s) - (f_haldane(s, valuem, valueK, valueI) * x1 + f_haldane(s, valuem2, valueK2, valueI2)*x2)
        dx1 = (f_haldane(s, valuem, valueK, valueI) - D) * x1
        dx2 = (f_haldane(s, valuem2, valueK2, valueI2) - D) * x2
        dy = [ds, dx1, dx2]

        return dy
    #Solución modelo anterior
    sol = odeint(model_haldane, c0, t)
    #Valores dispuestos en DataFrame
    values = pd.DataFrame(dict(
        t=t,
        s=sol[:, 0],
        x1=sol[:, 1],
        x2=sol[:, 2]
    ))
    #Configuración ejes y fondo de gráfica
    fig = px.line(values, x='t', y=['x1', 'x2', 's'])
    fig.update_layout(
        plot_bgcolor='white'
    )
    fig.update_xaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        zerolinecolor='lightgrey',
        gridcolor='lightgrey'
    )
    fig.update_yaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        zerolinecolor='lightgrey',
        gridcolor='lightgrey'
    )

    return fig





#----------EJECUCIÓN DEL SERVIDOR-----------
if __name__ == '__main__':
    app.run_server(debug=True)
