import dash
from dash.dependencies import Output, Input
from dash import dcc
from dash import html
import plotly.graph_objs as go
from plotly.subplots import make_subplots


import zmq
from struct import *

import threading

import numpy as np

template = "plotly_dark"
x_pos, y_pos = [0], [0]

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.connect("tcp://localhost:5550")


X, Y = [], []

X.append(-1)
Y.append(0)

Y2, Y3, Y4, Y_neg = [], [], [], []

Y2.append(0)
Y3.append(0)
Y4.append(0)
Y_neg.append(0)

tcp_1, tcp_2, tcp_3, tcp_4 = [], [], [], []

r = [3]

r1 = [3]


theta = [0]
theta1 = [0]


def real_time():
    while True:
        message = socket.recv()
        out = unpack('ffff', message)
        socket.send(b"World")

        tcp_1.append(out[0])
        tcp_2.append(out[1])
        tcp_3.append(out[2])
        tcp_4.append(out[3])


def update():
    app.run_server(port=8050, host="0.0.0.0")


app = dash.Dash(__name__)

app.layout = html.Div(children=[

    html.Div([
        html.H1('Dynamics Bicycle Model'),
        html.Hr(style={'border': '2px dotted'}),
        html.H3('A simple simulation'),
        html.Div([
            html.P(
                'This simple simulation aims to demonstrate the movement using the Dynamics Bicycle Model.'),
            # html.P("This conversion happens behind the scenes by Dash's JavaScript front-end")
        ])
    ], style={'display': 'inline-block'}),

    html.Img(
        src='https://3.bp.blogspot.com/-Q_MjiYzACpE/Uvy54VyY1PI/AAAAAAAAdrc/I42guqm7G2w/s800/car_suv.png',
        height=517/3,
        width=800/3,
        style={'display': 'inline-block'}),

    html.Hr(style={'border-top': '4px dashed purple', 'width': '100%'}),

    html.Div([
        html.H2('Real time data:'),
        html.Div([
            html.P('Dash converts Python classes into HTML')
        ])
    ], style={
        'color': 'white',
    }),


    dcc.Graph(id='live-graph', animate=False,
              style={'display': 'inline-block'}),
    dcc.Interval(
        id='graph-update',
        interval=500,
        n_intervals=0
    ),

    html.Div(children=[
        dcc.Graph(
            id='example-graph'
        )
    ], style={
        'display': 'inline-block'
    }),

    html.Hr(style={'border-bottom': '4px dashed purple', 'width': '100%'}),

    html.Div(children=[
        dcc.Graph(
            id='polar-graph', style={
                'display': 'inline-block'
            }
        ),
        dcc.Graph(
            id='steering-graph', style={'display': 'inline-block'}),
    ], style={
        'display': 'inline-block'
    }),



    html.Hr(style={'border-bottom': '4px dashed purple', 'width': '100%'}),
],
    style={
    'width': '100%',
    'display': 'inline-block',
    'color': 'white',
    'background-color': 'rgb(17, 17, 17)',
})


if __name__ == '__main__':
    job_1 = threading.Thread(target=real_time)
    job_2 = threading.Thread(target=update)
    job_1.start()

    @app.callback(
        Output('steering-graph', 'figure'),
        [Input('graph-update', 'n_intervals')]
    )
    def update_steering(n):
        r1.append(3)
        theta1_deg = np.degrees(tcp_3[-1])
        if theta1_deg < 0:
            theta1_deg = 360 + theta1_deg
        theta1.append(theta1_deg+90)
        fig = go.Figure(data=go.Scatterpolar(
            r=r1,
            theta=theta1[-2:-1],
            mode='markers',
            # marker_symbol = 'arrow-right',
            marker=dict(
                color="rgb(199,36,177)",
                size=14,
                symbol='x'
            ),
        ))
        fig.update_layout(
            showlegend=False,
            template='plotly_dark',
            title_text='steering',
            title_x=0.5,
            polar=dict(
                radialaxis=dict(range=[0, 5], showticklabels=False, ticks=''),
                angularaxis=dict(
                    showticklabels=True,
                    thetaunit="degrees",
                    ticks=''),
                sector=[45, 135]
            )
        )
        return fig

    @app.callback(
        Output('polar-graph', 'figure'),
        [Input('graph-update', 'n_intervals')]
    )
    def update_polar(n):
        r.append(3)
        theta_deg = np.degrees(tcp_4[-1])
        if theta_deg < 0:
            theta_deg = 360 + theta_deg
        theta.append(theta_deg)
        fig = go.Figure(data=go.Scatterpolar(
            r=r,
            theta=theta[-2:-1],
            mode='markers',
            # marker_symbol = 'arrow-right',
            marker=dict(
                color="rgb(199,36,177)",
                size=14,
                symbol='x'
            ),
        ))
        fig.update_layout(
            showlegend=False,
            template='plotly_dark',
            title_text='tlte',
            title_x=0.5,
            polar=dict(
                radialaxis=dict(range=[0, 5], showticklabels=False, ticks=''),
                angularaxis=dict(
                    showticklabels=True,
                    thetaunit="degrees",
                    ticks='',
                    # tick0=0,
                    # dtick=np.pi/2),
                )
            ))
        return fig

    @app.callback(
        Output('live-graph', 'figure'),
        [Input('graph-update', 'n_intervals')]
    )
    def update_graph_scatter(n):
        subplot_range = [n-50, n+50]

        X.append(X[-1]+1)
        Y.append(Y[0] + tcp_1[-1])

        Y2.append(tcp_2[0]+tcp_2[-1])
        Y3.append(tcp_3[0]+tcp_3[-1])
        Y4.append(tcp_4[0]+tcp_4[-1])

        fig = make_subplots(
            rows=2,
            cols=2,
            vertical_spacing=0.1,
            subplot_titles=("yaw", "x_pos", "delta", "y_pos"),
            shared_xaxes=True,
        )
        fig['layout']['margin'] = {
            'l': 20, 'r': 20, 'b': 30, 't': 30
        }

        fig.update_layout(height=450, width=600,
                          showlegend=False, template=template)

        fig.append_trace({
            'x': X,
            'y': Y4,
            # 'name': 'delta',
            'mode': 'lines+markers',
            'type': 'scatter'
        }, row=1, col=1)

        fig.append_trace({
            'x': X,
            'y': Y3,
            # 'name': 'delta',
            'mode': 'lines+markers',
            'type': 'scatter'
        }, row=2, col=1)

        fig.append_trace({
            'x': X,
            'y': Y,
            # 'name': 'yaw',
            'mode': 'lines+markers',
            'type': 'scatter'
        }, row=1, col=2)

        fig.append_trace({
            'x': X,
            'y': Y2,
            # 'name': 'delta',
            'mode': 'lines+markers',
            'type': 'scatter'
        }, row=2, col=2)

        fig.update_xaxes(
            # title_text="xaxis 1 title",
            row=1, col=1,
            range=subplot_range
        )

        fig.update_xaxes(
            title_text="time (ms)",
            row=2, col=1,
            range=subplot_range
        )

        fig.update_xaxes(
            # title_text="xaxis 1 title",
            row=1, col=2,
            range=subplot_range
        )

        fig.update_xaxes(
            title_text="time (ms)",
            row=2, col=2,
            range=subplot_range
        )

        return fig

    @app.callback(
        Output('example-graph', 'figure'),
        [Input('graph-update', 'n_intervals')]
    )
    def update_example_graph(n):

        Y_neg.append(tcp_2[-1])
        figure = go.Figure()

        figure.add_trace({
            'x': Y,
            'y': Y_neg,
            # 'name': 'delta',
            'mode': 'lines',
            'type': 'scatter',
            'line_width': 1,
            'line_color': 'rgb(199,36,177)'
        })
        figure.update_layout(
            autosize=True,
            width=600,
            height=450,
            margin=dict(
                l=10,
                r=10,
                b=10,
                t=35,
                pad=4
            ),
            template=template,
            # paper_bgcolor="LightSteelBlue",
            title_text='Title',
            title_x=0.5
        )

        figure.update_xaxes(range=(Y[-1]-250, Y[-1]+250))
        figure.update_yaxes(range=(Y_neg[-1]-250, Y_neg[-1]+250))

        return figure

    job_2.start()
    job_1.join()
    job_2.join()
