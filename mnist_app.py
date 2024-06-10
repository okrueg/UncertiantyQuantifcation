from dash import Dash, dcc, html, Input, Output, State, ctx
from dash_daq import BooleanSwitch

import plotly.graph_objects as go
import plotly.express as px

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from openTSNE import TSNE

from data_set_generation import exclusion_area, generate_data, generate_shifted
from model_architectures import basic_nn, dimension_increaser, shifted_dist_dropout
import model_train

class mnist_app():
    def __init__(self):
        
        self.model = basic_nn(input_dim=2,
                            hidden_dim=10,
                            output_dim=1)
        self.num_epochs = 150
        

        self.x = np.array([0.875, 0.98, 0.831, 0.852, 0.81, 0.978, 0.671, 0.931, 0.97, 0.955])
        self.y = np.arange(0,10,1)

        self.app = Dash(__name__)

        self.buttons = html.Div([
            html.Button('Confussion Matrix', id='corr', n_clicks=0, style={'margin-right': '10px'})
        ], style={
            'margin-top': '20px',
            'margin-bottom': '20px',
            'backgroundColor': '#b2b1b5',
            'padding': '10px',
            'borderRadius': '5px',
            'width': '30%'
        })
        #----Graph and hover detector -----
        self.graph_container = html.Div([
            dcc.Graph(id="graph"),
        ])

        # Throw HTML elements into app
        self.app.layout = html.Div([self.graph_container, self.buttons])
           
        @self.app.callback(Output("graph", "figure"),
                           Input('corr', 'n_clicks')) # output graph is what us updated
        def update_main_figure(corr): 
            fig  = px.bar(x=self.x, y=self.y,color=self.x, orientation='h', color_continuous_scale = 'sunsetdark')

            fig.update_layout(
                        showlegend=True,
                        autosize=False,
                        width=1200,
                        height=800,
                        )
            fig.update_layout(
                xaxis_title="Accuracy",
                yaxis_title="Label"
                )
            fig.update_yaxes(
                tickvals=self.y,
                #ticktext=['0', 'Five', 'Ten', 'Fifteen', 'Twenty']
            )
            return fig
        

    def run(self):
        self.app.run(debug=True)

vis = mnist_app()
vis.run()