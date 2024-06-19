from dash import Dash, dcc, html, Input, Output, State, ctx
from dash_daq import BooleanSwitch

import plotly.graph_objects as go
import plotly.express as px

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from data_set_generation import exclusion_area, generate_data, generate_shifted
from model_architectures import BasicNN, DimensionIncreaser, shiftedDistDropout
import model_train

class progress():
    def __init__(self, reduction, x, y, total_epochs):
        self.reduction = reduction
        self.x = x
        self.y = y
        self.total_epochs = total_epochs
        self.app = Dash(__name__)

    #------- UI elements -------#
        self.slider = html.Div([
                #---- text label for slider----#
                html.Label("Model Epoch:", htmlFor="epoch"),
                #---- Actual slider ----#
                dcc.Slider(
                    min=0,
                    max=self.total_epochs,
                    step=1,
                    value = 0, # starting value
                    id="epoch",
                    marks=None,
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
            ])
        self.graph_container = html.Div([
            dcc.Graph(id="graph"),
        ])
        # Throw HTML elements into app
        self.app.layout = html.Div([self.graph_container, self.slider])
           
        @self.app.callback(Output("graph", "figure"), # output graph is what us updated
                    Input("epoch", "value")) 
        #----- Interactive portion of Figure -----#
        # Called whenever an UI element is interated with
        def update_figure(epoch): 

            x = np.array([1, 2, 3, 4, 5])
            y = np.array([10, 11, 12, 13, 14])
            px_fig = px.scatter(x=x, y=y)
        

            # Ensure square aspect ratio
            px_fig.update_layout(
            showlegend=True,
            autosize=False,
            width=1000,
            height=1000
            )
            return px_fig
        
    def run(self):
        self.app.run(debug=True)


vis = progress()
vis.run()