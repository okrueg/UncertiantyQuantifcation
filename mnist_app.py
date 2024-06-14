from dash import Dash, dcc, html, Input, Output, State, ctx
from dash_daq import BooleanSwitch

import plotly.graph_objects as go
import plotly.express as px

import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from openTSNE import TSNE


from model_architectures import basic_cnn
import mnist_utils

class mnist_app():
    """
    if
        
    """
    def __init__(self, load_path: str | None, train_loss = None, val_loss= None):

        self.app = Dash(__name__)

        self.model = basic_cnn(num_classes=10)
        self.num_epochs = 2

        if load_path == None:
            self.train_loss, self.val_loss, self.drop_info = mnist_utils.train_fas_mnist(model=self.model,
                                                                        num_epochs=self.num_epochs)
        else:   
            self.model.load_state_dict(torch.load(load_path))
            self.train_loss = train_loss
            self.val_loss = val_loss

        
        self.label_acc, self.total = mnist_utils.test_fas_mnist(self.model)
        self.label_acc = np.array(self.label_acc)


        self.buttons = html.Div([
            html.Button('Update Figures', id='corr', n_clicks=0, style={'margin-right': '10px'}),
            html.Label('Selected Dropout: Special', id='dropout_label'),
            dcc.Dropdown(['None', 'Regular', 'Special'], 'Special', id='dropout_dropdown'),

            html.Label("Selected_Label", id='selected_Label'),
            dcc.Dropdown([x for x in range(10)], 0, id='label_dropdown')
        ], style={
            'margin-top': '20px',
            'margin-bottom': '20px',
            'backgroundColor': '#b2b1b5',
            'padding': '10px',
            'borderRadius': '5px',
            'width': '30%'
        })
        self.epoch_slider = html.Div([
            #---- text label for slider----#
            html.Label("Epoch:"),
            #---- Actual slider ----#
            dcc.Slider(
                min=0,
                max=self.num_epochs-1,
                step=1,
                value = 0, # starting value
                id="epoch",
                marks={i: str(i) for i in range(0,self.num_epochs-1,5)},
                tooltip={"placement": "bottom", "always_visible": True},
            ),
            ], style={'width': '50%'})
        #----Graphs about model training -----
        self.graph_container = html.Div([
            dcc.Graph(id="label_acc"),
            dcc.Graph(id="training_stats")
        ], style={'display': 'flex'})

        self.dropout_graphs = html.Div([
            dcc.Graph(id="dropped_channels"),
            dcc.Graph(id="confused_labels")
        ], style={'display': 'flex'})

        # Throw HTML elements into app
        self.app.layout = html.Div([self.graph_container, self.dropout_graphs, self.epoch_slider, self.buttons])

        @self.app.callback(Output("dropout_label", "children"),
                    Input('dropout_dropdown', 'value')) 
        def update_dropout(dropout_type):
            return dropout_type
           
        @self.app.callback(Output("label_acc", "figure"),
                           Input('corr', 'n_clicks')) # output graph is what us updated
        def update_bar(corr): 
            y= np.arange(0,10,1)
            fig  = px.bar(x=self.label_acc, y=y,color=self.label_acc, orientation='h', color_continuous_scale = 'sunsetdark')

            fig.update_layout(
                        showlegend=True,
                        autosize=False,
                        width=900,
                        height=500,
                        xaxis_title="Accuracy",
                        yaxis_title="Label"
                        )

            fig.update_yaxes(
                tickvals=y,
                #ticktext=['0', 'Five', 'Ten', 'Fifteen', 'Twenty']
            )
            return fig
        
        @self.app.callback(Output("training_stats", "figure"),
                           Input('corr', 'n_clicks')) # output graph is what us updated
        def update_train(corr):
            x = np.arange(self.num_epochs)
            fig = px.line(x,y=self.train_loss)
            fig.add_scatter(x=x, y= self.val_loss, mode='lines')

            fig.update_layout(
                        showlegend=False,
                        autosize=False,
                        width=900,
                        height=500,
                        xaxis_title="Epoch",
                        yaxis_title="Loss"
                        )
            
            return fig
        
        @self.app.callback(Output("dropped_channels", "figure"),
                           Output("confused_labels", "figure"),
                           Input('label_dropdown', 'value'),
                           Input('epoch', 'value')) # output graph is what us updated
        
        def update_drop_info(selected_label: str, epoch):
            selected_label = int(selected_label)

            selected_info = self.drop_info[selected_label]

            dropped_y = selected_info["dropped_channels"].numpy()
            weight_y = selected_info["weight_diffs"].numpy()

            dropped_y = dropped_y[epoch]
            weight_y = weight_y[epoch]

            channel_fig = px.imshow(img=np.transpose(dropped_y), title='Most Dropped Channels', color_continuous_scale='Sunsetdark', aspect="auto")
            
            weight_diff_fig = px.imshow(img=np.transpose(weight_y), title='Channels Weight diffs', color_continuous_scale='Sunsetdark', aspect="auto")

            channel_fig.update_layout(
                    showlegend=False,
                    #autosize=False,
                    width=900,
                    height=500,
                    xaxis_title="Batch idx",
                    yaxis_title="Channel"
                    )
            
            weight_diff_fig.update_layout(
                        showlegend=False,
                        #autosize=False,
                        width=900,
                        height=500,
                        xaxis_title="Batch idx",
                        yaxis_title="Channel"
                        )
            #print(selected_info["selected_weight_indexs"].shape, selected_info["weight_diffs"].shape)
            
            return (channel_fig, weight_diff_fig)

    def run(self):
        self.app.run(debug=False)

vis = mnist_app(load_path=None)
vis.run()