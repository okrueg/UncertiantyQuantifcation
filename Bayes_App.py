'''
Anti-Linter Measures
'''
from base64 import b64encode
import io
from dash import Dash, dcc, html, Input, Output
#import dash_bootstrap_components as dbc , external_stylesheets=[dbc.themes.DARKLY]
#import plotly.graph_objects as go
import plotly.express as px

#import pandas as pd
import numpy as np
import torch

from bayesArchetectures import BNN
import bayesUtils
from datasets import loadData
import plotly.figure_factory as ff

class MnistApp():
    """
    if
        
    """
    def __init__(self, train_loss = None, val_loss= None):

        self.app = Dash(__name__)

        self.buffer = io.StringIO()

        self.encoded = ""

        self.num_classes = 10

        self.train_loader, self.val_loader, self.test_loader = loadData('CIFAR-10',batch_size= 200)

        self.model = BNN(in_channels=3, in_feat= 32*32*3, out_feat= self.num_classes)
        
        # self.model.init_dropout(use_reg_dropout= False, use_activations= False, original_method= False, continous_dropout= True,
        #                         dropout_prob= 0.5, num_drop_channels=3, drop_certainty=1)
        self.num_epochs = 2

        #self.model = torch.load("model_20_BNN.path")

        (self.train_loss, self.val_loss),(self.train_acc, self.test_acc), self.best_model_path = bayesUtils.train_Bayes(model=self.model,
                                                                                                                    train_loader=self.test_loader,
                                                                                                                    test_loader=self.test_loader,
                                                                                                                    num_epochs=self.num_epochs,
                                                                                                                    num_mc= 1,
                                                                                                                    temperature= 1.0,
                                                                                                                    lr= 0.001,
                                                                                                                    save=True,
                                                                                                                    save_mode='accuracy')

        #self.model = torch.load(self.best_model_path)

        self.final_loss, self.final_acc, self.label_acc = bayesUtils.test_Bayes(self.model, test_loader=self.test_loader, num_mc=10)
        self.label_acc = np.array(self.label_acc)



        self.buttons = html.Div([
            html.Button('Update Figures', id='corr', n_clicks=0, style={'margin-right': '10px'}),
            html.Label('Selected Dropout: Special', id='dropout_label'),
            dcc.Dropdown(['None', 'Regular', 'Special'], 'Special', id='dropout_dropdown'),

            html.Label("Selected_Label", id='selected_Label'),
            dcc.Dropdown([x for x in range(10)], 0, id='label_dropdown'),
                            html.A(
                    html.Button("Download as HTML"), 
                    id="download",
                    href="data:text/html;base64,",
                    download="plotly_graph.html"
                )
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
            dcc.Graph(id="training_losses")
        ], style={'display': 'flex'})

        self.acc_graphs = html.Div([
            dcc.Graph(id="accuracys"),dcc.Graph(id="weights")
            ], style={'display': 'flex'})

        self.dropout_graphs = html.Div([
            dcc.Graph(id="dropped_channels"),
            dcc.Graph(id="confused_labels")
        ], style={'display': 'flex'})

        # Throw HTML elements into app
        self.app.layout = html.Div([self.graph_container,
                                    self.acc_graphs,
                                    #self.dropout_graphs,
                                    self.epoch_slider,
                                    self.buttons])

        @self.app.callback(Output("dropout_label", "children"),
                    Input('dropout_dropdown', 'value'))
        def update_dropout(dropout_type):
            return dropout_type


        @self.app.callback(Output("label_acc", "figure"),
                           Input('corr', 'n_clicks')) # output graph is what us updated
        def update_bar(corr):
            y= np.arange(0,self.num_classes,1)
            fig  = px.bar(x=self.label_acc,
                           y=y,
                           color=self.label_acc,
                           range_color= [0.0, 1.0],
                           orientation='h',
                           color_continuous_scale = 'sunsetdark')

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
            )
            return fig


        @self.app.callback(Output("training_losses", "figure"),
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


        @self.app.callback(Output("accuracys", "figure"),
                        Output("download", "href"),
                    Input('corr', 'n_clicks')) # output graph is what us updated
        def update_accuracys(corr):
            x = np.arange(self.num_epochs)
            fig = px.line(x,y=self.test_acc, range_y=[0,1])
            fig.add_scatter(x=x, y= self.train_acc, mode='lines')

            fig.update_layout(
                        showlegend=False,
                        autosize=False,
                        width=900,
                        height=500,
                        xaxis_title="Epoch",
                        yaxis_title="Accuracy"
                        )
            # For saving the Figure
            fig.write_html(self.buffer)
            html_bytes = self.buffer.getvalue().encode()
            encoded = b64encode(html_bytes).decode()

            return fig, "data:text/html;base64," + encoded


        @self.app.callback(Output("weights", "figure"),
                        Input('label_dropdown', 'value'))
        def update_weight_dist(selected_label:str):
            selected_label = int(selected_label)

            #weight_data = self.model.fc3.weight[selected_label].to('cpu').detach().numpy()
            means = torch.flatten(self.model.fc2.mu_weight[selected_label].to('cpu').detach())
            rho = torch.flatten(self.model.fc2.rho_weight[selected_label].to('cpu').detach())

            # if you wanna see them all
            #print(self.model.output_layer.mu_weight.to('cpu').detach().shape)
            # means = torch.flatten(self.model.output_layer.mu_weight.to('cpu').detach())
            # rho = torch.flatten(self.model.output_layer.rho_weight.to('cpu').detach())

            stds = torch.log1p(torch.exp(rho))

            samples = []

            for mean, std in zip(means, stds):

                distribution = torch.distributions.Normal(mean, std)

                samples.append(distribution.sample((means.shape[0],)).numpy())

            group_labels = [f'Weight {i+1}' for i in range(len(means))] # have to have labels :/

            fig = ff.create_distplot(samples, group_labels, show_hist=False, show_rug=False)

            fig.update_layout(
                    showlegend=False,
                    xaxis_title="Weight Mean",
                    yaxis_title="Probability"
                    )

            return fig
            

        @self.app.callback(Output("dropped_channels", "figure"),
                           Output("confused_labels", "figure"),
                           Input('label_dropdown', 'value'),
                           Input('epoch', 'value')) # output graph is what us updated

        def update_drop_info(selected_label: str, epoch):
            selected_label = int(selected_label)

            dropped_y = self.model.dropout.drop_handeler.forward_info(epoch,
                                                                      selected_label,
                                                                      'dropped_channels')
            weight_y = self.model.dropout.drop_handeler.forward_info(epoch,
                                                                     selected_label,
                                                                     'weight_diffs')

            dropped_y = dropped_y.numpy()
            print(np.count_nonzero(dropped_y))
            weight_y = weight_y.numpy()


            channel_fig = px.imshow(img=np.transpose(dropped_y),
                                    title='Most Dropped Channels',
                                    color_continuous_scale='gray_r',
                                    aspect="auto")

            weight_diff_fig = px.imshow(img=np.transpose(weight_y),
                                        title='Channels Weight diffs',
                                        color_continuous_scale='RdBu',
                                        aspect="auto")

            channel_fig.update_layout(
                    showlegend=False,
                    #autosize=False,
                    width=900,
                    height=700,
                    xaxis_title="Batch idx",
                    yaxis_title="Channel"
                    )

            weight_diff_fig.update_layout(
                        showlegend=False,
                        #autosize=False,
                        width=900,
                        height=700,
                        xaxis_title="Batch idx",
                        yaxis_title="Channel"
                        )
        #print(selected_info["selected_weight_indexs"].shape, selected_info["weight_diffs"].shape)

            return (channel_fig, weight_diff_fig)

        # @self.app.callback(Input('dowload', 'n_clicks'))
        # def save_figures(save_clicks):


    def run(self):
        '''
        runs the app
        '''
        self.app.run(debug=False)

def model_grid_heatmap(accuracy_path, losses_path, drops_path, features_path ):
    accuracys = np.loadtxt(fname= accuracy_path, delimiter= ',')
    losses = np.loadtxt(fname= losses_path, delimiter= ',')

    drops = np.loadtxt(fname= drops_path, delimiter= ',')
    features = np.loadtxt(fname= features_path, delimiter= ',')

    accuracy_map = px.imshow(img=accuracys,
                             labels=dict(x="Drop %", y="Number of Labels used", color="Accuracy"),
                             x= drops[0].astype(str),
                             y= features[:, 0].astype(str),
                             text_auto=True,
                             color_continuous_scale= 'BuGn',
                             #color_continuous_midpoint= 0.9,
                             #range_color=[0.85,0.95],
                            title='Model Grid Combinations: Accuracy',
                            aspect="auto")

    accuracy_map.show()


vis = MnistApp()
vis.run()

# train_loader,val_loader,test_loader = loadData('CIFAR-10',batch_size= 200)
# regular_model = torch.load("model_31_isreg_True_useAct_True_original_method_True.path")
# our_model = torch.load("model_30_isreg_False_useAct_True_original_method_True.path")

# model_survival_fig(regular_model, our_model, test_loader)

# model_grid_heatmap('acc_result.csv',
#                    'val_result.csv',
#                    'x.csv',
#                    'y.csv')
