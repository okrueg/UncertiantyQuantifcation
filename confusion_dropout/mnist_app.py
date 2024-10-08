
import io
from base64 import b64encode
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output

import numpy as np
import torch

import model_utils
from datasets import loadData
from model_architectures import BasicCNN, FromBayesCNN, DropoutDataHandler


class MnistApp():
    """
    Dashboard For Various Model Stats
        
    """
    def __init__(self, train_loss = None, val_loss= None):

        self.app = Dash(__name__)

        # For Saving Figures
        self.buffer = io.StringIO()
        self.encoded = ""

        # Set up the data
        self.train_loader, self.val_loader, self.test_loader = loadData('CIFAR-10',batch_size= 200)

        # Init the Model
        self.num_classes = 10
        self.model = BasicCNN(num_classes=self.num_classes,
                        in_channels=3,
                        out_feature_size=2048)
        
        self.model.init_dropout(use_reg_dropout= False, use_activations= True, original_method= True, continous_dropout= False,
                                dropout_prob= 0.5, num_drop_channels=3, drop_certainty=1.0,drop_handler = DropoutDataHandler())
        
        # Train the Model
        self.num_epochs = 2
        (self.train_loss, self.val_loss),(self.train_acc, self.test_acc), self.best_model_path = model_utils.train_fas_mnist(model=self.model,
                                                                                                                    train_loader=self.test_loader,
                                                                                                                    val_loader=self.val_loader,
                                                                                                                    test_loader=self.test_loader,
                                                                                                                    num_epochs=self.num_epochs,
                                                                                                                    activation_gamma = 0.001,
                                                                                                                    lr = 0.001,
                                                                                                                    save=True,
                                                                                                                    save_mode='accuracy')

        self.model = torch.load(self.best_model_path)

        self.final_loss, self.final_calib, self.final_acc, self.label_acc = model_utils.test_fas_mnist(self.model, test_loader=self.test_loader)
        self.label_acc = np.array(self.label_acc)


        #UI Element setup
        self.buttons = html.Div([
            html.Label("Selected_Label", id='selected_Label'),
            dcc.Dropdown([x for x in range(self.num_classes)], 0, id='label_dropdown'),
            html.A(
                    html.Button("Download Weight Dist as HTML",id='download_button',n_clicks=0), 
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
            'width': '40%'
        })

        #-----Epoch Slider-------#
        self.epoch_slider = html.Div([
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
            dcc.Graph(figure= self.label_acc_bar()),
            dcc.Graph(figure=self.update_train())
        ], style={'display': 'flex'})

        self.acc_graphs = html.Div([
            dcc.Graph(figure=self.update_accuracys(), id="accuracys"),
            dcc.Graph(id="weights")
            ], style={'display': 'flex'})
        
        #----Dropout Visualization------
        self.dropout_graphs = html.Div([
            dcc.Graph(id="dropped_channels"),
            dcc.Graph(id="confused_labels")
        ], style={'display': 'flex'})

        # Throw HTML elements into app
        self.title = html.H1(
                        "Confusion Dropout Dashboard",
                        style={
                            'textAlign': 'center',
                            'marginTop': '20px', 
                            'fontFamily': 'Arial, sans-serif',
                            'fontSize': '32px'  
                        }
                    )
        self.app.layout = html.Div([self.title,
                                    self.graph_container,
                                    self.acc_graphs,
                                    self.dropout_graphs,
                                    self.epoch_slider,
                                    self.buttons])

        #Functionality to download
        @self.app.callback(Output("download", "href"),
                           Input("download_button", "n_clicks"),
                           Input("weights", 'figure'))
        def download(clicks, fig):
            fig = go.Figure(fig)
            fig.write_html(self.buffer)

            html_bytes = self.buffer.getvalue().encode()
            encoded = b64encode(html_bytes).decode()

            return "data:text/html;base64," + encoded


        @self.app.callback(Output("weights", "figure"),
                        Input('label_dropdown', 'value'))
        def update_weight_dist(selected_label:str, all_weights=True):
            if all_weights:
                weight_data = torch.flatten(self.model.fc3.weight.to('cpu').detach()).numpy()

            else:
                selected_label = int(selected_label)

                weight_data = self.model.fc3.weight[selected_label].to('cpu').detach().numpy()


            fig = px.histogram(weight_data, nbins=200, range_x= [-0.6, 0.6], 
                               color_discrete_sequence=['indianred'], width= 900, title= f"Weight Dist for Label {selected_label}")

            fig.update_layout(
                    showlegend=False,
                    xaxis_title="Weight Value",
                    yaxis_title="Num weights"
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

            return (channel_fig, weight_diff_fig)

    def update_accuracys(self):
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
        return fig

    def label_acc_bar(self):
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
    

    def update_train(self):
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
    

    def run(self):
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


def model_survival_fig(regular_Model, our_Model, loader):

    reg_accs = model_utils.test_survival(regular_Model, test_loader=loader)

    our_accs = model_utils.test_survival(our_Model, test_loader=loader)

    x = [0.0, 0.2, 0.4, 0.6, 0.8, 0.95]
    fig = px.line(x=x, y=reg_accs, range_y=[0,1])
    fig.add_scatter(x=x, y= our_accs, mode='lines')

    fig.data[0].name = 'Regular Dropout'
    fig.data[1].name = 'Confussion Dropout'

    fig.update_layout(
                showlegend=True,
                xaxis_title="Drop amount",
                yaxis_title="Accuracy"
                )
    fig.show()

vis = MnistApp()
vis.run()

# train_loader,val_loader,test_loader = loadData('CIFAR-10',batch_size= 2000)
# # regular_model = torch.load("model_31_isreg_True_useAct_True_original_method_True.path")
# our_model = torch.load("model_30_isreg_False_useAct_True_original_method_True.path")

# images, labels = next(iter(test_loader))

# x = model_utils.feature_correlation(images,labels, our_model)
# print(x.shape)
# model_survival_fig(regular_model, our_model, test_loader)

# model_grid_heatmap('acc_result.csv',
#                    'val_result.csv',
#                    'x.csv',
#                    'y.csv')
