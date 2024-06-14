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


class feature_reduction():
    def __init__(self, red_type):
        self.red_type = red_type
        self.fitted = False
        match red_type:
            case 'pca':
                self.reduction = PCA(n_components=2)
            case 'tsne':
                self.reduction = TSNE(n_components=2,
                                      perplexity=20)
            case _:
                raise Exception("Not a valid reduction type")
    
    def fit(self, x, model, dim_red):
        x = torch.from_numpy(x).float()
        #x = dim_inc(x)
        x = model.forwardEmbeddings(x).detach().numpy()
        self.reduction = self.reduction.fit(x)
        self.fitted = True

    def apply(self, x, model, dim_red):
        assert self.fitted == True

        x = torch.from_numpy(x).float()
        #x = dim_inc(x)
        x = model.forwardEmbeddings(x).detach().numpy()
        transformed = self.reduction.transform(x)
        return transformed


class model_visualizer():
    def __init__(self):
        self.center = [2,1.5]
        self.base_x, self.base_y, self.user_points = generate_data(200)

        self.shifted_x, self.shifted_y, self.shifted_user_points = generate_shifted(num_points=5, center=self.center)

        self.selected_class = False
        self.num_points = 200

        self.feature_reducer = feature_reduction(red_type='pca')

        self.dim_inc = dimension_increaser(input_dim=2,
                                            output_dim=2,
                                            use_activation=False)
        
        self.model = basic_nn(input_dim=2,
                            hidden_dim=10,
                            output_dim=1)
        self.num_epochs = 150
        
        self.reduction = None

        self.app = Dash(__name__)

    #------- UI elements -------#
        #---- Slider for controling the Circles radius ----#
        self.radius_slider = html.Div([
                #---- text label for slider----#
                html.Label("Radius:", htmlFor="radius"),
                #---- Actual slider ----#
                dcc.Slider(
                    min=0,
                    max=3,
                    step=0.05,
                    value = 0.5, # starting value
                    id="radius",
                    marks=None,
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
            ], style={'width': '50%'})
        #---- Input field for circles X coordinates ----#
        self.circle_coords = html.Div([
            html.Label("Circle X Coordinates: ", htmlFor="x"),
            dcc.Input(
                id='x_coord',
                type='number',
                placeholder="circle: x",  # A hint to the user of what can be entered in the control
                debounce=False,                      # Changes to input are sent to Dash server only on enter or losing focus
                value =-1,
                min=-5, max=5, step=0.1,         # Ranges of numeric value. Step refers to increments
                minLength=0, maxLength=5,          # Ranges for character length inside input box
                autoComplete='off',
                disabled=False,                     # Disable input box
                readOnly=False,                     # Make input box read only
                required=True,                     # Require user to insert something into input box
                size="20",                          # Number of characters that will be visible inside box
                style={'background-color':'#FFFFFF', 
                    'width':'10%', 
                    'height':'10%', 
                    'margin':'0 auto'},
            ),
            html.Label("Circle Y Coordinates: ", htmlFor="y"),
            dcc.Input(
                id='y_coord',
                type='number',
                placeholder="circle: y",  # A hint to the user of what can be entered in the control
                debounce=False,                      # Changes to input are sent to Dash server only on enter or losing focus
                value = -1,
                min=-5, max=5, step=0.1,         # Ranges of numeric value. Step refers to increments
                minLength=0, maxLength=5,          # Ranges for character length inside input box
                autoComplete='off',
                disabled=False,                     # Disable input box
                readOnly=False,                     # Make input box read only
                required=True,                     # Require user to insert something into input box
                size="100",                          # Number of characters that will be visible inside box
                style={'background-color':'#FFFFFF', 
                    'width':'10%',
                    'height':'10%', 
                    'margin':'0 auto'},
            )
        ])
        #---- Button for generating new Data ----#
        self.buttons = html.Div([
            html.Button('Generate New', id='generate', n_clicks=0, style={'margin-right': '10px'}),
            html.Label("Number of points", style={'margin-right': '10px'}),
            dcc.Input(
                id='num-points',
                type='number',
                placeholder="200",  # A hint to the user of what can be entered in the control
                debounce=True,                      # Changes to input are sent to Dash server only on enter or losing focus
                value =200,
                min=2, max=10000, step=200,         # Ranges of numeric value. Step refers to increments
                minLength=1, maxLength=5,          # Ranges for character length inside input box
                autoComplete='off',
                required=True,                     # Require user to insert something into input box
                size="20",                          # Number of characters that will be visible inside box
                style={'background-color':'#FFFFFF', 
                    'width':'20%', 
                    'height':'10%', 
                    'margin-right': '10px'},
            ),
            html.Button('Correlation Matrix', id='corr', n_clicks=0, style={'margin-right': '10px'})

        ], style={
            'margin-top': '20px',
            'margin-bottom': '20px',
            'backgroundColor': '#b2b1b5',
            'padding': '10px',
            'borderRadius': '5px',
            'width': '30%'
        })
        #---- Toggle for selecting classes ----#
        self.class_switch = html.Div(
            [
            html.Label("Class 1 (Blue) "),
            BooleanSwitch(id="class_switch", on=False, color="red"),
            html.Label(" Class 0 (Orange)"),
            ],
        style={'display': 'flex', 
                'justify-content': 'flex-start',
                'backgroundColor': '#b2b1b5',
                'padding': '10px',
                'borderRadius': '5px',
                'width' : '20%'
        })
        #----Graph and hover detector -----
        self.graph_container = html.Div([
            dcc.Graph(id="graph"),
            html.Div(id='click-data')
        ])
        self.red_graph_container = html.Div([
            dcc.Graph(id="other"),
        ])
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

        # Throw HTML elements into app
        self.app.layout = html.Div([self.graph_container,
                                    self.class_switch, 
                                    self.buttons, 
                                    self.circle_coords, 
                                    self.radius_slider, 
                                    self.red_graph_container, 
                                    self.epoch_slider])

    #--- Functionality for switching classes ---#
        @self.app.callback(Input("class_switch", "on"))
        def switchClass(switch_state):
            self.selected_class = switch_state
           
        @self.app.callback(Output("graph", "figure"), # output graph is what us updated
                    Input("radius", "value"),  # radius sliders ID
                    Input("x_coord", "value"), # x coordinate field ID
                    Input("y_coord", "value"), # y coordinate field ID
                    Input('generate', 'n_clicks'), # generate button ID
                    Input('graph', 'clickData'), # click ID
                    Input("num-points", "value"),
                    Input('corr', 'n_clicks')) 
        #----- Interactive portion of Figure -----#
        # Called whenever an UI element is interated with
        def update_main_figure(radius, x_coord, y_coord, n_clicks, clickData, new_num_points, gen_corr): 

            match ctx.triggered_id:
                case "generate": # Upon button press generate new data
                    self.base_x, self.base_y, self.user_points = generate_data(self.num_points)
                    self.shifted_x, self.shifted_y, self.shifted_user_points = generate_shifted(num_points=5, center=self.center)
                
                case "graph":
                    self.add_point(clickData)
                
                case "num-points":
                    self.num_points = new_num_points               
            
            #-- Check if feilds are filled --#
            if x_coord == None:
                x_coord = 0
            if y_coord == None:
                y_coord = 0
            #calculate points to exclude
            curr_x, curr_y = exclusion_area(self.base_x,self.base_y, self.user_points, radius,np.array([x_coord, y_coord]))

            #calculate bounds of points (ensure square)
            lower_bounds = max(self.base_x[:, 0].min(), self.base_x[:, 1].min())
            upper_bounds = max(self.base_x[:, 0].max(), self.base_x[:, 1].max())

            # sets a grid of points to test for contour map
            xrange, yrange = build_range(curr_x, curr_y, lower_bounds, upper_bounds)
            xx, yy = np.meshgrid(xrange, yrange)
            test_input = np.c_[xx.ravel(), yy.ravel()]

            self.model = basic_nn(input_dim=2,
                            hidden_dim=10,
                            output_dim=1)

            #apply NN to points
            drop_hist = model_train.train_2d(curr_x,
                            curr_y,
                            self.shifted_x,
                            model= self.model,
                            dim_inc= self.dim_inc,
                            num_epochs=self.num_epochs)
            self.feature_reducer.fit(self.base_x,self.model,None)
            
            V = model_train.test_2d(test_input, self.model, self.dim_inc)

            Z = V.view(-1).numpy()

            Z = Z.reshape(xx.shape)

            if "corr" == ctx.triggered_id:
                # corr = model_train.feature_correlation(curr_x,self.model,self.dim_inc)
                # heatmap = px.imshow(corr, text_auto=True, color_continuous_scale= 'RdBu')
                # heatmap.show()
                drop_x = np.arange(start=0,stop=drop_hist.shape[0], step=1)
                drop = px.line(x=drop_x, y=drop_hist[:,0])

                for ind in range(drop_hist.shape[1]):

                    drop.add_scatter(x=drop_x, y=drop_hist[:,ind], mode='lines')
                    
                drop.show()

            #plot points
            plot_x = np.append(curr_x,self.shifted_x, axis=0)
            plot_y = np.append(curr_y,np.add(self.shifted_y, 2))
            #print(plot_x.shape)

            px_fig = px.scatter(x=plot_x[:,0], y=plot_x[:,1], color=np.char.mod('%s', plot_y),
                                 color_discrete_map={
                                    "0": "red",
                                    "1": "blue",})
            
            # add countour map
            px_fig.add_trace(
            go.Contour(
                x=xrange, y=yrange, z=Z,
                showscale=False, colorscale='RdBu',
                opacity=0.4, name='Score'
                )
            )
            # add circle with radius and coords specified by UI elements
            px_fig.add_shape(type="circle",
                    xref="x", yref="y",
                    x0= x_coord - radius, y0= y_coord - radius,
                    x1=x_coord + radius, y1=y_coord +radius,
                    opacity=0.3,
                    fillcolor="red",
                    line_color="black",)
            # Set axes
            px_fig.update_xaxes(range=[lower_bounds-1, upper_bounds+1])
            px_fig.update_yaxes(range=[lower_bounds-1, upper_bounds+1])

            # Ensure square aspect ratio
            px_fig.update_layout(
            showlegend=True,
            autosize=False,
            width=1000,
            height=1000
            )
            return px_fig
        
        @self.app.callback(Output("other", "figure"), # output graph is what us updated
            Input("epoch", "value"), prevent_inital_call = True)  # epoch sliders ID
        def update_feature_fig(epoch_num):
                path = f"models/model_{epoch_num}.path"
                self.model.load_state_dict(torch.load(path))
                
                plot_x = self.feature_reducer.apply( self.base_x,model=self.model,dim_red= None )

                other_fig = px.scatter(x=plot_x[:,0], y=plot_x[:,1], color=np.char.mod('%s', self.base_y),
                                 color_discrete_map={
                                    "0": "red",
                                    "1": "blue",})
                other_fig.update_layout(
                        showlegend=True,
                        autosize=False,
                        width=1000,
                        height=1000
                        )
                
                return other_fig
    #---- plot points ----
    def add_point(self, clickData):
        # if the user has clicked to add new point
        if clickData is None: # make sure they actually clicked
            return
        else:
            new_x = clickData['points'][0]['x']
            new_y = clickData['points'][0]['y']

            self.base_x = np.vstack([self.base_x, np.array([new_x, new_y])])
            if self.selected_class:
                self.base_y = np.append(self.base_y, 0)
            else:
                self.base_y = np.append(self.base_y, 1)

            self.user_points = np.append(self.user_points, 1)
            
    def add_test_point(self, clickData):
        # if the user has clicked to add new point
        if clickData is None: # make sure they actually clicked
            return
        else:
            new_x = clickData['points'][0]['x']
            new_y = clickData['points'][0]['y']

            self.shifted_x = np.vstack([self.shifted_x, np.array([new_x, new_y])])
            if self.selected_class:
                base_y = np.append(self.shifted_y, 0)
            else:
                base_y = np.append(self.shifted_y, 1)

            self.user_points = np.append(self.user_points, 1)
    
    def run(self):
        self.app.run(debug=True)
# ------ Countour HELPER FUNCTION ------
def build_range(X, y, lower_bounds, upper_bounds, mesh_size=.02, margin = 1):
    """
    Create an x range and a y range for building meshgrid
    """

    xrange = np.arange(lower_bounds - margin, upper_bounds + margin, mesh_size)
    yrange = np.arange(lower_bounds - margin, upper_bounds + margin, mesh_size)
    return xrange, yrange

def feature_correlation(x: np.ndarray, model: basic_nn, dim_inc: dimension_increaser):
    x = torch.from_numpy(x).float()
    #x = dim_inc(x)
    x = model.forwardEmbeddings(x).detach().numpy()
    cor = np.corrcoef(x, rowvar=False)
    return cor


vis = model_visualizer()
vis.run()