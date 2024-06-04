from dash import Dash, dcc, html, Input, Output, State, ctx
from dash_daq import BooleanSwitch

import plotly.graph_objects as go
import plotly.express as px

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from data_set_generation import exclusion_area, generate_data, generate_shifted
from model_architectures import basic_nn, dimension_increaser
import model_train

class model_visualizer():
    def __init__(self):
        self.base_x, self.base_y, self.user_points = generate_data(200)

        self.shifted_x, self.shifted_y, self.shifted_user_points = generate_shifted(num_points=5, shift=[0.5,0.5])

        self.selected_class = False
        self.num_points = 200

        self.dim_inc = dimension_increaser(input_dim=2,
                                            output_dim=2,
                                            use_activation=False)
        
        self.model = basic_nn(input_dim=2,
                            hidden_dim=20,
                            output_dim=1)

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
            ])
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

        # Throw HTML elements into app
        self.app.layout = html.Div([self.graph_container,self.class_switch, self.buttons, self.circle_coords, self.radius_slider])

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
        def update_figure(radius, x_coord, y_coord, n_clicks, clickData, new_num_points, gen_corr): 

            match ctx.triggered_id:
                case "generate": # Upon button press generate new data
                    self.base_x, self.base_y, self.user_points = generate_data(self.num_points)
                
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

            #apply NN to points
            model_train.train(curr_x,
                            curr_y,
                            model= self.model,
                            dim_inc= self.dim_inc)
            
            V = model_train.test(test_input, self.model, self.dim_inc)

            Z = V.view(-1).numpy()

            Z = Z.reshape(xx.shape)

            if "corr" == ctx.triggered_id:
                corr = model_train.feature_correlation(curr_x,self.model,self.dim_inc)
                heatmap = px.imshow(corr, text_auto=True, color_continuous_scale= 'RdBu')
                heatmap.show()

            #plot points
            plot_x = np.append(curr_x,self.shifted_x, axis=0)
            plot_y = np.append(curr_y,np.add(self.shifted_y, 2))
            #print(plot_x.shape)

            px_fig = px.scatter(x=plot_x[:,0], y=plot_x[:,1], color=np.char.mod('%s', plot_y))
            
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


vis = model_visualizer()
vis.run()