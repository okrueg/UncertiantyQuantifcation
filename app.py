from dash import Dash, dcc, html, Input, Output, State, ctx
from dash_daq import BooleanSwitch
import plotly.graph_objects as go
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split

from data_set_generation import exclusion_area, generate_data
import model_train
def interactive_plot():
    global base_x
    global base_y 
    global selected_class 

    selected_class = False
    base_x, base_y = generate_data()
    app = Dash(__name__)

#------- UI elements -------#
    #---- Slider for controling the Circles radius ----#
    radius_slider = html.Div([
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
    circle_coords = html.Div([
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
    buttons = html.Div([
        html.Button('Generate New', id='generate', n_clicks=0) 
    ])
    #---- Toggle for selecting classes ----#
    class_switch = html.Div(
        [
        html.Label("Class 1 (Blue) "),
        BooleanSwitch(id="class_switch", on=False, color="red"),
        html.Label(" Class 0 (Orange)"),
        
        ],
    style={'display': 'flex', 'justify-content': 'flex-start'}
    )
    #----Graph and hover detector -----
    graph_container = html.Div([
        dcc.Graph(id="graph"),
        html.Div(id='click-data')
    ])

    # Throw HTML elements into app
    app.layout = html.Div([graph_container,class_switch, buttons, circle_coords, radius_slider])

    #--- Functionality for switching classes ---#
    @app.callback(Input("class_switch", "on"))
    def switchClass(switch_state):
        global selected_class 
        selected_class = switch_state
           
    @app.callback(Output("graph", "figure"), # output graph is what us updated
                  Input("radius", "value"),  # radius sliders ID
                  Input("x_coord", "value"), # x coordinate field ID
                  Input("y_coord", "value"), # y coordinate field ID
                  [Input('generate', 'n_clicks')], # generate button ID
                  [Input('graph', 'clickData')]) # click ID
    #----- Interactive portion of Figure -----#
    # Called whenever an UI element is interated with
    def update_figure(radius, x_coord, y_coord, n_clicks, clickData): 
        global base_x
        global base_y
        global selected_class

        # Upon button press generate new data
        if "generate" == ctx.triggered_id:
            base_x,base_y = generate_data()
        
        # if the user has clicked to add new point
        elif "graph" == ctx.triggered_id:
            if clickData is None: # make sure they actually clicked
                return
            else:
                new_x = clickData['points'][0]['x']
                new_y = clickData['points'][0]['y']

                base_x = np.vstack([base_x, np.array([new_x, new_y])])
                if selected_class:
                    base_y = np.append(base_y, 0)
                else:
                    base_y = np.append(base_y, 1)
        
        #-- Check if feilds are filled --#
        if x_coord == None:
            x_coord = 0
        if y_coord == None:
            y_coord = 0
        #calculate points to exclude
        curr_x, curr_y = exclusion_area(base_x,base_y,radius,np.array([x_coord, y_coord]))

        #calculate bounds of points (ensure square)
        lower_bounds = max(base_x[:, 0].min(), base_x[:, 1].min())
        upper_bounds = max(base_x[:, 0].max(), base_x[:, 1].max())

        # sets a grid of points to test for contour map
        xrange, yrange = build_range(curr_x, curr_y, lower_bounds, upper_bounds)
        xx, yy = np.meshgrid(xrange, yrange)
        test_input = np.c_[xx.ravel(), yy.ravel()]

        #apply NN to points
        nn = model_train.train(curr_x,curr_y)
        V = model_train.test(test_input,nn)
        Z = V.view(-1).numpy()

        Z = Z.reshape(xx.shape)

        #plot points
        px_fig = px.scatter(x=curr_x[:,0], y=curr_x[:,1], color=curr_y, color_continuous_scale=["orange","purple"])

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
        showlegend=False,
        autosize=False,
        width=1000,
        height=1000
        )
        return px_fig

    app.run(debug=True)

# ------ Countour HELPER FUNCTION ------
def build_range(X, y, lower_bounds, upper_bounds, mesh_size=.02, margin = 1):
    """
    Create an x range and a y range for building meshgrid
    """

    xrange = np.arange(lower_bounds - margin, upper_bounds + margin, mesh_size)
    yrange = np.arange(lower_bounds - margin, upper_bounds + margin, mesh_size)
    return xrange, yrange

interactive_plot()