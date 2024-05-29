from dash import Dash, dcc, html, Input, Output, State, ctx
import plotly.graph_objects as go
import numpy as np
import plotly.express as px

from data_set_generation import exclusion_area, generate_data

def interactive_plot(x,y):
    app = Dash(__name__)

#------- UI elements -------#
    #---- Slider for controling the Circles radius ----#
    radius_slider = html.Div([
            #---- text label for slider----#
            html.Label("Radius:", htmlFor="radius"),
            #---- Actual slider ----#
            dcc.Slider(
                min=0,
                max=5,
                step=0.1,
                value = 1, # starting value
                id="radius",
                marks=None,
                tooltip={"placement": "bottom", "always_visible": True},
            ),
        ])
    #---- Input field for circles X coordinates ----#
    circle_x_coord = html.Div([
        html.Label("X Coordinates: ", htmlFor="x"),
        dcc.Input(
            id='x_coord',
            type='number',
            placeholder="circle: x",  # A hint to the user of what can be entered in the control
            debounce=False,                      # Changes to input are sent to Dash server only on enter or losing focus
            value =0,
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
        )
    ])

    #---- Input field for circles Y coordinates ----#
    circle_y_coord = html.Div([
        html.Label("Y Coordinates: ", htmlFor="y"),
        dcc.Input(
            id='y_coord',
            type='number',
            placeholder="circle: y",  # A hint to the user of what can be entered in the control
            debounce=False,                      # Changes to input are sent to Dash server only on enter or losing focus
            value =0,
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
    generate_button = html.Div([
        html.Button('Generate New', id='generate', n_clicks=0)
    ])

    # Throw HTML elements into app
    app.layout = html.Div([dcc.Graph(id="graph"),generate_button, circle_x_coord, circle_y_coord, radius_slider])


    @app.callback(Output("graph", "figure"), # output graph is what us updated
                  Input("radius", "value"),  # radius sliders ID
                  Input("x_coord", "value"), # x coordinate field ID
                  Input("y_coord", "value"), # y coordinate field ID
                  [Input('generate', 'n_clicks')],) # generate button ID
    #----- Interactive portion of Figure -----#
    # Called whenever an UI element is interated with
    def update_figure(radius, x_coord, y_coord, n_clicks, x=x, y=y): 
        # Upon button press generate new data
        if "generate" == ctx.triggered_id:
            x,y = generate_data()
        
        #-- Check if feilds are filled --#
        if x_coord == None:
            x_coord = 0
        if y_coord == None:
            y_coord = 0
        #calculate points to exclude
        x, y = exclusion_area(x,y,radius,np.array([x_coord, y_coord]))
        #plot points
        px_fig = px.scatter(x=x[:,0], y=x[:,1], color=y, color_continuous_scale=["orange","purple"])
        # add circle with radius and coords specified by UI elements
        px_fig.add_shape(type="circle",
                xref="x", yref="y",
                x0= x_coord - radius, y0= y_coord - radius,
                x1=x_coord + radius, y1=y_coord +radius,
                opacity=0.2,
                fillcolor="red",
                line_color="black",)
        # Set axes
        px_fig.update_xaxes(range=[-4, 4])
        px_fig.update_yaxes(range=[-4, 4])

        # Ensure square aspect ratio
        px_fig.update_layout(
        showlegend=False,
        autosize=False,
        width=1000,
        height=1000
        )
        return px_fig


    app.run(debug=True)

x, y = generate_data()

interactive_plot(x,y)