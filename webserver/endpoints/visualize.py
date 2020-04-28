# Adapted from https://github.com/Libardo1/dash-tsne

import h5py
import io
import dash
import base64
import pandas as pd
import dash_html_components as html
import dash_core_components as dcc
import plotly.graph_objs as go
from tempfile import NamedTemporaryFile
from dash.dependencies import Input, Output, State
from bio_embeddings.project import tsne_reduce

from webserver.database import get_file


def input_field(title, state_id, state_value, state_max, state_min):
    """Takes as parameter the title, state, default value and range of an input field, and output a Div object with
    the given specifications."""
    return html.Div([
        html.P(title,
               style={
                   'display': 'inline-block',
                   'verticalAlign': 'mid',
                   'marginRight': '5px',
                   'margin-bottom': '0px',
                   'margin-top': '0px'
               }),

        html.Div([
            dcc.Input(
                id=state_id,
                type='number',
                value=state_value,
                max=state_max,
                min=state_min,
                size=7
            )
        ],
            style={
                'display': 'inline-block',
                'margin-top': '0px',
                'margin-bottom': '0px'
            }
        )
    ]
    )


def _create_layout():
    return html.Div([
        # In-browser storage of global variables
        html.Div(
            id="data-df-and-message",
            style={'display': 'none'}
        ),

        html.Div(
            id="label-df-and-message",
            style={'display': 'none'}
        ),

        # represents the URL bar, doesn't render anything
        dcc.Location(id='url', refresh=False),

        # Main app
        html.Div([
            html.H2(
                't-SNE Explorer',
                id='title',
                style={
                    'float': 'left',
                    'margin-top': '20px',
                    'margin-bottom': '0',
                    'margin-left': '7px'
                }
            ),
            html.Img(
                src="https://s3-us-west-1.amazonaws.com/plotly-tutorials/logo/new-branding/dash-logo-by-plotly-stripe.png",
                style={
                    'height': '100px',
                    'float': 'right'
                }
            )
        ],
            className="row"
        ),

        html.Div([
            html.Div(
                id="plot-div",
                className="eight columns"
            ),

            html.Div([

                html.H4(
                    't-SNE Parameters',
                    id='tsne_h4'
                ),

                input_field("Perplexity:", "perplexity-state", 20, 50, 5),

                input_field("Number of Iterations:", "n-iter-state", 400, 1000, 250),

                input_field("Learning Rate:", "lr-state", 200, 1000, 10),

                html.Button(
                    id='tsne-train-button',
                    n_clicks=0,
                    children='Start Training t-SNE'
                ),

                dcc.Upload(
                    id='upload-label',
                    children=html.A('Upload your labels here.'),
                    style={
                        'height': '45px',
                        'line-height': '45px',
                        'border-width': '1px',
                        'border-style': 'dashed',
                        'border-radius': '5px',
                        'text-align': 'center',
                        'margin-top': '5px',
                        'margin-bottom': '5px'
                    },
                    multiple=False,
                    max_size=-1
                ),

                html.Div([
                    html.P(id='upload-data-message',
                           style={
                               'margin-bottom': '0px'
                           }),

                    html.P(id='upload-label-message',
                           style={
                               'margin-bottom': '0px'
                           }),

                    html.Div(id='training-status-message',
                             style={
                                 'margin-bottom': '0px',
                                 'margin-top': '0px'
                             }),

                    html.P(id='error-status-message')
                ],
                    id='output-messages',
                    style={
                        'margin-bottom': '2px',
                        'margin-top': '2px'
                    }
                )
            ],
                className="four columns"
            )
        ],
            className="row"
        ),

        html.Div([
            dcc.Markdown('''
**What is t-SNE?**
t-distributed stochastic neighbor embedding, created by van der Maaten and Hinton in 2008, is a visualization algorithm that reduce a high-dimensional space (e.g. an image or a word embedding) into two or three dimensions, so we can visualize how the data is distributed. A classical example is MNIST, a dataset of 60,000 handwritten digits of size 28x28 in black and white. When you reduce the MNIST dataset using t-SNE, you can clearly see all the digit clustered together, with the exception of a few that might have been poorly written. [You can read a detailed explanation of the algorithm on van der Maaten's personal blog.](https://lvdmaaten.github.io/tsne/)
**How to use the app**
To train your own t-SNE, you can input your own high-dimensional dataset and the corresponding labels inside the upload fields. For convenience, small sample datasets are included inside the data folder. The training can take a lot of time depending on the size of the dataset (the complete MNIST dataset could take 15-30 min), so it is **recommended to clone the repo and run the app locally if you want to use bigger datasets**. [You can find the repository containing this model here.](https://github.com/plotly/dash-tsne)''')
        ],
            style={
                'margin-top': '15px'
            },
            className="row"
        )
    ],
        className="container",
        style={
            'width': '90%',
            'max-width': 'none',
            'font-size': '1.5rem'
        }
    )


def parse_content(contents, filename):
    """This function parses the raw content and the file names, and returns the dataframe containing the data, as well
    as the message displaying whether it was successfully parsed or not."""

    if contents is None:
        return None, ""

    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)

    try:
        # Assume that the user uploaded a CSV file
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), index_col='identifier')
    except Exception as e:
        print(e)
        return None, 'There was an error processing this file.'

    return df, f'{filename} successfully processed.'


def create_dash_app(app):
    dash_app = dash.Dash(__name__, server=app, routes_pathname_prefix='/visualize/')

    dash_app.layout = _create_layout()

    # Load external CSS
    external_css = [
        "https://cdnjs.cloudflare.com/ajax/libs/normalize/7.0.0/normalize.min.css",
        "https://cdnjs.cloudflare.com/ajax/libs/skeleton/2.0.4/skeleton.min.css",
        "//fonts.googleapis.com/css?family=Raleway:400,300,600",
        "https://maxcdn.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css",
        "https://codepen.io/chriddyp/pen/brPBPO.css",
        "https://cdn.rawgit.com/plotly/dash-app-stylesheets/2cc54b8c03f4126569a3440aae611bbef1d7a5dd/stylesheet.css"
    ]

    # Uploaded data --> Hidden Data Div
    @dash_app.callback(Output('data-df-and-message', 'children'),
                       [Input('upload-data', 'contents'),
                        Input('upload-data', 'filename')])
    def parse_data(contents, filename):
        data_df, message = parse_content(contents, filename)

        if data_df is None:
            return [None, message]

        return [data_df.to_json(orient="split"), message]

    for css in external_css:
        dash_app.css.append_css({"external_url": css})

    # Uploaded labels --> Hidden Label div
    @dash_app.callback(Output('label-df-and-message', 'children'),
                       [Input('upload-label', 'contents'),
                        Input('upload-label', 'filename')])
    def parse_label(contents, filename):
        label_df, message = parse_content(contents, filename)

        if label_df is None:
            return [None, message]

        elif label_df.shape[1] != 1:
            return [None, f'The dimensions of {filename} are invalid.']

        return [label_df.to_json(orient="split"), message]

    # Hidden Data Div --> Display upload status message (Data)
    @dash_app.callback(Output('upload-data-message', 'children'),
                       [Input('data-df-and-message', 'children')])
    def output_upload_status_data(data):
        return data[1]

    # Hidden Label Div --> Display upload status message (Labels)
    @dash_app.callback(Output('upload-label-message', 'children'),
                       [Input('label-df-and-message', 'children')])
    def output_upload_status_label(data):
        return data[1]

    @dash_app.callback(dash.dependencies.Output('plot-div', 'children'),
                       [dash.dependencies.Input('url', 'pathname')])
    def display_page(pathname):
        if pathname:
            job_id = pathname.split('/')[-1]
            file = get_file(job_id, 'reduced_embeddings_file')

            if not file:
                return

            proteins = []
            temp_embeddings_file = NamedTemporaryFile()
            with file as grid_file:
                temp_embeddings_file.write(grid_file.read())
            with h5py.File(temp_embeddings_file.name, "r") as zipped_file:
                for protein_id in zipped_file.keys():
                    proteins.append((protein_id, list(zipped_file[protein_id])))

        projection_data = tsne_reduce([e[1] for e in proteins])

        ids = [e[0] for e in proteins]

        scatter = go.Scatter3d(
            name=str(ids),
            x=projection_data[:, 0],
            y=projection_data[:, 1],
            z=projection_data[:, 2],
            text=ids,
            textposition="middle center",
            showlegend=False,
            mode="markers",
            marker=dict(size=3, color="#3266c1", symbol="circle"),
        )

        # Layout for the t-SNE graph
        tsne_layout = go.Layout(
            margin=dict(
                l=0,
                r=0,
                b=0,
                t=0
            )
        )

        figure = go.Figure(data=[scatter], layout=tsne_layout)

        return html.Div([
            # Data about the graph
            html.Div(
                id="kl-divergence",
                style={'display': 'none'}
            ),

            html.Div(
                id="end-time",
                style={'display': 'none'}
            ),

            html.Div(
                id="error-message",
                style={'display': 'none'}
            ),

            # The graph
            dcc.Graph(
                id='tsne-3d-plot',
                figure=figure,
                style={
                    'height': '80vh',
                },
            )
        ])

    return dash_app


