import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from dash_extensions import Keyboard
import os
import sys

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        base_path = sys._MEIPASS  # PyInstaller temp folder
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# Load datasets
df_JA = pd.read_csv(
    resource_path("filtered_RNA_seq_time_series_urejeno_JA.csv"),
    index_col=0, header=[0, 1, 2]
)
df_SA = pd.read_csv(
    resource_path("filtered_RNA_seq_time_series_urejeno_SA.csv"),
    index_col=0, header=[0, 1, 2]
)
df_SAJA = pd.read_csv(
    resource_path("filtered_RNA_seq_time_series_urejeno_SAJA.csv"),
    index_col=0, header=[0, 1, 2]
)

# Define the gene and datasets
gene = 'AT1G32640'
gene_label = 'MYC2'
datasets = [(df_JA, "JA"), (df_SA, "SA"), (df_SAJA, "SAJA")]

# Prepare data for animation
timepoints_all = set()
data_dict = {}

for df, treatment_label in datasets:
    try:
        # Check if required columns and gene exist
        if "Mock" not in df.columns or treatment_label not in df.columns:
            print(f"Missing column in DataFrame for treatment: {treatment_label}")
            continue
        if gene not in df.index:
            print(f"Gene {gene} not found in DataFrame for treatment: {treatment_label}")
            continue

        # Extract data for the gene
        mock_data_gene = df["Mock"].loc[gene]
        treatment_data_gene = df[treatment_label].loc[gene]

        # Aggregate replicates for Mock and treatment
        mock_mean_values = mock_data_gene.groupby(level="timepoint").mean()
        treatment_mean_values = treatment_data_gene.groupby(level="timepoint").mean()

        # Calculate standard deviation for Mock and treatment
        mock_std_values = mock_data_gene.groupby(level="timepoint").std()
        treatment_std_values = treatment_data_gene.groupby(level="timepoint").std()

        # Convert the timepoints to numeric values and sort them
        timepoints = mock_mean_values.index.astype(float)
        sorted_indices = timepoints.argsort()
        timepoints = timepoints[sorted_indices]
        mock_mean_values = mock_mean_values.iloc[sorted_indices]
        treatment_mean_values = treatment_mean_values.iloc[sorted_indices]
        mock_std_values = mock_std_values.iloc[sorted_indices]
        treatment_std_values = treatment_std_values.iloc[sorted_indices]

        # Store data for each treatment
        timepoints_all.update(timepoints)
        data_dict[treatment_label] = {
            "timepoints": timepoints,
            "mock_mean_values": mock_mean_values,
            "mock_std_values": mock_std_values,
            "treatment_mean_values": treatment_mean_values,
            "treatment_std_values": treatment_std_values,
        }
    except Exception as e:
        print(f"Error processing treatment {treatment_label}: {e}")
        continue

timepoints_all = sorted(timepoints_all)

# Define colors for treatments and other elements in RGB format
colors = {
    "Mock": "rgb(0, 0, 0)",  # Black
    "JA": "rgb(201,196,84)",  # Updated color for JA
    "SA": "rgb(145,97,48)",  # Updated color for SA
    "SAJA": "rgb(179,144,61)",  # Updated color for SAJA
    "grid": "rgb(146, 159, 118)",  # Grid lines
    "background": "rgb(252, 255, 245)",  # Background color
    "above_area": "rgba(156, 126, 254, 0.4)",  # Light purple for "above"
    "below_area": "rgba(156, 126, 254, 0.4)"  # Light purple for "below"
}

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[
    "https://fonts.googleapis.com/css2?family=Lato:wght@400;700&display=swap"
])

# Layout
app.layout = html.Div(
    style={'backgroundColor': '#ffffff', 'fontFamily': 'Lato'},  # Set background color and font globally
    children=[
        html.H1(
            [
                html.Span("Jasmonic", style={'color': colors["JA"]}),
                " and ",
                html.Span("Salicylic", style={'color': colors["SA"]}),
                " acid have opposing effect on MYC2 expression"
            ],
            style={
                'textAlign': 'left',  # Align text to the left
                'marginLeft': '95px',  # Add left margin to align with the graph
                'fontFamily': 'Lato'  # Set font to Lato
            }
        ),
        html.H2(
            [
                "Time-dependent expression of MYC2 (AT1G32640) of ",
                html.I("Arabidopsis thaliana"),  # Wrap the text in html.I for italics
                " upon treatment with different hormones ♣ Source data: RNAseq timeseries (prof. Van Wees group, Utrecht University)"
            ],
            style={
                'textAlign': 'left',  # Align text to the left
                'marginLeft': '95px',  # Match the alignment with the main title
                'fontSize': '16px',  # Set font size for the subtitle
                'color': '#555',  # Set a subtle color for the subtitle
                'fontFamily': 'Lato'  # Set font to Lato
            }
        ),
        html.Div(
            [
                # Select Treatment to Display
                html.Div(
                    [
                        html.Label(
                            "Select Treatment to Display:",
                            style={'fontFamily': 'Lato'}  # Set font to Lato
                        ),
                        dcc.Checklist(
                            id='data-selection-checklist',
                            options=[
                                {'label': 'Jasmonic acid ', 'value': 'JA'},
                                {'label': 'Salicylic acid ', 'value': 'SA'},
                                {'label': 'Salicylic and Jasmonic acid', 'value': 'SAJA'}
                            ],
                            value=['JA', 'SA', 'SAJA'],  # Default to show all except Mock
                            inline=True,
                            style={'fontFamily': 'Lato'}  # Set font to Lato
                        )
                    ],
                    style={
                        'width': '50%',
                        'marginLeft': '95px',  # Align with the subtitle
                        'marginTop': '20px',
                        'textAlign': 'left'  # Left-align the content
                    }
                ),
                # Select Treatment to Highlight
                html.Div(
                    [
                        html.Label(
                            "Select Treatment to Highlight:",
                            style={'fontFamily': 'Lato'}  # Set font to Lato
                        ),
                        dcc.Dropdown(
                            id='treatment-dropdown',
                            options=[
                                {'label': 'None', 'value': 'None'},
                                {'label': 'Jasmonic acid', 'value': 'JA'},
                                {'label': 'Salicylic acid', 'value': 'SA'},
                                {'label': 'Salicylic and Jasmonic acid', 'value': 'SAJA'}
                            ],
                            value='None',  # Default to "None"
                            placeholder="Select a treatment",
                            style={'fontFamily': 'Lato'}  # Set font to Lato
                        )
                    ],
                    style={
                        'width': '50%',
                        'marginLeft': '10px',  # Reduced spacing between the two sections
                        'marginTop': '20px',
                        'textAlign': 'left'  # Left-align the content
                    }
                )
            ],
            style={
                'display': 'flex',  # Align the two sections horizontally
                'alignItems': 'flex-start'  # Align items at the top
            }
        ),
        dcc.Graph(
            id='time-series-plot',
            style={'backgroundColor': '#fcfff5'}  # Set background color for the graph container
        ),
        html.Div(
            [
                html.Button(
                    "Play",
                    id="play-button",
                    n_clicks=0,
                    style={'marginRight': '10px', 'fontFamily': 'Lato'}  # Set font to Lato
                ),
                html.Button(
                    "Pause",
                    id="pause-button",
                    n_clicks=0,
                    style={'marginRight': '10px', 'fontFamily': 'Lato'}  # Set font to Lato
                ),
                html.Button(
                    "Timepoint 0",
                    id="timepoint-0-button",
                    n_clicks=0,
                    style={'marginRight': '10px', 'fontFamily': 'Lato'}  # Set font to Lato
                ),
                html.Button(
                    "Last Timepoint",
                    id="last-timepoint-button",
                    n_clicks=0,
                    style={'fontFamily': 'Lato'}  # Set font to Lato
                )
            ],
            style={'textAlign': 'center', 'marginTop': '20px'}
        ),
        dcc.Interval(
            id="interval",
            interval=1000,  # Interval in milliseconds (1 second)
            n_intervals=0,  # Number of intervals that have passed
            disabled=True  # Start with the interval disabled
        )
    ]
)

# Adjust label positions to avoid overlap and move them to the right
def adjust_label_positions(labels, y_buffer=3000, x_offset=0.08, timepoint=0):
    """
    Adjust the label positions to avoid overlap, keeping the correct order.
    Labels are sorted by their y-values (expression levels) and stacked dynamically.
    For timepoint 0, no x_offset is applied.
    """
    sorted_labels = sorted(labels, key=lambda lbl: lbl['y'], reverse=True)  # Sort by y (expression level)
    adjusted_positions = []
    for label in sorted_labels:
        x_pos, y_pos = label['x'], label['y']
        # Adjust y position to avoid overlap
        for _, existing_y in adjusted_positions:
            if abs(y_pos - existing_y) < y_buffer:
                y_pos = existing_y - y_buffer  # Stack below the previous label
        adjusted_positions.append((x_pos, y_pos))
        label['adjusted_y'] = y_pos  # Store the adjusted y position
        # Apply x_offset only if timepoint is not 0
        label['adjusted_x'] = x_pos if timepoint == 0 else x_pos + x_offset
    return sorted_labels

# Function to calculate crossing points and divide into segments
def calculate_segments_with_crossings(x_values, y_treatment, y_mock):
    segments = []
    current_status = None
    segment_start = 0

    for i in range(1, len(x_values)):
        # Determine the status at the current point
        if y_treatment[i] > y_mock[i]:
            status = "above"
        elif y_treatment[i] < y_mock[i]:
            status = "below"
        else:
            status = current_status  # Maintain the previous status if equal

        # Check if the status changes between points
        if status != current_status:
            # If there's a change, calculate the crossing point
            if current_status is not None:
                crossing_x = x_values[i - 1] + (x_values[i] - x_values[i - 1]) * (
                    y_mock[i - 1] - y_treatment[i - 1]
                ) / ((y_mock[i - 1] - y_treatment[i - 1]) - (y_mock[i] - y_treatment[i]))
                crossing_y = y_treatment[i - 1] + (y_treatment[i] - y_treatment[i - 1]) * (
                    crossing_x - x_values[i - 1]
                ) / (x_values[i] - x_values[i - 1])

                # Add the segment before the crossing point
                segments.append({
                    "start": segment_start,
                    "end": i,
                    "crossing_x": crossing_x,
                    "crossing_y": crossing_y,
                    "status": current_status,  # Use the status of the previous timepoint
                })

                # Start a new segment from the crossing point
                segment_start = i - 1

            current_status = status

    # Add the final segment
    segments.append({
        "start": segment_start,
        "end": len(x_values),
        "crossing_x": None,
        "crossing_y": None,
        "status": current_status,
    })

    return segments

# Function to calculate crossing points and insert them into the dataset
def calculate_and_insert_crossing_points(x_values, y_treatment, y_mock):
    new_x = []
    new_y_treatment = []
    new_y_mock = []
    statuses = []

    for i in range(1, len(x_values)):
        # Add the current point
        new_x.append(x_values[i - 1])
        new_y_treatment.append(y_treatment[i - 1])
        new_y_mock.append(y_mock[i - 1])

        # Determine the status at the current point
        if y_treatment[i - 1] > y_mock[i - 1]:
            current_status = "above"
        elif y_treatment[i - 1] < y_mock[i - 1]:
            current_status = "below"
        else:
            current_status = "same"

        # Check if a crossing point exists between the current and next point
        if (y_treatment[i - 1] - y_mock[i - 1]) * (y_treatment[i] - y_mock[i]) < 0:
            # Calculate the crossing point
            crossing_x = x_values[i - 1] + (x_values[i] - x_values[i - 1]) * (
                y_mock[i - 1] - y_treatment[i - 1]
            ) / ((y_mock[i - 1] - y_treatment[i - 1]) - (y_mock[i] - y_treatment[i]))
            crossing_y = y_treatment[i - 1] + (y_treatment[i] - y_treatment[i - 1]) * (
                crossing_x - x_values[i - 1]
            ) / (x_values[i] - x_values[i - 1])

            # Add the crossing point
            new_x.append(crossing_x)
            new_y_treatment.append(crossing_y)
            new_y_mock.append(crossing_y)

            # Add the status for the region before the crossing point
            statuses.append(current_status)

            # Update the current status for the region after the crossing point
            if current_status == "above":
                current_status = "below"
            else:
                current_status = "above"

        # Add the status for the region
        statuses.append(current_status)

    # Add the last point
    new_x.append(x_values[-1])
    new_y_treatment.append(y_treatment[-1])
    new_y_mock.append(y_mock[-1])

    return new_x, new_y_treatment, new_y_mock, statuses

@app.callback(
    [Output('time-series-plot', 'figure'),
     Output('interval', 'disabled')],
    [Input('interval', 'n_intervals'),
     Input('play-button', 'n_clicks'),
     Input('pause-button', 'n_clicks'),
     Input('timepoint-0-button', 'n_clicks'),  # Add input for the "Timepoint 0" button
     Input('last-timepoint-button', 'n_clicks'),  # Add input for the "Last timepoint" button
     Input('treatment-dropdown', 'value'),
     Input('time-series-plot', 'clickData'),
     Input('data-selection-checklist', 'value')],
    [State('time-series-plot', 'figure')]
)
def update_graph_and_control_interval(n_intervals, play_clicks, pause_clicks, timepoint_0_clicks, last_timepoint_clicks, selected_treatment, click_data, selected_treatments, current_figure):
    # Update the colors dictionary here
    colors = {
        "Mock": "rgb(0, 0, 0)",  # Black
        "JA": "rgb(201,196,84)",  # Updated color for JA
        "SA": "rgb(145,97,48)",  # Updated color for SA
        "SAJA": "rgb(179,144,61)"  # Updated color for SAJA
    }

    # Determine the current timepoint index
    ctx = dash.callback_context

    # Initialize the current index to 0 if the figure is not rendered yet
    if not current_figure or 'title' not in current_figure['layout']:
        current_index = 0
    else:
        # Extract the current timepoint from the figure title
        current_title = current_figure['layout']['title']['text']
        current_timepoint = float(current_title.split(" ")[-2])  # Extract the timepoint from the title
        current_index = timepoints_all.index(current_timepoint)

    # Handle play/pause button clicks
    if ctx.triggered and 'play-button' in ctx.triggered[0]['prop_id']:
        interval_disabled = False  # Enable the interval
    elif ctx.triggered and 'pause-button' in ctx.triggered[0]['prop_id']:
        interval_disabled = True  # Disable the interval
    elif ctx.triggered and 'timepoint-0-button' in ctx.triggered[0]['prop_id']:
        current_index = 0  # Reset to timepoint 0
        interval_disabled = True  # Pause the interval
    elif ctx.triggered and 'last-timepoint-button' in ctx.triggered[0]['prop_id']:
        current_index = len(timepoints_all) - 1  # Set to the last timepoint
        interval_disabled = True  # Pause the interval
    else:
        interval_disabled = dash.no_update  # Keep the current state

    # Update the index based on the interval or clickData
    if ctx.triggered and 'interval' in ctx.triggered[0]['prop_id']:
        if current_index < len(timepoints_all) - 1:
            current_index += 1
        else:
            return dash.no_update, True  # Stop the interval when the last timepoint is reached
    elif ctx.triggered and 'time-series-plot' in ctx.triggered[0]['prop_id']:
        if click_data and 'points' in click_data:
            clicked_x = click_data['points'][0]['x']  # Get the x-value of the clicked point
            # Find the closest timepoint to the clicked x-coordinate
            closest_timepoint = min(timepoints_all, key=lambda t: abs(t - clicked_x))
            current_index = timepoints_all.index(closest_timepoint)

    selected_timepoint = timepoints_all[current_index]

    # Initialize the figure
    fig = go.Figure()

    # Collect labels for dynamic positioning
    labels = []

    for treatment_label, data in data_dict.items():
        # Skip treatments that are not selected in the checklist
        if treatment_label not in selected_treatments:
            continue

        timepoints = data["timepoints"]
        mock_mean_values = data["mock_mean_values"]
        treatment_mean_values = data["treatment_mean_values"]
        mock_std_values = data["mock_std_values"]
        treatment_std_values = data["treatment_std_values"]

        # Filter data up to the selected timepoint
        mask = timepoints <= selected_timepoint
        filtered_timepoints = timepoints[mask]
        filtered_mock_mean = mock_mean_values[mask]
        filtered_treatment_mean = treatment_mean_values[mask]
        filtered_mock_std = mock_std_values[mask]
        filtered_treatment_std = treatment_std_values[mask]

        # Add Mock trace with error bars (only one Mock trace)
        if not any(trace.name in ["Mock", "All"] for trace in fig.data):  # Ensure only one Mock/All trace is added
            fig.add_trace(go.Scatter(
                x=filtered_timepoints,
                y=filtered_mock_mean,
                mode='lines+markers',
                name="All" if selected_timepoint == 0 else "Mock",  # Dynamically set legend text
                line=dict(color=colors["Mock"], dash='solid', width=2),
                error_y=dict(
                    type='data',
                    array=filtered_mock_std,  # Standard deviation as error bars
                    visible=True
                )
            ))

            # Add label for the last point of Mock only once
            if len(labels) == 0 or labels[-1]['text'] != f"  All: {filtered_mock_mean[-1]:,.0f}":
                label_text = f"  All: {filtered_mock_mean[-1]:,.0f}" if selected_timepoint == 0 else f"Mock: {filtered_mock_mean[-1]:,.0f}"
                labels.append({
                    'x': filtered_timepoints[-1],
                    'y': filtered_mock_mean[-1],
                    'text': label_text,
                    'color': colors["Mock"]  # Assign the color for the label
                })

        # Skip adding treatment traces if the selected timepoint is 0
        if selected_timepoint == 0:
            continue

        # Add Treatment trace with error bars
        fig.add_trace(go.Scatter(
            x=filtered_timepoints,
            y=filtered_treatment_mean,
            mode='lines+markers',
            name=f"Jasmonic acid" if treatment_label == "JA" else
                 f"Salicylic acid" if treatment_label == "SA" else
                 f"Salicylic and<br>Jasmonic acid",  # Updated legend names
            line=dict(color=colors[treatment_label], dash='solid', width=2),
            error_y=dict(
                type='data',
                array=filtered_treatment_std,  # Standard deviation as error bars
                visible=True
            )
        ))

        # Add label for the last point of each treatment
        labels.append({
            'x': filtered_timepoints[-1],
            'y': filtered_treatment_mean[-1],
            'text': f"Jasmonic acid: {filtered_treatment_mean[-1]:,.0f}" if treatment_label == "JA" else
                    f"Salicylic acid: {filtered_treatment_mean[-1]:,.0f}" if treatment_label == "SA" else
                    f"Salicylic and<br>Jasmonic acid: {filtered_treatment_mean[-1]:,.0f}",  # Updated label names
            'color': colors[treatment_label]  # Assign the color for the label
        })

        # Highlight the area between treatment and mock if selected
        if selected_treatment != 'None' and treatment_label == selected_treatment:
            # Calculate crossing points and insert them into the dataset
            new_x, new_y_treatment, new_y_mock, statuses = calculate_and_insert_crossing_points(
                filtered_timepoints, filtered_treatment_mean, filtered_mock_mean
            )

            # Highlight regions based on statuses
            for i in range(1, len(new_x)):
                # Build x and y values for the region
                x_values = [new_x[i - 1], new_x[i], new_x[i], new_x[i - 1]]
                y1_values = [new_y_treatment[i - 1], new_y_treatment[i], new_y_mock[i], new_y_mock[i - 1]]

                # Determine fill color based on status
                if statuses[i - 1] == "above":
                    fillcolor = 'rgba(226,145,131, 0.6)'  # Light blue for "above"
                elif statuses[i - 1] == "below":
                    fillcolor = 'rgba(115, 170, 210, 0.6)'  # Light red for "below"
                else:
                    fillcolor = 'rgba(226,145,131, 0.6)'  # Default color for "same" or unexpected status

                # Add the region to the figure
                fig.add_trace(go.Scatter(
                    x=x_values,
                    y=y1_values,
                    fill='toself',
                    fillcolor=fillcolor,
                    line=dict(color='rgba(0, 0, 0, 0)'),  # Invisible line
                    showlegend=False,
                    name=f"{statuses[i - 1].capitalize()} Mock: {treatment_label}"
                ))

    # Adjust label positions to avoid overlap
    adjusted_labels = adjust_label_positions(labels, timepoint=selected_timepoint)

    # Add adjusted labels to the figure
    for label in adjusted_labels:
        fig.add_trace(go.Scatter(
            x=[label['adjusted_x']],  # Use adjusted x position
            y=[label['adjusted_y']],  # Use adjusted y position
            mode='text',
            text=[label['text']],
            textposition="middle right",
            textfont=dict(
                size=10,  # Set the font size for the label text
                color=label['color']  # Set the font color to match the treatment/mock
            ),
            showlegend=False
        ))

    # Dynamically adjust the x-axis range to include space for labels
    x_max = max(timepoints_all) + 1.35  # Add padding to the maximum x value

    # Update layout
    fig.update_layout(
        font=dict(
        family="Lato",  # Set the global font to Lato
        size=12,        # Optional: Set a default font size
        color="black"   # Optional: Set a default font color
        ),
        title=f"Expression Levels up to {selected_timepoint} hpt",
        xaxis_title="Hours post treatment",
        yaxis_title="Expression Level",
        legend_title="Treatments",
        xaxis=dict(
            range=[min(timepoints_all), x_max],  # Extend x-axis range
            showgrid=True,  # Enable grid lines
            gridcolor='#929F76',  # Set vertical grid line color
            gridwidth=1  # Set grid line width
        ),
        yaxis=dict(
            autorange=True,
            showgrid=True,  # Enable horizontal grid lines
            gridcolor='#929F76',  # Set horizontal grid line color
            gridwidth=1,  # Set grid line width
            title=dict(
                text="",  # Leave the default y-axis title empty
                standoff=50,  # Add space between the axis and the labels to prevent overlap
                font=dict(size=12)  # Set font size
            ),
            tickformat=",d"  # Display whole numbers on the y-axis
        ),
        plot_bgcolor='#ffffff',  # Background color for the graph area
        paper_bgcolor='#ffffff',  # Background color for the entire figure
        annotations=[
            dict(
                x=-0.071,  # Adjust x position to avoid overlap
                y=0.5,  # Center it vertically
                xref="paper",
                yref="paper",
                text="Expression<br>Level",  # The y-axis title text
                showarrow=False,
                textangle=0,  # Keep the text horizontal
                font=dict(size=12)  # Set font size
            )
        ],
        shapes=[
            dict(
                type="line",
                x0=0,  # Start of the x-axis
                x1=1,  # End of the x-axis
                xref="paper",  # Reference to the paper coordinates
                y0=0,  # y=0
                y1=0,  # y=0
                yref="y",  # Reference to the y-axis
                line=dict(
                    color='#929F76',  # Same color as grid lines
                    width=2  # Make it bold
                )
            )
        ],
        margin=dict(
            l=125,  # Increase the left margin to ensure space for the y-axis label
            r=40,   # Right margin
            t=40,   # Top margin
            b=40    # Bottom margin
        )
    )

    return fig, interval_disabled

if __name__ == '__main__':
    try:
        print("Starting Dash app...")
        print("Double-click the link to run the app in browser.")
        app.run(debug=True, port=8050)
    except Exception as e:
        print(f"Error starting the app: {e}")