from datetime import timedelta
from typing import Optional
import pandas as pd
import plotly.express as px

def plot_aggregated_time_series(
    features: pd.DataFrame,
    targets: pd.Series,
    row_id: int,
    predictions: Optional[pd.Series] = None,
):
    """
    Plots the time series data for a specific location from NYC taxi data.

    Args:
        features (pd.DataFrame): DataFrame containing feature data, including historical ride counts and metadata.
        targets (pd.Series): Series containing the target values (e.g., actual ride counts).
        row_id (int): Index of the row to plot.
        predictions (Optional[pd.Series]): Series containing predicted values (optional).

    Returns:
        plotly.graph_objects.Figure: A Plotly figure object showing the time series plot.  Returns None if no data is found.
    """

    # Extract the specific location's features and target
    location_features = features[features["pickup_location_id"] == row_id]

    if location_features.empty:
        print(f"No data found for location ID: {row_id}")
        return None  # Or handle the error as appropriate

    try:
        actual_target = targets[location_features.index].iloc[0]
    except IndexError:
        print(f"IndexError: No target found for location ID: {row_id}")
        return None

    # Identify time series columns (e.g., historical ride counts)
    time_series_columns = [
        col for col in features.columns if col.startswith("rides_t-")
    ]

    try:
        time_series_values = [location_features[col].iloc[0] for col in time_series_columns] + [
            actual_target
        ]
    except KeyError as e:
        print(f"KeyError accessing time series data: {e}")
        return None

    # Generate corresponding timestamps for the time series
    try:
        pickup_hour = location_features["pickup_hour"].iloc[0]  # Extract the single timestamp
        time_series_dates = pd.date_range(
            start=pickup_hour - timedelta(hours=len(time_series_columns)),
            end=pickup_hour,
            freq="h",
        )
    except KeyError as e:
        print(f"KeyError creating date range: {e}")
        return None

    # Create the plot title with relevant metadata
    try:
        title = f"Pickup Hour: {location_features['pickup_hour'].iloc[0]}, Location ID: {location_features['pickup_location_id'].iloc[0]}"
    except KeyError as e:
        print(f"KeyError creating title: {e}")
        title = "Time Series Plot"


    # Create the base line plot
    fig = px.line(
        x=time_series_dates,
        y=time_series_values,
        template="plotly_white",
        markers=True,
        title=title,
        labels={"x": "Time", "y": "Ride Counts"},
    )

    # Add the actual target value as a green marker
    fig.add_scatter(
        x=time_series_dates[-1:],  # Last timestamp
        y=[actual_target],  # Actual target value
        line_color="green",
        mode="markers",
        marker_size=10,
        name="Actual Value",
    )

    # Optionally add the prediction as a red marker
    if predictions is not None:
        try:
            predicted_value = predictions[location_features.index].iloc[0]
            fig.add_scatter(
                x=time_series_dates[-1:],  # Last timestamp
                y=[predicted_value],  # Predicted value
                line_color="red",
                mode="markers",
                marker_symbol="x",
                marker_size=15,
                name="Prediction",
            )
        except KeyError as e:
            print(f"KeyError adding prediction: {e}")
        except IndexError as e:
            print(f"IndexError adding prediction: {e}")

    return fig

def plot_prediction(features: pd.DataFrame, prediction: pd.Series):  # Changed prediction type
    """Plots a single time series with a prediction."""

    # Identify time series columns (e.g., historical ride counts)
    time_series_columns = [
        col for col in features.columns if col.startswith("rides_t-")
    ]
    try:
        time_series_values = [
            features[col].iloc[0] for col in time_series_columns
        ] + prediction["predicted_demand"].to_list()
    except KeyError as e:
        print(f"KeyError in plot_prediction: {e}")
        return None
    # Convert pickup_hour Series to single timestamp
    try:
        pickup_hour = pd.Timestamp(features["pickup_hour"].iloc[0])
    except KeyError as e:
        print(f"KeyError getting pickup_hour: {e}")
        return None


    # Generate corresponding timestamps for the time series
    time_series_dates = pd.date_range(
        start=pickup_hour - timedelta(hours=len(time_series_columns)),
        end=pickup_hour,
        freq="h",
    )

    # Create a DataFrame for the historical data
    historical_df = pd.DataFrame(
        {"datetime": time_series_dates, "rides": time_series_values}
    )

    # Create the plot title with relevant metadata
    try:
        title = f"Pickup Hour: {pickup_hour}, Location ID: {features['pickup_location_id'].iloc[0]}"
    except KeyError as e:
        print(f"Error: Could not find column: {e}. Setting default title.")
        title = "Time Series Plot"

    # Create the base line plot
    fig = px.line(
        historical_df,
        x="datetime",
        y="rides",
        template="plotly_white",
        markers=True,
        title=title,
        labels={"datetime": "Time", "rides": "Ride Counts"},
    )

    # Add prediction point
    fig.add_scatter(
        x=[pickup_hour],  # Last timestamp
        y=[prediction["predicted_demand"].iloc[0]], # Corrected line
        line_color="red",
        mode="markers",
        marker_symbol="x",
        marker_size=10,
        name="Prediction",
    )

    return fig