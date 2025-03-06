from datetime import timedelta
from typing import Optional

import pandas as pd
import plotly.express as px
import pytz


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
        plotly.graph_objects.Figure: A Plotly figure object showing the time series plot.
    """

    #location_features = features[features["pickup_location_id"] == 13817]
    location_features = features.iloc[row_id]
    actual_target = targets.iloc[row_id]
    #actual_target = targets[targets["pickup_location_id"] == row_id]

    # Identify time series columns (e.g., historical ride counts)
    time_series_columns = [
        col for col in features.columns if col.startswith("rides_t-")
    ]
    time_series_values = [location_features[col] for col in time_series_columns] + [
        actual_target
    ]

    # Generate corresponding timestamps for the time series
    time_series_dates = pd.date_range(
        start=location_features["pickup_hour"]
        - timedelta(hours=len(time_series_columns)),
        end=location_features["pickup_hour"],
        freq="h",
    )

    # Create the plot title with relevant metadata
    title = f"Pickup Hour: {location_features['pickup_hour']}, Location ID: {location_features['pickup_location_id']}"

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
        fig.add_scatter(
            x=time_series_dates[-1:],  # Last timestamp
            #y=predictions[predictions[row_id]],

            y=[predictions["pickup_location_id" == row_id]],  # Predicted value
            line_color="red",
            mode="markers",
            marker_symbol="x",
            marker_size=15,
            name="Prediction",
        )

    return fig

def plot_prediction2(features: pd.DataFrame, prediction: pd.DataFrame):
    # Validate if features DataFrame is empty
    if features.empty:
        raise ValueError("Error: 'features' DataFrame is empty! Ensure valid input.")

    # Identify time series columns (e.g., historical ride counts)
    time_series_columns = [col for col in features.columns if col.startswith("rides_t-")]

    # Ensure at least one time-series column is present
    if not time_series_columns:
        raise ValueError("Error: No historical ride count columns found in 'features'.")

    # Extract time series values safely
    time_series_values = [
        features[col].iloc[0] if col in features else None
        for col in time_series_columns
    ] + prediction["predicted_demand"].to_list()

    # Convert pickup_hour Series to single timestamp safely
    try:
        pickup_hour = pd.Timestamp(features["pickup_hour"].iloc[0])
    except IndexError:
        raise ValueError("Error: 'pickup_hour' column is empty or missing.")

    # Convert UTC to Eastern Time (EST)
    utc_timezone = pytz.utc
    est_timezone = pytz.timezone("US/Eastern")

    if pickup_hour.tzinfo is None:
        pickup_hour = pickup_hour.tz_localize(utc_timezone)  # Assign UTC if naive
    pickup_hour = pickup_hour.tz_convert(est_timezone)  # Convert to EST

    # Generate corresponding timestamps for the time series
    time_series_dates = pd.date_range(
        start=pickup_hour - timedelta(hours=len(time_series_columns)),
        end=pickup_hour,
        freq="h",
    )

    # Ensure length consistency between time and ride counts
    if len(time_series_dates) != len(time_series_values):
        raise ValueError("Mismatch: Time series length does not match ride counts.")

    # Create a DataFrame for historical data
    historical_df = pd.DataFrame({"datetime": time_series_dates, "rides": time_series_values})

    # Safely extract location ID
    location_id = features["pickup_location_id"].iloc[0] if "pickup_location_id" in features else "Unknown"

    # Create plot title with relevant metadata
    title = f"Pickup Hour: {pickup_hour}, Location ID: {location_id}"

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
        y=prediction["predicted_demand"].to_list(),
        line_color="red",
        mode="markers",
        marker_symbol="x",
        marker_size=10,
        name="Prediction",
    )

    return fig


def plot_prediction(features: pd.DataFrame, prediction: int):
    # Identify time series columns (e.g., historical ride counts)
    time_series_columns = [
        col for col in features.columns if col.startswith("rides_t-")
    ]
    time_series_values = [
        features[col].iloc[0] for col in time_series_columns
    ] + prediction["predicted_demand"].to_list()

    # Convert pickup_hour Series to single timestamp
    
    pickup_hour = pd.Timestamp(features["pickup_hour"].iloc[0])
    import pytz

    # Define UTC and EST timezones
    utc_timezone = pytz.utc
    est_timezone = pytz.timezone('US/Eastern')
    if pickup_hour.tzinfo is None:
        pickup_hour = pickup_hour.tz_localize('UTC')
    pickup_hour = pickup_hour.tz_convert(est_timezone)
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
    import pytz

    # Define UTC and EST timezones
    utc_timezone = pytz.utc
    est_timezone = pytz.timezone('US/Eastern')
    # Create the plot title with relevant metadata
    pickup_hour2=pickup_hour
    if pickup_hour.tzinfo is None:
        pickup_hour2 = pickup_hour.tz_localize('UTC')
    pickup_hour3 = pickup_hour2.tz_convert(est_timezone)
    title = f"Pickup Hour: {pickup_hour3}, Location ID: {features['pickup_location_id'].iloc[0]}"

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
        y=prediction["predicted_demand"].to_list(),
        line_color="red",
        mode="markers",
        marker_symbol="x",
        marker_size=10,
        name="Prediction",
    )

    return fig
