"""I/O utilities for loading and labeling datasets."""

from typing import List, Optional

import pandas as pd


def load_and_label(
    csv_path: str,
    columns: List[str],
    events_csv_path: Optional[str] = None,
    timestamp_column: str = "timestamp",
    event_begin_column: str = "begin_date",
    event_end_column: str = "end_date",
) -> tuple:
    """Load a CSV file and optionally label rows using an events file.

    Parameters
    ----------
    csv_path : str
        Path to the main CSV file.
    columns : list of str
        Column names to return for analysis.
    events_csv_path : str, optional
        Path to an events CSV with begin/end timestamps.
    timestamp_column : str
        Name of the timestamp column in the main CSV.
    event_begin_column : str
        Column in the events CSV for event start.
    event_end_column : str
        Column in the events CSV for event end.

    Returns
    -------
    df : pd.DataFrame
        The loaded (and optionally labeled) DataFrame.
    columns : list of str
        The column names passed in.
    """
    df = pd.read_csv(csv_path)

    if events_csv_path:
        events_df = pd.read_csv(events_csv_path)

        if timestamp_column in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df[timestamp_column]):
                df[timestamp_column] = pd.to_datetime(df[timestamp_column])

        df["label"] = 0

        for _, event in events_df.iterrows():
            begin_date = pd.to_datetime(event[event_begin_column])
            end_date = pd.to_datetime(event[event_end_column])
            mask = (df[timestamp_column] >= begin_date) & (df[timestamp_column] <= end_date)
            df.loc[mask, "label"] = 1

    return df, columns
