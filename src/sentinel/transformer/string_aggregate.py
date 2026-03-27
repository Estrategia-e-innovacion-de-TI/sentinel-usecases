import pandas as pd
import numpy as np
from typing import List, Dict, Union, Optional, Callable
from datetime import timedelta

class StringAggregator:
    """Flexible time-window aggregation for DataFrames with categorical data.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Input DataFrame.
    timestamp_column : str
        Name of the timestamp column.

    Raises
    ------
    TypeError
        If ``dataframe`` is not a DataFrame.
    ValueError
        If ``timestamp_column`` is not found in the DataFrame.
    """

    def __init__(self, dataframe: pd.DataFrame, timestamp_column: str):

        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        
        if timestamp_column not in dataframe.columns:
            raise ValueError(f"Column '{timestamp_column}' not found in DataFrame")

        self.df = dataframe.copy()
        self.df[timestamp_column] = pd.to_datetime(self.df[timestamp_column])
        self.timestamp_column = timestamp_column

    def create_time_aggregation(
        self, 
        time_window: str = '5min', 
        column_metrics: Optional[Dict[str, List[Union[str, Callable]]]] = None,
        custom_metrics: Optional[Dict[str, Callable]] = None,
        category_count_columns: Optional[Dict[str, List[str]]] = None
    ) -> pd.DataFrame:
        """Aggregate data over a time window with configurable metrics.

        Parameters
        ----------
        time_window : str, default='5min'
            Pandas offset alias for the grouping window.
        column_metrics : dict, optional
            Mapping of column names to lists of metrics (strings or callables).
            Defaults to ``'count'`` and ``'nunique'`` for all non-timestamp columns.
        custom_metrics : dict, optional
            Global custom metrics as ``{name: callable}``.
        category_count_columns : dict, optional
            Mapping of column names to category values to count.

        Returns
        -------
        pd.DataFrame
            Aggregated results indexed by time window.

        Raises
        ------
        ValueError
            If a specified column does not exist in the DataFrame.
        """
        # Keep a copy of the timestamp so inner functions can access it
        # even after pd.Grouper consumes the column as the grouping key.
        ts_col_backup = f"__{self.timestamp_column}_ts"
        self.df[ts_col_backup] = self.df[self.timestamp_column]

        # Group by time window
        grouped = self.df.groupby(pd.Grouper(key=self.timestamp_column, freq=time_window))
        
        # Results container
        results = pd.DataFrame(index=grouped.groups.keys())
        
        # Default metrics if none provided
        if column_metrics is None:
            column_metrics = {
                col: ['count', 'nunique'] 
                for col in self.df.columns 
                if col != self.timestamp_column
            }
        
        # Apply per-column metrics
        for column, metrics in column_metrics.items():
            if column not in self.df.columns:
                raise ValueError(f"Column '{column}' not found in DataFrame")
            
            for metric in metrics:
                col_name = f"{column}_{metric}"
                
                # Handle string metrics
                if isinstance(metric, str):
                    if metric == 'count':
                        results[col_name] = grouped[column].count()
                    elif metric == 'nunique':
                        results[col_name] = grouped[column].nunique()
                    elif metric == 'mode':
                        results[col_name] = grouped[column].agg(lambda x: x.mode().iloc[0] if not x.empty else np.nan)
                
                # Handle custom callable metrics
                elif callable(metric):
                    results[col_name] = grouped[column].agg(metric)
        
        # Category-specific counts
        if category_count_columns:
            for column, categories in category_count_columns.items():
                if column not in self.df.columns:
                    raise ValueError(f"Column '{column}' not found in DataFrame")
                
                for category in categories:
                    col_name = f"{column}_{category}_count"
                    results[col_name] = grouped.apply(lambda x: (x[column] == category).sum())
        
        # Time between events metrics (in seconds)
        def avg_time_between_events(group):
            if len(group) <= 1:
                return 0
            times = group[ts_col_backup].sort_values()
            diffs = times.diff().dropna()
            return diffs.dt.total_seconds().mean()
        
        # Additional time between events metrics
        def min_time_between_events(group):
            if len(group) <= 1:
                return 0
            times = group[ts_col_backup].sort_values()
            diffs = times.diff().dropna()
            return diffs.dt.total_seconds().min()
        
        def max_time_between_events(group):
            if len(group) <= 1:
                return 0
            times = group[ts_col_backup].sort_values()
            diffs = times.diff().dropna()
            return diffs.dt.total_seconds().max()
        
        results['avg_time_between_events_seconds'] = grouped.apply(avg_time_between_events)
        results['min_time_between_events_seconds'] = grouped.apply(min_time_between_events)
        results['max_time_between_events_seconds'] = grouped.apply(max_time_between_events)
        
        # Apply global custom metrics
        if custom_metrics:
            for metric_name, metric_func in custom_metrics.items():
                results[metric_name] = grouped.apply(metric_func)
        
        # Clean up the internal backup column
        self.df.drop(columns=[ts_col_backup], inplace=True, errors='ignore')

        return results

# Ejemplo de uso
def example_usage():
    # Crear un DataFrame de ejemplo con más variedad de datos
    np.random.seed(42)  # Para reproducibilidad
    
    # Generar timestamps con más densidad
    timestamps = pd.date_range(start='2024-01-01', end='2024-01-02', freq='2min')
    
    data = {
        'timestamp': timestamps,
        'category': np.random.choice(['web', 'mobile', 'desktop'], len(timestamps)),
        'level': np.random.choice(['info', 'warning', 'error'], len(timestamps)),
        'ip': [f'192.168.1.{i%20}' for i in range(len(timestamps))],
        'response_time': np.random.uniform(10, 500, len(timestamps))
    }
    df = pd.DataFrame(data)
    
    # Crear instancia del agregador
    aggregator = StringAggregator(df, 'timestamp')
    
    # Definir métricas personalizadas
    column_metrics = {
        'category': ['count', 'nunique'],
        'level': ['count', 'mode'],
        'ip': ['nunique'],
        'response_time': ['mean', 'max', 'min']
    }
    
    # Definir conteo por categorías
    category_count_columns = {
        'level': ['info', 'warning', 'error'],
        'category': ['web', 'mobile', 'desktop']
    }
    
    # Métrica personalizada global
    def count_high_latency(group):
        return (group['response_time'] > 300).sum()
    
    # Realizar agregación
    result = aggregator.create_time_aggregation(
        time_window='5min', 
        column_metrics=column_metrics,
        category_count_columns=category_count_columns,
        custom_metrics={'high_latency_count': count_high_latency}
    )
    
    # Imprimir resultados
    print(result)

if __name__ == "__main__":
    example_usage()