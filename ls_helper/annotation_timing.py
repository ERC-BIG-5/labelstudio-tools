import collections
import json
import tempfile
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from numpy.ma.extras import average

from ls_helper.my_labelstudio_client.models import TaskResultModel


def annotation_timing(annotations: list[TaskResultModel], min_annotations: int = 2) -> pd.DataFrame:
    counter = collections.Counter[date]()
    for ann in annotations:
        if ann.total_annotations < min_annotations:
            continue
        dt = ann.annotations[-1].created_at.date()
        counter.update((dt,))
    df = pd.DataFrame(list(counter.items()), columns=['date', 'count'])
    df = df.sort_values('date')
    # print(df)
    return df

def get_annotation_lead_times(annotations: list[TaskResultModel], min_annotations: int = 2) -> pd.DataFrame:
    lead_times: dict[date, list[float]] = {}
    for ann in annotations:
        if ann.total_annotations < min_annotations:
            continue
        dt = ann.annotations[-1].created_at.date()
        completed_annotations = [ann for ann in ann.annotations if not ann.was_cancelled]
        lead_times[dt] = [ann.lead_time for ann in completed_annotations]
    lead_times_avgs = {k: average(v) for k, v in lead_times.items()}
    df = pd.DataFrame(list(lead_times_avgs.items()), columns=['date', 'lead_time'])
    df = df.sort_values('date')
    return df

def annotation_total_over_time(annotations: list[TaskResultModel],  min_annotations: int = 2):
    df = annotation_timing(annotations, min_annotations)
    result_df = df.sort_values('date')

    # Calculate the cumulative sum
    result_df['cumulative_total'] = result_df['count'].cumsum()
    return result_df

def plot_date_distribution(df: pd.DataFrame, title: str = "Completed tasks per Day",
                           y_col: str = "count",
                           xlabel: str = "Date", ylabel: str = "Count"):
    """
    Plot the date distribution.

    Parameters:
        df: DataFrame with 'date' and 'count' columns
        title, xlabel, ylabel: Plot labels
    """
    plt.figure(figsize=(12, 6))
    plt.bar(df['date'], df[y_col])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    temp = tempfile.NamedTemporaryFile(suffix='.png', prefix='img_', delete=False)
    plt.savefig(temp)
    return temp



def plot_cumulative_annotations(cumulative_df: pd.DataFrame, title: str = "Cumulative Annotations Over Time") -> tempfile.NamedTemporaryFile:
    """
    Creates a line plot showing the cumulative total of annotations over time.

    Parameters:
    -----------
    cumulative_df : pd.DataFrame
        DataFrame with 'date' and 'cumulative_total' columns
    title : str, optional
        Title for the plot
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the cumulative total line
    ax.plot(cumulative_df['date'], cumulative_df['cumulative_total'],
            marker='o', linestyle='-', linewidth=2, color='#1f77b4')

    # Add labels and title
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Annotations')
    ax.set_title(title)

    # Add data labels on each point
    for x, y in zip(cumulative_df['date'], cumulative_df['cumulative_total']):
        ax.annotate(f'{y}', (x, y), textcoords="offset points",
                    xytext=(0, 10), ha='center')

    # Format the plot
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    temp = tempfile.NamedTemporaryFile(suffix='.png', prefix='img_', delete=False)
    plt.savefig(temp)
    return temp
    # Show plot
    # plt.show()

