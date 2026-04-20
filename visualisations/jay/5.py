import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

plt.style.use('_mpl-gallery')

def load_full_grouped():
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path, "..", "data", "full_grouped.csv")

    try:
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'])
        print("full_grouped.csv loaded successfully")
        return df
    except Exception as e:
        print("Error:", e)
        return None

def first_case_dates(df):
    first_cases = df[df['Confirmed'] > 0] \
        .groupby(['Country/Region', 'WHO Region'])['Date'] \
        .min() \
        .reset_index()

    first_cases = first_cases.sort_values(by='Date')

    x = first_cases['Date']  
    y = np.arange(len(first_cases))

    regions = first_cases['WHO Region'].astype('category').cat.codes

    fig, ax = plt.subplots(figsize=(10, 6))

    scatter = ax.scatter(
        x,
        y,
        c=regions,
        s=40,
        alpha=0.7
    )

    ax.set(
        title="First Recorded Case of COVID-19 by Country",
        xlabel="Date of First Confirmed Case",
        ylabel="Countries"
    )

    fig.autofmt_xdate()

    ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    df = load_full_grouped()

    if df is not None:
        first_case_dates(df)
