import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

plt.style.use('_mpl-gallery')

def load_data():
    path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "full_grouped.csv")
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def plot_fastest_growth(df):
    grouped = df.groupby('Country/Region')

    first = grouped.first()
    last = grouped.last()

    days = (df['Date'].max() - df['Date'].min()).days

    growth = pd.DataFrame({
        'first': first['Confirmed'],
        'last': last['Confirmed']
    })

    growth = growth[growth['first'] > 0]
    
    growth['daily_growth'] = (growth['last'] - growth['first']) / days

    top5 = growth.sort_values(by='daily_growth', ascending=False).head(5)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.barh(top5.index, top5['daily_growth'])

    ax.set(
        title="Top 5 Countries with Fastest Growth (Average Daily Cases)",
        xlabel="Average Daily Increase in Cases",
        ylabel="Country"
    )

    ax.grid(True, axis='x', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    df = load_data()
    plot_fastest_growth(df)