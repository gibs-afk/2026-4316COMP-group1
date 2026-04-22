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

def plot_heatmap(df):
    latest = df[df['Date'] == df['Date'].max()]

    region_deaths = latest.groupby('WHO Region')['Deaths'].sum().sort_values()

    data = region_deaths.values.reshape(-1, 1)

    fig, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(data, aspect='auto')

    ax.set(
        yticks=np.arange(len(region_deaths)),
        yticklabels=region_deaths.index,
        xticks=[],
        title="Global Death Density by WHO Region"
    )

    for i, val in enumerate(region_deaths.values):
        ax.text(0, i, f"{val:,}", ha='center', va='center')

    fig.colorbar(im)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    df = load_data()
    plot_heatmap(df)