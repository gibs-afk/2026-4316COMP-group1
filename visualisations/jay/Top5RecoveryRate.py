import pandas as pd
import matplotlib.pyplot as plt
import os

plt.style.use('_mpl-gallery')

def load_data():
    path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "full_grouped.csv")
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def plot_recovery_rate(df):
    latest = df[df['Date'] == df['Date'].max()].copy()
    latest = latest[latest['Confirmed'] > 0]
    latest['Recovery Rate'] = latest['Recovered'] / latest['Confirmed']

    top5 = latest.sort_values(by='Recovery Rate', ascending=False).head(5)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.barh(top5['Country/Region'], top5['Recovery Rate'])

    ax.set(
        title="Top 5 Countries by Recovery Rate",
        xlabel="Recovery Rate",
        ylabel="Country"
    )

    ax.grid(True, axis='x', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    df = load_data()
    plot_recovery_rate(df)