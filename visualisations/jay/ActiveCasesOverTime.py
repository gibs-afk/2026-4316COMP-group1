import pandas as pd
import matplotlib.pyplot as plt
import os

plt.style.use('_mpl-gallery')

def load_data():
    path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "full_grouped.csv")
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def plot_active_cases(df, country):
    country_df = df[df['Country/Region'] == country]

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(country_df['Date'], country_df['Active'])

    ax.set(
        title=f"Active COVID-19 Cases Over Time - {country}",
        xlabel="Date",
        ylabel="Active Cases"
    )

    ax.grid(True, linestyle='--', alpha=0.5)

    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    df = load_data()
    plot_active_cases(df, "United Kingdom")