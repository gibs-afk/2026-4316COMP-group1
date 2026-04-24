import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

data = pd.read_csv("../data/covid_19_clean_complete.csv")
data.columns = data.columns.str.strip()
data['Date'] = pd.to_datetime(data['Date'])

def getDeathsPerMonth(month):
    filtered = data[data['Date'].dt.month == month]
    deaths_by_country = filtered.groupby('Country/Region')['Deaths'].sum()
    top3 = deaths_by_country.sort_values(ascending=False).head(3)
    return list(top3.items())

def getHighestRatio():
    overall = data.groupby('Country/Region')[['Confirmed', 'Deaths']].sum()
    overall['Ratio'] = overall['Deaths'] / overall['Confirmed'].replace(0, 1)
    max_country = overall['Ratio'].idxmax()
    max_ratio = overall['Ratio'].max()
    return max_country, max_ratio

def getHighestDay():
    confirmed_by_date = data.groupby('Date')['Confirmed'].max()
    max_date = confirmed_by_date.idxmax()
    max_cases = confirmed_by_date.max()
    return max_date, max_cases

def getTop10RatioChart():
    overall = data.groupby('Country/Region')[['Confirmed', 'Deaths']].sum()
    overall['Ratio'] = overall['Deaths'] / overall['Confirmed'].replace(0, 1)
    top10 = overall.sort_values('Ratio', ascending=False).head(10)

    plt.figure(figsize=(10, 6))
    plt.barh(top10.index, top10['Ratio'], color='steelblue')
    plt.xlabel('Deaths to Confirmed Cases Ratio')
    plt.ylabel('Country')
    plt.title('Top 10 Countries by Deaths to Confirmed Cases Ratio')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

def getTop5Confirmed():
    overall = data.groupby('Country/Region')['Confirmed'].max()
    top5 = overall.sort_values(ascending=False).head(5)
    return list(top5.items())

def plot_active_cases(country):
    plt.style.use('_mpl-gallery')
    country_df = data[data['Country/Region'] == country]

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

def plot_heatmap():
    plt.style.use('_mpl-gallery')
    latest = data[data['Date'] == data['Date'].max()]

    if 'WHO Region' not in data.columns:
        print("WHO Region column not found in data.")
        return

    region_deaths = latest.groupby('WHO Region')['Deaths'].sum().sort_values()
    data_arr = region_deaths.values.reshape(-1, 1)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(data_arr, aspect='auto')

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

def plot_fastest_growth():
    plt.style.use('_mpl-gallery')
    grouped = data.groupby('Country/Region')

    first = grouped.first()
    last = grouped.last()

    days = (data['Date'].max() - data['Date'].min()).days

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

def first_case_dates():
    plt.style.use('_mpl-gallery')

    if 'WHO Region' not in data.columns:
        print("WHO Region column not found in data.")
        return

    first_cases = data[data['Confirmed'] > 0] \
        .groupby(['Country/Region', 'WHO Region'])['Date'] \
        .min() \
        .reset_index()

    first_cases = first_cases.sort_values(by='Date')

    x = first_cases['Date']
    y = np.arange(len(first_cases))
    regions = first_cases['WHO Region'].astype('category').cat.codes

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x, y, c=regions, s=40, alpha=0.7)

    ax.set(
        title="First Recorded Case of COVID-19 by Country",
        xlabel="Date of First Confirmed Case",
        ylabel="Countries"
    )

    fig.autofmt_xdate()
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def plot_recovery_rate():
    plt.style.use('_mpl-gallery')
    latest = data[data['Date'] == data['Date'].max()].copy()

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

def fastest_decline_after_peak():
    active_trend = data.groupby(["Country/Region", "Date"])["Active"].sum().reset_index()

    def decline_rate(group):
        group = group.sort_values("Date")
        peak = group["Active"].max()
        peak_position = group["Active"].idxmax()
        after_peak = group.loc[peak_position:]

        if len(after_peak) > 1:
            return (after_peak.iloc[-1]["Active"] - peak) / len(after_peak)
        return 0

    declines = active_trend.groupby("Country/Region").apply(decline_rate)
    top5 = declines.sort_values().head(5)

    print("Countries that reduced active cases fastest after peak:")
    print(top5)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(top5.index, top5.values)

    ax.set(
        title="Top 5 Countries with Fastest Active Case Decline After Peak",
        xlabel="Average Daily Decline After Peak",
        ylabel="Country"
    )

    ax.grid(True, axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

def find_anomalies():
    df_sorted = data.sort_values(["Country/Region", "Date"]).copy()

    df_sorted["Confirmed Change"] = df_sorted.groupby("Country/Region")["Confirmed"].diff()
    df_sorted["Deaths Change"] = df_sorted.groupby("Country/Region")["Deaths"].diff()

    anomalies = df_sorted[
        (df_sorted["Deaths Change"] > 0) & (df_sorted["Confirmed Change"] <= 0)
    ]

    anomaly_counts = anomalies["Country/Region"].value_counts().head(10)

    print("Countries with unusual reporting patterns:")
    print(anomaly_counts)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(anomaly_counts.index, anomaly_counts.values)

    ax.set(
        title="Top 10 Countries with Death Increases but No Confirmed Case Increase",
        xlabel="Number of Anomaly Days",
        ylabel="Country"
    )

    ax.grid(True, axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()    

def fastest_region_to_100k():
    region_daily = data.groupby(["WHO Region", "Date"])["Confirmed"].sum().reset_index()

    def time_to_threshold(group, threshold=100000):
        group = group.sort_values("Date")
        reached = group[group["Confirmed"] >= threshold]

        if not reached.empty:
            return (reached.iloc[0]["Date"] - group.iloc[0]["Date"]).days
        return np.nan

    threshold_times = region_daily.groupby("WHO Region").apply(time_to_threshold)
    threshold_times = threshold_times.dropna().sort_values()

    print("Fastest WHO regions to reach 100,000 confirmed cases:")
    print(threshold_times)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(threshold_times.index, threshold_times.values)

    ax.set(
        title="Days Taken for WHO Regions to Reach 100,000 Confirmed Cases",
        xlabel="Days Taken",
        ylabel="WHO Region"
    )

    ax.grid(True, axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

def longest_decline_streak():
    df_sorted = data.sort_values(["Country/Region", "Date"]).copy()

    def longest_decline(group):
        decline_streak = 0
        max_streak = 0

        changes = group["Active"].diff()

        for change in changes:
            if change < 0:
                decline_streak += 1
                max_streak = max(max_streak, decline_streak)
            else:
                decline_streak = 0

        return max_streak

    decline_streaks = df_sorted.groupby("Country/Region").apply(longest_decline)
    top5 = decline_streaks.sort_values(ascending=False).head(5)

    print("Countries with the longest sustained decrease in active cases:")
    print(top5)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(top5.index, top5.values)

    ax.set(
        title="Top 5 Countries with Longest Sustained Decline in Active Cases",
        xlabel="Longest Consecutive Decline Streak (Days)",
        ylabel="Country"
    )

    ax.grid(True, axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()    

while True:
    print("\nMenu:")
    print("1. Find country with most deaths in a specific month")
    print("2. Find country with highest deaths to confirmed cases ratio overall")
    print("3. Find the day with the most confirmed cases")
    print("4. View top 10 countries by deaths to confirmed ratio (bar chart)")
    print("5. Show top 5 countries with highest confirmed cases")
    print("6. View active cases over time for a country")
    print("7. View WHO region deaths heatmap")
    print("8. View top 5 countries with fastest growth")
    print("9. View first recorded case by country")
    print("10. View top 5 countries by recovery rate")
    print("11. Countries with fastest decline after peak")
    print("12. Detect unusual reporting patterns")
    print("13. Fastest region to 100k cases")
    print("14. Longest decline streak")
    print("15. Exit")

    choice = input("Enter your choice (1-15): ").strip()

    if choice == '1':
        month_input = input("Enter a month (1-12 or month name e.g. January): ").strip()
        try:
            import calendar
            if month_input.isdigit():
                month = int(month_input)
            else:
                month = list(calendar.month_name).index(month_input.capitalize())
            top_countries = getDeathsPerMonth(month)
            print(f"Top 3 countries with most deaths in month {month}:")
            for rank, (country, deaths) in enumerate(top_countries, start=1):
                print(f"  {rank}. {country}: {deaths} deaths")
        except Exception as e:
            print("Error:", e)

    elif choice == '2':
        max_country, max_ratio = getHighestRatio()
        print(f"Country with highest deaths to confirmed cases ratio overall: {max_country} (Ratio: {max_ratio:.4f})")

    elif choice == '3':
        max_date, max_cases = getHighestDay()
        print(f"Day with the most confirmed cases: {max_date.strftime('%Y-%m-%d')} ({max_cases} cases)")

    elif choice == '4':
        getTop10RatioChart()

    elif choice == '5':
        top5 = getTop5Confirmed()
        print("Top 5 countries with highest confirmed cases:")
        for rank, (country, cases) in enumerate(top5, start=1):
            print(f"  {rank}. {country}: {cases:,} cases")

    elif choice == '6':
        country = input("Enter country name: ").strip()
        if country in data['Country/Region'].values:
            plot_active_cases(country)
        else:
            print("Country not found in data.")

    elif choice == '7':
        plot_heatmap()

    elif choice == '8':
        plot_fastest_growth()

    elif choice == '9':
        first_case_dates()

    elif choice == '10':
        plot_recovery_rate()

    elif choice == '11':
        fastest_decline_after_peak()

    elif choice == '12':
        find_anomalies()

    elif choice == '13':
        fastest_region_to_100k()

    elif choice == '14':
        longest_decline_streak()

    elif choice == '15':
        print("Exiting...")
        break

    else:
        print("Invalid choice. Please enter 1-15.")