import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# More robust import
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.api import ExponentialSmoothing

# Ensure plots folder exists
os.makedirs("plots", exist_ok=True)

# Load Data
df = pd.read_csv(r"C:\Users\PAVITHRA\OneDrive\Desktop\Waste\waste_data_version4.csv", parse_dates=['Date'])

# --- sanity checks ---
# Ensure expected columns exist
expected_cols = {'Date', 'Weight_kg', 'Area', 'Type'}
missing = expected_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing expected columns in CSV: {missing}")

# Drop rows with missing date or weight
df = df.dropna(subset=['Date', 'Weight_kg']).copy()
# Ensure numeric
df['Weight_kg'] = pd.to_numeric(df['Weight_kg'], errors='coerce')
df = df.dropna(subset=['Weight_kg'])

# ----- DAILY AND WEEKLY TRENDS -----
# Daily Total Plot
daily = df.groupby('Date', as_index=False)['Weight_kg'].sum().reset_index(drop=True)
plt.figure(figsize=(10,5))
plt.plot(daily['Date'], daily['Weight_kg'], marker='o')
plt.title('Daily Waste Generation Trend')
plt.xlabel('Date')
plt.ylabel('Total Waste (kg)')
plt.grid(True)
plt.tight_layout()
plt.savefig('plots/daily_trend.png')
plt.close()

# Weekly Total Plot
# Use week start (Monday) as week label
df['week'] = df['Date'].dt.to_period('W').apply(lambda r: r.start_time)
weekly = df.groupby('week', as_index=False)['Weight_kg'].sum().reset_index(drop=True)
# Ensure sorted by week
weekly = weekly.sort_values('week').reset_index(drop=True)

plt.figure(figsize=(10,5))
plt.plot(weekly['week'], weekly['Weight_kg'], marker='o', color='darkorange')
plt.title('Weekly Waste Generation Trend')
plt.xlabel('Week Starting')
plt.ylabel('Total Waste (kg)')
plt.grid(True)
plt.tight_layout()
plt.savefig('plots/weekly_trend.png')
plt.close()

# ----- WASTE QUANTITY BY AREA AND TYPE -----
# By Area
area = df.groupby('Area', as_index=False)['Weight_kg'].sum().sort_values('Weight_kg', ascending=False)
plt.figure(figsize=(8,5))
plt.bar(area['Area'], area['Weight_kg'])
plt.title('Waste Quantity by Area')
plt.ylabel('Total Waste (kg)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('plots/by_area.png')
plt.close()

# By Type
wtype = df.groupby('Type', as_index=False)['Weight_kg'].sum().sort_values('Weight_kg', ascending=False)
plt.figure(figsize=(8,5))
plt.bar(wtype['Type'], wtype['Weight_kg'])
plt.title('Waste Quantity by Type')
plt.ylabel('Total Waste (kg)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('plots/by_type.png')
plt.close()

# ----- PREDICTION OF PEAK WASTE TIMES -----
# Use weekly data for better prediction
series = weekly.set_index('week')['Weight_kg'].astype(float).copy()
series = series.asfreq('W-MON') if series.index.inferred_freq is None else series  # try set freq if missing
series = series.sort_index()

# If series is very short, raise friendly error
if len(series) < 3:
    raise ValueError("Not enough weekly data points for forecasting (need >= 3).")

# Fit ExponentialSmoothing (simple)
model = ExponentialSmoothing(series, trend='add', seasonal=None)
res = model.fit(optimized=True)
forecast_values = res.forecast(steps=4)

# Create future week labels (week starting dates)
future_start = series.index.max() + pd.Timedelta(weeks=1)
future_weeks = pd.date_range(start=future_start, periods=4, freq='W-MON')

# Plot observed + forecast
plt.figure(figsize=(10,5))
plt.plot(series.index, series.values, label='Observed')
plt.plot(future_weeks, np.asarray(forecast_values), linestyle='--', marker='o', label='Forecast')
plt.title('Weekly Waste Generation: Prediction Next 4 Weeks')
plt.xlabel('Week Starting')
plt.ylabel('Total Waste (kg)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('plots/weekly_forecast.png')
plt.close()

# Safely get peak position (integer)
pos_peak = int(np.argmax(np.asarray(forecast_values)))
predicted_peak_week = future_weeks[pos_peak].strftime('%Y-%m-%d')
peak_value = float(np.asarray(forecast_values)[pos_peak])
print(f"Predicted peak waste generation is in the week starting {predicted_peak_week}, with an estimated value of {peak_value:.2f} kg.")

# ----- OUTPUT: CSVs for Dashboards -----
daily.to_csv('daily_trend.csv', index=False)
weekly.to_csv('weekly_trend.csv', index=False)
area.to_csv('area_total.csv', index=False)
wtype.to_csv('type_total.csv', index=False)
forecast_df = pd.DataFrame({'week': future_weeks, 'predicted_weight_kg': np.asarray(forecast_values)})
forecast_df.to_csv('forecast_weeks.csv', index=False)

print("All plots and summary CSVs are saved in the current directory.")
for f in ["daily_trend.csv", "weekly_trend.csv", "area_total.csv", "type_total.csv", "forecast_weeks.csv"]:
    print("✅ Saved:", os.path.abspath(f))