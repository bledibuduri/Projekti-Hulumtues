import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Load the CSV file
df = pd.read_csv('windspeed.csv')

# Print column names to verify the column names
print(df.columns)

# Convert the Timestamp column to datetime format and set it as the index
df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d.%m.%Y %H:%M:%S')
df.set_index('Timestamp', inplace=True)

# Plot the time series data
plt.figure(figsize=(14, 7))
plt.plot(df['WindSpeed'], label='Wind Speed')
plt.title('Wind Speed Time Series')
plt.xlabel('Timestamp')
plt.ylabel('Wind Speed')
plt.legend()
plt.show()

# Split the data into training and testing sets
train_size = int(len(df) * 0.8)
train, test = df.iloc[:train_size], df.iloc[train_size:]

# Train the SARIMA model
sarima_order = (1, 1, 1)
seasonal_order = (1, 1, 1, 12)  # Adjust the seasonal order as necessary
model = SARIMAX(train['WindSpeed'], order=sarima_order, seasonal_order=seasonal_order)
model_fit = model.fit(disp=False)

# Make predictions
predictions = model_fit.forecast(steps=len(test))

# Calculate RMSE and R Squared
rmse = np.sqrt(mean_squared_error(test['WindSpeed'], predictions))
r_squared = r2_score(test['WindSpeed'], predictions)

print(f"RMSE: {rmse}")
print(f"R Squared: {r_squared}")

# Plot the predictions
plt.figure(figsize=(14, 7))
plt.plot(test['WindSpeed'].values, label='Actual')
plt.plot(predictions, label='Predicted (SARIMA)')
plt.title('SARIMA Model Prediction')
plt.legend()
plt.show()

# Forecast from 2024-01-01 to 2024-12-31
start_date = '2024-01-01'
end_date = '2024-12-31'
future_dates = pd.date_range(start=start_date, end=end_date, freq='D')

# Number of steps to forecast
future_steps = len(future_dates)

# Generate future predictions
future_predictions = model_fit.get_forecast(steps=future_steps).predicted_mean

# Plot future predictions
plt.figure(figsize=(14, 7))
plt.plot(future_dates, future_predictions, label='Future Predictions (SARIMA)')
plt.title('SARIMA Future Prediction for 2024')
plt.xlabel('Date')
plt.ylabel('Wind Speed')
plt.legend()
plt.show()

# Prepare the DataFrame for the predicted values
future_df = pd.DataFrame({'Timestamp': future_dates, 'WindSpeed': future_predictions})

# Print the first few rows of the predictions DataFrame to verify
print(future_df.head())

# Save the predictions to a CSV file
output_path = 'windspeed_predictions_sarima2024.csv'
future_df.to_csv(output_path, index=False)

print(f"Predictions for 2024 have been saved to '{output_path}'")
