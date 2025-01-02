import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Load the CSV file
df = pd.read_csv('airpreasure.csv')

# Print column names to verify the column names
print(df.columns)

# Convert the Timestamp column to datetime format and rename for Prophet compatibility
df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d.%m.%Y %H:%M:%S')
df.rename(columns={'Timestamp': 'ds', 'Air_Pressure': 'y'}, inplace=True)

# Plot the time series data
plt.figure(figsize=(14, 7))
plt.plot(df['ds'], df['y'], label='Air Pressure')
plt.title('Air Pressure Time Series')
plt.xlabel('Timestamp')
plt.ylabel('Air Pressure')
plt.legend()
plt.show()

# Split the data into training and testing sets
train_size = int(len(df) * 0.8)
train, test = df.iloc[:train_size], df.iloc[train_size:]

# Initialize the Prophet model
model = Prophet()

# Fit the model to the training data
model.fit(train)

# Make predictions on the test data
future = test[['ds']]
forecast = model.predict(future)

# Evaluate the model
rmse = np.sqrt(mean_squared_error(test['y'], forecast['yhat']))
r2 = r2_score(test['y'], forecast['yhat'])
print(f"RMSE: {rmse}")
print(f"R-squared: {r2}")

# Plot the predictions
plt.figure(figsize=(14, 7))
plt.plot(test['ds'], test['y'], label='Actual')
plt.plot(test['ds'], forecast['yhat'], label='Predicted (Prophet)')
plt.title('Prophet Model Prediction')
plt.legend()
plt.show()

# Forecast for 2024
future_dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
future_df = pd.DataFrame({'ds': future_dates})

# Generate future predictions
future_forecast = model.predict(future_df)

# Plot future predictions
plt.figure(figsize=(14, 7))
plt.plot(future_forecast['ds'], future_forecast['yhat'], label='Future Predictions (Prophet)')
plt.title('Prophet Future Prediction for 2024')
plt.xlabel('Date')
plt.ylabel('Air Pressure')
plt.legend()
plt.show()

# Prepare the DataFrame for the predicted values
future_predictions = future_forecast[['ds', 'yhat']]
future_predictions.rename(columns={'ds': 'Timestamp', 'yhat': 'Air_Pressure'}, inplace=True)

# Print the first few rows of the predictions DataFrame to verify
print(future_predictions.head())

# Save the predictions to a CSV file
output_path = 'Air_Pressure_predictions_Prophet_2024.csv'
future_predictions.to_csv(output_path, index=False)

print(f"Predictions for 2024 have been saved to '{output_path}'")

# Plot histogram of residuals (difference between actual and predicted values)
residuals = test['y'] - forecast['yhat']
plt.figure(figsize=(10, 6))
plt.hist(residuals, bins=30, color='skyblue', edgecolor='black')
plt.title('Histogram of Residuals')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
