import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Load the CSV file
df = pd.read_csv('humidity.csv')

# Print column names to verify the column names
print(df.columns)

# Convert the Timestamp column to datetime format and rename for Prophet compatibility
df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%m/%d/%Y %H:%M')
df.rename(columns={'Timestamp': 'ds', 'Humidity': 'y'}, inplace=True)

# Plot the time series data
plt.figure(figsize=(14, 7))
plt.plot(df['ds'], df['y'], label='Humidity')
plt.title('Humidity Time Series')
plt.xlabel('Timestamp')
plt.ylabel('Humidity')
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

# Calculate RMSE and R Squared
rmse = np.sqrt(mean_squared_error(test['y'], forecast['yhat']))
r_squared = r2_score(test['y'], forecast['yhat'])

print(f"RMSE: {rmse}")
print(f"R Squared: {r_squared}")

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
plt.ylabel('Humidity')
plt.legend()
plt.show()

# Prepare the DataFrame for the predicted values
future_predictions = future_forecast[['ds', 'yhat']]
future_predictions.rename(columns={'ds': 'Timestamp', 'yhat': 'Humidity'}, inplace=True)

# Print the first few rows of the predictions DataFrame to verify
print(future_predictions.head())

# Save the predictions to a CSV file
output_path = 'Humidity_predictions_Prophet2024.csv'
future_predictions.to_csv(output_path, index=False)

print(f"Predictions for 2024 have been saved to '{output_path}'")
