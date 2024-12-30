import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Ngarkoni të dhënat nga CSV
df = pd.read_csv('airpreasure.csv')

# Përgatitja e të dhënave
df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d.%m.%Y %H:%M:%S')
df['Timestamp'] = df['Timestamp'].apply(lambda x: x.toordinal())
df['Air_Pressure'] = df['Air_Pressure'].astype(str).str.replace('hPa', '').astype(float)

X = df[['Timestamp']]
y = df['Air_Pressure']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modeli Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Parashikimet për vitin 2024
date_range = pd.date_range(start='2024-01-01 00:00:00', end='2024-12-31 23:59:59', freq='H')
date_ordinals = date_range.to_series().apply(lambda x: x.toordinal()).values.reshape(-1, 1)
predicted_pressures = model.predict(date_ordinals)

predicted_df = pd.DataFrame({
    'Timestamp': date_range,
    'Predicted_Air_Pressure': predicted_pressures
})

# Grafiku 1: Grafik Linjë (line plot)
plt.figure(figsize=(12, 6))
plt.plot(predicted_df['Timestamp'], predicted_df['Predicted_Air_Pressure'], label='Linear Regression Prediction', color='blue', linewidth=2)
plt.title('Grafiku Linjë: Parashikimi i Presionit të Ajrit për 2024', fontsize=14)
plt.xlabel('Data', fontsize=12)
plt.ylabel('Presioni i Ajrit (hPa)', fontsize=12)
plt.legend()
plt.grid(True)
plt.show()

# Grafiku 2: Grafik Pikat (scatter plot)
plt.figure(figsize=(12, 6))
plt.scatter(df['Timestamp'], df['Air_Pressure'], label='Të Dhënat Origjinale', color='darkblue', alpha=0.5, s=10)
plt.scatter(predicted_df['Timestamp'], predicted_df['Predicted_Air_Pressure'], label='Parashikimi 2024', color='red', alpha=0.7, s=10)
plt.title('Grafiku Pikat: Të Dhënat Origjinale dhe Parashikimet', fontsize=14)
plt.xlabel('Data dhe Ora (ordinal)', fontsize=12)
plt.ylabel('Presioni i Ajrit (hPa)', fontsize=12)
plt.legend()
plt.grid(True)
plt.show()


# Grafiku 4: Histogram (shpërndarja e parashikimeve)
plt.figure(figsize=(12, 6))
plt.hist(predicted_df['Predicted_Air_Pressure'], bins=50, color='green', alpha=0.7, edgecolor='black')
plt.title('Histogrami: Shpërndarja e Presionit të Parashikuar për 2024', fontsize=14)
plt.xlabel('Presioni i Ajrit (hPa)', fontsize=12)
plt.ylabel('Frekuenca', fontsize=12)
plt.grid(True)
plt.show()
