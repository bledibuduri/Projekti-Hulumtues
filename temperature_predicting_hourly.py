import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Ngarkoni të dhënat nga CSV për Temperature
df = pd.read_csv('temperature.csv')

# Konvertoni Timestamp në format datetime
df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%m/%d/%Y %H:%M')

# Shndërrojeni Timestamp në një numër (në këtë rast ditë numerike)
df['Timestamp'] = df['Timestamp'].apply(lambda x: x.toordinal())

# Sigurohuni që Temperature është një numër dhe hiqni çdo karakter të panevojshëm
df['Temperature'] = df['Temperature'].astype(float)

# Pjesëtimi i të dhënave në variabla të pavarur dhe të varur
X = df[['Timestamp']]  # variablat e pavarur (Timestamp)
y = df['Temperature']  # variabli i varur (Temperature)

# Pjesëtimi i të dhënave në trajnimin dhe testimin
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Krijimi i modelit Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# Parashikimi i të dhënave të testuara
y_pred = model.predict(X_test)

# Vlerësimi i modelit
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print('R²:', r2_score(y_test, y_pred))

# Parashikimi për vitin 2024
# Krijoni një seri datash për vitin 2024 (për çdo orë)
date_range = pd.date_range(start='2024-01-01', end='2024-12-31 23:00:00', freq='H')

# Shndërrojeni datat në format ordinal
date_ordinals = date_range.to_series().apply(lambda x: x.toordinal()).values.reshape(-1, 1)

# Parashikimi për secilën datë të vitit 2024
predicted_temperature = model.predict(date_ordinals)

# Krijoni një DataFrame të ri me datat dhe temperaturën e parashikuar
predicted_df = pd.DataFrame({
    'Timestamp': date_range,
    'Predicted_Temperature': predicted_temperature
})

# Ruani rezultatet në një skedar të ri CSV
predicted_df.to_csv('predicted_temperature_2024_hourly.csv', index=False)

print("Parashikimi për vitin 2024 u ruajt me sukses në 'predicted_temperature_2024_hourly.csv'")

# Shfaqni 5 rreshtat e para dhe të fundit të tabelës
print("5 rreshtat e para të tabelës origjinale: ")
print(df.head())

print("\n5 rreshtat e fundit të tabelës origjinale: ")
print(df.tail())

# Vizualizimi i të dhënave origjinale dhe të parashikuara
plt.figure(figsize=(10, 6))

# Të dhënat origjinale
plt.plot(df['Timestamp'], df['Temperature'], label='Të dhënat origjinale', color='blue', alpha=0.6)

# Të dhënat e parashikuara për 2024
plt.plot(predicted_df['Timestamp'], predicted_df['Predicted_Temperature'], label='Parashikimi 2024', color='red', alpha=0.6)

plt.title('Krahasimi i temperaturës - Të dhënat origjinale vs. Parashikimi 2024')
plt.xlabel('Timestamp')
plt.ylabel('Temperature (°C)')
plt.legend()

# Shfaqni grafikun
plt.show()
