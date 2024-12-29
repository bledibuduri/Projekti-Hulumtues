import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Ngarkoni të dhënat nga CSV
df = pd.read_csv('airpreasure.csv')

# Konvertoni Timestamp në format datetime
df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d.%m.%Y %H:%M:%S')

# Shndërrojeni Timestamp në një numër (në këtë rast ditë numerike)
df['Timestamp'] = df['Timestamp'].apply(lambda x: x.toordinal())

# Sigurohuni që Air_Pressure është një string dhe pastrojeni për të hequr "hPa"
df['Air_Pressure'] = df['Air_Pressure'].astype(str).str.replace('hPa', '').astype(float)

# Pjesëtimi i të dhënave në variabla të pavarur dhe të varur
X = df[['Timestamp']]  # variablat e pavarur (Timestamp)
y = df['Air_Pressure']  # variabli i varur (Air_Pressure)

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

# Parashikimi për vitin 2024
# Krijoni një seri datash për vitin 2024
date_range = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')

# Shndërrojeni datat në format ordinal
date_ordinals = date_range.to_series().apply(lambda x: x.toordinal()).values.reshape(-1, 1)

# Parashikimi për secilën datë të vitit 2024
predicted_pressures = model.predict(date_ordinals)

# Krijoni një DataFrame të ri me datat dhe presionin e parashikuar
predicted_df = pd.DataFrame({
    'Timestamp': date_range,
    'Predicted_Air_Pressure': predicted_pressures
})

# Ruani rezultatet në një skedar të ri CSV
predicted_df.to_csv('predicted_air_pressure_2024.csv', index=False)

print("Parashikimi për vitin 2024 u ruajt me sukses në 'predicted_air_pressure_2024.csv'")