import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Ngarkoni të dhënat nga CSV për Humidity
df = pd.read_csv('humidity.csv')

# Konvertoni Timestamp në format datetime
df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%m/%d/%Y %H:%M')

# Shndërrojeni Timestamp në një numër (në këtë rast ditë numerike)
df['Timestamp'] = df['Timestamp'].apply(lambda x: x.toordinal())

# Sigurohuni që Humidity është një numër dhe hiqni çdo karakter të panevojshëm
df['Humidity'] = df['Humidity'].astype(float)

# Pjesëtimi i të dhënave në variabla të pavarur dhe të varur
X = df[['Timestamp']]  # variablat e pavarur (Timestamp)
y = df['Humidity']  # variabli i varur (Humidity)

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
predicted_humidity = model.predict(date_ordinals)

# Krijoni një DataFrame të ri me datat dhe lagështirën e parashikuar
predicted_df = pd.DataFrame({
    'Timestamp': date_range,
    'Predicted_Humidity': predicted_humidity
})

# Ruani rezultatet në një skedar të ri CSV
predicted_df.to_csv('predicted_humidity_2024.csv', index=False)

print("Parashikimi për vitin 2024 u ruajt me sukses në 'predicted_humidity_2024.csv'")

# Shfaqni 5 rreshtat e parë dhe të fundit të tabelës
print("5 rreshtat e para të tabelës origjinale: ")
print(df.head())

print("\n5 rreshtat e fundit të tabelës origjinale: ")
print(df.tail())

# Vizualizimi i të dhënave origjinale dhe të parashikuara
plt.figure(figsize=(10, 6))

# Të dhënat origjinale
plt.plot(df['Timestamp'], df['Humidity'], label='Të dhënat origjinale', color='blue', alpha=0.6)

# Të dhënat e parashikuara për 2024
plt.plot(predicted_df['Timestamp'], predicted_df['Predicted_Humidity'], label='Parashikimi 2024', color='red', alpha=0.6)

plt.title('Krahasimi i lagështirës - Të dhënat origjinale vs. Parashikimi 2024')
plt.xlabel('Timestamp')
plt.ylabel('Humidity (%)')
plt.legend()

# Shfaqni grafikun
plt.show()
