# Parashikimi i Të Dhënave Klimatike për Vitin 2024 - Republika e Kosovës

Ky projekt përfshin përdorimin e **Regresionit Linearm** për të parashikuar katër lloje të të dhënave klimatike për vitin 2024, bazuar në të dhënat e grumbulluara për periudhën 2017-2023 nga Republika e Kosovës. Llojet e të dhënave përfshijnë:
- Temperatura
- Lagështira (Humidity)
- Shpejtësia e Erës (Wind Speed)
- Presioni Atmosferik (Air Pressure)

## Përmbajtja
- **temperature.csv**: Të dhënat për temperaturën për vitet 2017-2023.
- **humidity.csv**: Të dhënat për lagështirën për vitet 2017-2023.
- **windspeed.csv**: Të dhënat për shpejtësinë e erës për vitet 2017-2023.
- **airpressure.csv**: Të dhënat për presionin atmosferik për vitet 2017-2023.
- **Predicted Results**: Skedarët CSV për parashikimet e vitit 2024 për secilin nga këto katër lloje të të dhënave.
  
## Hapat Kryesorë të Projektit

### 1. Ngarkimi i të Dhënave
Të dhënat janë ngarkuar nga katër skedarë CSV: `temperature.csv`, `humidity.csv`, `windspeed.csv`, dhe `airpressure.csv`. Çdo skedar përmban të dhëna të grumbulluara për periudhën 2017-2023, me timestamp dhe vlerat përkatëse për secilin variabël klimatik.

### 2. Përshtatja e të Dhënave
Për secilën nga këto të dhëna:
- **Konvertimi i Timestamp**: Timestamp-i është konvertuar në formatin datetime dhe më pas është shndërruar në një numër ordinal për ta përdorur në modelin e regresionit.
- **Pastrimi i të Dhënave**: Secili variabël (Temperatura, Lagështira, Shpejtësia e Erës, dhe Presioni Atmosferik) është pastruar për të hequr vlerat e panevojshme (p.sh., njësitë si "m/s" për shpejtësinë e erës).

### 3. Modeli i Regresionit Linearm
Për secilën lloj të dhënash, është përdorur **Regresioni Linear** për të krijuar një model që parashikon vlerat për vitin 2024. Pjesëtimi i të dhënave është bërë në dy grupe: një grup për trajnim dhe një grup për testim.

### 4. Parashikimi për 2024
Modeli është përdorur për të parashikuar vlerat për çdo orë të vitit 2024 për secilën nga këto të dhëna. Rezultatet janë ruajtur në skedarë të rinj CSV për secilën kategori (p.sh., `predicted_temperature_2024.csv`, `predicted_humidity_2024.csv`, etj.).

### 5. Vizualizimi i Rezultateve
Vizualizimet janë krijuar për të krahasuar të dhënat origjinale dhe ato të parashikuara për vitin 2024, me grafikë që shfaqin ndryshimet e temperaturës, lagështirës, shpejtësisë së erës dhe presionit atmosferik.

## Teknologjitë dhe Bibliotekat e Përdorura

- **Pandas**: Për manipulimin e të dhënave dhe krijimin e DataFrame-ve.
- **Scikit-learn**: Për krijimin dhe trajnimin e modelit të regresionit linearm.
- **Matplotlib**: Për vizualizimin e të dhënave dhe parashikimeve.
- **NumPy**: Për manipulimin e të dhënave numerike dhe për llogaritjen e vlerave të modelit.
- **Python**: Programimi i përgjithshëm për këtë projekt.

## Hapat e Përdorimit

1. **Ngarkoni të dhënat**: Sigurohuni që të keni skedarët CSV të duhura në të njëjtin dosje si ky skript.
2. **Përgatitja e të Dhënave**: Sigurohuni që të dhënat të jenë të pastruara dhe të konvertuara në formatin e duhur për trajnim.
3. **Krijimi i Modelit të Regresionit**: Përdorni **LinearRegression** nga Scikit-learn për të krijuar modelin për secilin nga llojet e të dhënave.
4. **Parashikimi për 2024**: Përdorni modelin për të parashikuar vlerat për vitin 2024 për secilin variabël klimatik.
5. **Vizualizimi i Rezultateve**: Shfaqni grafiket që krahasojnë të dhënat origjinale dhe parashikimet për vitin 2024.

## Rezultatet

Të dhënat për vitin 2024 janë ruajtur në skedarët CSV përkatës:
- `predicted_temperature_2024.csv`
- `predicted_humidity_2024.csv`
- `predicted_windspeed_2024.csv`
- `predicted_airpressure_2024.csv`

### Vlerësimi i Modelit
Secili model është vlerësuar përmes **Mean Squared Error (MSE)**, që tregon sa larg janë parashikimet nga vlerat reale.

## Kontributet

Nëse dëshironi të kontribuoni në këtë projekt, ju lutemi hapni një **issue** ose bëni një **pull request** me përmirësime, sugjerime ose ndihmës për kodin ose dokumentimin.

## Licenca

Ky projekt është i licencuar nën **MIT License** - shihni [LICENSE.md](LICENSE.md) për më shumë detaje.

## Autorët

Ky projekt është krijuar nga Bledi Buduri. Të dhënat janë mbledhur nga Republika e Kosovës për periudhën 2017-2023 dhe janë përdorur për të parashikuar parametrat klimatike për vitin 2024.

