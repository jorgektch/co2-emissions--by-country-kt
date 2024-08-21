import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Cargar los datos del archivo Excel
@st.cache_data
def load_data():
    file_path = '/mnt/data/co2-emissions-by-country-kt.xlsx'
    data = pd.read_excel(file_path)
    return data

# Configuración inicial de Streamlit
st.title("Emisiones de CO2 por país (1960 - 2016)")
st.write("Seleccione un país para visualizar las emisiones de CO2 y la recta de regresión lineal.")

# Cargar los datos
data = load_data()

# Obtener la lista de países (primer columna)
countries = data['Country Name'].unique()

# Crear un widget para seleccionar el país
country_selected = st.selectbox("Selecciona un país:", countries)

# Filtrar los datos por el país seleccionado
country_data = data[data['Country Name'] == country_selected].T
country_data = country_data[1:].reset_index()
country_data.columns = ['Year', 'CO2 Emissions']
country_data['Year'] = country_data['Year'].astype(int)
country_data['CO2 Emissions'] = country_data['CO2 Emissions'].astype(float)

# Mostrar el dataframe filtrado (opcional)
st.write(f"Datos de emisiones de CO2 para {country_selected}:")
st.dataframe(country_data)

# Preparar los datos para la regresión lineal
X = country_data['Year'].values.reshape(-1, 1)
y = country_data['CO2 Emissions'].values

# Crear y entrenar el modelo de regresión lineal
model = LinearRegression()
model.fit(X, y)

# Predecir valores y calcular R²
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)

# Calcular la ecuación de la recta (y = mx + b)
m = model.coef_[0]
b = model.intercept_

# Mostrar la ecuación de la recta y el valor de R²
st.write(f"Ecuación de la recta: y = {m:.2f}x + {b:.2f}")
st.write(f"Valor de R²: {r2:.2f}")

# Graficar las emisiones de CO2 y la regresión lineal
plt.figure(figsize=(10, 6))
plt.scatter(country_data['Year'], country_data['CO2 Emissions'], color='blue', label='Datos reales')
plt.plot(country_data['Year'], y_pred, color='red', label='Regresión lineal')
plt.xlabel('Año')
plt.ylabel('Emisiones de CO2 (kt)')
plt.title(f'Emisiones de CO2 en {country_selected}')
plt.legend()
st.pyplot(plt)
