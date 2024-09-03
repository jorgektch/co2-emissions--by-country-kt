import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

# Cargar los datos del archivo Excel
@st.cache_data
def load_data():
    file_path = 'co2-emissions-by-country-kt.xlsx'
    data = pd.read_excel(file_path)
    return data

# Configuración inicial de Streamlit
st.title("Análisis de Emisiones de CO2 por país (1960 - 2016)")
st.write("Seleccione un país para visualizar las emisiones de CO2 y comparar diferentes modelos de regresión.")

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

# Preparar los datos para el análisis
X = country_data['Year'].values.reshape(-1, 1)
y = country_data['CO2 Emissions'].values

# Dividir los datos en entrenamiento (80%) y test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelos de regresión
models = {}

# Regresión Lineal
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)
r2_linear = r2_score(y_test, y_pred_linear)
mse_linear = mean_squared_error(y_test, y_pred_linear)
rmse_linear = np.sqrt(mse_linear)
models['Lineal'] = {'model': linear_model, 'r2': r2_linear, 'mse': mse_linear, 'rmse': rmse_linear}

# Regresión Polinomial (Grado 2)
poly = PolynomialFeatures(degree=2)
X_poly_train = poly.fit_transform(X_train)
X_poly_test = poly.transform(X_test)
poly_model = LinearRegression()
poly_model.fit(X_poly_train, y_train)
y_pred_poly = poly_model.predict(X_poly_test)
r2_poly = r2_score(y_test, y_pred_poly)
mse_poly = mean_squared_error(y_test, y_pred_poly)
rmse_poly = np.sqrt(mse_poly)
models['Polinomial'] = {'model': poly_model, 'r2': r2_poly, 'mse': mse_poly, 'rmse': rmse_poly}

# Regresión Logística
logistic_model = LogisticRegression(max_iter=10000)
logistic_model.fit(X_train, y_train.astype(int))  # Convertir a enteros para la regresión logística
y_pred_logistic = logistic_model.predict(X_test)
r2_logistic = r2_score(y_test.astype(int), y_pred_logistic)
mse_logistic = mean_squared_error(y_test, y_pred_logistic)
rmse_logistic = np.sqrt(mse_logistic)
models['Logística'] = {'model': logistic_model, 'r2': r2_logistic, 'mse': mse_logistic, 'rmse': rmse_logistic}

# Seleccionar el mejor modelo basado en R²
best_model_name = max(models, key=lambda x: models[x]['r2'])
best_model = models[best_model_name]

# Mostrar resultados
st.write(f"**Mejor modelo:** {best_model_name}")
st.write(f"R²: {best_model['r2']:.2f}")
st.write(f"MSE: {best_model['mse']:.2f}")
st.write(f"RMSE: {best_model['rmse']:.2f}")

# Graficar las emisiones de CO2 y la predicción del mejor modelo
plt.figure(figsize=(10, 6))
plt.scatter(country_data['Year'], country_data['CO2 Emissions'], color='blue', label='Datos reales')

if best_model_name == 'Polinomial':
    plt.plot(X_test, best_model['model'].predict(poly.transform(X_test)), color='red', label='Mejor modelo (Polinomial)')
else:
    plt.plot(X_test, best_model['model'].predict(X_test), color='red', label=f'Mejor modelo ({best_model_name})')

plt.xlabel('Año')
plt.ylabel('Emisiones de CO2 (kt)')
plt.title(f'Emisiones de CO2 en {country_selected}')
plt.legend()
st.pyplot(plt)
