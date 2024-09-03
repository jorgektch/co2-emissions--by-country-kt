import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.metrics import r2_score, mean_squared_error
import math

# Cargar los datos del archivo Excel
@st.cache_data
def load_data():
    file_path = '/mnt/data/co2-emissions-by-country-kt.xlsx'
    data = pd.read_excel(file_path)
    return data

# Configuración inicial de Streamlit
st.title("Emisiones de CO2 por país (1960 - 2016)")
st.write("Seleccione un país para visualizar las emisiones de CO2 y realizar el análisis de regresión.")

# Cargar los datos
data = load_data()

# Obtener la lista de países
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

# Preparar los datos para la regresión
X = country_data['Year'].values.reshape(-1, 1)
y = country_data['CO2 Emissions'].values

# Dividir los datos en entrenamiento y prueba (80% - 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Diccionario para almacenar modelos y sus resultados
models = {}
results = {}

# Función para calcular métricas
def calculate_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = math.sqrt(mse)
    return r2, mse, rmse

# 1. Regresión Lineal
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
models['Linear Regression'] = linear_model
results['Linear Regression'] = calculate_metrics(linear_model, X_test, y_test)

# 2. Regresión Polinomial (grado 2)
poly_features = PolynomialFeatures(degree=2)
X_train_poly = poly_features.fit_transform(X_train)
X_test_poly = poly_features.transform(X_test)
poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)
models['Polynomial Regression (degree 2)'] = poly_model
results['Polynomial Regression (degree 2)'] = calculate_metrics(poly_model, X_test_poly, y_test)

# 3. Regresión con Árboles de Decisión
tree_model = DecisionTreeRegressor(random_state=42)
tree_model.fit(X_train, y_train)
models['Decision Tree Regression'] = tree_model
results['Decision Tree Regression'] = calculate_metrics(tree_model, X_test, y_test)

# 4. Bosques Aleatorios
forest_model = RandomForestRegressor(n_estimators=100, random_state=42)
forest_model.fit(X_train, y_train)
models['Random Forest Regression'] = forest_model
results['Random Forest Regression'] = calculate_metrics(forest_model, X_test, y_test)

# 5. Gradient Boosting
gboost_model = GradientBoostingRegressor(random_state=42)
gboost_model.fit(X_train, y_train)
models['Gradient Boosting'] = gboost_model
results['Gradient Boosting'] = calculate_metrics(gboost_model, X_test, y_test)

# 6. AdaBoost
adaboost_model = AdaBoostRegressor(random_state=42)
adaboost_model.fit(X_train, y_train)
models['AdaBoost'] = adaboost_model
results['AdaBoost'] = calculate_metrics(adaboost_model, X_test, y_test)

# Selección del mejor modelo basado en R²
best_model_name = max(results, key=lambda k: results[k][0])
best_model = models[best_model_name]
best_model_metrics = results[best_model_name]

# Mostrar resultados
st.write(f"Mejor modelo: {best_model_name}")
st.write(f"R²: {best_model_metrics[0]:.2f}")
st.write(f"MSE: {best_model_metrics[1]:.2f}")
st.write(f"RMSE: {best_model_metrics[2]:.2f}")

# Graficar los resultados del mejor modelo
if 'Polynomial' in best_model_name:
    y_pred = best_model.predict(poly_features.transform(X))
else:
    y_pred = best_model.predict(X)

plt.figure(figsize=(10, 6))
plt.scatter(country_data['Year'], country_data['CO2 Emissions'], color='blue', label='Datos reales')
plt.plot(country_data['Year'], y_pred, color='red', label=f'{best_model_name}')
plt.xlabel('Año')
plt.ylabel('Emisiones de CO2 (kt)')
plt.title(f'Emisiones de CO2 en {country_selected} - {best_model_name}')
plt.legend()
st.pyplot(plt)
