# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression

st.title('Predicción del Precio de Alquiler de Propiedades en Brasil')
st.markdown("""
Esta aplicación permite ingresar los detalles de una propiedad y obtener una predicción del precio total de alquiler utilizando un modelo de Machine Learning entrenado.
""")

# Cargar el modelo entrenado
mejor_modelo = joblib.load('mejor_modelo.joblib')

# Cargar las columnas del modelo
model_columns = joblib.load('model_columns.pkl')

# Crear el formulario
st.sidebar.header('Ingrese los detalles de la propiedad')

# Variables numéricas
area = st.sidebar.number_input('Área (m²)', min_value=10, max_value=1000, value=50)
rooms = st.sidebar.number_input('Número de habitaciones', min_value=1, max_value=10, value=2)
bathroom = st.sidebar.number_input('Número de baños', min_value=1, max_value=10, value=1)
parking_spaces = st.sidebar.number_input('Espacios de estacionamiento', min_value=0, max_value=10, value=1)
floor = st.sidebar.number_input('Piso', min_value=0, max_value=50, value=1)
hoa = st.sidebar.number_input('Cuota de mantenimiento (hoa) (R$)', min_value=0, value=0)
rent_amount = st.sidebar.number_input('Monto de renta (R$)', min_value=0, value=1000)
property_tax = st.sidebar.number_input('Impuesto de propiedad (R$)', min_value=0, value=0)
fire_insurance = st.sidebar.number_input('Seguro contra incendios (R$)', min_value=0, value=0)

# Variables categóricas
city = st.sidebar.selectbox('Ciudad', options=['São Paulo', 'Porto Alegre', 'Rio de Janeiro', 'Campinas', 'Belo Horizonte'])
animal = st.sidebar.selectbox('¿Permite mascotas?', options=['aceptan mascotas', 'no aceptan mascotas'])
furniture = st.sidebar.selectbox('¿Está amueblado?', options=['amueblado', 'no amueblado'])

# Botón para predecir
if st.sidebar.button('Predecir Precio de Alquiler'):
    # Validaciones básicas
    errores = []
    if area <= 0:
        errores.append('El área debe ser mayor que cero.')
    if rooms <= 0:
        errores.append('El número de habitaciones debe ser mayor que cero.')
    if bathroom <= 0:
        errores.append('El número de baños debe ser mayor que cero.')
    if rent_amount <= 0:
        errores.append('El monto de renta debe ser mayor que cero.')
    
    if errores:
        for error in errores:
            st.error(error)
    else:
        try:
            # Crear un DataFrame con los datos ingresados
            data = {
                'area': [area],
                'rooms': [rooms],
                'bathroom': [bathroom],
                'parking spaces': [parking_spaces],
                'floor': [floor],
                'hoa (R$)': [hoa],
                'rent amount (R$)': [rent_amount],
                'property tax (R$)': [property_tax],
                'fire insurance (R$)': [fire_insurance],
                'city': [city],
                'animal': [animal],
                'furniture': [furniture]
            }
            input_df = pd.DataFrame(data)
            
            # Reemplazar valores '-' por 0 en las columnas de tipo objeto
            for column in input_df.columns:
                if input_df[column].dtype == 'object':
                    input_df[column] = input_df[column].apply(lambda x: 0 if '-' in str(x) else x)
            
            # Aplicar one-hot encoding a las variables categóricas
            input_df = pd.get_dummies(input_df, columns=['city', 'animal', 'furniture'])
            
            # Asegurarse de que las columnas del input coinciden con las del modelo
            missing_cols = set(model_columns) - set(input_df.columns)
            for col in missing_cols:
                input_df[col] = 0
            input_df = input_df[model_columns]
            
            # Realizar la predicción
            prediction = mejor_modelo.predict(input_df)
            
            # Mostrar el resultado
            st.success(f'El precio total estimado de alquiler es: R$ {prediction[0]:.2f}')
        except Exception as e:
            st.error(f'Ocurrió un error al realizar la predicción: {e}')
            st.error('Por favor, verifique los datos ingresados e intente nuevamente.')

# Visualizaciones
st.header('Importancia de las Características en el Modelo')

# Obtener las importancias de las características si el modelo lo permite
if hasattr(mejor_modelo, 'feature_importances_'):
    importances = mejor_modelo.feature_importances_
    features = model_columns
    feature_importance = pd.DataFrame({'Característica': features, 'Importancia': importances})
    feature_importance = feature_importance.sort_values(by='Importancia', ascending=False)
    
    # Gráfico de barras
    plt.figure(figsize=(10,6))
    sns.barplot(x='Importancia', y='Característica', data=feature_importance)
    plt.title('Importancia de las Características')
    st.pyplot(plt)
else:
    st.write('El modelo no proporciona información sobre la importancia de las características.')

st.header('Tendencias de Alquiler en Diferentes Ciudades')

# Cargar los datos originales
df = pd.read_csv('houses_to_rent_v2.csv')

# Convertir columnas numéricas si es necesario
numeric_columns = ['total (R$)']
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Gráfico interactivo con Plotly
import plotly.express as px

# Filtrar datos si es necesario
city_avg = df.groupby('city')['total (R$)'].mean().reset_index()

fig = px.bar(city_avg, x='city', y='total (R$)', title='Promedio de Precio Total de Alquiler por Ciudad')
st.plotly_chart(fig)

# Mensajes informativos
st.info('Ingrese los datos de la propiedad en el formulario de la izquierda y presione "Predecir Precio de Alquiler".')

# Agregar una barra lateral adicional
st.sidebar.header('Acerca de')
st.sidebar.markdown("""
Esta aplicación utiliza un modelo de Machine Learning para predecir el precio total de alquiler de una propiedad en Brasil basado en las características ingresadas.
""")
