import numpy as np
import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier

st.write(''' # Predicción de temperatura ''')
st.image("Temperatura.jpg", caption="Vamos a intentar predecir la temperatura.")

st.header('Datos de evaluación')

def user_input_features():
  City = st.number_input('Ciudad (0 = Acapulco, 1 = Acuña, 2 = Aguascalientes):', min_value=0.0, max_value=2.0, value = 0.0, step = 1.0)
  Year = st.number_input('Año:', min_value=0.0, max_value=100000.0, value = 0.0, step = 1.0)
  Month = st.number_input('Mes:', min_value=0.0, max_value=100.0, value = 0.0, step = 1.0)

  user_input_data = {'Ciudad (0 = Acapulco, 1 = Acuña, 2 = Aguascalientes)': City,
                     'Año': Year,
                     'Mes': Month}

  features = pd.DataFrame(user_input_data, index=[0])

  return features

df = user_input_features()
datos =  pd.read_csv('Temperatura.csv', encoding='latin-1')
X = datos.drop(columns='AverageTemperature')
y = datos['AverageTemperature']

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1613555)
LR = LinearRegression()
LR.fit(X_train,y_train)

b1 = LR.coef_
b0 = LR.intercept_
prediccion = b0 + b1[0]*df['City'] + b1[1]*df['Year'] + b1[2]*df['Month']

st.subheader('Calculo de Temperatura')
st.write('La temperatura será: ', prediccion)
