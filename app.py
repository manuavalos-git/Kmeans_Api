import numpy as np
from flask import Flask, request, jsonify, render_template, url_for
import pickle
from sklearn import svm
import streamlit as st


# Path del modelo preentrenado
MODEL_PATH = 'models/kmeans_model.pkl'


# Se recibe la imagen y el modelo, devuelve la predicción
def model_prediction(x_in, model):

    x = np.asarray(x_in).reshape(1,-1)
    preds=model.predict(x)

    return preds


def main():

    model=''

    # Se carga el modelo
    if model=='':
        with open(MODEL_PATH, 'rb') as file:
            model = pickle.load(file)

    # Título
    html_temp = """
    <h1 style="color:#181082;text-align:center;">SISTEMA DE CLASIFICACION DE RIESGO DE SUICIDIO</h1>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)

    # Lecctura de datos
    #Datos = st.text_input("Ingrese los valores : N P K Temp Hum pH lluvia:")
    N = st.text_input("SUIC RISK:")
    P = st.text_input("PROM SUIC:")

    # El botón predicción se usa para iniciar el procesamiento
    if st.button("Predicción :"):
        #x_in = list(np.float_((Datos.title().split('\t'))))
        x_in =[np.float_(N.title()),
                    np.float_(P.title()),
        ]
        predictS = model_prediction(x_in, model)
        st.success('EL GRUPO DE RIESGO DE SUICIDIO AL QUE PERTENECE ES ES: {}'.format(predictS[0]).upper())

if __name__ == '__main__':
    main()
