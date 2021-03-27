# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 22:59:00 2021

@author: unlu
"""

import pandas as pd
from funciones_dataset import cargar_dataset, get_clases
from funciones_preprocesamiento import preprocesar_correos
from funciones_clasificacion_texto import gridsearch_por_estrategia_representacion
import warnings
warnings.filterwarnings("ignore")

# Defino la cantidad de clases con las que se va a trabajar
CANTIDAD_CLASES = 4

# Cargo el dataset
etiquetas = get_clases()
df_train, df_test, etiquetas = cargar_dataset('https://raw.githubusercontent.com/jumafernandez/clasificacion_correos/main/data/consolidado_jcc/', 'correos-train-80.csv', 'correos-test-20.csv', 'clase', etiquetas, CANTIDAD_CLASES, "Otras Consultas", 'COLAB')

# Se ejecuta el preprocesamiento de correos sobre el campo Consulta de train y test
df_train['Consulta'] = pd.Series(preprocesar_correos(df_train['Consulta']))
df_test['Consulta'] = pd.Series(preprocesar_correos(df_test['Consulta']))

# Defino una lista con los esquemas de representación
estrategias_representacion = ['BASELINE', 'BOW', 'TFIDF', '3-4-NGRAM-CHARS', '1-2-NGRAM-WORDS']

for estrategia in estrategias_representacion:
  # Llamo a la función que realiza el gridsearch por estrategia  
  gridsearch_por_estrategia_representacion(df_train, df_test, estrategia)
