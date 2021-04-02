# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 22:59:00 2021

@author: unlu
"""

import pandas as pd
import pathlib
from funciones_dataset import cargar_dataset, get_clases
from funciones_preprocesamiento import preprocesar_correos
from funciones_clasificacion_texto import gridsearch_por_estrategia_representacion
import warnings
warnings.filterwarnings("ignore")

# Defino la cantidad de clases con las que se va a trabajar
CANTIDAD_CLASES = 4

# Cargo el dataset
etiquetas = get_clases()
path_file = str(pathlib.Path().absolute())
train_df, test_df, etiquetas = cargar_dataset('https://raw.githubusercontent.com/jumafernandez/clasificacion_correos/main/data/consolidado_jcc/', 'correos-train-80.csv', 'correos-test-20.csv', path_file, 'clase', etiquetas, CANTIDAD_CLASES, 'Otras Consultas', 'COLAB')

# Se ejecuta el preprocesamiento de correos sobre el campo Consulta de train y test
train_df['Consulta'] = pd.Series(preprocesar_correos(train_df['Consulta']))
test_df['Consulta'] = pd.Series(preprocesar_correos(test_df['Consulta']))

# Defino una lista con los esquemas de representación
estrategias_representacion = ['BASELINE', 'BOW', 'TFIDF', '3-4-NGRAM-CHARS', '1-2-NGRAM-WORDS']
modelo = 'SVM'

# Defino los parámetros para GridSearchCV
params_svm = {'SVM__C': [0.01, 0.1, 1, 10, 100, 1000], 
              'SVM__gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'SVM__class_weight': [None, 'balanced'],
              'SVM__kernel': ['rbf', 'linear', 'poly', 'sigmoid']
              }

for estrategia in estrategias_representacion:
  # Llamo a la función que realiza el gridsearch por estrategia  
  gridsearch_por_estrategia_representacion(train_df, test_df, estrategia, modelo, params_svm)
