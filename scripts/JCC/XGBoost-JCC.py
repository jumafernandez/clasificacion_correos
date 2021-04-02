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

# Se renombran las etiquetas
train_df.clase = etiquetas[train_df.clase]
test_df.clase = etiquetas[test_df.clase]

# Las vuelvo a pasar a números 0-N para evitar conflictos con simpletransformers
# Este paso está fijo para estos experimentos
dict_clases_id = {'Otras Consultas': 0,
                            'Ingreso a la Universidad': 1,
                            'Boleto Universitario': 2,
                            'Requisitos de Ingreso': 3}

train_df['clase'].replace(dict_clases_id, inplace=True)
test_df['clase'].replace(dict_clases_id, inplace=True)

# Se ejecuta el preprocesamiento de correos sobre el campo Consulta de train y test
train_df['Consulta'] = pd.Series(preprocesar_correos(train_df['Consulta']))
test_df['Consulta'] = pd.Series(preprocesar_correos(test_df['Consulta']))

# Defino una lista con los esquemas de representación
estrategias_representacion = ['BASELINE', 'BOW', 'TFIDF', '3-4-NGRAM-CHARS', '1-2-NGRAM-WORDS']
tecnica_aprendizaje = 'XGBoost'

# A parameter grid for XGBoost
params_xg = {
          'XGBoost__min_child_weight': [1, 5, 10],
          'XGBoost__gamma': [0.5, 1, 1.5, 2, 5],
          'XGBoost__subsample': [0.6, 0.8, 1.0],
          'XGBoost__colsample_bytree': [0.6, 0.8, 1.0],
          'XGBoost__max_depth': [3, 4, 5],
          'XGBoost__eval_metric': ['merror'],
          'XGBoost__use_label_encoder': [False]
        }


for estrategia in estrategias_representacion:
  # Llamo a la función que realiza el gridsearch por estrategia  
  gridsearch_por_estrategia_representacion(train_df, test_df, estrategia, tecnica_aprendizaje, params_xg)
