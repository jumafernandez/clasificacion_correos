# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 20:50:16 2021

@author: unlu
"""

# Cargo la librería con la función propia para la consulta a Elastic
from elasticsearch import Elasticsearch
from functions_elastic import terms2df_tfidf_ss3, limpiar_clase_ss3
from os import listdir
import pandas as pd

# Genero la conexión al Cluster
es = Elasticsearch([{'host': 'localhost', 'port': 9200}])

DIRECTORIO_TERMINOS = 'C:/Users/unlu/Desktop/JAIIO50/etiquetado_jaiio/features/txts_ss3'
INSTANCIAS = 200
BOOSTING = True

# Creamos un dataframe para ir guardando las instancias
dataset = pd.DataFrame()

for archivo in listdir(DIRECTORIO_TERMINOS):

    # Tomo la clase del nombre del archivo
    clase = limpiar_clase_ss3(archivo.split('.csv')[0])

    # Levanto cada archivo
    ubicacion_file = DIRECTORIO_TERMINOS + '/' + archivo  
    file_terms = open(ubicacion_file)
    
    # Proceso los términos del archivo
    terminos = ''
    cantidad_terminos = 0
    for linea in file_terms:
        if cantidad_terminos < 20:
            terminos += linea.split(',')[0] + ' '
            cantidad_terminos+=1
        else:
            break
    
    terminos = terminos.strip()

    df_clase = terms2df_tfidf_ss3(es, 'correos_jaiio', terminos, INSTANCIAS)
    df_clase['clase'] = clase
    
    dataset = pd.concat([dataset, df_clase])

DIRECTORIO = 'C:/Users/unlu/Documents/GitHub/jumafernandez/clasificacion_correos/data/50jaiio/consolidados/feature-extraction/'
dataset.to_csv(DIRECTORIO + 'dataset-ss3-' + str(INSTANCIAS) + '.csv', index=False)

