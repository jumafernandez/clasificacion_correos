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

# Creamos un dataframe para ir guardando las instancias
dataset = pd.DataFrame()

print(listdir(DIRECTORIO_TERMINOS))

    # Tomo la clase del nombre del archivo
#    print(archivo)
#    clase = limpiar_clase_ss3(archivo.split('.csv')[0])
#    print(clase)
