# -*- coding: utf-8 -*-
"""
Created on Sun Jun 20 00:01:45 2021

@author: unlu
"""

# -*- coding: utf-8 -*-

# Cargo la librería con la función propia para la consulta a Elastic
from elasticsearch import Elasticsearch
from functions_elastic import terms2df_lr
from os import listdir
import pandas as pd

# Genero la conexión al Cluster
es = Elasticsearch([{'host': 'localhost', 'port': 9200}])

DIRECTORIO_TERMINOS = 'C:/Users/unlu/Desktop/JAIIO50/etiquetado_jaiio/features/txts_lr'
DIRECTORIO_DESTINO = 'C:/Users/unlu/Documents/GitHub/jumafernandez/clasificacion_correos/data/50jaiio/consolidados/feature-extraction/'
INSTANCIAS = 200
BOOSTING = True

# Creamos un dataframe para ir guardando las instancias
dataset = pd.DataFrame()

for archivo in listdir(DIRECTORIO_TERMINOS):
    
    # Tomo la clase del nombre del archivo
    clase = archivo.split('.txt')[0]

    # Levanto cada archivo
    ubicacion_file = DIRECTORIO_TERMINOS + '/' + archivo  
    file_terms = open(ubicacion_file)
    
    # Proceso los términos del archivo
    terminos_pos = ''
    terminos_neg = ''
    for linea in file_terms:       
        idx, termino, score, clase_pos_neg = linea.split(',')
        
        if clase_pos_neg.strip() == 'positivo':
            if BOOSTING:
                # Boosting de términos
                # https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl-query-string-query.html
                terminos_pos += termino + '^' + str(round(1 + float(score), 2)) + ' '
            else:
                terminos_pos += termino + ' '

        elif clase_pos_neg.strip() == 'negativo':
            terminos_neg += termino
            
    terminos_pos = terminos_pos.strip()
    terminos_neg = terminos_neg.strip()
    
    # print(terminos_pos)
    
    df_clase = terms2df_lr(es, 'correos_jaiio', terminos_pos, terminos_neg, INSTANCIAS)
    df_clase['clase'] = clase
    
    dataset = pd.concat([dataset, df_clase])

    # column_names = ['score', 'clase', 'consulta']
    # dataset = dataset.reindex(columns=column_names)

# Se formatean las opciones para que aparezcan en el nombre del archivo
if not(INSTANCIAS):
    INSTANCIAS = 'ilimitado'
    
if BOOSTING:
    BOOSTING = '-boosting'
else:
    BOOSTING = ''
# Se guarda el csv
dataset.to_csv(f'{DIRECTORIO_DESTINO}dataset-lr-{INSTANCIAS}{BOOSTING}.csv', index=False)