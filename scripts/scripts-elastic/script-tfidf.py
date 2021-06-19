# Cargo la librería con la función propia para la consulta a Elastic
from elasticsearch import Elasticsearch
from functions_elastic import terms2df_tfidf
from os import listdir
import pandas as pd

# Genero la conexión al Cluster
es = Elasticsearch([{'host': 'localhost', 'port': 9200}])

DIRECTORIO_TERMINOS = 'C:/Users/unlu/Desktop/JAIIO50/etiquetado_jaiio/features/txts_tfidf'

# Creamos un dataframe para ir guardando las instancias
dataset = pd.DataFrame()

for archivo in listdir(DIRECTORIO_TERMINOS):

    # Tomo la clase del nombre del archivo
    clase = archivo.split('.txt')[0]

    # Levanto cada archivo
    ubicacion_file = DIRECTORIO_TERMINOS + '/' + archivo  
    file_terms = open(ubicacion_file)
    
    # Proceso los términos del archivo
    terminos = ''
    for linea in file_terms:
        terminos += linea.split(',')[0] + ' '
    
    terminos = terminos.strip()

    df_clase = terms2df_tfidf(es, 'correos_jaiio', terminos, 20)
    df_clase['clase'] = clase
    
    dataset = pd.concat([dataset, df_clase])

DIRECTORIO = 'C:/Users/unlu/Documents/GitHub/jumafernandez/clasificacion_correos/data/50jaiio/consolidados/feature-extraction/'
dataset.to_csv(DIRECTORIO + 'dataset-tfidf.csv', index=False)



