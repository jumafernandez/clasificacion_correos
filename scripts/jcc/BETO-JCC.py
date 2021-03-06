# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 02:42:57 2021

@author: unlu
"""

from funciones_dataset import get_clases, cargar_dataset
from os import path
import warnings
warnings.filterwarnings("ignore")

# Cantidad de clases
CANTIDAD_CLASES = 4

# Constantes con los datos
DS_DIR = 'https://raw.githubusercontent.com/jumafernandez/clasificacion_correos/main/data/consolidado_jcc/'
TRAIN_FILE = 'correos-train-80.csv'
TEST_FILE = 'correos-test-20.csv'

# Chequeo sobre si los archivos están en el working directory
download_files = not(path.exists(TRAIN_FILE))

etiquetas = get_clases()
train_df, test_df, etiquetas = cargar_dataset(DS_DIR, TRAIN_FILE, TEST_FILE, download_files, 'clase', etiquetas, CANTIDAD_CLASES, 'Otras Consultas')

train_df = train_df[['Consulta', 'clase']]
train_df.columns = ['text', 'labels']
test_df = test_df[['Consulta', 'clase']]
test_df.columns = ['text', 'labels']

# Preprocesamiento del texto
from funciones_preprocesamiento import preprocesar_correos_bert

# Se ejecuta el preprocesamiento de correos sobre el campo Consulta de train y test
# En caso de Bert es solo normalización sin eliminar palabras vacías
import pandas as pd
train_df['Consulta'] = pd.Series(preprocesar_correos_bert(train_df['text']))
test_df['Consulta'] = pd.Series(preprocesar_correos_bert(test_df['text']))

# Cambio los integers por las etiquetas
train_df.labels = etiquetas[train_df.labels]
test_df.labels = etiquetas[test_df.labels]


# Las vuelvo a pasar a números 0-N para evitar conflictos con simpletransformers
# Este paso está fijo para estos experimentos
dict_clases_id = {'Otras Consultas': 0,
                            'Ingreso a la Universidad': 1,
                            'Boleto Universitario': 2,
                            'Requisitos de Ingreso': 3}

train_df['labels'].replace(dict_clases_id, inplace=True)
test_df['labels'].replace(dict_clases_id, inplace=True)

# Muestro salida por consola
print('Existen {} clases: {}.'.format(len(train_df.labels.unique()), train_df.labels.unique()))

##################################################################
########## Comenzamos con las instrucciones del modelo ###########

from simpletransformers.classification import ClassificationModel

# Cantidad de epochs
epocas = 4

# Hiperparámetros
train_args = {
        'overwrite_output_dir': True,
        'num_train_epochs': epocas,
        'fp16': True,
        'learning_rate': 4e-5,
        'do_lower_case': True,
        'use_early_stopping': True,
        
        }

# Creamos el ClassificationModel
model = ClassificationModel(
    model_type='bert', 
    model_name='bert-base-multilingual-cased',
#    model_name='dccuchile/bert-base-spanish-wwm-cased',
    num_labels=CANTIDAD_CLASES,
    use_cuda=False,
    args=train_args
)

# Entrenamos el modelo
model.train_model(train_df)

# Evaluamos el modelo
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Ejecutamos las predicciones sobre testing
predictions, raw_outputs = model.predict(list(test_df.text))

# Calculo las métricas sobre test para el paper
acc_test = accuracy_score(test_df.labels, predictions)
precision_test = precision_score(test_df.labels, predictions, average='macro')
recall_test = recall_score(test_df.labels, predictions, average='macro')
f1_test = f1_score(test_df.labels, predictions, average='macro')

# Genero un diccionario con los parámetro y el acc en test
dict_test = {}
dict_test['clasificador'] = 'BETO'
dict_test['accuracy'] = acc_test
dict_test['precision'] = precision_test
dict_test['recall'] = recall_test
dict_test['f1_score'] = f1_test
 
# Paso el diccionario a dataframe y lo guardo en un archivo con fecha/hora
results_test = pd.DataFrame([dict_test])
print(results_test)

# Lo guardo en un archivo
from datetime import datetime
now = datetime.now()
nombre_results_test = 'resultados/results_test-'+ str(now) + '.csv'
results_test.to_csv(nombre_results_test, mode='w')
