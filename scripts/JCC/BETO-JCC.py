# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 02:42:57 2021

@author: unlu
"""
def get_clases():
  '''
  Esta función retorna las etiquetas de las clases sobre el total de los correos. 
  Tomado de notebooks/jcc/de train_test_data.ipynb
  '''
  import numpy as np

  etiquetas_clases = np.array(['Boleto Universitario', 
                               'Cambio de Carrera', 
                               'Cambio de Comisión',
                               'Carga de Notas', 
                               'Certificados Web', 
                               'Consulta por Equivalencias',
                               'Consulta por Legajo', 
                               'Consulta sobre Título Universitario',
                               'Cursadas', 
                               'Datos Personales', 
                               'Exámenes',
                               'Ingreso a la Universidad',
                               'Inscripción a Cursadas',
                               'Pedido de Certificados',
                               'Problemas con la Clave',
                               'Reincorporación', 
                               'Requisitos de Ingreso',
                               'Simultaneidad de Carreras', 
                               'Situación Académica',
                               'Vacunas Enfermería'])

  return etiquetas_clases

def cargar_dataset(URL_data, file_train, file_test, nombre_clase, class_labels, cantidad_clases, texto_otras, origen_ds):
  '''
  Carga los train y test set y genera la reducción de clases, en caso que sea necesario
  '''
  import pandas as pd
  import numpy as np
  import warnings
  warnings.filterwarnings("ignore")

  # Genero el enlace completo
  URL_file_train = URL_data + file_train
  URL_file_test = URL_data + file_test
  
  # Me traigo los archivos de train y test
  if origen_ds == 'COLAB':  
    import wget
    wget.download(URL_file_train)
    wget.download(URL_file_test)
  else:
    file_train = URL_file_train
    file_test = URL_file_test
    
  # Leemos el archivo en un dataframe
  df_train = pd.read_csv(file_train)
  df_test = pd.read_csv(file_test)

  # Agrupamiento de clases
  # Se realiza un conteo de frecuencia por clase y se toman los correos que pertenecen a 
  # las N-cantidad_clases menos observadas
  clases = df_train.clase.value_counts()
  clases_minoritarias = clases.iloc[cantidad_clases-1:].keys().to_list()

  # Agrego a las etiquetas la etiqueta "Otras Consultas" para el agrupamiento
  etiquetas_clases = np.append(class_labels, texto_otras)

  # Genero una nueva clave de clases para "Otras Consultas" a modo de agrupar las que poseen menos apariciones
  df_train.clase[df_train[nombre_clase].isin(clases_minoritarias)] = np.where(etiquetas_clases == texto_otras)[0]
  df_test.clase[df_test[nombre_clase].isin(clases_minoritarias)] = np.where(etiquetas_clases == texto_otras)[0]

  print("\nEl conjunto de entrenamiento tiene la dimensión: " + str(df_train.shape))
  print("El conjunto de testeo tiene la dimensión: " + str(df_test.shape))

  return df_train, df_test, etiquetas_clases

import warnings
warnings.filterwarnings("ignore")

# Cantidad de clases
CANTIDAD_CLASES = 4

# Cargo el dataset
etiquetas = get_clases()
train_df, test_df, etiquetas = cargar_dataset('https://raw.githubusercontent.com/jumafernandez/clasificacion_correos/main/data/consolidado_jcc/', 'correos-train-80.csv', 'correos-test-20.csv', 'clase', etiquetas, CANTIDAD_CLASES, "Otras Consultas", 'COLAB')

train_df = train_df[['Consulta', 'clase']]
train_df.columns = ['text', 'labels']
test_df = test_df[['Consulta', 'clase']]
test_df.columns = ['text', 'labels']

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
    'bert', 'dccuchile/bert-base-spanish-wwm-cased',
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

from datetime import datetime
now = datetime.now()
nombre_results_test = 'resultados/results_test.csv'+ str(now)
results_test.to_csv(nombre_results_test, mode='w')
print(results_test)
