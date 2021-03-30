# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 08:48:33 2021

@author: unlu
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 02:42:57 2021

@author: unlu
"""

from funciones_dataset import get_clases, cargar_dataset
from funciones_preprocesamiento import get_max_length
import pathlib
import warnings
warnings.filterwarnings("ignore")

# Cantidad de clases
CANTIDAD_CLASES = 4

# Cargo el dataset
etiquetas = get_clases()
path_file = str(pathlib.Path().absolute())
train_df, test_df, etiquetas = cargar_dataset('https://raw.githubusercontent.com/jumafernandez/clasificacion_correos/main/data/consolidado_jcc/', 'correos-train-80.csv', 'correos-test-20.csv', path_file, 'clase', etiquetas, CANTIDAD_CLASES, 'Otras Consultas', 'COLAB')

train_df = train_df[['Consulta', 'clase']]
train_df.columns = ['sentence', 'label']
test_df = test_df[['Consulta', 'clase']]
test_df.columns = ['sentence', 'label']

# Cambio los integers por las etiquetas
train_df.label = etiquetas[train_df.label]
test_df.label = etiquetas[test_df.label]

# Las vuelvo a pasar a números 0-N para evitar conflictos con simpletransformers
# Este paso está fijo para estos experimentos
dict_clases_id = {'Otras Consultas': 0,
                            'Ingreso a la Universidad': 1,
                            'Boleto Universitario': 2,
                            'Requisitos de Ingreso': 3}

train_df['label'].replace(dict_clases_id, inplace=True)
test_df['label'].replace(dict_clases_id, inplace=True)

# Muestro salida por consola
print('Existen {} clases: {}, que representan a {}.'.format(len(train_df.label.unique()), train_df.label.unique(), etiquetas[train_df.label.unique()]))

sentences_train = train_df['sentence'].values
sentences_test = test_df['sentence'].values
y_train = train_df['label'].values
sentences_test = test_df['sentence'].values
y_test = test_df['label'].values

#################################################################
##### Comenzamos con las instrucciones del preprocesamiento #####

# Tokenizamos los textos y los pasamos a indices
from keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(num_words=2500)
tokenizer.fit_on_texts(sentences_train)

X_train = tokenizer.texts_to_sequences(sentences_train)
X_test = tokenizer.texts_to_sequences(sentences_test)

# Calculamos el tamaño del vocabulario (se reserva el índice 0)
vocab_size = len(tokenizer.word_index) + 1
print('El tamaño del vocabulario es: {}.'.format(vocab_size))

# Prueba de escritorio con las diferencias entre:
# una consulta raw:
print('Ejemplo de consulta: {}'.format(sentences_train[0]))
# y otra tokenizada:
print('Ejemplo de consulta tokenizada: {}'.format(X_train[0]))

# Aquí se realiza el padding, se unifica el tamaño de todas las consultas
maxlen = get_max_length(sentences_train)
print('La cantidad máxima de tokens encontrada en una consulta es {}.'.format(maxlen))

# Se realiza el padding en función de maxlen
from keras.preprocessing.sequence import pad_sequences
X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

print('Ejemplo de consulta con padding:\n {}'.format(X_train[0, :]))

################################################################
###### Comenzamos con las instrucciones de los embeddings ######

# Genero la matriz de embeddings en función del vocabulario
from funciones_preprocesamiento import create_embedding_matrix

# Embeddings a utilizar en este modelo
EMBEDDINGS_DIM = 300
EMBEDDINGS = 'SBW-vectors-300-min5.bin.gz'
URL_EMBEDDINGS = 'http://cs.famaf.unc.edu.ar/~ccardellino/SBWCE/'+EMBEDDINGS

# Inicializamos los embeddings a utilizar en este modelo (dim, nombre, url y path de descarga)
EMBEDDINGS_DIM = 300
EMBEDDINGS = 'SBW-vectors-300-min5.bin.gz'
URL_EMBEDDINGS = 'http://cs.famaf.unc.edu.ar/~ccardellino/SBWCE/'+EMBEDDINGS
EMBEDDINGS_PATH = 'embeddings/'+EMBEDDINGS

# Los descargo solo en caso que no lo haya hecho (Size =~ 1GB)
import pathlib
from os import path
import wget
path_file = str(pathlib.Path().absolute())

if not(path.exists(EMBEDDINGS_PATH)):
  print('Los embeddings {} no existen. Se procede a la descarga.'.format(EMBEDDINGS))
  wget.download(URL_EMBEDDINGS, path_file+'\\'+EMBEDDINGS_PATH)

# Inicializamos los embeddings
import gensim
embeddings_model = gensim.models.KeyedVectors.load_word2vec_format(EMBEDDINGS_PATH, binary=True)

# Ejemplo de obtención de un vector de embeddings
# vector = embeddings['London']
# print(vector)
       
embedding_matrix = create_embedding_matrix(embeddings_model, tokenizer.word_index, EMBEDDINGS_DIM)

import numpy as np
nonzero_elements = np.count_nonzero(np.count_nonzero(embedding_matrix, axis=1))
print('Existen embeddings para el {}% de las palabras del vocabulario.'.format(nonzero_elements/vocab_size*100))

##################################################################
######## Comenzamos con las instrucciones para el modelo #########
from keras.models import Sequential
from keras import layers
from keras.initializers import Constant

# Se le aplican las capas al modelo
model = Sequential()
model.add(layers.Embedding(vocab_size, 
                           EMBEDDINGS_DIM, 
                           embeddings_initializer=Constant(embedding_matrix),
                           input_length=maxlen, 
                           trainable=False))
model.add(layers.LSTM(EMBEDDINGS_DIM))
model.add(layers.Dropout(0.1))
model.add(layers.Dense(CANTIDAD_CLASES, activation='softmax'))


model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()

EPOCAS = 10
BATCH_SIZE = 32

lstm = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCAS)

score, accuracy = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE)

print('El score es {} y el accuracy {} :'.format(score, accuracy))

##################################################################
######## Comenzamos con las instrucciones de evaluación ##########

# predictions = ACA VA EL MODEL.PREDICTIONS

# Calculo las métricas sobre test para el paper
# acc_test = accuracy_score(test_df.labels, predictions)
# precision_test = precision_score(test_df.labels, predictions, average='macro')
# recall_test = recall_score(test_df.labels, predictions, average='macro')
# f1_test = f1_score(test_df.labels, predictions, average='macro')

# Genero un diccionario con los parámetro y el acc en test
# dict_test = {}
# dict_test['clasificador'] = 'BETO'
# dict_test['accuracy'] = acc_test
# dict_test['precision'] = precision_test
# dict_test['recall'] = recall_test
# dict_test['f1_score'] = f1_test
 
# Paso el diccionario a dataframe y lo guardo en un archivo con fecha/hora
# results_test = pd.DataFrame([dict_test])

# from datetime import datetime
# now = datetime.now()
# nombre_results_test = 'resultados/results_test-'+ str(now) + '.csv'
# results_test.to_csv(nombre_results_test, mode='w')
# print(results_test)
