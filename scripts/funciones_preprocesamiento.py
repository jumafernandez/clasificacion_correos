# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 11:25:52 2021

@author: unlu
"""

# Aqui están las funciones que trabajan sobre el preprocesamiento de los correos

def limpiar_texto(texto): # Estrategia tomada de UNSL (https://github.com/hjthomp/tf-thompson)
  '''
  toma un texto y lo devuelve limpio (pasa a minúsculas, elimina símbolos, 
  dobles ocurrencias de letras, palabras y espacios).
  '''
  import re
  import unicodedata

  # Pasamos todo a minúsculas
  texto = texto.lower()
  # Se Elimina TODO menos: \w = alphanum y ¿?%/
  texto = re.sub(r'[^\w %/]', " ", texto)   
    
  # Elimina acentos
  texto = ''.join(c for c in unicodedata.normalize('NFD', texto) if unicodedata.category(c) != 'Mn') 

  #Remover letras con 2 ocurrencias (con excepciones)      
  letras_dobles = "abdfghijkmnñpqstuvwxyz" # Excepciones: ee-cc-ll-rr-oo (y mayus)
  letras_dobles += letras_dobles.upper()          
  texto = re.sub("(?P<char>[" + re.escape(letras_dobles) + "])(?P=char)+", r"\1", texto) 

  #remover caracteres que se repiten al menos 3 veces
  texto = re.sub(r'([\w\W])\1{2,}', r'\1', texto) 

  #remover palabras que se repiten 
  texto = re.sub(r'\b(\w+)(\b\W+\b\1\b)*', r'\1', texto) 

  #Eliminar repetición de espacios
  texto = re.sub(r"\s{2,}", " ", texto) 

  return texto


def preprocesar_correo(correo, remove_stopwords):
  '''
  Se eliminan las stopwords del texto del correo
  '''
  import unicodedata

  texto = limpiar_texto(correo) 

  tokens = texto.split(' ')
  tkns_limpios = [] 
  stopwords = []
  
  if remove_stopwords:
    import nltk 
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    from nltk.corpus import stopwords
    stop_words_set = set(stopwords.words('spanish')) 
  
    #Eliminar acentos de stopwords
    stopwords_final = []
    for st in list(stop_words_set):
      stopwords_final.append(''.join(c for c in unicodedata.normalize('NFD', st) if unicodedata.category(c) != 'Mn'))        
    stopwords = set(stopwords_final)

  for tk in tokens: 
    if tk not in stopwords:
      if tk.count('/')>0 or tk.count('%')>0 or tk.isdigit() or (tk.isalpha() and (len(tk)>1)): 
        tkns_limpios.append(tk) 
        
  texto_preprocesado = ' '.join(tkns_limpios)
  
  return texto_preprocesado


def preprocesar_correos(correos):
  '''
  Esta función toma los correos y los va preprocesando uno a uno para devolverlos
  '''
  correos_limpios = []
  for correo in correos:
    correo_limpio = preprocesar_correo(correo, True)
    correos_limpios.append(correo_limpio)
    
  return correos_limpios

def preprocesar_correos_bert(correos):
  '''
  Esta función toma los correos y los va preprocesando uno a uno para devolverlos
  '''
  correos_limpios = []
  for correo in correos:
    correo_limpio = preprocesar_correo(correo, False)
    correos_limpios.append(correo_limpio)
    
  return correos_limpios

def get_max_length(text):
  """
  devuelve la cantidad máxima de tokens para las consultas que se le pasan.
  Esta acción es importante para el padding en LSTM.
  """
  max_length = 0
  for row in text:
    if len(row.split(" ")) > max_length:
      max_length = len(row.split(" "))
  return max_length



def create_embedding_matrix(embeddings_model, word_index, embedding_dim):
  '''
  Esta función crea la matriz de embeddings para el modelo en función
  del vocabulario de consultas, omitiendo el resto de los términos
  '''
  import numpy as np

  vocab_size = len(word_index) + 1  # Se incorpora 1 porque se reserva el 0
  embedding_matrix = np.zeros((vocab_size, embedding_dim))

  for word, idx in word_index.items():
    try:
      vector = embeddings_model[word]
      if not vector is None:
        embedding_matrix[idx] = vector
    except:
      pass

  return embedding_matrix

def generar_train_test_set(train, test, estrategia, atr_consulta='consulta', atr_clase='clase'):
  from funciones_dataset import consolidar_df
  from funciones_clasificacion_texto import representacion_documentos
  
  from sklearn.preprocessing import MinMaxScaler
  import pandas as pd

  # Esta función va dentro de un iterador entre las 5 estrategias    
  print('Estrategia de representación: {}' . format(estrategia))
  correos_train_vec, correos_test_vec = representacion_documentos(train[atr_consulta], test[atr_consulta], estrategia, None)

  # Separo en x e y - train y test- (además consolido feature estáticas con dinámicas)
  x_train, y_train = consolidar_df(train, correos_train_vec, atr_consulta, atr_clase)
  x_test, y_test = consolidar_df(test, correos_test_vec, atr_consulta, atr_clase)
  
  # Escalado de datos: Se probó scale y MinMaxScaler y dió mejores resultados el último
  scaler = MinMaxScaler()
  x_train_scaled = scaler.fit_transform(x_train)
  x_test_scaled = scaler.fit_transform(x_test)

  return x_train_scaled, y_train, x_test_scaled, y_test
