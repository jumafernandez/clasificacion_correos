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


def preprocesar_correo(correo):
  '''
  Se eliminan las stopwords del texto del correo
  '''
  import unicodedata
  import nltk 
  nltk.download('stopwords', quiet=True)
  nltk.download('punkt', quiet=True)
  from nltk.corpus import stopwords

  texto = limpiar_texto(correo) 

  tokens = texto.split(' ')
  tkns_limpios = [] 
       
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
    correo_limpio = preprocesar_correo(correo)
    correos_limpios.append(correo_limpio)
    
  return correos_limpios

