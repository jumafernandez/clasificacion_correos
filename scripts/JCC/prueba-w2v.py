# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 10:52:35 2021

@author: unlu
"""

# Embeddings a utilizar en este modelo
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

import gensim
from gensim.models import KeyedVectors
 
embeddings = gensim.models.KeyedVectors.load_word2vec_format(EMBEDDINGS_PATH, binary=True)

model = KeyedVectors.load_word2vec_format(EMBEDDINGS_PATH, binary=True)

#Obtener el vector de palabras de Google News Corpus
vector = model['London']
print(vector)