# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 02:42:57 2021

@author: jumafernandez
"""

# Me guardo los logs
import logging
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

# Inicializo los parámetros del sweeping
import wandb
sweep_config = {
    "method": "bayes",
    "metric": {"name": "train_loss", "goal": "minimize"},
    "parameters": {
        "num_train_epochs": {"values": [2, 3, 4, 5]},
        "learning_rate": {"min": 4e-5, "max": 4e-4},
    },
}

sweep_id = wandb.sweep(sweep_config, project="Grid_Search_Bert")

from funciones_dataset import get_clases, cargar_dataset
import warnings
warnings.filterwarnings("ignore")

# Cantidad de clases
CANTIDAD_CLASES = 4

# Cargo el dataset
etiquetas = get_clases()
train_df, test_df, etiquetas = cargar_dataset('https://raw.githubusercontent.com/jumafernandez/clasificacion_correos/main/data/consolidado_jcc/', 'correos-train-80.csv', 'correos-test-20.csv', '/home/jmfernandez/', 'clase', etiquetas, CANTIDAD_CLASES, 'Otras Consultas', 'COLAB')

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
        'reprocess_input_data': True,
        'evaluate_during_training': True,
        'manual_seed':4,
        'fp16': True,
        'do_lower_case': True,
        'use_early_stopping': True,
        'wandb_project': "Grid_Search_Bert"
        }

def train():
  # Initialize a new wandb run
  wandb.init()

  # Creamos el ClassificationModel
  model = ClassificationModel(
    #'bert', 'bert-base-multilingual-cased',
    model_type='bert',
    model_name='dccuchile/bert-base-spanish-wwm-cased',
    num_labels=CANTIDAD_CLASES,
    use_cuda=False,
    args=train_args
  )

  # Entrenamos el modelo
  model.train_model(train_df, eval_df=test_df)

  # Evalúo el modelo
  model.eval_model(test_df)
  # Sync wandb
  wandb.join()

wandb.agent(sweep_id, train)

