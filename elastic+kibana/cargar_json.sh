#!/bin/bash

echo "Importando correos..."
python jsonpyes.py --data C:\Users\unlu\Documents\GitHub\jumafernandez\clasificacion_correos\elastic+kibana\correos-procesados.json --bulk http://localhost:9200  --import --index correos --type correos
echo "Datos cargados en elastic"
