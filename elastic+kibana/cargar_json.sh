#!/bin/bash

echo "Importando users..."
jsonpyes --data users.json --bulk http://localhost:9200  --import --index users --type users
echo "Importando tweets..."
jsonpyes --data tweets.json --bulk http://localhost:9200  --import --index tweets --type tweets
echo "Datos cargados en elastic"
