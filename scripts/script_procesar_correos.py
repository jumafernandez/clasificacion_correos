#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 10:26:46 2019

@author: juan
"""

#%% Inicio

import pypff

pst_file = '/home/juan/Escritorio/Enlace hacia Trabajo_Final_Especializacion/data/Consultas.pst'

pst = pypff.file()
pst.open(pst_file)

root_node = pst.get_root_folder()
carpeta_Outlook = root_node.get_sub_folder(1)
print('Carpetas del Outlook:')
for i in range(0, carpeta_Outlook.get_number_of_sub_folders()):
    folder=carpeta_Outlook.get_sub_folder(i)
    print(str(i) + "-" + folder.get_name() + ": " + str(folder.get_number_of_sub_messages()))


#%% Función para limpiar las consultas
def limpiar_correo(text):
    '''Se limpian las cadenas de texto'''
    
    # Paso a minusculas
    text = str(text).lower()

    # Reemplazo los tildes
    text = text.replace("\\xc3\\xa1", "á")
    text = text.replace("\\xc3\\xa9", "é")
    text = text.replace("\\xc3\\xad", "í")
    text = text.replace("\\xc3\\xb3", "ó")
    text = text.replace("&uacute;", "ú")
    text = text.replace("\\xc3\\xb1", "ñ")
    text = text.replace("&aacute;", "á")
    text = text.replace("&eacute;", "é")
    text = text.replace("&iacute;", "í")
    text = text.replace("&oacute;", "ó")
    text = text.replace("&uacute;", "ú")
    text = text.replace("&ntilde;", "ñ")
    text = text.replace("&ordm", "°")

    # Quito los fin de linea y caracteres especiales
    text = text.replace("\\n", " ")
    text = text.replace("\\r", "")
    text = text.replace("\\", "")
    text = text.replace("b\'", "")

    return text

def limpiar_consulta(text):
    '''Se limpian las consultas'''   
    # Separo la consulta en encabezado y cuerpo
    COMIENZO_CORREO = "de: u.n.lu. [mailto:consultasweb@mail.unlu.edu.ar]"
    GUIONES_CUERPO  = "-------------------------"
    
    text = str(text).replace(" >", "")
    text = text.split("---------------------------------")
    if len(text)>1:
        # El encabezado solo posee la fecha como dato importante
        encabezado = text[0]
        encabezado = encabezado.replace(COMIENZO_CORREO, "")
        #fecha = encabezado[len(encabezado)-len("08.23.2015-00:57:19")-2:len(encabezado)].strip()
        inicio_fecha = encabezado.find("enviado :")
        fecha        = encabezado[inicio_fecha+len("enviado :"):len(encabezado)].strip()
        fecha        = fecha[0:len("08.20.2019-20:48:53")]
        hora         = fecha.split("-")[1]
        fecha        = fecha.split("-")[0].replace(".", "-")
        
        # Cuerpo        
        cuerpo = text[1]
        cuerpo = cuerpo.replace(GUIONES_CUERPO, "")
        
        # Busco el inicio de cada dato para estructurarlos
        inicio_ap_nom    = cuerpo.find("nombre y apellido: ")
        inicio_legajo    = cuerpo.find("legajo: ")
        inicio_documento = cuerpo.find("documento: ")
        inicio_carrera   = cuerpo.find("carrera: ")
        inicio_telefono  = cuerpo.find("teléfono: ")
        inicio_email     = cuerpo.find("e-mail: ")
        inicio_consulta  = cuerpo.find("mensaje / consulta: ")

        apellido_nombre = cuerpo[inicio_ap_nom+len("nombre y apellido: "):inicio_legajo-1]
        legajo          = cuerpo[inicio_legajo+len("legajo: "):inicio_documento-1]
        documento       = cuerpo[inicio_documento+len("documento: "):inicio_carrera-1]
        carrera         = cuerpo[inicio_carrera+len("carrera: "):inicio_telefono-1]
        telefono        = cuerpo[inicio_telefono+len("teléfono: "):inicio_email-1]
        email           = cuerpo[inicio_email+len("e-mail: "):inicio_consulta-1]
        consulta        = cuerpo[inicio_consulta+len("mensaje / consulta: "):len(cuerpo)]
        
    else:
        fecha = text[0]
        hora  = -1
        apellido_nombre = -1
        legajo          = -1
        documento       = -1
        carrera         = -1
        telefono        = -1
        email           = -1
        consulta        = -1
        
    return fecha, hora, apellido_nombre, legajo, documento, carrera, telefono, email, consulta


#%% Recupero respuestas (que también poseen las consultas)


respuestas = carpeta_Outlook.get_sub_folder(3)

CANTIDAD_CORREOS = respuestas.get_number_of_sub_messages()
#CANTIDAD_CORREOS = 10
SEPARADOR_CONSULTA_RESPUESTA = "-----mensaje original-----"
PATH_ARCHIVO_CORREOS = "/home/juan/Escritorio/Enlace hacia Trabajo_Final_Especializacion/scripts/correos-procesados.csv"
INICIO_DISTINTO = "de: u.n.lu. [mailto:consultasweb@mail.unlu.edu.ar]"

archivo_correos = open(PATH_ARCHIVO_CORREOS, "w")
archivo_correos.write("Fecha" + "|" + "Hora" + "|" + "Apellido y Nombre" + "|" + "Legajo" + "|" + "Documento" + "|" + "Carrera" + "|" + "Teléfono" + "|" + "E-mail" + "|" + "Consulta" + "|" + "Respuesta" + "\n")

for i in range(0, CANTIDAD_CORREOS):
    correo=respuestas.get_sub_message(i)
    cuerpo  =   correo.get_plain_text_body()
    if cuerpo:
        # Se hace una limpieza inicial del texto
        cuerpo = limpiar_correo(cuerpo)      
        # Se separa la consulta de la respuesta, solo acepta un ida y vuelta
        cuerpo    =   cuerpo.split(SEPARADOR_CONSULTA_RESPUESTA)
        # Se deciden desechar los correos no separables, con varios idas/vueltas
        if len(cuerpo)>1:
            respuesta =   cuerpo[0]
            consulta  =   cuerpo[1]
            
            # Si es un ida/vuelta UNICO
            if consulta.find(INICIO_DISTINTO)!=-1:
                fecha, hora, apellido_nombre, legajo, documento, carrera, telefono, email, consulta = limpiar_consulta(consulta)

                if apellido_nombre!=-1:
                    # Se guarda en el archivo si la consulta es separable
                    archivo_correos.write(fecha + "|" + hora + "|" + apellido_nombre + "|" + legajo + "|" + documento + "|" + carrera + "|" + telefono + "|" + email + "|" + consulta + "|" + respuesta + "\n")


archivo_correos.close()
