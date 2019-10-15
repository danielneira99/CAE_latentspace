#Imports

import pandas as pd
import numpy as np
import gzip
import io
import pyfits as fits
import matplotlib.pyplot as plt
import pickle as pk


#Funcion para pasar de bits a imagen

def get_image_from_bytes_stamp(stamp_byte):
    with gzip.open(io.BytesIO(stamp_byte), 'rb') as f:
        with fits.open(io.BytesIO(f.read())) as hdul:
            img = hdul[0].data
    return img



data = pd.read_pickle("D:\\vuchef\\electrica\\dirigido\\training_set_isdiffpos.pkl")
Data = data.to_numpy()
print('Clases distintas',np.unique(Data[:,0]))

#Conteo de imagenes corruptas

d = 0
for i in range(len(Data[:,0])):
    if np.shape(get_image_from_bytes_stamp(Data[0,3])) == np.shape(get_image_from_bytes_stamp(Data[i,3])):
        d = d+1


#Filtrado de imagenes corruptas

datos_originales = np.zeros((d,63,63))
j=0
n=0
indices = np.zeros(d)
indices2 = np.zeros(len(Data[:,0]) - d)
for i in range(len(Data[:,0])):
    if np.shape(get_image_from_bytes_stamp(Data[0,3])) == np.shape(get_image_from_bytes_stamp(Data[i,3])):
        datos_originales[j,:,:]=get_image_from_bytes_stamp(Data[i,3])
        indices[j] = i
        j=j+1
    else:
        indices2[n] = i
        n=n+1

print(np.shape(datos_originales))
print(len(indices2))
plt.imshow(datos_originales[320,:,:])
plt.title('Imagen de prueba')
plt.show()

#Guardar las otras imagenes dentro del dataset

datos_segundos = np.zeros((d,63,63))
datos_diferencia = np.zeros((d,63,63))
for i in range(d):
    datos_segundos[i,:,:]=get_image_from_bytes_stamp(Data[int(indices[i]),2])
    datos_diferencia[i,:,:] = get_image_from_bytes_stamp(Data[int(indices[i]),4])

#Funcion para cambiar NaN por 0 en una imagen

def arreglar_imagen(imagen):
    auxiliar = imagen
    for i in range(63):
        for j in range(63):
            if(np.isnan(auxiliar[i,j]).any()):
                auxiliar[i,j] = 0
    return auxiliar

#Iteracion para arreglar esos datos

for i in range(np.shape(datos_diferencia)[0]):
    if(np.isnan(datos_diferencia[i,:,:]).any()):
        arreglado = arreglar_imagen(datos_diferencia[i,:,:])
        datos_diferencia[i,:,:] = arreglado

#Funcion para tener las clases de los datos buenos

def clase(indice):
    nombre = Data[indice,0]
    if nombre == 'AGN':
        return 0
    elif nombre == 'SN':
        return 1
    elif nombre == 'VS':
        return 2
    elif nombre == 'asteroid':
        return 3
    else:
        return 4

#Iteracion para guardar esos datos

y = np.zeros(int(np.shape(datos_diferencia)[0]))
for i in range(len(y)):
    ind = int(indices[i])
    cl = clase(ind)
    y[i] = cl
print(np.unique(y).astype(int))

#Guardar las clases

np.savetxt('clases.txt',y,delimiter=',')
pk.dump(datos_diferencia, open( "datos_filtrados.p", "wb" ))