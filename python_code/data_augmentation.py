#Imports
import pickle as pk
import numpy as np
import matplotlib.pyplot as plt

#Lectura de datos
with open("datos_filtrados.p","rb") as f:
    datos = pk.load(f)

clases = np.loadtxt('clases.txt',delimiter=',')

#Funcion que genera 3 rotaciones y la imagen original

def rotacion(imagen):
    dimensiones = np.shape(imagen)
    salida = np.zeros((4,dimensiones[0],dimensiones[1]))
    salida[0,:,:] = imagen
    salida[1,:,:] = np.rot90(imagen,k=1)
    salida[2,:,:] = np.rot90(imagen,k=2)
    salida[3,:,:] = np.rot90(imagen,k=3)
    return salida

#Separacion por clases
a = []
b = []
c = []
d = []
e = []
for i in range(len(clases)):
    if clases[i] == 0:
        a.append(i)
    elif clases[i] == 1:
        b.append(i)
    elif clases[i] == 2:
        c.append(i)
    elif clases[i] == 3:
        d.append(i)
    elif clases[i] == 4:
        e.append(i)
a = np.asarray(a)
b = np.asarray(b)
c = np.asarray(c)
d = np.asarray(d)
e = np.asarray(e)

#Aumento de la clase 1 y clase 3. Se guarda en una nueva variable para luego juntarlos con los otros

BOGUS = np.zeros((4*int(len(e)),63,63))
for i in range(len(e)):
    auxiliar = rotacion(datos[int(e[i]),:,:])
    BOGUS[i*4:i*4+4,:,:] = auxiliar

SUPER_NOVA = np.zeros((4*int(len(b)),63,63))
for i in range(len(b)):
    auxiliar = rotacion(datos[int(b[i]),:,:])
    SUPER_NOVA[i*4:i*4+4,:,:] = auxiliar

#Se crea una nueva matriz de datos, con las dimensiones necesarias

total_datos = len(a)+4*len(b)+len(c)+len(d)+4*len(e)
datos_nuevos = np.zeros((total_datos,63,63))
indices = []


for i in range(len(a)):
    datos_nuevos[i,:,:] = datos[int(a[i]),:,:]
    indices.append(0)

for i in range(len(c)):
    datos_nuevos[i+len(a),:,:] = datos[int(c[i]),:,:]
    indices.append(2)

for i in range(len(d)):
    datos_nuevos[i+len(a)+len(c),:,:] = datos[int(d[i]),:,:]
    indices.append(3)

for i in range(4*len(b)):
    datos_nuevos[i+len(a)+len(c)+len(d),:,:] = SUPER_NOVA[i,:,:]
    indices.append(1)

for i in range(4*len(e)):
    datos_nuevos[i+len(a)+len(c)+len(d)+4*len(b),:,:] = BOGUS[i,:,:]
    indices.append(4)

indices = np.array(indices)
print(np.shape(indices),np.shape(datos_nuevos))
pk.dump(datos_nuevos, open( "datos_aumentados.pkl", "wb" ))
np.savetxt('indices.txt',indices,delimiter=',')