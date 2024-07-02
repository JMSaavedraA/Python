#Función Complementaria al Graficador y Graficador con Zoom, dar un ejemplo de como correr ambos códigos. Correr en Python 3, se probó en python 3.8 y 3.10, sin ninguna diferencia, en Windows 10 y en Ubuntu 20.04 LTS
#José Miguel Saavedra Aguilar

#Se pueden comentar o descomentar lineas de código para ver las diferentes funciones

#El graficador requiere tener instaladas las librerias numpy y graphics (se instala graphics.py en pip3)

#Se corren de la siguiente forma:

#Graficar(funcion,a,b,n,pX,pY,enmarcar,showGrid,showAxis,Título)

#GraficadorConZoom(funcion,a,b,n,pX,pY,enmarcar,showGrid,showAxis,Título)

#Los argumentos son idénticos para ambas funciones:
# funcion es una función evaluable para numpy arrays
# a, b son floats que indican el intervalo [a,b] donde se grafica la función
# n es el número de puntos dentro del intervalo
# Px, Py son el tamaño en pixeles de la ventana gráfica sobre x, y respectivamente

# Opcionales:
# enmarcar Booleano que activa el marco en la gráfica, default True
# showGrid Booleano que activa la malla en la gráfica, default True
# showAxis Booleano que activa los ejes coordenados en la gráfica, default True


from Graficador import *

#Debemos definir las funciones previamente, por ejemplo

def f(x):
    a=x**3+x**2+1
    return a
def g(x):
    y=np.sin(10*x)
    return y
def h(x):
    y=1/(x-1)
    return y

#O en su caso utilizar funciones ya incluidas en las librerias. Para salir de la gráfica, presionar cualquier botón

Graficar(np.sin,-5,5,800,800,800,True,True,True,"f(x)=sin(x)")

#Para hacer uso del zoom, deberá presionar + para acercar o - para alejar la gráfica con la ventana de la gráfica ACTIVA, y elegir el nuevo centro en la consola. En caso de no querer un nuevo centro, solo presione Enter en la consola.
# Para únicamente mover la gráfica sin hacer zoom, presionar Enter con la ventana de la gráfica ACTIVA, e ingresar el nuevo centro en la consola. Para salir presionar Escape con la ventana de la gráfica ACTIVA.
# Para más información, ver el Reporte

#GraficadorConZoom(funcion=f,a=-20,b=20,n=800,pX=800,pY=800,enmarcar=True,showGrid = True,showAxis = True,Título="f(x)=x³+x²+1")

#GraficadorConZoom(funcion=g,a=-3,b=3,n=1600,pX=800,pY=800,enmarcar=True,showGrid = True,showAxis = True,Título="g(x)=1/(x+1)")

#GraficadorConZoom(funcion=h,a=-20,b=20,n=800,pX=800,pY=800,enmarcar=True,showGrid = True,showAxis = True,Título="h(x)=1/(x+1)")