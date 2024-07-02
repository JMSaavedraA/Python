#Graficador y Graficador con Zoom, construidos especialmente para la clase de Métodos Numéricos, Maestría en Matemáticas Aplicadas, CIMAT
#José Miguel Saavedra Aguilar
from math import ceil, floor, log10
import numpy as np

from graphics import *

def Graficar(funcion,a,b,n,pX,pY,enmarcar=True,showGrid=True,showAxis=True,Título="Gráfica 1"):
    
    # Iniciamos si el usuario eligió mostrar el marco para la gráfica
    if enmarcar:
        marco=50
    else:
        marco=0

    # Generamos la malla donde evaluamos la función, de tamaño especificado por el usuario
    x=np.linspace(a,b,n)

    # Evaluamos la función en la malla anterior, y=f(x)
    
    yTol=1e+10
    
    y=funcion(x)
    isPrinted=np.repeat(True,n)
    
       
    for i in range(n):
        if np.isinf(y[i]) or np.isnan(y[i]) or abs(y[i])>yTol:
            y[i]=0
            isPrinted[i]=False
            isPrinted[i-1]=False

    # Ahora, necesitamos saber el mínimo y el máximo de la función para ajustar la pantalla0
    yMin=min(y)

    yMax=max(y)

    for i in range(n-1):
        if abs(y[i]-y[i+1])==yMax-yMin:
            isPrinted[i]=False

    dX=b-a
    
    dY=yMax-yMin
    
    # Ahora calculamos el reescalamiento de x, y
    mY=-pY/(dY)

    bY=-yMax*mY + marco

    scaledY=mY*y+bY

    mX=pX/dX

    bX=-a*mX + marco

    scaledX=mX*x+bX
    
    # Inicializamos el gráfico
    win = GraphWin(Título, pX+2*marco, pY+2*marco)

    win.setBackground('white')

    # Vamos a mostrar la grilla si así lo especifica el usuario
    if showGrid:
        # Vamos a hacer la grilla de acuerdo a la magnitud (base 10) de la función f y de x. Guardamos la magnitud para usarla después
        cX=ceil(log10(dX/2))-1
        
        sX=10**cX
        
        cY=ceil(log10(dY/2))-1
        
        sY=10**cY
        
        # Calculamos la grilla que se mostrará para x
        minXgrid=(round(np.floor(x[0]/sX))+1)
        
        maxXgrid=(ceil(x[-1]/sX))
        
        xMeshGrid=range(minXgrid,maxXgrid,1)
        
        for j in xMeshGrid:
            
            # Ahora, dibujamos la grilla de x(vertical)
            auxDraw = Line(Point(sX*mX*j+bX, marco), Point(sX*mX*j+bX, pY + marco))
        
            auxDraw.setFill('Light Gray')
        
            auxDraw.setWidth(1)
        
            auxDraw.draw(win)
        
        # Ahora, calculamos la grilla que se mostrará para y
        minYgrid=(round(np.floor(yMin/sY))+1)
        
        maxYgrid=(ceil(yMax/sY))
        
        yMeshGrid=range(minYgrid,maxYgrid,1)
        
        for j in yMeshGrid:
        
            # Dibujamos la grilla de y(horizontal)
            auxDraw = Line(Point(marco,sY*mY*j+bY), Point(pX + marco,sY*mY*j+bY))
        
            auxDraw.setFill('Light Gray')
        
            auxDraw.setWidth(1)
        
            auxDraw.draw(win)

    # Ahora, pintamos el marco
    if enmarcar:
        
        auxDraw=Rectangle(Point(marco,marco), Point(pX+marco,pY+marco))
        
        auxDraw.setWidth(1)
        
        auxDraw.draw(win)

    # Si el usuario lo especificó, se muestran los ejes x, y
    if showAxis:
        
        # Cada uno de los ejes se mostrará sí y solamente sí será visible en la pantalla
        showYaxis=((a*b)<0)
        
        showXaxis=((yMin*yMax)<0)
        
        if showXaxis:
            
            # Una vez que se mostrará el eje x, lo pintamos       
            auxDraw=Line(Point(marco,bY),Point(pX+marco,bY))
        
            auxDraw.setWidth(2)
        
            auxDraw.setArrow('last')
        
            auxDraw.draw(win)
        
        if showYaxis:
            
            # Una vez que se mostrará el eje x, lo pintamos
            auxDraw=Line(Point(bX,marco),Point(bX,pY+marco))
        
            auxDraw.setWidth(2)
        
            auxDraw.setArrow('first')
        
            auxDraw.draw(win)

    # Ahora, vamos a poner la escala de la malla en la gráfica
    if showAxis and showGrid:
        
        # La escala será diferente dependiendo de cuáles ejes se muestran. Si se muestra un eje, se muestra junto al eje. De otra forma, 
        if showXaxis:
        
            if showYaxis:
                
                # Este es el caso que en se muestren ambos ejes, solo se mostrará una vez el origen        
                xList=list(xMeshGrid)
        
                yList=list(yMeshGrid)
        
                xList.remove(0)
        
                yList.remove(0)
        
                for j in xList:
        
                    auxDraw=Text(Point(sX*mX*j+bX,bY+10), str(round(j*sX,-cX)))
        
                    auxDraw.draw(win)
        
                for j in yList:
        
                    yS=str(round(j*sY,-cY))
        
                    auxDraw=Text(Point(bX-(10*len(yS)/2),sY*mY*j+bY+10), yS)
        
                    auxDraw.draw(win)
        
                auxDraw=Text(Point(bX-10,bY+10), '0')
        
                auxDraw.draw(win)
        
            else:
                
                # En este caso solo se muestra el eje x
                for j in xMeshGrid:
        
                    auxDraw=Text(Point(sX*mX*j+bX,bY+10), str(round(j*sX,-cX)))
        
                    auxDraw.draw(win)
        
                for j in yMeshGrid:
        
                    yS=str(round(j*sY,-cY))
        
                    auxDraw=Text(Point(marco-(10*len(yS)/2),sY*mY*j+bY), yS)
        
                    auxDraw.draw(win)
        
        elif showYaxis:
            
            # El caso en que solo se muestra el eje y
            for j in yMeshGrid:
        
                yS=str(round(j*sY,-cY))
        
                auxDraw=Text(Point(bX-(10*len(yS)/2),sY*mY*j+bY+10), yS)
        
                auxDraw.draw(win)
        
            for j in xMeshGrid:
        
                auxDraw=Text(Point(sX*mX*j+bX,pY+marco+15), str(round(j*sX,-cX)))
        
                auxDraw.draw(win)
        
        else:
            
            # El caso en que no se muestra ningún eje pero el usuario si pidió ver los ejes
            for j in xMeshGrid:
        
                auxDraw=Text(Point(sX*mX*j+bX,pY+marco+15), str(round(j*sX,-cX)))
        
                auxDraw.draw(win)
        
            for j in yMeshGrid:
        
                yS=str(round(j*sY,-cY))
        
                auxDraw=Text(Point(marco-(10*len(yS)/2),sY*mY*j+bY), yS)
        
                auxDraw.draw(win)
    
    elif showGrid:
    
        # El caso en que el usuario especificó no ver los ejes pero si la malla
        for j in xMeshGrid:
    
            auxDraw=Text(Point(sX*mX*j+bX,pY+marco+15), str(round(j*sX,-cX)))
    
            auxDraw.draw(win)
    
        for j in yMeshGrid:
    
            yS=str(round(j*sY,-cY))
    
            auxDraw=Text(Point(marco-(10*len(yS)/2),sY*mY*j+bY), yS)
    
            auxDraw.draw(win)

    if Título:
        auxDraw=Text(Point(marco+pX/2.0,marco/2.0), Título)
        auxDraw.draw(win)
    
    # Finalmente, vamos a pintar la función y=f(x)
    for k in range(n-1):
        
        if isPrinted[k]:
                    
            mainDraw = Line(Point(scaledX[k], scaledY[k]), Point(scaledX[k+1], scaledY[k+1]))
        
            mainDraw.setWidth(1)
        
            mainDraw.setFill('Dark Blue')
        
            mainDraw.draw(win)

    # Por último, regresamos el botón presionado en la gráfica.
    win.postscript(file="image.eps", colormode='color')

    g=win.getKey()
    
    win.close()
    
    return g


def GraficadorConZoom(funcion,a,b,n,pX,pY,enmarcar=True,showGrid=True,showAxis=True,Título="Gráfica 1"):
    
    # Ahora, vamos a añadirle un zoom al graficador. Para esto, calculamos el centro y la amplitud del intervalo [a,b].
    cX=(a+b)/2.0
    
    dX=b-cX
    
    while True:
        
        print("Presiona + para acercar (2X), - para alejar (0.5X), Enter para mover")
        
        a=cX-dX
        
        b=cX+dX
        
        # Graficamos y leemos el botón presionado en la gráfica
        g=Graficar(funcion,a,b,n,pX,pY,enmarcar,showGrid,showAxis,Título)    
        
        # Presionar Esc para cerrar
        if g=="Escape":
            
            print('Cerrando')

            break
        # Presionar + para hacer zoom (2x)
        elif g=="KP_Add" or g=="plus":
            
            print('Centro actual es '+ str(cX))
            
            # Podemos hacer zoom con centro en un nuevo punto de x, o simplemente mantener el punto anterior
            val = input("Introduce nuevo centro de x. En caso de no querer un nuevo centro, solo presione Enter: ")
            
            try:
                
                float(val)
                
                cX=float(val)
                
                dX /= 2.0
            
            except ValueError:
                
                if val:                  
                    
                    print("Introduce un número válido")
                
                else:
                    
                    print("Centro no actualizado")
                    
                    dX /= 2.0
        # Presionar - para hacer zoom (0.5x)
        elif g=="KP_Subtract" or g=="minus":
            
            print('Centro actual es '+ str(cX))
            
            # Podemos hacer zoom con centro en un nuevo punto de x, o simplemente mantener el punto anterior            
            val = input("Introduce nuevo centro de x. En caso de no querer un nuevo centro, solo presione Enter: ")
            
            try:
                
                float(val)
                
                cX=float(val)
                
                dX *= 2.0
            
            except ValueError:
                
                if val:                  
                    
                    print("Introduce un número válido")
                
                else:
                    
                    print("Centro no actualizado")
                    
                    dX *= 2.0
        
        # Presionar Enter para mover el centro de la gráfica
        elif g=="KP_Enter" or g=="Return":
            
            print('Centro actual es '+ str(cX))
            
            # Elegimos un nuevo punto de x, o mantenemos el punto anterior en caso de error de dedo
            val = input("Introduce nuevo centro de x. En caso de no querer un nuevo centro, solo presione Enter: ")
            
            try:
                
                float(val)
                
                cX=float(val)
            
            except ValueError:
                
                if val:                  
                    
                    print("Introduce un número válido")
                
                else:
                    
                    print("Centro no actualizado")

