from Muestras import Muestras
from Red_Neuronal import Red_Neuronal
import numpy as np
from getCaract import Caracteristicas

#error-> 0.01

class Entrenamiento:
	def __init__ (self):
		self.muestras,self.resultados=Muestras().getMuestras() ##obtengo la muestra y el resultado esperado
		self.parada=0
		self.salidas=4 	
		self.red=Red_Neuronal(400,[100,50,30],self.salidas)

	def entrenar(self):
		print self.muestras,self.resultados
		tam=len(self.muestras)
		print tam
		self.parada=tam #sumatoria ideal de errores en las salidas de las muestras
		errorSum=0
		while errorSum != self.parada:
			errorSum=0
			for i in range(tam): #modifica pesos acorde a las muestras introducicas i -> muestra
				self.red.genera_pesos()
				print "\nSalida muestra",i+1,":\n",self.red.getSalida(self.muestras[i]),"\nresultado esperado: \n",self.resultados[i]
				e=self.red.modifica_pesos(self.muestras[i],self.resultados[i])
				print "\nError: ",e
				#sumatoria del error de la salida de la muestra actual (en este caso: 4 muestras error considerado 0.01 (lo maximo de error que podria tener 1 salida de las 4 que tiene seria 0.03999999))
				suma=sum(np.absolute(e))
				#print "suma de errores: ",suma
				if suma <= 0.01 * self.salidas:
					errorSum+=1
			

	def clasificar(self,entrada):
		self.red.genera_pesos()
   		print "\nSalida muestra\n",self.red.getSalida(entrada)
   		return self.red.getSalida(entrada)
'''
cad="0"
for i in range(10):
	if i==10:
		cad=""
	temp=Caracteristicas(5).getCaract("setMuestras/imagenes/0"+cad+str(i)+".jpg")
	Muestras().addMuestra(temp,[[0,0,0,0]]) #neutro
	temp=Caracteristicas(5).getCaract("setMuestras/imagenes/1"+cad+str(i)+".jpg")
	Muestras().addMuestra(temp,[[0,0,0,1]]) #arriba
	temp=Caracteristicas(5).getCaract("setMuestras/imagenes/2"+cad+str(i)+".jpg")
	Muestras().addMuestra(temp,[[0,0,1,0]]) #arriba der
	temp=Caracteristicas(5).getCaract("setMuestras/imagenes/3"+cad+str(i)+".jpg")
	Muestras().addMuestra(temp,[[0,0,1,1]]) #derecha
	temp=Caracteristicas(5).getCaract("setMuestras/imagenes/4"+cad+str(i)+".jpg")
	Muestras().addMuestra(temp,[[0,1,0,0]]) #derecha abjo
	temp=Caracteristicas(5).getCaract("setMuestras/imagenes/5"+cad+str(i)+".jpg")
	Muestras().addMuestra(temp,[[0,1,0,1]]) #abajo
	temp=Caracteristicas(5).getCaract("setMuestras/imagenes/6"+cad+str(i)+".jpg")
	Muestras().addMuestra(temp,[[0,1,1,0]]) #abjo izq
	temp=Caracteristicas(5).getCaract("setMuestras/imagenes/7"+cad+str(i)+".jpg")
	Muestras().addMuestra(temp,[[0,1,1,1]]) #izq
	temp=Caracteristicas(5).getCaract("setMuestras/imagenes/8"+cad+str(i)+".jpg")
	Muestras().addMuestra(temp,[[1,0,0,0]]) #izq arriba
	print ""+cad+str(i)

'''
#Entrenamiento().entrenar()
Entrenamiento().clasificar(Caracteristicas(5).getCaract("225.jpg"))

