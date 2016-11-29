from Muestras import Muestras
from Red_Neuronal import Red_Neuronal
import numpy as np
from getCaract import Caracteristicas

class Entrenamiento:
	def __init__ (self):
		self.muestras,self.resultados=Muestras().getMuestras() ##obtengo la muestra y el resultado esperado
		self.errorFinal=0
		self.salidas=4 	
		self.red=Red_Neuronal(400,[100,50,30],self.salidas)

	def entrenar(self):
		print self.muestras,self.resultados
		tam=len(self.muestras)
		print tam
		self.errorFinal=0.0001*self.salidas*tam #sumatoria ideal de errores en las salidas de las muestras
		errorSum=1
		while errorSum > self.errorFinal:
			errorSum=0
			for i in range(tam): #modifica pesos acorde a las muestras introducicas i -> muestra
				self.red.genera_pesos()
				print "\nSalida muestra",i+1,":\n",self.red.getSalida(self.muestras[i]),"\nresultado esperado: \n",self.resultados[i]
				e=self.red.modifica_pesos(self.muestras[i],self.resultados[i])
				print "\nError: ",e
				errorSum+=sum(np.absolute(e))

	def clasificar(self,entrada):
		self.red.genera_pesos()
   		print "\nSalida muestra\n",self.red.getSalida(entrada)
   		return self.red.getSalida(entrada)


'''
for i in range(10):
	temp=Caracteristicas(5).getCaract("setMuestras/imagenes/0"+str(i)+".jpg")
	Muestras().addMuestra(temp,[[0,0,0,0]]) #neutro
	temp=Caracteristicas(5).getCaract("setMuestras/imagenes/1"+str(i)+".jpg")
	Muestras().addMuestra(temp,[[0,0,0,1]]) #arriba
'''


Entrenamiento().entrenar()
Entrenamiento().clasificar(Caracteristicas(5).getCaract("prueba.jpg"))

