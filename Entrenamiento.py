import numpy as np
import Muestras

class Entrenamiento:
	def __init__(self):
		self.numeroEntradas=3
		self.numeroCapaOculta=[3] #Numero de neuronas en cada capa oculta
		self.numeroSalidas=2
		self.pesos=[] #w0=matriz de pesos[0](in 1) , w1=Matriz de pesos[1] (1 out)
 		self.salidaCapas=[] #lista de listas de resultados de la sumatoria y la sigmoide
 		#la primera salida, es la salida de la primera capa oculta (datos ya con la funcion sigmoide)
 		#la ultima salida, es el resultado final de la capa


	def genera_pesos(self):
		try: #Si ya estan guardados, los lee
			for i in range(len(self.numeroCapaOculta)+1):
				self.pesos.append(np.loadtxt("pesos/w"+str(i)+".txt"))
		except: #si es la primera vez que se crean
			self.pesos.append(2* np.random.rand(self.numeroEntradas,self.numeroCapaOculta[0])-1)
			for i in range(len(self.numeroCapaOculta)-1):
				self.pesos.append(2* np.random.rand(self.numeroCapaOculta[i],self.numeroCapaOculta[i+1])-1)
			self.pesos.append(2* np.random.rand(self.numeroCapaOculta[-1],self.numeroSalidas)-1)
			for i in range(len(self.numeroCapaOculta)+1):
				temp=open("pesos/w"+str(i)+".txt",'w+')
				np.savetxt(temp,self.pesos[i])
				temp.close
	

	def avanzar(self,x):
		self.salidaCapas.append( self.sigmoid( np.dot(x,self.pesos[0]) ) ) #Tal vez y no sea global
		for i in range(len(self.numeroCapaOculta)):
			self.salidaCapas.append( self.sigmoid( np.dot(self.salidaCapas[i],self.pesos[i+1]) ) )
		print self.salidaCapas
	
	def sigmoid(self,z):
		return 1/(1+np.exp(-z))

	def entrenar(self,x,y):
		#leer las muestras (ya sea en una funcion a parte, o blablabla)
		error= np.multiply( self.salidaCapas[-1], np.multiply((1-self.salidaCapas[-1]),y-self.salidaCapas[-1]))
		#error en salida
		print error



n=Entrenamiento()
n.genera_pesos()
n.avanzar([2,5,1])
n.entrenar([2,5,1],[1,4])
