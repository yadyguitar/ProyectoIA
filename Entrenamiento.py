import numpy as np
import Muestras

class Entrenamiento:
	def __init__(self):
		self.numeroEntradas=3
		self.numeroCapaOculta=[3,3] #Numero de neuronas en cada capa oculta
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

	
	def sigmoid(self,z):
		return 1/(1+np.exp(-z))

	def entrenar(self,x,y):
		#leer las muestras (ya sea en una funcion a parte, o blablabla)
		error=np.multiply( self.salidaCapas[-1], np.multiply((1-self.salidaCapas[-1]),y-self.salidaCapas[-1]))
		errorOut= error[:]
		#Calculo los pesos cercanos y los errores de capa oculta
		for capas in range(len(self.numeroCapaOculta)):
			n=(capas*-1)-1
			filas,columnas=np.shape(self.pesos[n])
			for i in range(columnas):
				for j in range(filas):
					self.pesos[n][j][i]+=(error[i]*self.salidaCapas[n-1][j])
					
			
			errorTemp=[]
			for i in range(len(self.salidaCapas[n-1])):
				suma=0
				for e in range(len(error)):
					suma+=error[e]*self.pesos[n][i][e]
				errorTemp.append( self.salidaCapas[n-1][i]*(1-self.salidaCapas[n-1][i])*suma )
			error=[]
			error=errorTemp[:]

			#pesos entre la entrada y la primera capa oculta
		filas,columnas=np.shape(self.pesos[0])
		for i in range(columnas):
			for j in range(filas):
				self.pesos[0][i][j]+=(error[j]*x[i])

		##guarda pesos forma oculta##
		c=0
		for p in self.pesos:
			temp=open(".pesos/w"+str(c)+".txt",'w+')
			np.savetxt(temp,p)
			temp.close
			c+=1

		
	def guardarPesos(self):
		c=0
		for p in self.pesos:
			temp=open("pesos/w"+str(c)+".txt",'w+')
			np.savetxt(temp,p)
			temp.close
			c+=1

	def run(self):#aqui es donde tomará las muestras y las pasará por la red modificando los pesos
		pass


n=Entrenamiento()
n.genera_pesos()
n.avanzar([2,5,1])
n.entrenar([2,5,1],[1,0])


print "guardar los pesos?"
opc=int(input())
if opc==1: 
	n.guardarPesos()