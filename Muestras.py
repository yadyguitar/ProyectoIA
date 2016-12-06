import numpy as np

class Muestras:
	def __init__ (self):
		self.muestras=None
		self.resultados_muestras=None
		

	def addMuestra(self,muestra,resultado):
		self.abreArchivo()
		np.savetxt(self.muestras,muestra)
		np.savetxt(self.resultados_muestras,resultado)
		self.cierraArchivo()

	def getMuestras(self):
		self.abreArchivo()
		m,r=np.loadtxt('setMuestras/muestras.txt') , np.loadtxt('setMuestras/resultados_muestras.txt')
		self.cierraArchivo()
		return m,r

	def cierraArchivo(self):
		self.muestras.close
		self.resultados_muestras.close

	def abreArchivo(self):
		self.muestras=open('setMuestras/muestras.txt','a+')
		self.resultados_muestras=open('setMuestras/resultados_muestras.txt','a+')
		
'''


x,y=Muestras().getMuestras()
print "Muestras:" ,np.shape(x),"resultados:",np.shape(y)'''