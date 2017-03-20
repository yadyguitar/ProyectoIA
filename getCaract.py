from pylab import imread,subplot,imshow,title,gray,figure,show,NullLocator
from numpy import mean,cov,cumsum,dot,linalg,size,flipud,argsort,shape,reshape,asmatrix
import cv2
import numpy as np
import matplotlib.pylab as plt

class Caracteristicas:
  def __init__(self, numpc):
    self.numpc=numpc

  def getCaract(self,ruta):
    A = imread(ruta) # load an image
    # A = mean(A,axis=2) # to get a 2-D array
    A=cv2.cvtColor(A,cv2.COLOR_BGR2GRAY)
    A = cv2.equalizeHist(A)
    cv2.imshow("1",A)
    full_pc = size(A,axis=1) # numbers of all the principal components
    coeff= self.princomp(A,self.numpc)
    return asmatrix(coeff.flatten())
  
  def princomp(self,A,numpc=0):
    M = (A-mean(A.T,axis=1)).T # subtract the mean (along columns)
    [latent,coeff] = linalg.eig(cov(M))
    p = size(coeff,axis=1)
    idx = argsort(latent) # sorting the eigenvalues
    idx = idx[::-1]       # in ascending order
   # sorting eigenvectors according to the sorted eigenvalues
    coeff = coeff[:,idx]
    latent = latent[idx] # sorting eigenvalues
    if numpc < p and numpc >= 0:
      coeff = coeff[:,range(numpc)] # cutting some PCs if needed
    score = dot(coeff.T,M) # projection of the data in the new space
    #fig = plt.figure()
    #ax=fig.add_subplot(1,1,1)
    #ax.set_aspect('equal')
    #plt.imshow(coeff, interpolation='nearest', cmap=plt.cm.ocean)
    #plt.colorbar()
    #plt.show()
    return coeff

Caracteristicas(5).getCaract("a.jpg")