from pylab import imread,subplot,imshow,title,gray,figure,show,NullLocator
from numpy import mean,cov,cumsum,dot,linalg,size,flipud,argsort,shape
import cv2


def princomp(A,numpc=0):
 # computing eigenvalues and eigenvectors of covariance matrix
	M = (A-mean(A.T,axis=1)).T # subtract the mean (along columns)
	[latent,coeff] = linalg.eig(cov(M))
	p = size(coeff,axis=1)
	print "size de p: ",p
	idx = argsort(latent) # sorting the eigenvalues
	idx = idx[::-1]       # in ascending order
 # sorting eigenvectors according to the sorted eigenvalues
	coeff = coeff[:,idx]
	latent = latent[idx] # sorting eigenvalues
	if numpc < p and numpc >= 0:
		coeff = coeff[:,range(numpc)] # cutting some PCs if needed
	score = dot(coeff.T,M) # projection of the data in the new space
	return coeff,score,latent


A = imread('prueba.jpg') # load an image

A = mean(A,axis=2) # to get a 2-D array
print "A en 2D:,",A
full_pc = size(A,axis=1) # numbers of all the principal components
print "Numero de todos los componentes principales",full_pc
i = 1
dist = []
numpc=5
coeff, score, latent = princomp(A,numpc)
print shape(coeff)
print coeff

Ar = dot(coeff,score).T+mean(A,axis=0) # image reconstruction
# difference in Frobenius norm
dist.append(linalg.norm(A-Ar,'fro'))
# showing the pics reconstructed with less than 50 PCs

ax = subplot(2,3,i,frame_on=False)
Ar=Ar.astype(int)
print Ar

print size(Ar)
imshow(Ar)
gray()

show()
print "fin"