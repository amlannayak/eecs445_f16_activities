import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from numpy.matlib import repmat

degrees = [1,2,3,4,5]


#define data
n = 20
sub = 1000
mean = 0
std = 0.25

#define test set
Xtest = np.random.random((n,1))*2*np.pi
ytest = np.sin(Xtest) + np.random.normal(mean,std,(n,1))

#pre-allocate variables
preds = np.zeros((n,sub))
bias = np.zeros(len(degrees))
variance = np.zeros(len(degrees))
mse = np.zeros(len(degrees))
values = np.expand_dims(np.linspace(0,2*np.pi,100),1)

for j,degree in enumerate(degrees):

    poly = PolynomialFeatures(degree)    
    
    for i in range(sub):
            
        #create data - sample from sine wave     
        x = np.random.random((n,1))*2*np.pi
        y = np.sin(x) + np.random.normal(mean,std,(n,1))
        
        #get features
        A = poly.fit_transform(x)
        
        #fit model
        coeffs = np.linalg.pinv(A).dot(y)
                

        preds[:,i] = poly.fit_transform(Xtest).dot(coeffs)[:,0]
        
        #plot 
        if i < 9:
            plt.subplot(3,3,i+1)
            plt.plot(values,poly.fit_transform(values).dot(coeffs),x,y,'.b')

    plt.axis([0,2*np.pi,-2,2])
    plt.suptitle('PolyFit = %i' % (degree))
    plt.show()

    #Calculate average bias and variance
    bias[j] = np.mean(np.mean(preds,1) - ytest)**2 
    variance[j] = np.mean(np.var(preds,1))
    mse[j] = np.mean(np.mean(np.square(preds - repmat(ytest,1,sub))))

plt.subplot(3,1,1)
plt.plot(degrees,bias)
plt.title('bias')
plt.subplot(3,1,2)
plt.plot(degrees,variance)
plt.title('variance')
plt.subplot(3,1,3)
plt.plot(degrees,mse)
plt.title('MSE')
plt.show()

