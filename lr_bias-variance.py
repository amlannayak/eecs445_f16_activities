import numpy as np
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
    
    for i in range(sub):
            
        #create data - sample from sine wave     
        x = np.random.random((n,1))*2*np.pi
        y = np.sin(x) + np.random.normal(mean,std,(n,1))
        
        #TODO
        #create features corresponding to degree - ex: 1, x, x^2, x^3...
        A = 
        
        #TODO:        
        #fit model using least squares solution (linear regression)
        #later include ridge regression/normalization
        coeffs = 
                
        #store predictions for each sampling
        preds[:,i] = poly.fit_transform(Xtest).dot(coeffs)[:,0]
        
        #plot 9 images
        if i < 9:
            plt.subplot(3,3,i+1)
            plt.plot(values,poly.fit_transform(values).dot(coeffs),x,y,'.b')

    plt.axis([0,2*np.pi,-2,2])
    plt.suptitle('PolyFit = %i' % (degree))
    plt.show()

    #TODO
    #Calculate mean bias, variance, and MSE
    bias[j] = 
    variance[j] = 
    mse[j] = 

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

