import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt

#SVM kernel - linear, poly, rbf, etc.
kernel='rbf'

#Cost values
Cs = [1,2,5,10,20,50,100,200,500,1000]

def plot_boundary(clf,g):
    #adapted from http://scikit-learn.org/stable/modules/svm.html
    h = 0.2
    xx, yy = np.meshgrid(np.arange(-g,g, h),np.arange(-g,g, h))    
    Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

#define data
n = 100
sub = 50
var = 2
mean = 1
mean1 = [mean, mean]
cov1 = [[2*var, 0], [0, 2*var]]
mean2 = [-mean, -mean]
cov2 = [[var,0],[0,var]]
g = 8

#define test set
x1, y1 = np.random.multivariate_normal(mean1, cov1, n).T
x2, y2 = np.random.multivariate_normal(mean2, cov2, n).T
Xtest = np.concatenate((np.asarray([x1,y1]).T,np.asarray([x2,y2]).T),axis=0)
ytest = np.asarray([-1]*(n) + [1]*(n))


biasSquared = np.zeros(sub)
preds = np.zeros((2*n,sub))

b = np.zeros(len(Cs))
v = np.zeros(len(Cs))

for j,C in enumerate(Cs):
    for i in range(sub):
        
        #create data - 2D Gaussians     
        x1, y1 = np.random.multivariate_normal(mean1, cov1, n).T
        x2, y2 = np.random.multivariate_normal(mean2, cov2, n).T
        X = np.concatenate((np.asarray([x1,y1]).T,np.asarray([x2,y2]).T),axis=0)
        y = np.asarray([-1]*(n) + [1]*(n))
        
        #fit SVM
        svc = svm.SVC(kernel=kernel,C=C).fit(X,y)
        preds[:,i] = svc.predict(Xtest)
        
        #plot 
        if i < 4:
            plt.subplot(2,2,i+1)
            plot_boundary(svc,g)
            plt.plot(x1, y1, 'b.', x2, y2, 'r.')

    plt.axis([-g,g,-g,g])
    plt.suptitle('Cost = %i' % (C))
    plt.show()
    b[j] = np.mean(np.mean(preds,1) - ytest) 
    v[j] = np.mean(np.var(preds,1))

plt.subplot(2,1,1)
plt.plot(Cs,b)
plt.title('bias')
plt.subplot(2,1,2)
plt.plot(Cs,v)
plt.title('variance')
plt.show()

