import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
import sklearn.neural_network as nn
import bonnerlib2 as graph
import sklearn

print ("\n\nQuestion 1(a)")

def genData(mu0,mu1,Sigma0,Sigma1,N):
    
    data0 = rnd.multivariate_normal(mu0,Sigma0, (N,1)).reshape((N,2))
    data1 = rnd.multivariate_normal(mu1,Sigma1, (N,1)).reshape((N,2))
    
    t0 = np.zeros(N, dtype = int) 
    t1 = np.ones(N, dtype = int)
    t = np.concatenate([t0,t1])
    X = np.concatenate([data0,data1])

    return sklearn.utils.shuffle(X,t)

mu0 = [0,-1]
mu1 = [-1,1]
sigma0 = np.array([(2.0, 0.5), (0.5,1.0)])
sigma1 = np.array([(1.0,-1.0), (-1.0,2.0)])
X_train,t_train = genData(mu0,mu1,sigma0,sigma1,1000)
X_test,t_test = genData(mu0,mu1,sigma0,sigma1,10000)

#--------------------------------------------------------------------------------------------------------------------------------
print ("\n\nQuestion 1(b)")

#Classifier 
clf = nn.MLPClassifier((1,), "tanh", "sgd", learning_rate_init=0.01, max_iter= 10000,tol=10**(-10))
clf = clf.fit(X_train,t_train)

colors = np.array(['r','b'])
plt.figure()
plt.suptitle('Question 1(b): Neural net with 1 hidden unit' )
plt.scatter(X_train[:,0],X_train[:,1], s = 2, c = colors[t_train])
graph.dfContour(clf)
print("Explain this similarity")


#-------------------------------------------------------------------------------------------------------------------------

print ("\n\nQuestion 1(c)")

colors = np.array(['r','b'])
plt.figure()
plt.suptitle('Question 1(c): Neural net with 2 hidden units' )
bestAcc = 0
num = 0
bestNum = 0
while(num != 9):
    plt.subplot(3,3, num+1)
    clf = nn.MLPClassifier((2,), "tanh", "sgd", learning_rate_init=0.01, max_iter= 10000,tol=10**(-10))
    clf = clf.fit(X_train,t_train)
    plt.scatter(X_train[:,0],X_train[:,1], s = 2, c = colors[t_train])
    graph.dfContour(clf)
    #Accuracy
    acc = clf.score(X_test,t_test)
    if (acc>bestAcc):
        bestAcc = acc
        bestNum = num
        bestclfC = clf
    
    num += 1
    
plt.figure()
plt.suptitle('Question 1(c): Best neural net with 2 hidden unit' )
plt.scatter(X_train[:,0],X_train[:,1], s = 2, c = colors[t_train])
graph.dfContour(bestclfC)
print(bestAcc, bestNum)

#------------------------------------------------------------------------------------------------------

print ("\n\nQuestion 1(d)")

colors = np.array(['r','b'])
plt.figure()
plt.suptitle('Question 1(d): Neural net with 3 hidden units' )
bestAcc = 0
num = 0
bestNum = 0
while(num != 9):
    plt.subplot(3,3, num+1)
    clf = nn.MLPClassifier((3,), "tanh", "sgd", learning_rate_init=0.01, max_iter= 10000,tol=10**(-10))
    clf = clf.fit(X_train,t_train)
    plt.scatter(X_train[:,0],X_train[:,1], s = 2, c = colors[t_train])
    graph.dfContour(clf)
    #Accuracy
    acc = clf.score(X_test,t_test)
    if (acc>bestAcc):
        bestAcc = acc
        bestNum = num
        bestclfD = clf
    
    num += 1
    
plt.figure()
plt.suptitle('Question 1(d): Best neural net with 3 hidden unit' )
plt.scatter(X_train[:,0],X_train[:,1], s = 2, c = colors[t_train])
graph.dfContour(bestclfD)
print(bestAcc, bestNum)

#-----------------------------------------------------------------------------------------------

print ("\n\nQuestion 1(e)")

colors = np.array(['r','b'])
plt.figure()
plt.suptitle('Question 1(e): Neural net with 4 hidden units' )
bestAcc = 0
num = 0
bestNum = 0
while(num != 9):
    plt.subplot(3,3, num+1)
    clf = nn.MLPClassifier((4,), "tanh", "sgd", learning_rate_init=0.01, max_iter= 10000,tol=10**(-10))
    clf = clf.fit(X_train,t_train)
    plt.scatter(X_train[:,0],X_train[:,1], s = 2, c = colors[t_train])
    graph.dfContour(clf)
    #Accuracy
    acc = clf.score(X_test,t_test)
    if (acc>bestAcc):
        bestAcc = acc
        bestNum = num
        bestclfE = clf
    
    num += 1
    
plt.figure()
plt.suptitle('Question 1(e): Best neural net with 4 hidden unit' )
plt.scatter(X_train[:,0],X_train[:,1], s = 2, c = colors[t_train])
graph.dfContour(bestclfE)
print(bestAcc, bestNum)

#-----------------------------------------------------------------------------------------------

print ("\n\nQuestion 1(g)")

#Bias
w0 = bestclfD.intercepts_
#weight Vector
w = bestclfD.coefs_

plt.figure()
plt.suptitle('Question 1(g): Decision boundaries for 3 hidden units' )
plt.scatter(X_train[:,0],X_train[:,1], s = 2, c = colors[t_train])
x = np.linspace(-5.0, 5.0, num=(1000))
plt.xlim((-5,5))
plt.ylim((-7,7))
plt.plot(x,-(w[0][0][0]*x+w0[0][0])/(w[0][1][0]), c='black', linestyle='dashed')
plt.plot(x,-(w[0][0][1]*x+w0[0][1])/(w[0][1][1]), c='black', linestyle='dashed')
plt.plot(x,-(w[0][0][2]*x+w0[0][2])/(w[0][1][2]), c='black', linestyle='dashed')
graph.dfContour(bestclfD)

#-----------------------------------------------------------------------------------------------

print ("\n\nQuestion 1(h)")

#Bias
w0 = bestclfC.intercepts_
#weight Vector
w = bestclfC.coefs_

plt.figure()
plt.suptitle('Question 1(h): Decision boundaries for 2 hidden units' )
plt.scatter(X_train[:,0],X_train[:,1], s = 2, c = colors[t_train])
x = np.linspace(-5.0, 5.0, num=(1000))
plt.xlim((-5,5))
plt.ylim((-7,7))
plt.plot(x,-(w[0][0][0]*x+w0[0][0])/(w[0][1][0]), c='black', linestyle='dashed')
plt.plot(x,-(w[0][0][1]*x+w0[0][1])/(w[0][1][1]), c='black', linestyle='dashed')
graph.dfContour(bestclfC)

#-----------------------------------------------------------------------------------------------

print ("\n\nQuestion 1(i)")

#Bias
w0 = bestclfE.intercepts_
#weight Vector
w = bestclfE.coefs_

plt.figure()
plt.suptitle('Question 1(i): Decision boundaries for 4 hidden units' )
plt.scatter(X_train[:,0],X_train[:,1], s = 2, c = colors[t_train])
x = np.linspace(-5.0, 5.0, num=(1000))
plt.xlim((-5,5))
plt.ylim((-7,7))
plt.plot(x,-(w[0][0][0]*x+w0[0][0])/(w[0][1][0]), c='black', linestyle='dashed')
plt.plot(x,-(w[0][0][1]*x+w0[0][1])/(w[0][1][1]), c='black', linestyle='dashed')
plt.plot(x,-(w[0][0][2]*x+w0[0][2])/(w[0][1][2]), c='black', linestyle='dashed')
plt.plot(x,-(w[0][0][3]*x+w0[0][3])/(w[0][1][3]), c='black', linestyle='dashed')
graph.dfContour(bestclfE)

#--------------------------------------------------------------------------------------------------

print('\n\nQuestion 1(k)')

#Generate the curve

M = 1000
t = np.linspace(0,1,M)
precision = np.zeros([M])
recall = np.zeros([M])
#TruePos = np.zeros([20000,4])
numPos = np.sum(t_test)
lol = w[0]
z= bestclfD.predict_proba(X_test)[:,1]
for n in range(M):
    PredictedPos = z>=t[n]
    TruePos = t_test & PredictedPos
    numPP = np.sum(PredictedPos)
    numTP = np.sum(TruePos)
    
    precision[n] = numTP/np.float(numPP)
    recall[n] = numTP/np.float(numPos)

plt.figure()
plt.suptitle('Question 1(k): Precision/recall curve')
plt.plot(recall,precision)
plt.xlabel('Recall')
plt.ylabel('Precision')

#----------------------------------------------------------------------------------------------------

print('\n\nQuestion 1(l)')

area = 0.0
for m in range(1, len(recall)):
    area+= precision[m] * (recall[m-1]-recall[m])
print(area*100)

#--------------------------------------------------------------------------------------------------

print('\n\nQuestion 3(a)')

X_train,t_train = genData(mu0,mu1,sigma0,sigma1,10000)
X_test,t_test = genData(mu0,mu1,sigma0,sigma1,10000)


def forward(X,V,v0,W,w0):
    u = np.matmul(X,V)+v0
    h = np.tanh(u)
    z = np.matmul(h,W)+w0
    o = 1 /(1 + np.exp(-z))
    o = o.reshape(-1,1)[:,0]
    return u,h,z,o

nn3 = bestclfD
w0 = nn3.intercepts_
w = nn3.coefs_
u,h,z,o1 = forward(X_test, w[0],w0[0],w[1],w0[1])

o2 = nn3.predict_proba(X_test)

diff = np.sum((o1-o2[:,1])**2)
print(diff)

#-----------------------------------------------------------------------------------------------------

print('\n\nQuestion 3(b)')


# *** modified ***
    
def MYdfContour(V,v0,W,w0):
    ax = plt.gca()
    # The extent of xy space
    x_min,x_max = ax.get_xlim()
    y_min,y_max = ax.get_ylim()
     
    # form a mesh/grid over xy space
    h = 0.02    # mesh granularity
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    mesh = np.c_[xx.ravel(),yy.ravel()]
    
    # evaluate the decision functrion at the grid points
    Z = forward(mesh, V,v0,W,w0)[3]
    
    # plot the contours of the decision function
    Z = Z.reshape(xx.shape)
    mylevels=np.linspace(0.0,1.0,11)
    ax.contourf(xx, yy, Z, levels=mylevels, cmap=cm.RdBu, alpha=0.5)
    
    # draw the decision boundary in solid black
    ax.contour(xx, yy, Z, levels=[0.5], colors='k', linestyles='solid')

def gradient(h,o,T,X,W):
    l = (o-T)/ np.shape(h)[0]
    w = np.matmul(np.transpose(h), l)
    w0 = np.sum(l,axis= 0)
    W = W.reshape([W.shape[0], 1])
    l = l.reshape([l.shape[0],1])
    l2 = np.matmul(l,np.transpose(W))*(1-h**2)
    v = np.matmul(np.transpose(l2), X)
    v0 = np.sum(l2,axis=0)
    return w, w0, np.transpose(v), v0

def bgd(J,K,lrate):
    
    #Ttrain = np.zeros([np.shape(X_train)[0]])
    #Ttest = np.zeros([np.shape(X_test)[0],K])

    W_weights = rnd.randn(J)
    W_bias = 0.0
    V_weights = rnd.randn(2,J)
    V_bias = np.zeros(J)
    
    Etrain = []

    accTrain = []
    accTest = []

    for i in range(0,K):
       
        u,h,z,o1 = forward(X_train, V_weights,V_bias,W_weights,W_bias)
     
        NW, Nw0, NV, Nv0  = gradient(h,o1, t_train, X_train, W_weights)

        W_weights = W_weights - lrate*NW
        W_bias = W_bias - lrate*Nw0
        V_weights = V_weights - lrate*NV
        V_bias = V_bias- lrate*Nv0
        
        
         
        if (i%10 == 0):
            o_test = forward(X_test, V_weights,V_bias,W_weights,W_bias)[3]
            o_train = forward(X_train, V_weights,V_bias,W_weights,W_bias)[3]
            predictions_test = o_test > 0.5 
            predictions_train = o_train > 0.5
            
            Etrain.append(-np.sum((t_train*np.log(o1)))/np.shape(t_train)[0])
            print('Loss Train ', -np.sum((t_train*np.log(o1)))/np.shape(t_train)[0])
 
            accTrain.append(np.mean(t_train == predictions_train)*100.0)
            print('Acc Train ', np.mean(t_train == predictions_train)*100.0)
            accTest.append(np.mean(t_test == predictions_test)*100.0)
            print('Acc Test ', np.mean(t_test == predictions_test)*100.0)
 

    print('Final Test Accuracy', accTest[-1])

    plt.figure()
    plt.suptitle('Question 3(b):training and test accuracy for bgd')
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.semilogx(range(10,K+1,10),accTrain, '#FFA500')
    plt.semilogx(range(10,K+1, 10),accTest, 'b')
    
    plt.figure()
    plt.suptitle('Question 3(b):training loss for bgd')
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.semilogx(range(10,K+1,10),Etrain, '#FFA500')
                 
    plt.figure()
    plt.suptitle('Question 3(b): final test accuracy for bgd')
    plt.ylabel("accuracy")
    plt.xlabel("training time")
    plt.plot(accTest[-K//2:])
    
    plt.figure()
    plt.suptitle('Question 3(b): final training loss for bgd')
    plt.ylabel("accuracy")
    plt.xlabel("training time")
    plt.plot(Etrain[-K//2:])
    
    plt.figure()
    plt.suptitle('Question 3(b): Decisiom boundary for my neural net' )
    plt.scatter(X_train[:,0],X_train[:,1], s = 2, c = colors[t_train])
    MYdfContour(V_weights,V_bias,W_weights,W_bias)
     
bgd(3,1000,1.0)

#--------------------------------------------------------------------------------------

print('\n\nQuestion 3(c)')

def sgd(J,K,lrate):
    W_weights = rnd.randn(J)
    W_bias = 0.0
    V_weights = rnd.randn(2,J)
    V_bias = np.zeros(J)
    
    Etrain = []

    accTrain = []
    accTest = []
    
    for i in range(0,K):
               
        N1 = 0
        while (N1 < np.shape(t_train)[0]):
            
            N2 = np.min([N1+50,np.shape(t_train)[0]])
            X = X_train[N1:N2]
            T = t_train[N1:N2]
            N1 = N2
            
            u,h,z,o1 = forward(X, V_weights,V_bias,W_weights,W_bias)
            NW, Nw0, NV, Nv0  = gradient(h,o1, T, X, W_weights)
            
            W_weights = W_weights - lrate*NW
            W_bias = W_bias - lrate*Nw0
            V_weights = V_weights - lrate*NV
            V_bias = V_bias- lrate*Nv0

        o_test = forward(X_test, V_weights,V_bias,W_weights,W_bias)[3]
        o_train = forward(X_train, V_weights,V_bias,W_weights,W_bias)[3]
        predictions_test = o_test > 0.5 
        predictions_train = o_train > 0.5
        
        Etrain.append(-np.sum((T*np.log(o1)))/np.shape(T)[0])
        print('Loss Train ', -np.sum((T*np.log(o1)))/np.shape(T)[0])
     
        accTrain.append(np.mean(t_train == predictions_train)*100.0)
        print('Acc Train ', np.mean(t_train == predictions_train)*100.0)
        accTest.append(np.mean(t_test == predictions_test)*100.0)
        print('Acc Test ', np.mean(t_test == predictions_test)*100.0)
 

    print('Final Test Accuracy', accTest[-1])

    plt.figure()
    plt.suptitle('Question 3(c):training and test accuracy for bgd')
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.semilogx(range(1,K+1),accTrain, '#FFA500')
    plt.semilogx(range(1,K+1),accTest, 'b')
    
    plt.figure()
    plt.suptitle('Question 3(c):training loss for bgd')
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.semilogx(range(1,K+1),Etrain, '#FFA500')
                 
    plt.figure()
    plt.suptitle('Question 3(c): final test accuracy for bgd')
    plt.ylabel("accuracy")
    plt.xlabel("training time")
    plt.plot(accTest[-K//2:])
    
    plt.figure()
    plt.suptitle('Question 3(c): final training loss for bgd')
    plt.ylabel("accuracy")
    plt.xlabel("training time")
    plt.plot(Etrain[-K//2:])
    
    plt.figure()
    plt.suptitle('Question 3(c): Decisiom boundary for my neural net' )
    plt.scatter(X_train[:,0],X_train[:,1], s = 2, c = colors[t_train])
    MYdfContour(V_weights,V_bias,W_weights,W_bias)
    
sgd(3,20,1.0)