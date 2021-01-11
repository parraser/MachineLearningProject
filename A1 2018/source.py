import numpy as np
import numpy.random as rnd
import time
import pickle
import matplotlib.pyplot as plt
import sklearn.linear_model as lin

with open("data1.pickle","rb") as f:
    dataTrain,dataTest = pickle.load(f)
    
print ("\n\nQuestion 1")

print ("\nQuestion 1(a):")
A =np.random.rand(4,3)
print (A)

print ("\nQuestion 1(b):")
x = np.random.rand(4,1)
print (x)

print ("\nQuestion 1(c):")
B = np.reshape(A, (2,6))
print (B)

print ("\nQuestion 1(d):")
C = A + x
print (C)

print ("\nQuestion 1(e):")
y = np.reshape(x,(4))
print (y)

print ("\nQuestion 1(f):")
A[ : ,0] = y
print (A)

print ("\nQuestion 1(g):")
A[ : ,0] = y + A[ : ,2]
print (A)

print ("\nQuestion 1(h):")
print(A[ : ,0:2])

print ("\nQuestion 1(i):")
print (A[0,:])
print (A[2,:])

print ("\nQuestion 1(j):")
print(A.sum())

print ("\nQuestion 1(k):")
print(A.max())

print ("\nQuestion 1(l):")
print(A.mean())

print ("\nQuestion 1(m):")
print(np.log(A**2))

print ("\nQuestion 1(n):")
print(np.matmul(A.transpose(),x))



def cube(A):
	square = np.zeros((A.shape))
        cube = np.zeros((A.shape))
        
        for row_count in range(0,A.shape[0]):
            
            for column_count in range(0,A.shape[1]):
                    P = 0
                    for element_count in range(0,A.shape[1]):  
                            
                            P = P + (A[row_count,element_count] * A[element_count,column_count])
                            square[row_count,column_count] = P 
            
            for column_count in range(0,A.shape[1]):
                    P = 0
                    for element_count in range(0,A.shape[1]):
                            
                            P= P + (square[row_count,element_count] * A[element_count,column_count])
                            cube[row_count,column_count] = P

            
	return cube


def mymeasure(N):
	
	A =np.random.rand(N,N)
        startnp = time.time()
        sqr = np.matmul(A,A)
        Cube1 = np.matmul(sqr,A)
        endnp = time.time()
        amountnp = endnp - startnp
        print(amountnp)
        startcube = time.time()
    	Cube2= cube(A)
        endcube = time.time()
        amountcube = endcube - startcube
        print(amountcube)
        print((np.absolute(Cube1 - Cube2).max()))


print("\nQuestion 2(c):")
'''
print("\n mymeasure(200)")
mymeasure(200)
print("\n mymeasure(2000)")
mymeasure(2000)
'''
print ("\nQuestion 4(a):")	

def kernelMatrix(X,S,sigma):
    
    S = np.reshape(S,[-1])
    K = np.zeros((X.shape[0], S.shape[0]+1))
    K[:,0] = 1
    kernel = np.zeros((X.shape[0], S.shape[0]))

    for s in range(0,S.shape[0]):
        kernel[:,s] = X - S[s]
    
    kernel= -(kernel)**2
    kernel = kernel/(2*(sigma)**2)
    K[:,1:S.shape[0]+1] = np.exp((kernel))
    
    return K

    
print ("\nQuestion 4(b):")	

def plotBasis(S,sigma):
    
    plt.figure()
    plt.suptitle('Question 4(b): some basis functions with sigma = 0.2')
    plt.xlabel("x")
    plt.ylabel("y")
    x = np.linspace(0,1,1000)
    k = kernelMatrix(x,S,sigma) 
    plt.plot(x,k[:,1:])
    plt.show()

plotBasis(dataTrain[0:5,0],0.2)


def myfit(S, sigma):
    
    kTrain = kernelMatrix(dataTrain[:,0],S,sigma) 
    w = np.linalg.lstsq(kTrain,dataTrain[:,1])[0]
    errTrain = np.mean((dataTrain[:,1] - np.matmul(kTrain,w))**2)
    
    kTest = kernelMatrix(dataTest[:,0],S,sigma)
    errTest = np.mean((dataTest[:,1] - np.matmul(kTest,w))**2)
    
    return w, errTrain, errTest


def plotY(w,S,sigma):
    
    x = np.linspace(0,1,1000)
    k = kernelMatrix(x,S,sigma) 
    w = np.reshape(w,[-1])
    y = np.matmul(k,w)
    plt.plot(x,y,"r")
    plt.plot(dataTrain[:,0], dataTrain[:,1],"b.")
    plt.ylim(-15,15)
    
    
print ("\nQuestion 4(e):")

plt.figure()
plt.suptitle('Question 4(e): the fitted function (5 basis functions)')
plt.xlabel("x")
plt.ylabel("y")
plotY(myfit(dataTrain[0:5,0],0.2)[0], dataTrain[0:5,0],0.2)
plt.show()

print ("\nQuestion 4(f):")

def bestM(sigma):
    
    plt.figure()
    plt.suptitle('Question 4(f): best-fitting functions with 0-15 basis functions')
    errTestMin = np.Inf
    errTrainVec = []
    errTestVec = []
    for M in range(0,16):
        w,errTrain,errTest = myfit(dataTrain[0:M,0],sigma)
        plt.subplot(4,4,M+1)
        plt.ylabel("y")
        plt.xlabel("x")
        plotY(w,dataTrain[0:M,0],0.2)
        errTrainVec.append(errTrain)
        errTestVec.append(errTest)
        
        if errTest< errTestMin:
            errTestMin = errTest
            w_fit = w
            m = M
    
    plt.figure()
    plt.suptitle('Question 4(f): training and test error')
    plt.ylabel("error")
    plt.xlabel("M")
    plt.plot(errTrainVec,'b')
    plt.plot(errTestVec, 'r')
    plt.ylim(0,250)
    
    plt.figure()
    plt.suptitle('Question 4(f): best-fitting function (8 basis functions)')
    plt.ylabel("y")
    plt.xlabel("x")
    plotY(w_fit,dataTrain[0:m,0],0.2)
    
    print("best-fitting w: " + str(w_fit))
    print("Value of M: " + str(m))
    print("Training Error " + str(errTrainVec[m]))
    print("Test Error " + str(errTestVec[m]))
    
    
bestM(0.2)
    
    

with open("data2.pickle","rb") as f:
    dataVal,dataTest = pickle.load(f)
        
print ("\nQuestion 5(a):")
    
def regFit(S,sigma,alpha):
    
    kTrain = kernelMatrix(dataTrain[:,0],S,sigma) 
    kVal = kernelMatrix(dataVal[:,0],S,sigma)
    
    ridge = lin.Ridge(alpha)
    ridge.fit(kTrain,dataTrain[:,1])
    w = ridge.coef_
    w[0] = ridge.intercept_

    errVal = np.mean((dataVal[:,1] - np.matmul(kVal,w))**2) #kVal
    errVal = errVal + alpha*np.sum(w**2)
    
    errTrain = np.mean((dataTrain[:,1] - np.matmul(kTrain,w))**2)
    errTrain = errTrain + alpha*np.sum(w**2)
    return w, errVal, errTrain

print ("\nQuestion 5(b):")

plt.figure()
plt.suptitle('Question 5(b): the fitted function (alpha=1)')
plt.ylabel("y")
plt.xlabel("x")
plotY(regFit(dataTrain[:,0],0.2, 1)[0], dataTrain[:,0],0.2)
plt.show() 


print ("\nQuestion 5(c):")

def bestAlpha(S,sigma):
    
    plt.figure()
    plt.suptitle('Question 5(c): best-fitting functions with 0-15 basis functions')
    errValMin = np.Inf
    errValVec = []
    errTrainVec = []
    
    a = -12
    m = 0
    alpha = 10.0**np.arange(-12,4)
    while(a != 4):
        
        w,errVal,errTrain = regFit(S,sigma,10**a) #regFit
        plt.subplot(4,4,m+1)
        plt.suptitle('Question 5(c): best-fitting functions for log(alpha) = -12,-11,...,1,2,3')
        plt.ylabel("y")
        plt.xlabel("x")
        plotY(w,dataTrain[:,0],0.2)
        
        errValVec.append(errVal)
        errTrainVec.append(errTrain)
        
        a+=1
        m+=1
        
        if errVal< errValMin:
            errValMin = errVal
            w_fit = w
            A = a
            M=m
    plt.show()
        
    plt.figure()
    plt.suptitle('Question 5(c): training and validation error')
    plt.ylabel("error")
    plt.xlabel("alpha")
    plt.semilogx(alpha,errValVec,'r')
    plt.plot(alpha, errTrainVec, 'b')
    plt.ylim(0,250)
    plt.show()
    
    plt.figure()
    plt.suptitle('Question 5(c): best-fitting function (-6 basis functions)')
    plt.ylabel("y")
    plt.xlabel("x")
    plotY(w_fit,dataTrain[0:m,0],0.2)
    plt.show()
    
    kTest = kernelMatrix(dataTest[:,0],S,sigma)
    errTest = np.mean((dataTest[:,1] - np.matmul(kTest,w_fit))**2)
    
    print("Optimal w: " + str(w_fit))
    print("Optimal alpha " + str(A))
    print("Training Error: " + str(errTrainVec[M]))
    print("Validation Error: " + str(errValVec[M]))
    print("Test Error: " + str(errTest)) 
    
    
bestAlpha(dataTrain[:,0],0.2)
    
print("\n“Question 6(a): Training data for Question 6")

dataTrain = np.copy(dataVal)
rnd.shuffle(dataTrain)

plt.figure()
plt.suptitle('Question 6(a): Training data for Question 6')
plt.ylabel("y")
plt.xlabel("x")
plt.plot(dataTrain[:,0], dataTrain[:,1],"b.")
plt.show()
    
def cross_val(K,S,sigma,alpha,X,Y):
    
    errTrainVec = []
    errValVec = []
    fold=X.shape[0]//K
 
    for i in range(0,K):
        
        inpVal = X[fold*i:(fold*i)+fold]
        inpTrain = X[0:fold*i]
        inpTrain = np.concatenate((inpTrain, X[(fold*i)+fold: ]), axis=None)
        tarVal = Y[fold*i:(fold*i)+fold]
        tarTrain = Y[0:fold*i]
        tarTrain = np.concatenate((tarTrain, Y[(fold*i)+fold: ]), axis=None)
        
        kVal= kernelMatrix(inpVal,S,sigma) 
        kTrain= kernelMatrix(inpTrain,S,sigma) 
    
        ridge = lin.Ridge(alpha)
        ridge.fit(kTrain,tarTrain)
        w = ridge.coef_
        w[0] = ridge.intercept_

        errVal = np.mean((tarVal - np.matmul(kVal,w))**2)

        errTrain = np.mean((tarTrain - np.matmul(kTrain,w))**2)
        
        errTrainVec.append(errTrain)
        errValVec.append(errVal)
    
    return errTrainVec,errValVec


errTrain,errVal = cross_val(5,dataTrain[0:10,0], 0.2,1.0, dataTrain[:,0],dataTrain[:,1])
plt.figure()
plt.suptitle('Question 6(c): training and validation errors during cross validation')
plt.ylabel("error")
plt.xlabel("fold")
plt.plot(errTrain,'b')
plt.plot(errVal, 'r')
plt.show()
print("Mean Training Error: " + str(np.mean(errTrain)))
print("Mean Validation Error: " + str(np.mean(errVal)))


def bestAlphaCV(K,S,sigma,X,Y):
    
    errValMin = np.Inf    
    
    errTrainAll = []
    errValAll = []
    A = 0
    M=0
    for a in range(-11,5):
        
        alpha = 10.0**a
        errTrain,errVal = cross_val(K,S,sigma,alpha,X,Y)
        
        errTrainAll.append(np.mean(errTrain))
        errValAll.append(np.mean(errVal))
        M+=1
        if np.mean(errVal) < errValMin:
            errValMin = np.mean(errVal)
            A = alpha
            m=M

    w, errVal, errTrain = regFit(S,sigma, A)
    
    kTest = kernelMatrix(dataTest[:,0],S,sigma)
    errTest = np.mean((dataTest[:,1] - np.matmul(kTest,w))**2)
    
    plt.figure()
    plt.suptitle('Question 6(d): training and validation error')
    alpha = 10.0**np.arange(-11,5)
    plt.ylabel("error")
    plt.xlabel("alpha")
    plt.semilogx(alpha,errTrainAll,'b')
    plt.semilogx(alpha, errValAll, 'r')
    plt.show()
    
    plt.figure()
    plt.suptitle('Question 6(d): best-fitting function ('+str(A)+')')
    plt.ylabel("y")
    plt.xlabel("x")
    plotY(w,S,sigma)
    plt.show()
    
    print("Training Error: " + str(np.mean(errTrainAll[m])))
    print("Mean Validation Error: " + str(np.mean(errValAll[m])))
    print("Test Error: " +str(errTest))

bestAlphaCV(5,dataTrain[0:15,0], 0.2, dataTrain[:,0],dataTrain[:,1])




with open("data1.pickle","rb") as f:
    dataTrain,dataTest = pickle.load(f)
    
    
def gradient(w,S,sigma,alpha):
    
    K = kernelMatrix(dataTrain[:,0],S,sigma)
    
    lxDerv = 2*(np.matmul(K,w) - dataTrain[:,1])
    lxDerv = np.matmul(np.transpose(K), lxDerv)
    grad = alpha*(2*w)
    grad[0] = 0.0
    grad = lxDerv + grad
   
    return grad
    
def fitRegGD(S,sigma,alpha,lrate):
    
    w = np.random.rand(S.shape[0]+1)
    
    errTrainVec = []
    errTestVec = []

    m=0
    pow_four = 4**np.array([0,1,2,3,4,5,6,7,8,9])
    plt.figure()
    for i in range(0, 100000):
       
        grad = gradient(w,S,sigma,alpha)
        w = w - lrate*(grad)
        
        kTrain = kernelMatrix(dataTrain[:,0],S,sigma) 
        errTrain = np.mean((dataTrain[:,1] - np.matmul(kTrain,w))**2)
        
        kTest = kernelMatrix(dataTest[:,0],S,sigma)
        errTest = np.mean((dataTest[:,1] - np.matmul(kTest,w))**2)
        
        errTestVec.append(errTest)
        errTrainVec.append(errTrain)
        
        if i == pow_four[m]:
            m += 1
            
            plt.suptitle('Question 7: fitted function as iterations increase')
            plt.subplot(3,3, m)
            plotY(w,S,sigma)
            
    plt.show()

            
    plt.figure()
    plt.suptitle('Question 7: fitted function')
    plt.ylabel("y")
    plt.xlabel("x")
    plotY(w,S,sigma)
    plt.show()
            
    plt.figure()
    plt.suptitle('Question 7: training and test error v.s. iteration')
    plt.ylabel("y")
    plt.xlabel("x")
    plt.plot(range(0,100000),errTrainVec,'b')
    plt.plot(range(0,100000),errTestVec, 'r')
    plt.show()
    
    plt.figure()
    plt.suptitle('Question 7: training and test error v.s. iterations (log scale)')
    plt.ylabel("error")
    plt.xlabel("alpha")
    plt.semilogx(range(0,100000),errTrainVec,'b')
    plt.semilogx(range(0,100000),errTestVec, 'r')
    plt.show()
    
    plt.figure()
    plt.suptitle('Question 7: last 10,000 training errors')
    plt.plot(range(90000,100000),errTrainVec[-10000:],'b')
    plt.show()
    
    kTrain = kernelMatrix(dataTrain[:,0],S,sigma) 
    kTest = kernelMatrix(dataTest[:,0],S,sigma)

    errTest = np.mean((dataTest[:,1] - np.matmul(kTest,w))**2)
    
    errTrain = np.mean((dataTrain[:,1] - np.matmul(kTrain,w))**2)
    
    
    print("w: " + str(w))
    print("Training Error: " + str(errTrain))
    print("Test Error: " + str(errTest))
    w2 = regFit(S,sigma,alpha)[0]

    print("w-w2 Magnitud: "+ str(np.absolute(w - w2).max()))
    print("Learning Rate: 0.01")
    print("Alpha: 0.01")

#fitRegGD(dataTrain[:,0], 0.2, 0.01, 0.01)