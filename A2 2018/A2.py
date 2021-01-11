import numpy as np
import numpy.random as rnd
from random import randint
import pickle
import matplotlib.pyplot as plt
import sklearn.linear_model as lin
import sklearn


print ("\n\nQuestion 1")


def genData(mu0,mu1,Sigma0,Sigma1,N):
    
    data0 = rnd.multivariate_normal(mu0,Sigma0, (N,1)).reshape((N,2))
    data1 = rnd.multivariate_normal(mu1,Sigma1, (N,1)).reshape((N,2))
    
    t0 = np.zeros(N, dtype = int) 
    t1 = np.ones(N, dtype = int)
    t = np.concatenate([t0,t1])
    X = np.concatenate([data0,data1])

    return sklearn.utils.shuffle(X,t)

print ("\n\nQuestion 1 (b)")

mu0 = [0,-1]
mu1 = [-1,1]
sigma0 = np.array([(2.0, 0.5), (0.5,1.0)])
sigma1 = np.array([(1.0,-1.0), (-1.0,2.0)])
X,t = genData(mu0,mu1,sigma0,sigma1,10000)

print ("\n\nQuestion 1(c): sample cluster data (10,000 points per cluster)")
colors = np.array(['r','b'])
plt.figure()
plt.suptitle('Question 1(c): sample cluster data (10,000 points per cluster)' )
plt.scatter(X[:,0],X[:,1], s = 2, c = colors[t])


print ("\n\nQuestion 2(b)")

mu0 = [0,-1]
mu1 = [-1,1]
sigma0 = np.array([(2.0, 0.5), (0.5,1.0)])
sigma1 = np.array([(1.0,-1.0), (-1.0,2.0)])
X,t = genData(mu0,mu1,sigma0,sigma1,1000)
#Classifier
clf = lin.LogisticRegression().fit(X,t)
#Bias
w0 = clf.intercept_
print(w0)
#weight Vector
w = clf.coef_
print(w)
#Accuracy
acc = clf.score(X,t)
print(acc)

print ("\n\nQuestion 2(c)")

colors = np.array(['r','b'])
plt.figure()
plt.suptitle('Question 2(c): training data and decision boundary' )
plt.scatter(X[:,0],X[:,1], s = 2, c = colors[t])
x = np.linspace(-5.0, 5.0, num=(1000))
plt.plot(x, -(w[0][0]*x+w0)/(w[0][1]), c='black')

print ("\n\nQuestion 2(d)")

colors = np.array(['r','b'])
plt.figure()
plt.suptitle('Question 2(d): decision boundaries for seven thresholds')
plt.scatter(X[:,0],X[:,1], s = 2, c = colors[t])
x = np.linspace(-5.0, 5.0, num=(1000))
t = -3
for s in range(0,7):
    
    if(t<0):
        color = 'r'
    if(t==0):
        color = 'black'
    if(t>0):
        color = 'b'
    
    plt.plot(x, -(w[0][0]*x+w0-t)/(w[0][1]), c=color)
    t += 1

print ("\n\nQuestion 2(e)  I Don't Know")

print ("\n\nQuestion 2(g)")

mu0 = [0,-1]
mu1 = [-1,1]
sigma0 = np.array([(2.0, 0.5), (0.5,1.0)])
sigma1 = np.array([(1.0,-1.0), (-1.0,2.0)])
X,t = genData(mu0,mu1,sigma0,sigma1,10000)

print("\n\nQuestion 2(h)")

clf = lin.LogisticRegression()
clf.fit(X,t.ravel())
clf.intercept_ -= 1
count = np.unique(clf.predict(X), return_counts=True)[1]

print('Predicted Positives: ', count[1])
print('Predicted Negatives: ', count[0])

truth = np.add(t.ravel(),clf.predict(X))
truthCount = np.unique(truth, return_counts=True)[1]

print('Number of True Positives: ', truthCount[2])
print('Number of True Negatives: ', truthCount[0])

false = np.subtract(t.ravel(),clf.predict(X))
falseCount = np.unique(false, return_counts=True)[1]

print('Number of False Negatives: ', falseCount[2])
print('Number of False Positives: ', falseCount[0])
precision = truthCount[2]/float(truthCount[2]+falseCount[0])
print('Precision: ', precision)
recall = truthCount[2]/float(truthCount[2]+falseCount[2])
print('Recall: ', recall)

print ("\n\nQuestion 2(i)  I Don't Know")
print ("\n\nQuestion 2(j)  I Don't Know")
print ("\n\nQuestion 2(k)  I Don't Know")
print ("\n\nQuestion 2(l)  I Don't Know")

print ("\n\nQuestion 3")

with open('mnist.pickle','rb') as f:
    Xtrain,Ytrain,Xtest,Ytest = pickle.load(f)

def plot_36():   
    plt.figure()
    plt.suptitle('Question 3(a):36 random MNIST images' )
    num = 0
    while(num != 36):
        X = np.reshape(Xtrain[randint(0, 60000),:],(28,28))
        plt.subplot(6,6, num+1)
        plt.axis('off')
        plt.imshow(X, interpolation='nearest', cmap='Greys')
        num += 1

plot_36()

print ("\n\nQuestion 3(b)")
#Multi-Classifier
def means():
    clf = lin.LogisticRegression(multi_class='multinomial',solver='lbfgs')
    clf = clf.fit(Xtrain,Ytrain)
    accTrain = clf.score(Xtrain,Ytrain)*100
    accTest = clf.score(Xtest,Ytest)*100
    
    print("Training mean " + str(accTrain))
    print("Test mean " + str(accTest))

means()
    
def bruteKN():
    
    TrainMean = []
    
    for K in range(1,21):
        clf = sklearn.neighbors.KNeighborsClassifier(n_neighbors=K, algorithm='brute')
        clf = clf.fit(Xtrain,Ytrain)
        accTrain = clf.score(Xtrain,Ytrain)*100
        TrainMean.append(accTrain)
        
    plt.figure()
    plt.suptitle('Figure 3(c): KNN test accuracy')
    plt.plot(range(1,21),TrainMean)
    
#bruteKN()
    
print ("\n\nQuestion 5(a)")


def softmax1(z):
    y = np.exp(z)/np.sum(np.exp(z))
    return y

print(softmax1((0,0)))
print(softmax1((1000,0)))
print(np.log(softmax1((-1000,0))))

print ("\n\nQuestion 5(c)")

def softmax2(z):
    y = np.exp(z - np.max(z))/np.sum(np.exp(z - np.max(z)))
    logy = z - np.max(z) - np.log(np.sum(np.exp(z - np.max(z))))
    return y, logy
    
print(softmax2((0,0)))
print(softmax2((1000,0)))
print(softmax2((-1000,0)))


print ("\n\nQuestion 6")

def calculateT(Y):
    
    T = np.zeros((Y.shape[0],10), dtype = int)
    for i in range(0,Y.shape[0]):
        T[i,Y[i]] = 1
    return T

def softmax(z):
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis]
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis]
    y = e_x / div
    return y, np.log(y)

def batch_update(lrate):
    
    w = np.random.rand(784,10)*0.1
    bias = np.zeros(10)
    Ttrain = calculateT(Ytrain) 
    Ttest = calculateT(Ytest)
    Etrain = []
    Etest = []
    accTrain = []
    accTest = []

    for i in range(0,5000):
       
        zTrain = np.matmul(Xtrain, w) + bias
        zTest = np.matmul(Xtest, w) + bias
        yTrain, logyTrain = softmax(zTrain)
        yTest, logyTest = softmax(zTest)
        w = w - lrate*(np.matmul(np.transpose(Xtrain),yTrain-Ttrain)/Xtrain.shape[0])
        bias = bias - lrate*(np.sum(yTrain-Ttrain)/Xtrain.shape[0])
        
        if i%10 == 0:
            
            Etrain.append(-np.sum((Ttrain*logyTrain))/Xtrain.shape[0])
            print('Loss Train ', -np.sum((Ttrain*logyTrain))/Xtrain.shape[0])
            Etest.append(-np.sum((Ttest*logyTest))/Xtest.shape[0])
            print('Loss Test ', -np.sum((Ttest*logyTest))/Xtest.shape[0])
            accTrain.append(np.mean(np.argmax(yTrain,axis=1) == np.argmax(Ttrain,axis=1))*100.0)
            print('Acc Train ', np.mean(np.argmax(yTrain,axis=1) == np.argmax(Ttrain,axis=1))*100.0)
            accTest.append(np.mean(np.argmax(yTest,axis=1) == np.argmax(Ttest,axis=1))*100.0)
            print('Acc Test ',np.mean(np.argmax(yTest,axis=1) == np.argmax(Ttest,axis=1))*100.0)
            
    print('Learning Rate =', lrate)
    print('Last Training Loss and Accuracy', Etrain[-1], accTrain[-1])
    print('Last Test Loss and Accuracy', Etest[-1], accTest[-1])
    
    plt.figure()
    plt.suptitle('Question 6(e): training and test accuracy for batch gradient descent')
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.semilogx(range(0,5000,10),accTrain, '#FFA500')
    plt.semilogx(range(0,5000, 10),accTest, 'b')
    
    plt.figure()
    plt.suptitle('Question 6(e): training and test loss for batch gradient descent')
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.semilogx(range(0,5000,10),Etrain, '#FFA500')
    plt.semilogx(range(0,5000,10),Etest, 'b')

    plt.figure()
    plt.suptitle('Question 6(g): training accuracy for last 2000 epochs of bgd')
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.plot(range(3000,5000,10),accTrain[-2000:])
    
    plt.figure()
    plt.suptitle('“Question 6(h): training loss for last 2000 epochs of bgd')
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.plot(range(3000,5000,10),Etest[-2000:])
    
   
batch_update(1)
    
print ("\n\nQuestion 7  I Don't Know")