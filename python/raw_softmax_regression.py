import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse

import get_my_mnist_data
mnist, mnist_train_labels, mnist_train_images, mnist_valid_images, mnist_valid_labels, mnist_test_labels, mnist_test_images = get_my_mnist_data.get_data()  

# ...
# Here we determine the probabilities and predictions for each class when given a set of input data
# ...
def getProbsAndPreds(someX):
    probs = softmax(np.dot(someX,w))
    preds = np.argmax(probs,axis=1)
    return probs,preds
# ...
# Here we perform the softmax transformation: This allows us to get probabilities for each class score that sum to 100%.
# ...
def softmax(z):
    z -= np.max(z)
    sm = (np.exp(z).T / np.sum(np.exp(z),axis=1)).T
    return sm
# ...
# Here we calculate the accuracy of the predicted data
# ...
def getAccuracy(someX,someY):
    prob,prede = getProbsAndPreds(someX)
    c = 0;
    for i in range(len(someY)):
        test = someY[i, prede[i]]
        if  test == 1:
            c+=1
    accuracy = c/(float(len(someY)))        
    return accuracy
# ...
# Here we define the loss function for softmax regression.
# ...
def getLoss(w,x,y,lam):
    m = x.shape[0] #First we get the number of training examples    
    scores = np.dot(x,w) #Then we compute raw class scores given our input and current weights
    prob = softmax(scores) #Next we perform a softmax on these scores to get their probabilities
    loss = (-1 / m) * np.sum(y * np.log(prob)) + (lam/2)*np.sum(w*w) #We then find the loss of the probabilities
    grad = (-1 / m) * np.dot(x.T,(y - prob)) + lam*w #And compute the gradient for that loss
    return loss,grad

# ...
# This is the main loop of the softmax regression.
# Here we initialize our weights, regularization factor, number of iterations,
# and learning rate. We then loop over a computation of the loss and gradient,
# and application of gradient.
# ...
x = mnist_train_images
y = mnist_train_labels
w = np.zeros([x.shape[1], 10])
lam = 1
iterations = 100
learningRate = 1e-5
losses = []
for i in range(0,iterations):
    loss,grad = getLoss(w,x,y,lam)
    losses.append(loss)
    w = w - (learningRate * grad)
    print("Loss for iteration : ", i, "is : ", loss)

print(loss)
plt.plot(losses)
#Visualize predicted Weights
classWeightsToVisualize = 8
plt.imshow(scipy.reshape(w[:,classWeightsToVisualize],[28,28]))
#Print Accuracy
print ('Training Accuracy: ', getAccuracy(x,y))
testX = mnist_test_images
testY = mnist_test_labels
print ( 'Test Accuracy: ', getAccuracy(testX,testY))
