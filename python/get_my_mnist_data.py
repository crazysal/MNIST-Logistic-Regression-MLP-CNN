from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


def get_data():
	mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

	#Extract feature values and labels from the data
	mnist_train_labels = np.array(mnist.train.labels)
	mnist_train_images =  np.array(mnist.train.images)
	mnist_valid_images =  np.array(mnist.validation.images)
	mnist_valid_labels =  np.array(mnist.validation.labels)
	mnist_test_labels =  np.array(mnist.test.labels)
	mnist_test_images =  np.array(mnist.test.images)

	return mnist,mnist_train_labels, mnist_train_images, mnist_valid_images, mnist_valid_labels, mnist_test_labels, mnist_test_images

#Uncomment to test if loaded data is correct
# example = mnist_train_images[100]
# mnist_train_images.shape
# plt.imshow(np.reshape(example,[28,28]))
