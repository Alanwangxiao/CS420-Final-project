import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.utils import to_categorical
from tensorflow.contrib.layers import flatten
from sklearn.metrics import classification_report, confusion_matrix


NUM_ITERS=10000
DISPLAY_STEP=100

tf.set_random_seed(0)

# Download images and labels 
x_train= np.fromfile("mnist_train/mnist_train_data",dtype=np.uint8)
x_train=x_train/255
x_train=x_train.reshape(60000,45*45)
y_train = np.fromfile("mnist_train/mnist_train_label",dtype=np.uint8)
y_train = to_categorical(y_train, num_classes=10)
print(x_train.shape)
print(y_train.shape)
x_test=np.fromfile("mnist_test/mnist_test_data",dtype=np.uint8)
x_test=x_test/255
x_test=x_test.reshape(10000,45*45)
y_test=np.fromfile("mnist_test/mnist_test_label",dtype=np.uint8)
y_testorg=y_test
y_test = to_categorical(y_test, num_classes=10)
print(x_test.shape)
print(y_test.shape)
 

def KNN():   

	K=5
	xtr = tf.placeholder(tf.float32,[None,2025])  
	xte = tf.placeholder(tf.float32,[2025])  
	ytr=tf.placeholder(tf.float32,[None,10])

	distance = tf.negative(tf.reduce_sum(tf.pow(tf.subtract(xtr, xte),2),axis=1))  
	
	values,indices=tf.nn.top_k(distance,k=K,sorted=False)

	nn = []
	nearest_neighbors=tf.Variable(tf.zeros([K]))
	for i in range(K):
		nn.append(tf.argmax(ytr[indices[i]], 0)) #taking the result indexes
	nearest_neighbors=nn

	y, idx, count = tf.unique_with_counts(nearest_neighbors)
	pred = tf.slice(y, begin=[tf.argmax(count, 0)], size=tf.constant([1], dtype=tf.int64))[0] 
  
	init = tf.global_variables_initializer() 
	sess = tf.Session()  
	sess.run(init)  
  
	right = 0  
	prediction=[]
	for i in range(NUM_ITERS):  
		ansIndex = sess.run(pred,{xtr:x_train,xte:x_test[i,:],ytr:y_train})
		prediction.append(ansIndex)
		#print (ansIndex," ",np.argmax(y_test[i]))  
		#print ('prediction is ',np.argmax(y_train[ansIndex]))  
		#print ('true value is ',np.argmax(y_test[i]))  
		if np.argmax(y_test[i])==ansIndex: 
			right += 1.0
		if i % DISPLAY_STEP == 0 and i!=0:
			print(right/i)
	print("Optimization Finished!")
	accracy = right/NUM_ITERS 
	print(accracy)

	print(classification_report(y_testorg, prediction))
	cm=confusion_matrix(y_testorg, prediction)
	print(cm)
	plot_confusion_matrix(cm,target_names=["0","1","2","3","4","5","6","7","8","9"])

def plot_confusion_matrix(cm, target_names, title='Confusion matrix', cmap="YlGnBu"):
	plt.figure(1, figsize=(8, 6))
	plt.imshow(cm/cm.sum(axis=1), interpolation='nearest', cmap=cmap)
	#plt.title(title)
	plt.colorbar()
	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	tick_marks = np.arange(len(target_names))
	plt.xticks(tick_marks, target_names)
	plt.yticks(tick_marks, target_names)
	plt.show()
	plt.savefig("KNN.png")

if __name__ == "__main__":   
    KNN()


