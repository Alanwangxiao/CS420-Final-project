import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.utils import to_categorical
from tensorflow.contrib.layers import flatten
from sklearn.metrics import classification_report, confusion_matrix

NUM_ITERS=5000
DISPLAY_STEP=100
BATCH=200

tf.set_random_seed(0)

#import images and labels 
x_train= np.fromfile("mnist_train/mnist_train_data",dtype=np.uint8)
x_train=x_train/255
x_train=x_train.reshape(60000,45,45,1)
y_train = np.fromfile("mnist_train/mnist_train_label",dtype=np.uint8)
y_train = to_categorical(y_train, num_classes=10)
print(x_train.shape)
print(y_train.shape)
x_test=np.fromfile("mnist_test/mnist_test_data",dtype=np.uint8)
x_test=x_test/255
x_test=x_test.reshape(10000,45,45,1)
y_test=np.fromfile("mnist_test/mnist_test_label",dtype=np.uint8)
y_test = to_categorical(y_test, num_classes=10)
print(x_test.shape)
print(y_test.shape)

# Parameters
learning_rate = 0.001

# layers sizes
c1 = 6  # first convolutional layer output depth
c2 = 8  # second convolutional layer output depth
c3 = 16 # third convolutional layer output depth

# fully connected layer
FC1 = 120  
FC2 = 84
FC3 = 10

# tf Graph input
X = tf.placeholder(tf.float32, [None, 45, 45, 1])
Y = tf.placeholder(tf.float32, [None, 10])
dropout1 = tf.placeholder(tf.float32)

# Store layers weight & bias
W1= tf.Variable(tf.truncated_normal([5,5,1,c1],stddev=0.1))
W2= tf.Variable(tf.truncated_normal([4,4,c1,c2],stddev=0.1))
W3= tf.Variable(tf.truncated_normal([6,6,c2,c3],stddev=0.1))
Wfc1= tf.Variable(tf.truncated_normal([7*7*c3,FC1],stddev=0.1))
Wfc2= tf.Variable(tf.truncated_normal([FC1,FC2],stddev=0.1))
Wfc3= tf.Variable(tf.truncated_normal([FC2,FC3],stddev=0.1))

b1= tf.Variable(tf.truncated_normal([c1],stddev=0.1))
b2= tf.Variable(tf.truncated_normal([c2],stddev=0.1))
b3= tf.Variable(tf.truncated_normal([c3],stddev=0.1))
bfc1= tf.Variable(tf.truncated_normal([FC1],stddev=0.1))
bfc2= tf.Variable(tf.truncated_normal([FC2],stddev=0.1))
bfc3= tf.Variable(tf.truncated_normal([FC3],stddev=0.1))

# Create model
def neural_net(x):
    stride=1
    k=2
    
    rand_index1 =np.random.choice(len(x_train), size=BATCH)
    rand_x1 = x_train[rand_index1]
    print(X.shape)

    Y1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1, stride, stride, 1], padding='VALID') + b1)
    print(Y1.shape)
    
    Y2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1, stride, stride, 1], padding='VALID') + b2)
    print(Y2.shape)
    # max pool filter size and stride, will reduce input by factor of 2
    Y2 = tf.nn.max_pool(Y2, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='VALID')
    print(Y2.shape)

    Y3 = tf.nn.relu(tf.nn.conv2d(Y2, W3, strides=[1, stride, stride, 1], padding='VALID') + b3)
    print(Y3.shape)

    Y4 = tf.nn.max_pool(Y3, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='VALID')
    print(Y4.shape)
    
    # reshape the output from the pooling layer for the fully connected layer
    Y5 = flatten(Y4)
    print(Y5.shape)

    fc1 = tf.nn.relu(tf.matmul(Y5, Wfc1) + bfc1)
    print(fc1.shape)

    fc2 = tf.nn.relu(tf.matmul(fc1, Wfc2) + bfc2)
    print(fc2.shape)

    Ylogits = tf.nn.sigmoid(tf.matmul(fc2, Wfc3) + bfc3)
    Y_ = tf.nn.softmax(Ylogits)
    return Ylogits,Y_
# loss function: cross-entropy = - sum( Y_i * log(Yi) )
# here we end up with the total cross-entropy for all images in the batch
# cross_entropy = -tf.reduce_mean(Y_ * tf.log(Y))
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

# Construct model
Ylogits,Y_ = neural_net(X)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y)
cross_entropy = tf.reduce_mean(cross_entropy)

correct_prediction = tf.equal(tf.argmax(Y_, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

 # Define loss and optimizer 
loss_op = cross_entropy
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

loss_vec = []
train_accuracy = []
test_accuracy = []

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:
    # Run the initializer
    sess.run(init)

    for step in range(NUM_ITERS+1):
        rand_index =np.random.choice(len(x_train), size=BATCH)
        rand_x = x_train[rand_index]
        rand_y = y_train[rand_index]
        sess.run(train_op, feed_dict={X:rand_x, Y:rand_y, dropout1:1.0})
        if step % DISPLAY_STEP == 0 :
            # Calculate batch loss and accuracy
            loss_trn = sess.run(loss_op, feed_dict={X: rand_x, Y: rand_y, dropout1:1.0})
            
            train_acc_temp=sess.run(accuracy,feed_dict={X: rand_x,Y: rand_y,dropout1:1.0})

            test_acc_temp=sess.run(accuracy,feed_dict={X: x_test,Y: y_test,dropout1:1.0})

            train_accuracy.append(train_acc_temp)
            test_accuracy.append(test_acc_temp)
            loss_vec.append(loss_trn)
            print("#{} Trn acc={} , Trn loss={} Tst acc={}".format(step,train_acc_temp,loss_trn,test_acc_temp))

    print("Optimization Finished!")

    print("Testing Accuracy:", sess.run(accuracy, feed_dict={X: x_test,Y: y_test,dropout1:1.0}))
    y_pred=sess.run(Y_,feed_dict={X: x_test,Y: y_test})
    
    plt.plot(train_accuracy)
    plt.plot(test_accuracy)
    plt.legend(['train_accuracy','test_accuracy'])
    plt.ylim(0.,1.)
    plt.show()

    plt.plot(loss_vec)
    plt.title('Loss per Generation')
    plt.xlabel('Generation')
    plt.ylabel('Loss')
    plt.show()

    print(classification_report(np.argmax(y_test,1), np.argmax(y_pred,1)))
    cm=confusion_matrix(np.argmax(y_test,1), np.argmax(y_pred,1))
    print(cm)
    plot_confusion_matrix(cm,target_names=["0","1","2","3","4","5","6","7","8","9"])
