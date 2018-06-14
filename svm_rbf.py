# Multi-class RBF SVM
#
# Gaussian Kernel:
# K(x1, x2) = exp(-gamma * abs(x1 - x2)^2)
#
# one vs all classification.

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.utils import to_categorical
from tensorflow.python.framework import ops
ops.reset_default_graph()

# Create graph
sess = tf.Session()

data_num = 60000 #The number of figures
fig_w = 45       #width of each figure
# Load the data
x_train= np.fromfile("mnist_train/mnist_train_data",dtype=np.uint8)
x_train=x_train/255
x_train=x_train.reshape(60000,45*45)
y_train = np.fromfile("mnist_train/mnist_train_label",dtype=np.uint8)

x_test=np.fromfile("mnist_test/mnist_test_data",dtype=np.uint8)
x_test=x_test/255
x_test=x_test.reshape(10000,45*45)
y_test=np.fromfile("mnist_test/mnist_test_label",dtype=np.uint8)

#generate multiclass labels
labels = np.hstack((y_train,y_test))
print(labels.shape)
y_vals0 = np.array([1 if y == 0 else -1 for y in labels])
y_vals1 = np.array([1 if y == 1 else -1 for y in labels])
y_vals2 = np.array([1 if y == 2 else -1 for y in labels])
y_vals3 = np.array([1 if y == 3 else -1 for y in labels])
y_vals4 = np.array([1 if y == 4 else -1 for y in labels])
y_vals5 = np.array([1 if y == 5 else -1 for y in labels])
y_vals6 = np.array([1 if y == 6 else -1 for y in labels])
y_vals7 = np.array([1 if y == 7 else -1 for y in labels])
y_vals8 = np.array([1 if y == 8 else -1 for y in labels])
y_vals9 = np.array([1 if y == 9 else -1 for y in labels])
y_vals = np.array([y_vals0, y_vals1, y_vals2, y_vals3, y_vals4, y_vals5, y_vals6, y_vals7, y_vals8, y_vals9])
print(y_vals.shape)
ytr=y_vals[:,0:60000]
yte=y_vals[:,60000:70000]
y_train=np.transpose(ytr)
y_test=np.transpose(yte)

# Declare batch size
batch_size = 2000
num_features = 2025
# Initialize placeholders
x_data = tf.placeholder(shape=[None, num_features], dtype=tf.float32)
y_target = tf.placeholder(shape=[None,10], dtype=tf.float32)
prediction_grid = tf.placeholder(shape=[None, num_features], dtype=tf.float32)

# Create variables for svm
b = tf.Variable(tf.random_normal(shape=[10, batch_size]))

# Gaussian (RBF) kernel
gamma = tf.constant(-10.0)
x_square = tf.reshape(tf.reduce_sum(tf.square(x_data), 1),shape=[-1,1])
sq_dists = tf.add(tf.subtract(x_square,tf.multiply(2., tf.matmul(x_data, tf.transpose(x_data)))),tf.transpose(x_square))
rbf_kernel = tf.exp(tf.multiply(gamma, tf.abs(sq_dists)))

# Compute SVM Model
first_term = tf.reduce_sum(b)

b_cross = tf.matmul(tf.transpose(b), b)
tmp=tf.expand_dims(y_target,axis=0)
y_cross = tf.matmul(tf.transpose(tmp, perm=[2,1,0]), tf.transpose(tmp, perm=[2,0,1]))
second_term = tf.reduce_sum(tf.multiply(rbf_kernel, tf.multiply(b_cross, y_cross)), [1, 2])

loss = tf.reduce_sum(tf.negative(tf.subtract(first_term, 0.5*second_term)))

# Gaussian (RBF) prediction kernel
data_square = tf.reshape(tf.reduce_sum(tf.square(x_data), 1), [-1, 1])
grid_square = tf.reshape(tf.reduce_sum(tf.square(prediction_grid), 1), [-1, 1])
pred_sq_dist = tf.add(tf.subtract(data_square, tf.multiply(2., tf.matmul(x_data, tf.transpose(prediction_grid)))), tf.transpose(grid_square))
pred_kernel = tf.exp(tf.multiply(gamma, tf.abs(pred_sq_dist)))
prediction_output = tf.matmul(tf.multiply(tf.transpose(y_target), b), pred_kernel)

prediction = tf.argmax(prediction_output - tf.expand_dims(tf.reduce_mean(prediction_output, 1), 1), 0)

accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, tf.argmax(y_target, 1)), tf.float32))

# Declare optimizer
opt = tf.train.GradientDescentOptimizer(0.001)
train_step = opt.minimize(loss)

# Initialize variables
init = tf.global_variables_initializer()
sess.run(init)

# Training loop
loss_vec = []
train_accuracy = []
test_accuracy = []

for i in range(400):
    rand_index = np.random.choice(len(x_train), size=batch_size)
    rand_x = x_train[rand_index]
    rand_y = y_train[rand_index]
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
    
    temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
    loss_vec.append(temp_loss)
    
    train_acc_temp=sess.run(accuracy,feed_dict={x_data: rand_x,y_target: rand_y, prediction_grid:rand_x})
    train_accuracy.append(train_acc_temp)

    #evaluate test set accuracy
    low=0
    accuracy_test=0
    for i in range(5):
        xt=x_test[low:low+batch_size,:]
        yt=y_test[low:low+batch_size,:]
        acc_test = sess.run(accuracy, feed_dict={x_data:xt, y_target:yt, prediction_grid:xt})
        accuracy_test=accuracy_test+acc_test
        low=low+batch_size
    test_accuracy.append(accuracy_test/5)
    if (i + 1) % 50 == 0:
        print('Step #' + str(i+1))
        print('Loss = ' + str(temp_loss))
        print("Train acc:",train_acc_temp)
        print("Test acc:",accuracy_test/5)

print(sess.run(b))

# Plot batch accuracy
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