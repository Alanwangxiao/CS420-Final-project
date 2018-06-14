import tensorflow as tf
import numpy as np
import scipy.io as io 
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix


data_num = 60000 #The number of figures
fig_w = 45       #width of each figure

x_train= np.fromfile("mnist_train/mnist_train_data",dtype=np.uint8)
x_train=x_train/255
x_train=x_train.reshape(60000,45*45)
y_train = np.fromfile("mnist_train/mnist_train_label",dtype=np.uint8)
#y_train=np.expand_dims(y_train,1)
y_train = to_categorical(y_train, num_classes=10)
print(x_train.shape)
print(y_train.shape)
x_test=np.fromfile("mnist_test/mnist_test_data",dtype=np.uint8)
x_test=x_test/255
x_test=x_test.reshape(10000,45*45)
y_test=np.fromfile("mnist_test/mnist_test_label",dtype=np.uint8)
#y_test=np.expand_dims(y_test,1)
y_test = to_categorical(y_test, num_classes=10)
#print(x_test.shape)
print(y_test.shape)

'''
pca=PCA(n_components=0.95)
X=np.vstack((x_train,x_test))
data=pca.fit_transform(X)
x_train=data[0:60000]
x_test=data[60000:70000]
print(x_train.shape)
print(x_test.shape)
'''

np.random.seed(1)
tf.set_random_seed(1)

# The number of training examples to use per training step.
BATCH_SIZE = 5000  

# Define the flags useable from the command line.

tf.app.flags.DEFINE_integer('num_epochs', 300, 'Number of training epochs.')
tf.app.flags.DEFINE_float('svmC', 5, 'The C parameter of the SVM cost function.')
FLAGS = tf.app.flags.FLAGS

def main(argv=None):

    # Get the shape of the training data.
    train_size, num_features= x_train.shape

    # Get the number of epochs for training.
    num_epochs = FLAGS.num_epochs

    # Get the C param of SVM
    svmC = FLAGS.svmC

    # This is where training samples and labels are fed to the graph.
    x = tf.placeholder("float", shape=[None, num_features])
    y = tf.placeholder("float", shape=[None,10])

    # Define and initialize the network.

    # These are the weights that inform how much each feature contributes to the classification.
    W = tf.Variable(tf.random_normal(shape=[num_features,10]))
    b = tf.Variable(tf.random_normal(shape=[1,10]))
    y_raw = tf.matmul(x,W) + b
    print(y_raw.shape)
    #predicted = tf.nn.log_softmax(y_raw)
    pred = tf.nn.softmax(y_raw)
    print(pred.shape)
    # Optimization.
    l2_norm = 0.5*tf.reduce_sum(tf.square(W)) 
    print("y",y.shape)
    print("pred",pred.shape)
    #cross_entropy=-tf.reduce_sum(y*predicted,reduction_indices=[1])
    svm_loss=tf.reduce_sum(svmC*tf.nn.softmax_cross_entropy_with_logits(logits=y_raw,labels=y)+l2_norm)

    train_step = tf.train.GradientDescentOptimizer(0.000001).minimize(svm_loss)

    # Evaluation.
    correct_prediction = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    loss_vec = []
    train_accuracy = []
    test_accuracy = []

    # Create a local session to run this computation.
    with tf.Session() as sess:
        # Run all the initializers to prepare the trainable parameters.
        tf.global_variables_initializer().run()
        print ('Initialized!')
        print ('Training.')
        print(train_size)

        # Iterate and train.
        for step in range(num_epochs * train_size // BATCH_SIZE):
            rand_index =np.random.choice(len(x_train), size=BATCH_SIZE)
            rand_x = x_train[rand_index]
            rand_y = y_train[rand_index]
            
            sess.run(train_step, feed_dict={x:rand_x, y:rand_y})
            temp_loss=svm_loss.eval(feed_dict={x: rand_x, y: rand_y})
            loss_vec.append(temp_loss)
                        
            if step % 50==0:
                print ('loss: ', temp_loss)
                test_acc_temp=sess.run(accuracy,feed_dict={x: x_test,y: y_test})
                test_accuracy.append(test_acc_temp)
                print(test_acc_temp)
                train_acc_temp=accuracy.eval(feed_dict={x: rand_x,y: rand_y})
                train_accuracy.append(train_acc_temp)
                print(train_acc_temp)


        y_pred=sess.run(pred,feed_dict={x: x_test,y: y_test})
        print ("Accuracy on test:", accuracy.eval(feed_dict={x: x_test, y: y_test}))

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
    
if __name__ == '__main__':
    tf.app.run()
