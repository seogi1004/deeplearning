import tensorflow as tf
import time
import logging.handlers
from aries.cpu_utility import get_matrix_data
from aries.cpu_utility import batch_norm_layer
from aries.cpu_utility import generate_batch_data

epoch_limit = 1000
batch_size = 10
data_size = 10000
ln_rate = 0.002
x_classes = 6
h_layer_1 = 60
h_layer_2 = 30
nb_classes = 100
model_path = "models/cpu_softmax.ckpt"

train_data = get_matrix_data("data/cpu_train.csv", nb_classes)
test_data = get_matrix_data("data/cpu_test.csv", nb_classes)
train_x_batch, train_y_batch = generate_batch_data(train_data[0], train_data[1], data_size, batch_size, True)

# input place holders
X = tf.placeholder(tf.float32, [None, x_classes])
Y = tf.placeholder(tf.int32, [None, nb_classes])
is_training = tf.placeholder(tf.bool)

# weights & bias for nn layers
W1 = tf.Variable(tf.random_normal([x_classes, h_layer_1]))
b1 = tf.Variable(tf.random_normal([h_layer_1]))
W2 = tf.Variable(tf.random_normal([h_layer_1, h_layer_2]))
b2 = tf.Variable(tf.random_normal([h_layer_2]))
W3 = tf.Variable(tf.random_normal([h_layer_2, nb_classes]))
b3 = tf.Variable(tf.random_normal([nb_classes]))

# Hidden layer with RELU activation
layer_1 = tf.add(tf.matmul(X, W1), b1)
layer_1 = batch_norm_layer(layer_1, is_training=is_training, scope='layer_1_bn')
layer_1 = tf.nn.relu(layer_1)
# Hidden layer with RELU activation
layer_2 = tf.add(tf.matmul(layer_1, W2), b2)
layer_2 = batch_norm_layer(layer_2, is_training=is_training, scope='layer_2_bn')
layer_2 = tf.nn.relu(layer_2)
# Output layer with linear activation
out_layer = tf.matmul(layer_2, W3) + b3

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out_layer, labels=Y))
train_step = tf.train.AdamOptimizer(ln_rate).minimize(loss)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
if update_ops:
    train_ops = [train_step] + update_ops
    train_op_final = tf.group(*train_ops)

# Get accuracy of model
prediction = tf.argmax(out_layer, 1)
correct_prediction = tf.equal(prediction, tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# cache data
tf.add_to_collection("is_training", is_training)
tf.add_to_collection("input", X)
tf.add_to_collection("input", Y)
tf.add_to_collection("vars", W1)
tf.add_to_collection("vars", b1)
tf.add_to_collection("vars", W2)
tf.add_to_collection("vars", b2)
tf.add_to_collection("vars", W3)
tf.add_to_collection("vars", b3)
tf.add_to_collection("cost", loss)
tf.add_to_collection("prediction", prediction)
tf.add_to_collection("accuracy", accuracy)
tf.add_to_collection("optimizer", train_op_final)

saver = tf.train.Saver()
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

# Start input enqueue threads.
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

# Logging
logger = logging.getLogger('mylogger')
fileHandler = logging.FileHandler("logs/cpu.log")
streamHandler = logging.StreamHandler()
logger.addHandler(fileHandler)
logger.addHandler(streamHandler)
logger.setLevel(logging.INFO)

for epoch in range(epoch_limit):
    stime = time.time()

    for step in range(batch_size):
        x_batch, y_batch = sess.run([train_x_batch, train_y_batch])

        _, train_accuracy = sess.run([train_op_final, accuracy],
                                     feed_dict={X: x_batch[step], Y: y_batch[step], is_training: True})

        _, train_accuracy, train_loss, train_pred =\
            sess.run([train_op_final, accuracy, loss, prediction],
                     feed_dict={X: x_batch[step], Y: y_batch[step], is_training: False})

        global_loss = train_loss
        global_accuracy = train_accuracy

    logger.info("Epoch: {:5}\t\tLoss: {:.3f}\t\tAccuracy : {:.2%}\t\tElapsed Time : {:.2f}sec\t\tRemaining : {:.2f}min"
              .format(epoch, global_loss, global_accuracy, (time.time()-stime), ((time.time()-stime)*(epoch_limit - epoch))/60))

coord.request_stop()
coord.join(threads)

# Calculate accuracy for all mnist test images
logger.info("Test accuracy for the latest result: %g" % accuracy.eval(
    feed_dict={X: test_data[0], Y: test_data[1], is_training: False}))

# save data
saver.save(sess, model_path)
sess.close()
