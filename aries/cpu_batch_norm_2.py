import tensorflow as tf
import time
import logging.handlers
from aries.cpu_utility import get_matrix_data
from aries.cpu_utility import generate_batch_data

epoch_limit = 1000
batch_size = 10
data_size = 5000
nb_classes = 100
global_loss = 999999
loss_limit = 0.9

model_dir_path = "models/"
model_path = "cpu_softmax.ckpt"
train_data = get_matrix_data("data/cpu_train.csv", nb_classes)
test_data = get_matrix_data("data/cpu_test.csv", nb_classes)
train_x_batch, train_y_batch = generate_batch_data(train_data[0], train_data[1], data_size, batch_size, True)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

saver = tf.train.import_meta_graph(model_dir_path + model_path + ".meta")
saver.restore(sess, tf.train.latest_checkpoint(model_dir_path))

inputs = tf.get_collection("input")
X = inputs[0]
Y = inputs[1]
train_op_final = tf.get_collection('optimizer')[0]
prediction = tf.get_collection('prediction')[0]
loss = tf.get_collection('cost')[0]
is_training = tf.get_collection('is_training')[0]
accuracy = tf.get_collection('accuracy')[0]

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

    if global_loss < loss_limit:
        logger.info("Break Training... {}, {}".format(global_loss, loss_limit))
        break

    for step in range(batch_size):
        x_batch, y_batch = sess.run([train_x_batch, train_y_batch])

        _, train_accuracy = sess.run([train_op_final, accuracy],
                                     feed_dict={X: x_batch[step], Y: y_batch[step], is_training: True})

        _, train_accuracy, train_loss, train_pred = \
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
saver.save(sess, model_dir_path + model_path)
sess.close()
