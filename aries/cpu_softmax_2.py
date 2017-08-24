import tensorflow as tf
from aries.cpu_utility import get_matrix_data

nb_classes = 100
x_classes = 6
model_path = "cpu_softmax.ckpt"

trainData = get_matrix_data("data/cpu_train.csv", nb_classes)
testData = get_matrix_data("data/cpu_test.csv", nb_classes)

x_data = trainData[0]
y_data = trainData[1]

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    saver = tf.train.import_meta_graph("model/" + model_path + ".meta")
    saver.restore(sess, tf.train.latest_checkpoint("model/"))

    inputs = tf.get_collection("input")
    X = inputs[0]
    Y = inputs[1]
    dropout_rate = inputs[2]
    hypothesis = tf.get_collection('hypothesis')[0]
    prediction = tf.get_collection('prediction')[0]
    cost = tf.get_collection('cost')[0]
    optimizer = tf.get_collection('optimizer')[0]

    correct_prediction = tf.equal(prediction, tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    for step in range(2000):
        sess.run(optimizer, feed_dict={X: x_data, Y: y_data, dropout_rate: 1})
        if step % 100 == 0:
            loss, acc, pred = sess.run([cost, accuracy, prediction], feed_dict={X: x_data, Y: y_data, dropout_rate: 1})
            print("Step: {:5}\tLoss: {:.3f}\tAcc: {:.2%}".format(step, loss, acc), pred)

    # Test model and check accuracy
    print('Accuracy:', sess.run(accuracy, feed_dict={X: testData[0], Y: testData[1], dropout_rate: 1}))
    saver_path = saver.save(sess, "model/" + model_path)