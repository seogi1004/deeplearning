import tensorflow as tf
from aries.cpu_utility import get_matrix_data

l_rate = 0.001
d_rate = 0.5

x_classes = 6
h_layer_1 = 40
h_layer_2 = 20
nb_classes = 100
model_path = "cpu_softmax.ckpt"

trainData = get_matrix_data("data/cpu_train.csv", nb_classes)
testData = get_matrix_data("data/cpu_test.csv", nb_classes)

x_data = trainData[0]
y_data = trainData[1]

# input place holders
X = tf.placeholder(tf.float32, [None, x_classes])
Y = tf.placeholder(tf.int32, [None, nb_classes])
dropout_rate = tf.placeholder(tf.float32)

# weights & bias for nn layers
W1 = tf.Variable(tf.random_normal([x_classes, h_layer_1]))
b1 = tf.Variable(tf.random_normal([h_layer_1]))
_L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
L1 = tf.nn.dropout(_L1, dropout_rate)

W2 = tf.Variable(tf.random_normal([h_layer_1, h_layer_2]))
b2 = tf.Variable(tf.random_normal([h_layer_2]))
_L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
L2 = tf.nn.dropout(_L2, dropout_rate)

W3 = tf.Variable(tf.random_normal([h_layer_2, nb_classes]))
b3 = tf.Variable(tf.random_normal([nb_classes]))
hypothesis = tf.matmul(L2, W3) + b3

# define cost/loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=l_rate).minimize(cost)

prediction = tf.argmax(hypothesis, 1)
correct_prediction = tf.equal(prediction, tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# restore를 위한 변수 저장
tf.add_to_collection("input", X)
tf.add_to_collection("input", Y)
tf.add_to_collection("input", dropout_rate)
tf.add_to_collection("vars", W1)
tf.add_to_collection("vars", b1)
tf.add_to_collection("vars", _L1)
tf.add_to_collection("vars", L1)
tf.add_to_collection("vars", W2)
tf.add_to_collection("vars", b2)
tf.add_to_collection("vars", _L2)
tf.add_to_collection("vars", L2)
tf.add_to_collection("vars", W3)
tf.add_to_collection("vars", b3)
tf.add_to_collection("hypothesis", hypothesis)
tf.add_to_collection("prediction", prediction)
tf.add_to_collection("cost", cost)
tf.add_to_collection("optimizer", optimizer)

with tf.Session() as sess:
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    for step in range(1000):
        sess.run(optimizer, feed_dict={X: x_data, Y: y_data, dropout_rate: d_rate})
        if step % 100 == 0:
            loss, acc, pred = sess.run([cost, accuracy, prediction], feed_dict={X: x_data, Y: y_data, dropout_rate: 1})
            print("Step: {:5}\tLoss: {:.3f}\tAcc: {:.2%}".format(step, loss, acc), pred)

    # Test model and check accuracy
    print('Accuracy:', sess.run(accuracy, feed_dict={X: testData[0], Y: testData[1], dropout_rate: 1}))
    saver_path = saver.save(sess, "model/" + model_path)
