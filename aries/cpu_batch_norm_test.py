import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from aries.cpu_utility import get_merged_matrix_data

# 입력 데이터 가져오기
reality_data = get_merged_matrix_data("data/cpu_today.csv", "data/cpu_yesterday.csv")
reality_data_x = reality_data[0]
reality_data_y = reality_data[1]

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    saver = tf.train.import_meta_graph('models/cpu_softmax.ckpt.meta')
    saver.restore(sess, tf.train.latest_checkpoint('models/'))

    inputs = tf.get_collection('input')
    X = inputs[0]
    Y = inputs[1]
    prediction = tf.get_collection('prediction')[0]
    is_training = tf.get_collection('is_training')[0]

    predicted_data_y = sess.run(prediction, feed_dict={X: reality_data_x, is_training: True})

    plt.gca().set_color_cycle(['red', 'green'])
    plt.plot(reality_data_y[360:720])
    plt.plot(predicted_data_y[360:720])
    plt.ylim(0, 20)
    plt.show()