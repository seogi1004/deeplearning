import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from aries.cpu_utility import get_merged_matrix_data

# 입력 데이터 가져오기
merged_data = get_merged_matrix_data("data/cpu_today.csv", "data/cpu_yesterday.csv")
merged_data_x = merged_data[0]
merged_data_y = merged_data[1]

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    saver = tf.train.import_meta_graph('model/cpu_softmax2.ckpt.meta')
    saver.restore(sess, tf.train.latest_checkpoint('model/'))

    inputs = tf.get_collection('input')
    X = inputs[0]
    Y = inputs[1]
    dropout_rate = inputs[2]
    prediction = tf.get_collection('prediction')[0]

    y_data = sess.run(prediction, feed_dict={X: merged_data_x, dropout_rate: 1})
    y_min = []
    y_max = []
    y_range = np.arange(100)

    for i in range(len(y_data)):
        level = y_data[i] + 1
        y_min.append(level)
        y_max.append(level * 5)

    plt.gca().set_color_cycle(['red', 'green', 'green'])
    plt.plot(merged_data_y)
    plt.plot(y_min)
    plt.plot(y_max)
    plt.ylim(0, 100)
    plt.show()