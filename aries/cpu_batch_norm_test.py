import tensorflow as tf
import matplotlib.pyplot as plt
from aries.cpu_utility import get_merged_matrix_data
from aries.cpu_utility import get_original_matrix_data

# 입력 데이터 가져오기
half_data = get_merged_matrix_data("data/cpu_today_half.csv", "data/cpu_yesterday.csv", "data/cpu_train.csv")
real_data = get_original_matrix_data("data/cpu_today.csv")

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    saver = tf.train.import_meta_graph('models/cpu_softmax.ckpt.meta')
    saver.restore(sess, tf.train.latest_checkpoint('models/'))

    inputs = tf.get_collection('input')
    X = inputs[0]
    Y = inputs[1]
    prediction = tf.get_collection('prediction')[0]
    is_training = tf.get_collection('is_training')[0]

    predicted_data_y = sess.run(prediction, feed_dict={X: half_data[0], is_training: True})
    # predicted_data_y = sess.run(prediction, feed_dict={X: real_data[0], is_training: True})

    plt.gca().set_color_cycle(['red', 'green'])
    plt.plot(real_data[1][0:1440])
    plt.plot(predicted_data_y[0:1440])
    plt.ylim(0, 50)
    plt.show()