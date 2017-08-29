import numpy as np
import pandas as pd
import datetime
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm

def data_normalization(xdata):
    return (xdata - xdata.min()) / (xdata.max() - xdata.min())

def data_standardization(xdata):
    return (xdata - xdata.mean()) / xdata.std()

def one_hot_encode(x, n_classes):
    """
    One hot encode a list of sample labels. Return a one-hot encoded vector for each label.
    : x: List of sample Labels
    : return: Numpy array of one-hot encoded labels
     """
    return np.eye(n_classes)[x]

def get_matrix_data(fileName, nb_classes):
    data = pd.read_csv(fileName)
    count = len(data["시간"])

    hours = []
    minutes = []
    weekdays = []
    outputs = []

    for i in range(count):
        dateObj = datetime.datetime.strptime(data["시간"][i], "%m/%d/%Y %H:%M")
        hours.append(dateObj.hour)
        minutes.append(dateObj.minute)
        weekdays.append(dateObj.weekday())

        cpu = data["프로세스 CPU사용률 (%)"][i]
        level = int(round(cpu))
        if(level >= nb_classes):
            level = nb_classes - 1

        outputs.append(level)

    data["구간"] = pd.Series(outputs, index=data.index)
    data["시"] = pd.Series(hours, index=data.index)
    data["분"] = pd.Series(minutes, index=data.index)
    data["요일"] = pd.Series(weekdays, index=data.index)

    cols = data.columns.tolist()
    xcols = cols[-3:] + cols[1:-5]
    ycols = cols[-4]

    print(xcols)
    print(ycols)

    xdata = data[xcols]
    ydata = data[ycols]

    x_data = xdata.as_matrix()
    y_data = ydata.as_matrix()
    one_hot_data = []

    for value in y_data:
        one_hot_data.append(
            one_hot_encode([ value ], nb_classes)[0]
        )

    return x_data, np.array(one_hot_data)

def get_original_matrix_data(fileName):
    data = pd.read_csv(fileName)
    count = len(data["시간"])

    hours = []
    minutes = []
    weekdays = []

    for i in range(count):
        dateObj = datetime.datetime.strptime(data["시간"][i], "%m/%d/%Y %H:%M")
        hours.append(dateObj.hour)
        minutes.append(dateObj.minute)
        weekdays.append(dateObj.weekday())

    data["시"] = pd.Series(hours, index=data.index)
    data["분"] = pd.Series(minutes, index=data.index)
    data["요일"] = pd.Series(weekdays, index=data.index)

    cols = data.columns.tolist()
    xcols = cols[-3:] + cols[1:-4]
    ycols = cols[-4]

    print(xcols)
    # print(ycols)

    xdata = data[xcols]
    ydata = data[ycols]

    x_data = xdata.as_matrix()
    y_data = ydata.as_matrix()

    return x_data, y_data

def get_merged_matrix_data(todayName, yesterdayName, allName):
    today_data = get_original_matrix_data(todayName)
    yesterday_data = get_original_matrix_data(yesterdayName)
    all_data = get_original_matrix_data(allName)

    today_new_x = []
    today_new_y = []
    count_x = len(today_data[0])
    count_y = len(yesterday_data[0])

    for i in range(count_y):
        today_x = today_data[0]
        today_y = today_data[1]
        yesterday_x = yesterday_data[0]

        if i >= count_x:
            row_x = today_x[count_x - 1]
            row_y = yesterday_x[i]

            # 시/분/요일이 동일한 지난 데이터의 각각의 평균을 x값으로 넣어준다.
            filter_data = pd.DataFrame(all_data[0])
            filter_data = filter_data[filter_data[0] == row_y[0]]
            filter_data = filter_data[filter_data[1] == row_y[1]]
            filter_data = filter_data[filter_data[2] == row_x[2]]

            row_pred = [
                row_y[0], row_y[1], row_x[2],
                filter_data[3].mean(), filter_data[4].mean(), filter_data[5].mean(),
                filter_data[6].mean(), filter_data[7].mean(), filter_data[8].mean(),
                filter_data[9].mean(), filter_data[10].mean(), filter_data[11].mean()
            ]

            today_new_x.append(row_pred)
            today_new_y.append(0)
        else:
            today_new_x.append(today_x[i])
            today_new_y.append(today_y[i])

    return np.array(today_new_x), np.array(today_new_y)

def batch_norm_layer(inputT, is_training=True, scope=None):
    # Note: is_training is tf.placeholder(tf.bool) type
    return tf.cond(is_training,
                   lambda: batch_norm(inputT, is_training=True, center=True, scale=True, activation_fn=tf.nn.relu, decay=0.9, scope=scope),
                   lambda: batch_norm(inputT, is_training=False, center=True, scale=True, activation_fn=tf.nn.relu, decay=0.9, scope=scope, reuse=True))


def generate_batch_data(image, label, min_queue_examples, batch_size, shuffle):
    """Construct a queued batch of images and labels.

    Args:
        image: 3-D Tensor of [height, width, 3] of type.float32.
        label: 1-D Tensor of type.int32
        min_queue_examples: int32, minimum number of samples to retain
          in the queue that provides of batches of examples.
        batch_size: Number of images per batch.
        shuffle: boolean indicating whether to use a shuffling queue.

    Returns:
        images: Images. 4D tensor of [batch_size, height, width, 3] size.
        labels: Labels. 1D tensor of [batch_size] size.
    """
    # Create a queue that shuffles the examples, and then
    # read 'batch_size' images + labels from the example queue.
    num_preprocess_threads = 16
    if shuffle:
        images, labels = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=int(min_queue_examples))
    else:
        images, labels = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size)
    return images, labels



