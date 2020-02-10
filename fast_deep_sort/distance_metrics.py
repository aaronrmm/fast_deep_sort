# from https://stackoverflow.com/questions/48485373/pairwise-cosine-similarity-using-tensorflow
import tensorflow as tf
import numpy as np

def pairwise_cosine_distance(a, b):
    with tf.Session() as sess:

        # input
        input_a = tf.placeholder(tf.float32, shape = a.shape)
        input_b = tf.placeholder(tf.float32, shape = b.shape)

        # normalize each row
        normalized_a = tf.nn.l2_normalize(a, dim = 1)
        normalized_b = tf.nn.l2_normalize(b, dim = 1)

        # multiply row i with row j using transpose
        # element wise product
        prod = tf.matmul(normalized_a, normalized_b, transpose_b=True)

        dist = 1 - prod
        
        output = sess.run(dist, feed_dict = { input_a : a, input_b: b})
        return output