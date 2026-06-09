import tensorflow as tf

def kld(y_true, y_pred, eps=1e-7):
    """KL Divergence loss — giống hệt baseline."""
    sum_per_image = tf.reduce_sum(y_true, axis=(1,2,3), keepdims=True)
    y_true = y_true / (eps + sum_per_image)
    sum_per_image = tf.reduce_sum(y_pred, axis=(1,2,3), keepdims=True)
    y_pred = y_pred / (eps + sum_per_image)
    loss = y_true * tf.math.log(eps + y_true / (eps + y_pred))
    return tf.reduce_mean(tf.reduce_sum(loss, axis=(1,2,3)))
