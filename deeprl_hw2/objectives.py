"""Loss functions."""

# import tensorflow as tf
from keras import backend as K
import semver


def huber_loss(y_true, y_pred):
    """Calculate the huber loss.

    See https://en.wikipedia.org/wiki/Huber_loss

    Parameters
    ----------
    y_true: np.array, tf.Tensor
      Target value.
    y_pred: np.array, tf.Tensor
      Predicted value.
    max_grad: float, optional
      Positive floating point value. Represents the maximum possible
      gradient magnitude.

    Returns
    -------
    tf.Tensor
      The huber loss.
    """
    loss1 = 0.5 * K.square(y_pred - y_true)
    loss2 = K.abs(y_pred - y_true) - 0.5
    return K.minimum(loss1, loss2)


def mean_huber_loss(y_true, y_pred):
    """Return mean huber loss.

    Same as huber_loss, but takes the mean over all values in the
    output tensor.

    Parameters
    ----------
    y_true: np.array, tf.Tensor
      Target value.
    y_pred: np.array, tf.Tensor
      Predicted value.
    max_grad: float, optional
      Positive floating point value. Represents the maximum possible
      gradient magnitude.

    Returns
    -------
    tf.Tensor
      The mean huber loss.
    """
    return K.mean(huber_loss(y_true, y_pred), axis=-1)
