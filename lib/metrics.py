import numpy as np
import tensorflow as tf

import pdb


def masked_mse_tf(preds, labels, null_val=np.nan):
    """
    Accuracy with masking.
    :param preds:
    :param labels:
    :param null_val:
    :return:
    """
    if np.isnan(null_val):
        mask = ~tf.is_nan(labels)
    else:
        mask = tf.not_equal(labels, null_val)
    mask = tf.cast(mask, tf.float32)
    mask /= tf.reduce_mean(mask)
    mask = tf.where(tf.is_nan(mask), tf.zeros_like(mask), mask)
    loss = tf.square(tf.subtract(preds, labels))
    loss = loss * mask
    loss = tf.where(tf.is_nan(loss), tf.zeros_like(loss), loss)
    return tf.reduce_mean(loss)


def masked_mae_tf(preds, labels, alpha=None, null_val=np.nan):
    """
    Accuracy with masking.
    :param preds:
    :param labels:
    :param null_val:
    :return:
    """
    if np.isnan(null_val):
        mask = ~tf.is_nan(labels)
    else:
        mask = tf.not_equal(labels, null_val)

    # mask gives true to all values that are not equal to 0.
    mask = tf.cast(mask, tf.float32)
    mask /= tf.reduce_mean(mask)
    mask = tf.where(tf.is_nan(mask), tf.zeros_like(mask), mask)
    # According to whether it is training/validation,
    # we might be able to associate different weights.
    loss = tf.abs(tf.subtract(preds, labels))
    loss = loss * mask
    loss = tf.where(tf.is_nan(loss), tf.zeros_like(loss), loss)

    if alpha:
        alpha_tensor = tf.expand_dims(
            tf.expand_dims(
                tf.expand_dims(tf.constant([
                    alpha, 1 - alpha]),
                    axis=0),
                axis=0),
            axis=0)

        alpha_tensor = tf.tile(alpha_tensor, [loss.shape[0], loss.shape[1], loss.shape[2], 1])
        loss = tf.math.multiply(alpha_tensor, loss, name=None)

    return tf.reduce_mean(loss)


def masked_rmse_tf(preds, labels, null_val=np.nan):
    """
    Accuracy with masking.
    :param preds:
    :param labels:
    :param null_val:
    :return:
    """
    return tf.sqrt(masked_mse_tf(preds=preds, labels=labels, null_val=null_val))


def masked_rmse_np(preds, labels, null_val=np.nan):
    return np.sqrt(masked_mse_np(preds=preds, labels=labels, null_val=null_val))


def masked_mse_np(preds, labels, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        rmse = np.square(np.subtract(preds, labels)).astype('float32')
        rmse = np.nan_to_num(rmse * mask)
        return np.mean(rmse)


def masked_mae_np(preds, labels, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mae = np.abs(np.subtract(preds, labels)).astype('float32')
        mae = np.nan_to_num(mae * mask)
        return np.mean(mae)


def masked_mape_np(preds, labels, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mape = np.abs(np.divide(np.subtract(preds, labels).astype('float32'), labels))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape)


# Builds loss function.
def masked_mse_loss(scaler, null_val):
    def loss(preds, labels):
        if scaler:
            preds = scaler.inverse_transform(preds)
            labels = scaler.inverse_transform(labels)
        return masked_mse_tf(preds=preds, labels=labels, null_val=null_val)

    return loss


def masked_rmse_loss(scaler, null_val):
    def loss(preds, labels):
        if scaler:
            preds = scaler.inverse_transform(preds)
            labels = scaler.inverse_transform(labels)
        return masked_rmse_tf(preds=preds, labels=labels, null_val=null_val)

    return loss


def masked_mae_loss(scaler, null_val):
    def loss(preds, labels):
        """
        :type alpha: float
        """
        if scaler:
            preds = scaler.inverse_transform(preds)
            labels = scaler.inverse_transform(labels)
        mae = masked_mae_tf(preds=preds, labels=labels, null_val=null_val)

        return mae

    return loss


def calculate_metrics(df_pred, df_test, null_val):
    """
    Calculate the MAE, MAPE, RMSE
    :param df_pred:
    :param df_test:
    :param null_val:
    :return:
    """
    mape = masked_mape_np(preds=df_pred.as_matrix(), labels=df_test.as_matrix(), null_val=null_val)
    mae = masked_mae_np(preds=df_pred.as_matrix(), labels=df_test.as_matrix(), null_val=null_val)
    rmse = masked_rmse_np(preds=df_pred.as_matrix(), labels=df_test.as_matrix(), null_val=null_val)
    return mae, mape, rmse
