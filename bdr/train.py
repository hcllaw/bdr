from __future__ import division
from functools import partial

import numpy as np
from sklearn.externals.six.moves import xrange
import tensorflow as tf

from .utils import loop_batches


def train_network(sess, net, train_f, val_f, checkpoint_path,
                  batch_pts, batch_bags=np.inf,
                  eval_batch_pts=None, eval_batch_bags=None,
                  lr=0.01, first_early_stop_epoch=None, max_epochs=1000,
                  optimizer=tf.train.AdamOptimizer, display_every=1):
    if first_early_stop_epoch is None:
        first_early_stop_epoch = max_epochs // 3
    if eval_batch_pts is None:
        eval_batch_pts = batch_pts
    if eval_batch_bags is None:
        eval_batch_bags = batch_bags

    looper = partial(
        loop_batches, max_pts=batch_pts, max_bags=batch_bags,
        stack=True, shuffle=True)
    eval_looper = partial(
        loop_batches, max_pts=eval_batch_pts, max_bags=eval_batch_bags,
        stack=True, shuffle=False)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        # Make sure we do update_ops, if we do batch_norm, before stepping
        optimize_step = optimizer(lr).minimize(net.loss)

    cur_min = np.inf  # used for early stopping
    countdown = np.inf

    saver = tf.train.Saver(max_to_keep=1)
    sess.run(tf.global_variables_initializer())

    print("Training up to {} epochs; considering early stopping from epoch {}."
          .format(max_epochs, first_early_stop_epoch))
    if hasattr(net, 'print_out_names'):
        print("Last columns are: " + ', '.join(net.print_out_names))

    # Print out initialization quality
    s = "Before training: "
    s += " " * (4 + 20 + 8 + 8 + 8 + 2 - len(s))
    val_loss = 0
    val_sse = 0
    for batch_i, batch in enumerate(eval_looper(val_f)):
        l, preds = sess.run(
            [net.early_stopper, net.output],
            feed_dict=net.feed_dict(batch, batch.y, training=False))
        val_loss += l
        val_sse += np.sum((preds - batch.y) ** 2)
    val_loss /= batch_i + 1
    val_mse = val_sse / len(val_f)
    s += "estop crit = {:8.5f}, MSE = {:8.5f}".format(val_loss, val_mse)
    if hasattr(net, 'print_out'):
        s += "; " + ' '.join(
            '{:8.5f}'.format(x) for x in sess.run(net.print_out))
    print(s)

    # Training loop
    epoch = -np.inf
    for epoch in xrange(max_epochs):
        avg_loss = 0
        train_sse = 0
        for batch_i, batch in enumerate(looper(train_f)):
            # Take an optimization step
            _, l, preds = sess.run(
                [optimize_step, net.loss, net.output],
                feed_dict=net.feed_dict(batch, batch.y, training=True))
            train_sse += np.sum((preds - batch.y) ** 2)
            avg_loss += l
        avg_loss /= batch_i + 1
        train_mse = train_sse / len(train_f)

        if epoch >= first_early_stop_epoch or epoch % display_every == 0:
            val_loss = 0
            val_sse = 0
            for batch_i, batch in enumerate(eval_looper(val_f)):
                l, preds = sess.run(
                    [net.early_stopper, net.output],
                    feed_dict=net.feed_dict(batch, batch.y, training=False))
                val_loss += l
                val_sse += np.sum((preds - batch.y) ** 2)
            val_loss /= batch_i + 1
            val_mse = val_sse / len(val_f)

            if val_loss <= cur_min and epoch >= first_early_stop_epoch:
                countdown = 10
                cur_min = val_loss
                save_path = saver.save(sess, checkpoint_path)
                # TODO: this saving is a huge portion of training time.
                #       figure out another way to do it...
                best_epoch = epoch
            else:
                countdown -= 1

        if epoch % display_every == 0:
            s = ("{: 4d}: mean train loss = {:8.5f}, MSE = {:8.5f}; "
                 "estop crit = {:8.5f}, MSE = {:8.5f}"
                 ).format(epoch, avg_loss, train_mse, val_loss, val_mse)
            if hasattr(net, 'print_out'):
                s += "; " + ' '.join(
                    '{:8.5f}'.format(x) for x in sess.run(net.print_out))
            print(s)

        if epoch >= first_early_stop_epoch and countdown <= 0:
            break

    if epoch >= first_early_stop_epoch:
        print(("Stopping at epoch {} with val loss {:.8}\n"
               "Using model from epoch {} with estop loss {:.8}").format(
                   epoch, val_loss, best_epoch, cur_min))
        saver.restore(sess, save_path)
    else:
        print("Using final model.")


def eval_network(sess, net, test_f, batch_pts, batch_bags=np.inf, do_var=False):
    preds = np.zeros_like(test_f.y)
    if do_var:
        pred_vars = np.zeros_like(test_f.y)
    i = 0
    for batch in loop_batches(test_f, max_pts=batch_pts, max_bags=batch_bags,
                              stack=True, shuffle=False):
        d = net.feed_dict(batch, training=False)
        if do_var:
            preds[i:i + len(batch)], pred_vars[i:i + len(batch)] = sess.run(
                [net.output, net.output_var], feed_dict=d)
        else:
            preds[i:i + len(batch)] = net.output.eval(feed_dict=d)
        i += len(batch)
    return (preds, pred_vars) if do_var else preds
