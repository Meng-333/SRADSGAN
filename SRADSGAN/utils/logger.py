# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
import tensorflow as tf
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
import os
import sys
from utils.utils import get_timestamp

try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO  # Python 3.x


# print to file and std_out simultaneously
class PrintLogger(object):
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log = open(os.path.join(log_path, 'print_log.txt'), 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

class Logger(object):
    def __init__(self, name, log_dir, print_freq):
        """Create a summary writer logging to log_dir."""
        self.writer = tf.summary.FileWriter(log_dir)              #tensorflow 1.0版本
        #self.writer = tf.compat.v1.summary.FileWriter(log_dir)   #tensorflow 1.0版本
        # self.writer = tf.summary.create_file_writer(log_dir)    #tensorflow 2.0版本
        self.exp_name = name
        self.print_freq = print_freq
        self.log_dir = log_dir
        # loss log file
        self.loss_log_path = os.path.join(self.log_dir, 'loss_log.txt')
        with open(self.loss_log_path, 'a') as log_file:
            log_file.write('=============== Time: ' + get_timestamp() + ' =============\n')
            log_file.write('================ Training Losses ================\n')
        # val results log file
        self.val_log_path = os.path.join(self.log_dir, 'val_log.txt')
        with open(self.val_log_path, 'a') as log_file:
            log_file.write('================ Time: ' + get_timestamp() + ' ===============\n')
            log_file.write('================ Validation Results ================\n')
        # if self.use_tb_logger and 'debug' not in self.exp_name:
        #     from tensorboard_logger import Logger as TensorboardLogger
        #     self.tb_logger = TensorboardLogger('../tb_logger/' + self.exp_name)

    def scalar_summary(self, tag, value, step):
         """Log a scalar variable."""
         #############################tensorflow=1.0###############################
         summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
         self.writer.add_summary(summary, step)
         self.writer.flush()

         #####################tensorflow=2.0#############################
         # with self.writer.as_default():
         #     tf.summary.scalar(tag, value, step=step)
         #     self.writer.flush()

    def image_summary(self, tag, images, step):
        """Log a list of images."""

        img_summaries = []
        for i, img in enumerate(images):
            # Write the image to a string
            try:
                s = StringIO()
            except:
                s = BytesIO()
            # scipy.misc.toimage(img).save(s, format="png")
            plt.imsave(s, img, format='png')

            # Create an Image object
            img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                       height=img.shape[0],
                                       width=img.shape[1])
            # Create a Summary value
            img_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, i), image=img_sum))

        # Create and write Summary
        summary = tf.Summary(value=img_summaries)
        self.writer.add_summary(summary, step)
        self.writer.flush()

    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""

        # Create a histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill the fields of the histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values ** 2))

        # Drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()

    def print_format_results(self, mode, rlt):
        epoch = rlt.pop('epoch')
        iters = rlt.pop('iters')
        time = rlt.pop('time')
        model = rlt.pop('model')
        if 'lr' in rlt:
            lr = rlt.pop('lr')
            message = '<epoch:{:3d}, iter:{:8,d}, time:{:.2f}, lr:{:.1e}> '.format(
                epoch, iters, time, lr)
        else:
            message = '<epoch:{:3d}, iter:{:8,d}, time:{:.2f}> '.format(epoch, iters, time)

        message += '{:s}: {:s} '.format('dataset', model)
        for label, value in rlt.items():
            if mode == 'train':
                message += '{:s}: {:.2e} '.format(label, value)
            elif mode == 'val':
                message += '{:s}: {:.4e} '.format(label, value)
            # tensorboard logger
            # if self.use_tb_logger and 'debug' not in self.exp_name:
            #     self.tb_logger.log_value(label, value, iters)

        # print in console
        print(message)
        # write in log file
        if mode == 'train':
            with open(self.loss_log_path, 'a') as log_file:
                log_file.write(message + '\n')
        elif mode == 'val':
            with open(self.val_log_path, 'a') as log_file:
                log_file.write(message + '\n')