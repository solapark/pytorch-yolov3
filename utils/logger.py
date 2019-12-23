import tensorflow as tf


class Logger(object):
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        #self.writer = tf.summary.FileWriter(log_dir)
        self.writer = tf.summary.create_file_writer(log_dir)

    def scalar_summary(self, name, value, step):
        """Log a scalar variable."""
        #summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        with self.writer.as_default():
            tf.summary.scalar(name, value, step=step)    

    def list_of_scalars_summary(self, tag_value_pairs, step):
        """Log scalar variables."""
        #summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value) for tag, value in tag_value_pairs])
        #summary = tf.summary.(value=[tf.compat.v1.Summary.Value(tag=tag, simple_value=value) for tag, value in tag_value_pairs])
        with self.writer.as_default():
            for name, value in tag_value_pairs : 
                tf.summary.scalar(name, value, step=step)
