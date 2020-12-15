import tensorflow as tf


class FrameProcessor(object):
    """Resizes and converts RGB Atari frames to grayscale"""

    def __init__(self, frame_height=84, frame_width=84):
        """
        Args:
            frame_height: Integer, Height of a frame of an Atari game
            frame_width: Integer, Width of a frame of an Atari game
        """
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.frame = tf.compat.v1.placeholder(shape=[210, 160, 3], dtype=tf.uint8)
        self.processed = tf.image.rgb_to_grayscale(self.frame)
        self.processed = tf.image.crop_to_bounding_box(self.processed, 34, 0, 160, 160)
        self.processed = tf.image.resize(self.processed,
                                         [self.frame_height, self.frame_width],
                                         method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    def __call__(self, session, frame):
        """
        Args:
            session: A Tensorflow session object
            frame: A (210, 160, 3) frame of an Atari game in RGB
        Returns:
            A processed (84, 84, 1) frame in grayscale
        """
        return session.run(self.processed, feed_dict={self.frame: frame})
