import tensorflow as tf
import vgg


def main():
    batch_size = 1
    height, width = 224, 224
    with tf.Session() as sess:
        inputs = tf.random_uniform((batch_size, height, width, 3))
        logits, _ = vgg.vgg_16(inputs)
        sess.run(tf.global_variables_initializer())
        output = sess.run(logits)


if __name__ == "__main__":
    main()