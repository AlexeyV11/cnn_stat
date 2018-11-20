import tensorflow as tf
import vgg


def main():
    with tf.Session() as sess:

        batch_size = 1
        height, width = 224, 224
        kernel_depth = 3

        inputs = tf.random_uniform((batch_size, height, width, kernel_depth))
        # here we are missing the final classification layer
        logits, _ = vgg.vgg_16(inputs)

        # FC can be replaced with CONV, btw official VGG model in TF rep uses only convs
        target_ops = [op for op in tf.get_default_graph().get_operations()
                      if op.type == 'Conv2D' or op.type == "MaxPool"]

        for op in target_ops:
            strides = op.get_attr('strides')


            # check if w and h has the same stride
            assert(strides[1] == strides[2])


            #in case of pool stride == kernel size; kernel_depth stays the same
            kernel_size = strides[1]

            if op.type == "Conv2D":
                outs = tf.get_default_graph().get_operation_by_name(op.name.replace(op.type, "weights")).outputs[0]
                dim1, dim2, prev_layer_depth, current_layer_depth = outs.get_shape()

                # check that our conv is square
                assert(dim1 == dim2)
                assert(kernel_depth == prev_layer_depth)

                kernel_size = dim1
                kernel_depth = current_layer_depth


            print("{} {} {}x{}x{} stride[{}] padding[{}]"
                  .format(op.name, op.type, kernel_size, kernel_size, kernel_depth, strides[1], op.get_attr('padding')))



        sess.run(tf.global_variables_initializer())
        output = sess.run(logits)



if __name__ == "__main__":
    main()
