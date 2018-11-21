import tensorflow as tf
import vgg
import math


class LayerBase():
    def __init__(self, prev_h, prev_w, prev_d, filter_h, filter_w, filter_d, filter_stride, padding):

        self.prev_h = prev_h
        self.prev_w = prev_w
        self.prev_d = prev_d
        self.filter_h = filter_h
        self.filter_w = filter_w
        self.filter_d = filter_d
        self.filter_stride = filter_stride
        self.padding = padding

    def compute_feature_shape(self):

        # for pool the same rules
        # https://www.tensorflow.org/api_guides/python/nn#Convolution
        if self.padding.decode() == "SAME":
            n_h = math.ceil(float(self.prev_h) / float(self.filter_stride))
            n_w = math.ceil(float(self.prev_w) / float(self.filter_stride))
        elif self.padding.decode() == "VALID":
            n_h = math.ceil((1 + self.prev_h - self.filter_h) / float(self.filter_stride))
            n_w = math.ceil((1 + self.prev_w - self.filter_w) / float(self.filter_stride))
        else:
            # not implemeted
            assert (False)

        n_d = self.filter_d

        return (n_h, n_w, n_d)

    def compute_flops(self):
        # not implemented
        assert (False)

    def get_type(self):
        # not implemented
        assert (False)

class LayerMaxPool(LayerBase):
    def __init__(self, prev_h, prev_w, prev_d, filter_h, filter_w, filter_d, filter_stride, padding):
        LayerBase.__init__(self, prev_h, prev_w, prev_d, filter_h, filter_w, filter_d, filter_stride, padding)

    def compute_flops(self):
        # not implemented
        assert(False)

    def get_type(self):
        return "MaxPool"


class LayerConv(LayerBase):
    def __init__(self, prev_h, prev_w, prev_d, filter_h, filter_w, filter_d, filter_stride, padding):
        LayerBase.__init__(self, prev_h, prev_w, prev_d, filter_h, filter_w, filter_d, filter_stride, padding)

    def computer_flops(self):
        # not implemented
        assert (False)

    def get_type(self):
        return "Conv2D"


def create_layer_obj(op, prev_layer_shape):
    strides = op.get_attr('strides')

    # check if w and h has the same stride
    assert (strides[1] == strides[2])
    kernel_stride = strides[1]

    if op.type == "MaxPool":
        # in case of pool stride == kernel size; kernel_depth stays the same
        kernel_size = strides[1]
        kernel_depth = prev_layer_shape[2]

        return LayerMaxPool(prev_layer_shape[0], prev_layer_shape[1], prev_layer_shape[2],
                            kernel_size, kernel_size, kernel_depth, kernel_stride, op.get_attr('padding'))

    elif op.type == "Conv2D":
        outs = tf.get_default_graph().get_operation_by_name(op.name.replace(op.type, "weights")).outputs[0]
        h, w, prev_layer_depth_tmp, current_layer_depth = outs.get_shape()

        # check that our conv is square
        assert(h == w)
        assert(prev_layer_depth_tmp == prev_layer_shape[2])

        kernel_size = int(h)
        kernel_depth = int(current_layer_depth)

        return LayerConv(prev_layer_shape[0], prev_layer_shape[1], prev_layer_shape[2],
                            kernel_size, kernel_size, kernel_depth, kernel_stride, op.get_attr('padding'))
    else:
        # not implemented
        assert (False)

def check_shape_againt_groundtruth(op, prev_layer_shape):
    gt = op.outputs[0].get_shape()
    return tuple([int(i) for i in gt[1:]]) == prev_layer_shape

def process_ops(target_ops, input_shape):

    prev_layer_shape = input_shape

    for op in target_ops:


        layer = create_layer_obj(op, prev_layer_shape)

        prev_layer_shape = layer.compute_feature_shape()

        if not check_shape_againt_groundtruth(op, prev_layer_shape):
            print("issues with dimension computation for operation {}".format(op.name))

        print("{} {}".format(op.name, prev_layer_shape))

        #print("{} {} {}x{}x{} stride[{}] padding[{}]"
        #      .format(op.name, op.type, kernel_size, kernel_size, kernel_depth, strides[1], op.get_attr('padding')))


def main():
    with tf.Session() as sess:

        batch_size = 1

        input_shape = (224, 224, 3)

        inputs = tf.random_uniform((batch_size, *input_shape))
        # here we are missing the final classification layer
        logits, _ = vgg.vgg_16(inputs)

        # FC can be replaced with CONV, btw official VGG model in TF rep uses only convs
        target_ops = [op for op in tf.get_default_graph().get_operations()
                      if op.type == 'Conv2D' or op.type == "MaxPool"]

        process_ops(target_ops, input_shape)


        # sess.run(tf.global_variables_initializer())
        # output = sess.run(logits)



if __name__ == "__main__":
    main()
