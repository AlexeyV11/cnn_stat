import tensorflow as tf
import vgg
import math

import functools
import operator

import term_plot

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

    @property
    def stride(self):
        return self.filter_stride

    @property
    def kernel_spatial_size(self):
        assert(self.filter_h == self.filter_w)
        return self.filter_h

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

    def compute_params(self):
        # not implemented
        assert (False)

    def get_type(self):
        # not implemented
        assert (False)

class LayerMaxPool(LayerBase):
    def __init__(self, prev_h, prev_w, prev_d, filter_h, filter_w, filter_d, filter_stride, padding):
        LayerBase.__init__(self, prev_h, prev_w, prev_d, filter_h, filter_w, filter_d, filter_stride, padding)

    def compute_flops(self):
        # in case of max pooling we do only comparisons
        return 0

    def compute_params(self):
        return 0

    def get_type(self):
        return "MaxPool"


class LayerConv(LayerBase):
    def __init__(self, prev_h, prev_w, prev_d, filter_h, filter_w, filter_d, filter_stride, padding):
        LayerBase.__init__(self, prev_h, prev_w, prev_d, filter_h, filter_w, filter_d, filter_stride, padding)

    def compute_flops(self):
        h,w,d = self.compute_feature_shape()

        mult_adds_per_kernel = self.filter_h * self.filter_w * self.prev_d  # we need to take a kernel and multiply/add to get just 1 output

        # repeat for each possible output location
        return mult_adds_per_kernel * self.filter_d * h * w


    def compute_params(self):
        # weight + bias
        return self.filter_h * self.filter_w * self.filter_d * self.prev_d + self.filter_d

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

    total_flops = 0
    total_params = 0

    receptive_field = 1
    stride_k = 1


    layer_stat = []

    for op in target_ops:

        layer = create_layer_obj(op, prev_layer_shape)

        receptive_field += (layer.kernel_spatial_size - 1) * stride_k
        stride_k *= layer.stride

        prev_layer_shape = layer.compute_feature_shape()

        if not check_shape_againt_groundtruth(op, prev_layer_shape):
            print("issues with dimension computation for operation {}".format(op.name))

        flops = layer.compute_flops()
        params = layer.compute_params()
        feats_volume = functools.reduce(operator.mul, layer.compute_feature_shape())
        layer_stat.append((op.name, flops, params, feats_volume))

        print("{:35} {:26} {:20} {:20} {:20}".format(op.name,
                                            "shape[{}]".format(prev_layer_shape),
                                            "params[{}]".format(params),
                                            "flops[{}]".format(flops),
                                            "receptive_f[{}]".format(receptive_field)))

        total_flops += flops
        total_params += params


    print()

    charts =  \
        [term_plot.BarChart([a[1] for a in layer_stat], header = [a[0] for a in layer_stat], name = "Computation (layer flops) distribution"),
        term_plot.BarChart([a[2] for a in layer_stat], header=[a[0] for a in layer_stat], name="Params (layer weights) distribution"),
        term_plot.BarChart([a[3] for a in layer_stat], header = [a[0] for a in layer_stat], name = "Information (layer feature volume) distribution")]

    for chart in charts:
        print(chart.plot())

    print()
    print("Total GFLOPs for CONV layers : {0:.2f} ".format(total_flops / 1000000000))
    print("Total params for CONV layers : {} ".format(total_params))

def process_tf_flops():
    # supress output
    opts = (tf.profiler.ProfileOptionBuilder(
        tf.profiler.ProfileOptionBuilder.float_operation())
            .with_empty_output()
            .build())

    # show default output with useful info
    # opts = tf.profiler.ProfileOptionBuilder.float_operation()

    flops = tf.profiler.profile(tf.get_default_graph(), run_meta=tf.RunMetadata(), cmd='op', options=opts)

    if flops is not None:
        # NOTE: division by two of the result
        # we divide by two because we consider add and mult as one operation in our calculations!
        print("Total GFLOPs from TF graph : {0:.2f} ".format(flops.total_float_ops / 1000000000 / 2))

def process_tf_variables():
    total_parameters = 0
    for variable in tf.trainable_variables():
        total_parameters += functools.reduce(operator.mul, variable.get_shape().as_list())

    print("Total params from TF graph : {} ".format(total_parameters))


def main():
    # TODO: add check that we are working with the right batch ordering (GPU format)
    # TODO: check if every graph op we analyze have only 1 input and output or impliment handling of more complex cases like resnet
    # TODO: process pool flops
    # TODO: process activations layers altough they will have a minor effect

    #tf.logging.set_verbosity(tf.logging.ERROR)

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


        process_tf_flops()

        process_tf_variables()

        # sess.run(tf.global_variables_initializer())
        # output = sess.run(logits)


if __name__ == "__main__":
    main()
