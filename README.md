# cnn_stat
Example of how to compute simple network statistic on top of tensorflow graph (computing using simple mathematics and not TF api/internal state).


For each layer we compute the following:
- params count 
- flops
- receptive field
- shape


Things to improve:
- add check that we are working with the right batch ordering (GPU format)
- check if every graph op we analyze have only 1 input and output or impliment handling of more complex cases like resnet
- process pool flops
- process activations layers altough they will have a minor effect
- add comparison of computed flops/params with number extracted directly from TF graph
- visualise the distribution of the params/computations over the net

example output for VGG16 - https://pastebin.com/raw/5jLCCf8r

