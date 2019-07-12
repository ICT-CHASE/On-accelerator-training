# Neural network training for accelerator with computing errors

Background:

About: 是一个基于caffe和CNN accelerator的联合训练框架。其中CNN accelerator负责前向传播算法，剩余部分由Caffe框架完成。结合Xilinx SDAccel和Caffe，实现FPGA和CPU的切换。通过Xilinx SDAccel环境对CNN accelerator进行编译，其中CNN accelerator可以由用户自定义。本例中的SDAccel的版本为2017.1,CNN accelerator为PipeCNN，binary_container_1.xclbin为PipeCNN编译过后的二进制文件。(https://github.com/doonny/PipeCNN)。

User guide:

1、Install Caffe and Xilinx SDAccel 2017.1 See xxx and xxx for the installation details.

2、Update SDAcccel path to Makefile, run "make all" completes the compilation

3. Download AlexNet model from http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel to ./alexNet 

4 Setup the ImageNet dataset path in train_val_caffe.prototxt

5、Run “alexNet/alexnet.sh" to performance the inference.

Reference:

