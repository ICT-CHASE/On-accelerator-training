# Caffe-Accelerator
About：

Caffe-accelerator是一个基于caffe和CNN accelerator的联合训练框架。其中CNN accelerator负责前向传播算法，剩余部分由Caffe框架完成。结合Xilinx SDAccel和Caffe，实现FPGA和CPU的切换。通过Xilinx SDAccel环境对CNN accelerator进行编译，其中CNN accelerator可以由用户自定义。本例中的SDAccel的版本为2017.1,CNN accelerator为PipeCNN，binary_container_1.xclbin为PipeCNN编译过后的二进制文件。(https://github.com/doonny/PipeCNN)。

How to use:

1、安装Caffe所需要的环境和Xilinx SDAccel。

2、修改Makefile中的SDAccel的安装路径,终端下输入make all完成Caffe-accelerator的编译。

3、下载AlexNet的Caffe模型至alexNet文件夹。下载地址：http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel

4、准备好数据集，并在train_val_caffe.prototxt中修改数据集的路径。

5、运行alexNet/alexnet.sh即可执行AlexNet的inference。

