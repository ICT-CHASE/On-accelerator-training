# Caffe-accelerator
About：
Caffe-accelerator是一个基于caffe和CNN accelerator的联合训练框架。其中CNN accelerator负责前向传播算法，剩余部分由Caffe框架完成。结合Xilinx SDAccel和Caffe，实现FPGA和CPU的切换。通过Xilinx SDAccel环境对CNN accelerator进行编译，其中CNN accelerator可以由用户自定义。

How to use:
1、修改Makefile中的SDAccel的安装路径。
2、运行alexnet.sh即可。	

