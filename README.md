# Neural network training for accelerator with computing errors

Background: With the advancements of neural networks, customized accelerators
are increasingly adopted in massive AI applications. To gain higher energy
efficiency or performance, many hardware design optimizations
such as near-threshold logic or overclocking can be utilized.
In these cases, computing errors may happen and the computing errors 
are difficult to be captured by conventional training on general 
purposed processors (GPPs). Applying the offline trained neural 
network models to the accelerators with errors directly may 
lead to considerable prediction accuracy loss.

To address the above problem, we opt to integrate the CNN accelerator 
into Caffe such that the application data and the computing errors can be trained 
and exposed in the rsulting model. In this project, we take overcloked CNN 
accelerator as an example. Basically, the CNN accelerator (PipeCNN) is overclocked 
regardless the timing errors. While we have forward computing deployed on the 
overclocked CNN accelerator and backward computing on host CPU, the retrained models 
can tolerate the computing errors according to our experiments on ImageNet.

通过Xilinx SDAccel环境对CNN accelerator进行编译，其中CNN accelerator可以由用户自定义。本例中的SDAccel的版本为2017.1,CNN accelerator为PipeCNN，binary_container_1.xclbin为PipeCNN编译过后的二进制文件。(https://github.com/doonny/PipeCNN)。

User guide:

1. Install Caffe and Xilinx SDAccel 2017.1 on ubuntu16.04.
2. Compile PepeCNN models to bitstream. We have compiled bitstream stored in ./bitstreams. You can 
find the implementations with different frequency. Relace "binary_container_1.xclbin" file with the 
one you want to test. Do keep the "binary_container_1.xclbin" when you change the bitstream.

2、Update SDAcccel path to Makefile, run "make all" completes the compilation

3. Download AlexNet model from http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel to ./alexNet 
You can compile the model and generate the bistream by yourself.

4 Setup the ImageNet dataset path in train_val_caffe.prototxt

5、Run “alexNet/alexnet.sh" to performance the inference.

Reference:
[1] Xing, Kouzi, Dawen Xu, Cheng Liu, Ying Wang, Huawei Li and Xiaowei Li, Squeezing the Last MHz for CNN Acceleration on FPGAs, The 3rd International Test Conference in Asia, 2019 (to appear)
[2] Xu, Dawen, Kouzi Xing, Cheng Liu, Ying Wang, Yulin Dai, Long Cheng, Huawei Li, Lei Zhang, Resilient Neural Network Training for Accelerators with Computing Errors, The 30th IEEE International Conference on Application-specific Systems, Architectures and Processors (ASAP), July 15-17, 2019 (to appear)

