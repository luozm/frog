#!/usr/bin/env bash

CUDA_PATH=/usr/local/cuda

echo "Compiling nms kernels by nvcc..."
cd box/nms/torch_nms/src/
$CUDA_PATH/bin/nvcc -c -o nms_kernel.cu.o nms_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_61
cd ../
python3 build.py


echo "Building nms by cython..."
cd ../cython_nms/
python3 setup.py build_ext --inplace
rm -rf build


echo "Building nms by CUDA..."
cd ../gpu_nms/
python3 setup.py build_ext --inplace
rm -rf build


echo "Building box_overlap by cython..."
cd ../../overlap/cython_overlap/
python3 setup.py build_ext --inplace
rm -rf build


echo "Compiling crop_and_resize kernel by nvcc..."
cd ../../../roi_align_pool_tf/src/
$CUDA_PATH/bin/nvcc -c -o crop_and_resize_kernel.cu.o crop_and_resize_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_61
cd ../
python3 build.py

