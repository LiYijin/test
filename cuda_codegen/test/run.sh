echo "-------test noinline---------"
cp noinline.cu ../kernels/kernel_func_def_22.cu
cd ..
cmake .
make -j
./main_test
cd test
echo "------test forceinline-------"
cp inline.cu ../kernels/kernel_func_def_22.cu
cd ..
make -j
./main_test