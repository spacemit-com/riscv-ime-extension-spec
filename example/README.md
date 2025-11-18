## Introduction
This is a sample demo to use spacemit IME extension instruction. 

## quick start

```
# prepare the tools
wget https://archive.spacemit.com/toolchain/spacemit-toolchain-linux-glibc-x86_64-v1.1.2.tar.xz
tar -xvf spacemit-toolchain-linux-glibc-x86_64-v1.1.2.tar.xz

# build 
riscv64-unknown-linux-gnu-gcc  -march=rv64gcv vmadot-gemm-demo.c -o gemm-vmadot-4x8x4


# run on qemu
wget https://archive.spacemit.com/spacemit-ai/qemu/jdsk-qemu-v10.0.2.tar.gz
tar -xzvf jdsk-qemu-v10.0.2.tar.gz
qemu-riscv64 -cpu max,vlen=256 gemm-vmadot-4x8x4

# run on k1
# copy the bin file to K1
./gemm-vmadot-4x8x4

```
