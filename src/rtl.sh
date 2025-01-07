#Don't forget to set the XILINX_PLATFORM environment variable.
#In my case ./bashrc file has the following line: export XILINX_PLATFORM=xilinx_u55c_gen3x16_xdma_3_202211_1
mkdir build
tapa \
  --work-dir build \
compile \
  --top Chason \
  --platform $XILINX_PLATFORM\
  --clock-period 3.33 \
  -o build/Chason.xo \
  -s build/newScript.sh \
  -f chason.cpp

#newScript.sh will be used to generate the bitstream (.xclbin) file. Refer to tapa/rapidstream official documnetation for more details