# Chasoň: Supporting Cross HBM Channel Data Migration to Enable Efficient Sparse Algebraic Acceleration

Chason is an out-of-order HBM-based sparse algebraic accelerator.  It is built on top of Serpens(https://dl.acm.org/doi/abs/10.1145/3489517.3530420) and introduces a novel non-zero scheduling scheme called **Cross-HBM Channel OoO Scheduling** along with relevant architectural support.

## Dependencies
- **Rapidstream Tapa** [link](https://tapa.readthedocs.io/en/main/)
- **Xilinx Vitis 2023.2**
- **Xilinx xilinx_u55c_gen3x16_xdma_3_202210_1 shell**
- **Xilinx U55c FPGA**
- *(Optional)* **Rapidstream Autobridge and Gurobi** - For bitstream synthesis and floorplanning. Xilinx Vitis can also be used for HW synthesis, but it might not replicate our results.

## To Do Software Emulation
```bash
sh run.sh 
./chason <matrix_file.mtx> <rp_time> <alpha> <beta>
```

## To Run High-Level Synthesis
```bash
sh rtl.sh
```
The relevant files will be generated in the `build` directory.

## To Generate the Bitstream
Go to the `build` directory and inspect the `newScript.sh` file. If the following flag is not already present, add it:
```bash
--config ../link_config_a16.ini
```
Then run:
```bash
sh newScript.sh
```

You can also switch between hardware (`hw`) for bitstream generation or hardware emulation (`hw_emu`) using the `TARGET` variable in `newScript.sh`. The bitstream file (`.xclbin`) will be generated in the `build/vitis_run_<TARGET>` directory.

## To Run Chason on Xilinx Alveo U55c
```bash
TAPAB=<path_to_bitstream>/Chason_xilinx_u55c_gen3x16_xdma_3_202210_1.xclbin <path_to_binary_file>/chason <matrix_file>
```

Alternatively, you can run Chason on the Alveo U55c using the provided bitstream file by executing:
```bash
sh run_fpga.sh
```

You can read more about Chason on this [link](www.google.com).

If you find Chason useful, please cite:
```plaintext
leaving for later
