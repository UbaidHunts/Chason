# mkdir op
# OP_FILE="$(pwd)/op/fpga.log"
INPUT_DIR="../bitstream"
BITSTREAM=Chason_xilinx_u55c_gen3x16_xdma_3_202210_1
MAT_LIB="../matrices"
rp_time=1000

###___________________________test___________________________###
TAPAB="${INPUT_DIR}/${BITSTREAM}".xclbin ./chason "${MAT_LIB}"/dynamicSoaringProblem_8.mtx "${rp_time}"
sh reset.sh

###___________________________SuiteSparse Matrices___________________________###
# TAPAB="${INPUT_DIR}/${BITSTREAM}".xclbin ./chason "${MAT_LIB}"/dynamicSoaringProblem_8.mtx "${rp_time}"  >> "${OP_FILE}"
# sh reset.sh
# TAPAB="${INPUT_DIR}/${BITSTREAM}".xclbin ./chason "${MAT_LIB}"/reorientation_4.mtx "${rp_time}"          >> "${OP_FILE}"
# sh reset.sh
# TAPAB="${INPUT_DIR}/${BITSTREAM}".xclbin ./chason "${MAT_LIB}"/c-52.mtx "${rp_time}"                     >> "${OP_FILE}"
# sh reset.sh
# TAPAB="${INPUT_DIR}/${BITSTREAM}".xclbin ./chason "${MAT_LIB}"/mycielskian12.mtx "${rp_time}"            >> "${OP_FILE}"
# sh reset.sh
# TAPAB="${INPUT_DIR}/${BITSTREAM}".xclbin ./chason "${MAT_LIB}"/vsp_c-30_data_data.mtx "${rp_time}"       >> "${OP_FILE}"
# sh reset.sh
# TAPAB="${INPUT_DIR}/${BITSTREAM}".xclbin ./chason "${MAT_LIB}"/TSC_OPF_300.mtx "${rp_time}"              >> "${OP_FILE}"
# sh reset.sh
# TAPAB="${INPUT_DIR}/${BITSTREAM}".xclbin ./chason "${MAT_LIB}"/lowThrust_7.mtx  "${rp_time}"             >> "${OP_FILE}"
# sh reset.sh
# TAPAB="${INPUT_DIR}/${BITSTREAM}".xclbin ./chason "${MAT_LIB}"/hangGlider_3.mtx "${rp_time}"             >> "${OP_FILE}"
# sh reset.sh
# TAPAB="${INPUT_DIR}/${BITSTREAM}".xclbin ./chason "${MAT_LIB}"/trans5.mtx "${rp_time}"                   >> "${OP_FILE}"
# sh reset.sh
# TAPAB="${INPUT_DIR}/${BITSTREAM}".xclbin ./chason "${MAT_LIB}"/ckt11752_dc_1.mtx "${rp_time}"            >> "${OP_FILE}"
# sh reset.sh

###___________________________SNAP Matrices___________________________###
# TAPAB="${INPUT_DIR}/${BITSTREAM}".xclbin ./chason "${MAT_LIB}"/wiki-Vote.mtx "${rp_time}"                >> "${OP_FILE}"
# sh reset.sh
# TAPAB="${INPUT_DIR}/${BITSTREAM}".xclbin ./chason "${MAT_LIB}"/email-Enron.mtx "${rp_time}"              >> "${OP_FILE}"
# sh reset.sh
# TAPAB="${INPUT_DIR}/${BITSTREAM}".xclbin ./chason "${MAT_LIB}"/as-caida.mtx "${rp_time}"                 >> "${OP_FILE}"
# sh reset.sh
# TAPAB="${INPUT_DIR}/${BITSTREAM}".xclbin ./chason "${MAT_LIB}"/Oregon-2.mtx "${rp_time}"                 >> "${OP_FILE}"
# sh reset.sh
# TAPAB="${INPUT_DIR}/${BITSTREAM}".xclbin ./chason "${MAT_LIB}"/wiki-RfA.mtx "${rp_time}"                 >> "${OP_FILE}"
# sh reset.sh
# TAPAB="${INPUT_DIR}/${BITSTREAM}".xclbin ./chason "${MAT_LIB}"/soc-Slashdot0811.mtx "${rp_time}"         >> "${OP_FILE}"
# sh reset.sh
# TAPAB="${INPUT_DIR}/${BITSTREAM}".xclbin ./chason "${MAT_LIB}"/as-735.mtx "${rp_time}"                   >> "${OP_FILE}"
# sh reset.sh
# TAPAB="${INPUT_DIR}/${BITSTREAM}".xclbin ./chason "${MAT_LIB}"/CollegeMsg.mtx "${rp_time}"               >> "${OP_FILE}"
# sh reset.sh
# TAPAB="${INPUT_DIR}/${BITSTREAM}".xclbin ./chason "${MAT_LIB}"/wb-cs-stanford.mtx "${rp_time}"           >> "${OP_FILE}"
# sh reset.sh
# TAPAB="${INPUT_DIR}/${BITSTREAM}".xclbin ./chason "${MAT_LIB}"/Reuters911.mtx "${rp_time}"               >> "${OP_FILE}"
# sh reset.sh


