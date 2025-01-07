#include <ap_int.h>
#include <cstdio>
#include <cstring>
#include <cassert>
#include <tapa.h>
#include "chason.h"

constexpr int FIFO_DEPTH = 2;
const int NUM_CH_SPARSE_div_8 = NUM_CH_SPARSE / 8;
const int NUM_CH_SPARSE_mult_16 = NUM_CH_SPARSE * 16;
const int NUM_CH_SPARSE_mult_2 = NUM_CH_SPARSE * 2;
const int WINDOW_SIZE_div_16 = WINDOW_SIZE >> 4;
using float_v8 = tapa::vec_t<float, 8>;
using float_v2 = tapa::vec_t<float, 2>;

struct MultXVec { //Data structure for HBM Channel
    tapa::vec_t<ap_uint<15>, 8> row;
    tapa::vec_t<ap_uint<1>,  8> pvt;
    tapa::vec_t<ap_uint<3>,  8> PE_src;

    float_v8 axv;
};

template <typename T, typename R>
inline void async_read(tapa::async_mmap<T> & A,
                       tapa::ostream<T> & fifo_A,
                       const R A_len,
                       R & i_req,
                       R & i_resp) {
#pragma HLS inline
    if ((i_req < A_len) &
        !A.read_addr.full()) {
        A.read_addr.try_write(i_req);
        ++i_req;
    }
    if (!fifo_A.full() & !A.read_data.empty()) {
        T tmp;
        A.read_data.try_read(tmp);
        fifo_A.try_write(tmp);
        ++i_resp;
    }
}

void read_edge_list_ptr(const int num_ite,
                        const int M,
                        const int P_N,
                        const int K,
                        tapa::async_mmap<int> & edge_list_ptr,
                        tapa::ostream<int> & PE_inst
                        ) {
    const int rp_time = (P_N == 0)? 1 : P_N;
    
    PE_inst.write(num_ite);
    PE_inst.write(M);
    PE_inst.write(rp_time);
    PE_inst.write(K);
    
    const int num_ite_plus1 = num_ite + 1;
l_rp:
    for(int rp = 0; rp < rp_time; rp++) {
#pragma HLS loop_flatten off
#pragma HLS loop_tripcount min=1 max=16
    rd_ptr:
        for (int i_req = 0, i_resp = 0; i_resp < num_ite_plus1;) {
#pragma HLS loop_tripcount min=1 max=800
#pragma HLS pipeline II=1
            async_read(edge_list_ptr,
                       PE_inst,
                       num_ite_plus1,
                       i_req, i_resp);
        }
    }
}

void read_X(const int P_N,
            const int K,
            tapa::async_mmap<float_v16> & vec_X,
            tapa::ostream<float_v16> & fifo_X
            ) {
    const int rp_time = (P_N == 0)? 1 : P_N;
    const int num_ite_X = (K + 15) >> 4;
    
l_rp:
    for(int rp = 0; rp < rp_time; rp++) {
#pragma HLS loop_flatten off
#pragma HLS loop_tripcount min=1 max=16
    rd_X:
        for(int i_req = 0, i_resp = 0; i_resp < num_ite_X;) {
#pragma HLS loop_tripcount min=1 max=500000
#pragma HLS pipeline II=1
            async_read(vec_X,
                       fifo_X,
                       num_ite_X,
                       i_req, i_resp);
        }
    }  
}

void read_A(const int P_N,
            const int A_len,
            tapa::async_mmap<ap_uint<512>> & A,
            tapa::ostream<ap_uint<512>> & fifo_A
            ) {
    const int rp_time = (P_N == 0)? 1 : P_N;
l_rp:
    for(int rp = 0; rp < rp_time; rp++) {
#pragma HLS loop_flatten off
#pragma HLS loop_tripcount min=1 max=16
    rd_A:
        for(int i_req = 0, i_resp = 0; i_resp < A_len;) {
#pragma HLS loop_tripcount min=1 max=10000
#pragma HLS pipeline II=1
            async_read(A,
                       fifo_A,
                       A_len,
                       i_req, i_resp);
        }
    } 
}

void PEG_Xvec(tapa::istream<int> & fifo_inst_in,
              tapa::istream<ap_uint<512>> & fifo_A,
              tapa::istream<float_v16> & fifo_X_in,
              tapa::ostream<int> & fifo_inst_out,
              tapa::ostream<float_v16> & fifo_X_out,
              // to PEG_Yvec
              tapa::ostream<int> & fifo_inst_out_to_Yvec,
              tapa::ostream<MultXVec> & fifo_aXvec
              ) {

    const int NUM_ITE = fifo_inst_in.read();
    const int M = fifo_inst_in.read();
    const int rp_time = fifo_inst_in.read();
    const int K = fifo_inst_in.read();
    
    fifo_inst_out.write(NUM_ITE);
    fifo_inst_out.write(M);
    fifo_inst_out.write(rp_time);
    fifo_inst_out.write(K);
    
    fifo_inst_out_to_Yvec.write(NUM_ITE);
    fifo_inst_out_to_Yvec.write(M);
    fifo_inst_out_to_Yvec.write(rp_time);
    
l_rp:
    for(int rp = 0; rp < rp_time; rp++) {
#pragma HLS loop_flatten off
#pragma HLS loop_tripcount min=1 max=16
        
        float local_X[4][WINDOW_SIZE];
#pragma HLS bind_storage variable=local_X latency=2
#pragma HLS array_partition variable=local_X complete dim=1
#pragma HLS array_partition variable=local_X cyclic factor=X_PARTITION_FACTOR dim=2
        
        auto start_32 = fifo_inst_in.read();
        fifo_inst_out.write(start_32);
        fifo_inst_out_to_Yvec.write(start_32);
        
    main:
        for (int i = 0; i < NUM_ITE; ++i) {
#pragma HLS loop_tripcount min=1 max=49
            
            // fill onchip X
        read_X:
            for (int j = 0; (j < WINDOW_SIZE_div_16) & (j < ((K + 15) >> 4) - i * WINDOW_SIZE_div_16); ) {
#pragma HLS loop_tripcount min=1 max=512
#pragma HLS pipeline II = 1
                if (!fifo_X_in.empty() & !fifo_X_out.full()) {
                    float_v16 x; fifo_X_in.try_read(x);
                    fifo_X_out.try_write(x);
                    for (int kk = 0; kk < 16; ++kk) {
                        for (int l = 0; l < 4; ++l) {
                            local_X[l][(j << 4) + kk] = x[kk];
                        }
                    }
                    ++j;
                }
            }
            
            // computation
            const auto end_32 = fifo_inst_in.read();
            fifo_inst_out.write(end_32);
            fifo_inst_out_to_Yvec.write(end_32);
            
        computation:
            for (int j = start_32; j < end_32; ) {
#pragma HLS loop_tripcount min=1 max=200
#pragma HLS pipeline II=1
                if (!fifo_A.empty()) {
                    ap_uint<512> a_pes; fifo_A.try_read(a_pes);
                    MultXVec raxv;
                    //coalesced eight 64-bits sparse matrix values 
                    for (int p = 0; p < 8; ++p) { //eight packed values of sparse A
                        ap_uint<64> a = a_pes(63 + p * 64, p * 64); //seperating coalesced values. (63,0)-->(127,64)
                        
                        ap_uint<32> a_val    = a(31,  0);
                        ap_uint<15> a_row    = a(46, 32);
                        ap_uint<1>  a_pvt    = a[47];
                        ap_uint<3>  a_PE_src = a(50, 48);
                        ap_uint<13> a_col    = a(63, 51);
                        
                        raxv.row[p] = a_row;
                        raxv.pvt[p] = a_pvt;
                        raxv.PE_src[p] = a_PE_src;

                        if (a_row[14] == 0) { 
                            float a_val_f = tapa::bit_cast<float>(a_val);
                            raxv.axv[p] = a_val_f * local_X[p/2][a_col];    //Why p/2::: this allows two reads from a same BRAM.
                        }
                    }
                    fifo_aXvec.write(raxv);
                    ++j;
                }
            }
            start_32 = end_32;
        }
    }
}

inline void PUcore_Ymtx(ap_uint<15> addr_c,
                        float val_d0_f,
                        ap_uint<64> local_C_pe0[URAM_DEPTH]
                        ) {
#pragma HLS inline
    ap_uint<64> c_val_u64 = local_C_pe0[addr_c(14, 1)]; //local C is URAM and has depth eqaul to number of pairs of rows in PE list
    ap_uint<32> c_val_d0_u = c_val_u64(31,  0);
    ap_uint<32> c_val_d1_u = c_val_u64(63, 32);
    ap_uint<32> c_val_u = (addr_c[0])? c_val_d1_u : c_val_d0_u; 
    float c_val_plus_d0_f = tapa::bit_cast<float>(c_val_u) + val_d0_f;
    c_val_u = tapa::bit_cast<ap_uint<32>>(c_val_plus_d0_f);
    
    if (addr_c[0]) {
        c_val_d1_u = c_val_u;
    } else {
        c_val_d0_u = c_val_u;
    }
    
    c_val_u64(63, 32) = c_val_d1_u;
    c_val_u64(31,  0) = c_val_d0_u;
    local_C_pe0[addr_c(14, 1)] = c_val_u64;
}

inline void ext_PUcore_Ymtx(ap_uint<1>idx,
                        ap_uint<14> addr_c,
                        ap_uint<1> pvt,
                        ap_uint<3> PE_src,
                        float val_d0_f,
                        ap_uint<64> PE0_PEsrc[2][URAM_DEPTH]
                        ) {
#pragma HLS inline

    ap_uint<64> c_val_u64 = PE0_PEsrc[PE_src(2,0)>>2][addr_c]; //local C is URAM and has depth eqaul to number of pairs of rows in PE list
    ap_uint<32> c_val_d0_u = c_val_u64(31,  0);
    ap_uint<32> c_val_d1_u = c_val_u64(63, 32);
    ap_uint<32> c_val_u = (idx)? c_val_d1_u : c_val_d0_u; 
    float c_val_plus_d0_f = tapa::bit_cast<float>(c_val_u) + val_d0_f;
    c_val_u = tapa::bit_cast<ap_uint<32>>(c_val_plus_d0_f);

    if (idx) {
        c_val_d1_u = c_val_u;
    } else {
        c_val_d0_u = c_val_u;
    }
    
    c_val_u64(63, 32) = c_val_d1_u;
    c_val_u64(31,  0) = c_val_d0_u;
    PE0_PEsrc[PE_src(2,0)>>2][addr_c] = c_val_u64;
}

inline void add_64_4(ap_uint<64> p1, ap_uint<64> p2,
                   ap_uint<64>sum[4], int pe){
#pragma HLS inline    
    ap_uint<64> c_val_u64 = sum[pe];

    ap_uint<32> p1_a0 = p1(31,  0);
    ap_uint<32> p1_a1 = p1(63, 32);
    
    ap_uint<32> p2_a0 = p2(31,  0);
    ap_uint<32> p2_a1 = p2(63, 32);

    float p1_p2_a0 = tapa::bit_cast<float> (p1_a0) + tapa::bit_cast<float> (p2_a0);
    float p1_p2_a1 = tapa::bit_cast<float> (p1_a1) + tapa::bit_cast<float> (p2_a1);

    ap_uint<32> uint_p1_p2_a0 = tapa::bit_cast<ap_uint<32>>(p1_p2_a0);
    ap_uint<32> uint_p1_p2_a1 = tapa::bit_cast<ap_uint<32>>(p1_p2_a1);

    c_val_u64(31, 0 ) = uint_p1_p2_a0;
    c_val_u64(63, 32) = uint_p1_p2_a1;
    sum[pe] = c_val_u64;    
}

inline void add_64_2(ap_uint<64> p1, ap_uint<64> p2,
                   ap_uint<64>sum[2], int pe){
#pragma HLS inline    
    ap_uint<64> c_val_u64 = sum[pe];

    ap_uint<32> p1_a0 = p1(31,  0);
    ap_uint<32> p1_a1 = p1(63, 32);
    
    ap_uint<32> p2_a0 = p2(31,  0);
    ap_uint<32> p2_a1 = p2(63, 32);

    float p1_p2_a0 = tapa::bit_cast<float> (p1_a0) + tapa::bit_cast<float> (p2_a0);
    float p1_p2_a1 = tapa::bit_cast<float> (p1_a1) + tapa::bit_cast<float> (p2_a1);

    ap_uint<32> uint_p1_p2_a0 = tapa::bit_cast<ap_uint<32>>(p1_p2_a0);
    ap_uint<32> uint_p1_p2_a1 = tapa::bit_cast<ap_uint<32>>(p1_p2_a1);
    
    c_val_u64(31, 0 ) = uint_p1_p2_a0;
    c_val_u64(63, 32) = uint_p1_p2_a1;
    sum[pe] = c_val_u64;    
}

inline void aggregate(ap_uint<64> PE_PEsrc[8][2][URAM_DEPTH], 
                      ap_uint<64> PE_PEsrc_agg[8][URAM_DEPTH],
                      int num_v_init, const int num_v_out){

#pragma HLS inline 
    partial_sums: for (int idx=0, pe_src=0; idx<num_v_out; ){
#pragma HLS loop_tripcount min=1 max=1800
#pragma HLS pipeline II=1
            //First adder stage
            ap_uint<64>sum[4];
            Add_64_4: for (int pe=0; pe<4; pe++){
                sum[pe] = 0 ;  
                add_64_4(PE_PEsrc[2*pe][pe_src >> 2][(idx >> 3)+ (pe_src%4)*num_v_init ], //PE_PEsrc[0][0][0]
                        PE_PEsrc[2*pe+1][pe_src >> 2][(idx >> 3)+ (pe_src%4)*num_v_init ], //PE_PEsrc[1][0][0]
                        sum, pe);
            }
            //Second adder stage
            ap_uint<64>sum2[2];
            Add_64_2: for (int pe=0; pe<2; pe++){
                sum2[pe] = 0 ;  
                add_64_2(sum[2*pe],
                        sum[2*pe+1],
                        sum2, pe);
            }

            ap_uint<32> p0_a0 = sum2[0](31,  0);
            ap_uint<32> p0_a1 = sum2[0](63, 32);
            
            ap_uint<32> p1_a0 = sum2[1](31,  0);
            ap_uint<32> p1_a1 = sum2[1](63, 32);

            //Last adder stage
            float p0_p1_a0 = tapa::bit_cast<float> (p0_a0) + tapa::bit_cast<float> (p1_a0);
            float p0_p1_a1 = tapa::bit_cast<float> (p0_a1) + tapa::bit_cast<float> (p1_a1);

            ap_uint<32> uint_p0_p1_a0 = tapa::bit_cast<ap_uint<32>>(p0_p1_a0);
            ap_uint<32> uint_p0_p1_a1 = tapa::bit_cast<ap_uint<32>>(p0_p1_a1); 

            PE_PEsrc_agg[pe_src][idx>>3](31,0) = uint_p0_p1_a0;
            PE_PEsrc_agg[pe_src][idx>>3](63, 32) = uint_p0_p1_a1;
            idx++;
            pe_src++;
            if (pe_src==8){pe_src =0;}
    }
}

/*
//BRUTE FORCE ADDER. INEFFICIENT. USEFUL FOR DEBUGGING
inline void aggregate_simple(ap_uint<64> PE_PEsrc[8][2][URAM_DEPTH], 
                             ap_uint<64> PE_PEsrc_agg[8][URAM_DEPTH],
                             int num_v_init, int num_v_out) {

    // Loop to iterate over the num_v_out values
    chain1: for (int idx = 0; idx < num_v_out;) {
#pragma HLS loop_tripcount min=1 max=1800
//#pragma HLS pipeline II=1
        // Loop over the 8 PE sources for the summation
        chain2: for (int j = 0; j < 8; ++j) {
            ap_uint<64> sum = 0;
            chain3: for (int pe = 0; pe < 8; ++pe) {
                
                ap_uint<32> p0_a0 = sum(31,  0);
                ap_uint<32> p0_a1 = sum(63, 32);
                
                ap_uint<32> p1_a0 = PE_PEsrc[pe][j >> 2][(idx >> 3) + (j%4)*num_v_init](31,  0);
                ap_uint<32> p1_a1 = PE_PEsrc[pe][j >> 2][(idx >> 3) + (j%4)*num_v_init](63,  32);

                float p0_p1_a0 = tapa::bit_cast<float> (p0_a0) + tapa::bit_cast<float> (p1_a0);
                float p0_p1_a1 = tapa::bit_cast<float> (p0_a1) + tapa::bit_cast<float> (p1_a1);
                // if(p0_p1_a0==0)std::cout<<"\nhere_agg_3_0=0";
                // if(p0_p1_a0!=0)std::cout<<"\nhere_agg_3_0=1";
                
                // if(p0_p1_a1==0)std::cout<<"\nhere_agg_3_1=0";
                // if(p0_p1_a1!=0)std::cout<<"\nhere_agg_3_1=1";
                // std::cout<<"\n in_agg ";
                ap_uint<32> uint_p0_p1_a0 = tapa::bit_cast<ap_uint<32>>(p0_p1_a0);
                ap_uint<32> uint_p0_p1_a1 = tapa::bit_cast<ap_uint<32>>(p0_p1_a1); 

                // ap_uint<32> f_0 = uint_p0_p1_a0;
                // ap_uint<32> f_1 = uint_p0_p1_a1;

                ap_uint<64> c_val_u64;
                c_val_u64(31, 0 ) = uint_p0_p1_a0;
                c_val_u64(63, 32) = uint_p0_p1_a1;
                sum = c_val_u64;


                //sum(31,  0) = sum (31,  0)+  PE_PEsrc[pe][j][idx >> 3](31,  0);  // idx >> 3 for correct URAM_DEPTH indexing
                //sum(63,  32) = sum (63,  32)+  PE_PEsrc[pe][j][idx >> 3](63,  32);
            }
            PE_PEsrc_agg[j][idx >> 3] = sum;
            // Store the sum into the aggregated output
        }
        // Increment the main loop counters
        idx++;
    }
}
*/

void PEG_Yvec(tapa::istream<int> & fifo_inst_in,
              tapa::istream<MultXVec> & fifo_aXvec,
              tapa::ostream<float_v2> & fifo_Y_out,
              tapa::ostream<float_v2> & fifo_Y_p_out
              ) {
    const int NUM_ITE = fifo_inst_in.read();
    const int M = fifo_inst_in.read();
    const int rp_time = fifo_inst_in.read();
    
    const int num_v_init = (M + NUM_CH_SPARSE_mult_16 - 1) / NUM_CH_SPARSE_mult_16; 
    const int num_v_out = (M + NUM_CH_SPARSE_mult_2 - 1) / NUM_CH_SPARSE_mult_2; 
    
    ap_uint<64> local_C[8][URAM_DEPTH]; //URAM_pvt
#pragma HLS bind_storage variable=local_C type=RAM_2P impl=URAM latency=1
#pragma HLS array_partition complete variable=local_C dim=1
    
    ap_uint<64> PE_PEsrc_agg[8][URAM_DEPTH]; //URAM_sh_PE_k
#pragma HLS bind_storage variable=PE_PEsrc_agg type=RAM_2P impl=URAM latency=1
#pragma HLS array_partition complete variable=PE_PEsrc_agg dim=1

     ap_uint<64> PE_PEsrc[8][2][URAM_DEPTH]; //Shared Channel URAM Group (ScUG). Each PE has one ScUG and each ScUG has 2 URAMs. 
#pragma HLS bind_storage variable=PE_PEsrc type=RAM_2P impl=URAM latency=1 
#pragma HLS array_partition complete variable=PE_PEsrc dim=1            //Each PE will have its own 8xURAM_depth URAMs
#pragma HLS array_partition complete variable=PE_PEsrc dim=2            //newly added

l_rp:
    for(int rp = 0; rp < rp_time; rp++) {
#pragma HLS loop_flatten off
#pragma HLS loop_tripcount min=1 max=16
        
        //init local C
    init_C:
        for (int i = 0; i < num_v_init; ++i) {
#pragma HLS loop_tripcount min=1 max=800
#pragma HLS pipeline II=1
            for (int p = 0; p < 8; ++p) {
                local_C[p][i] = 0;
                PE_PEsrc_agg[p][i] = 0;
            }
        }
        
         //init shared_URAMs
    init_PE_PEsrc:
        for (int i = 0; i < num_v_init*4; ++i) {  //Last dimension of URAM
#pragma HLS loop_tripcount min=1 max=800
#pragma HLS pipeline II=1
            for (int pe_dst = 0; pe_dst < 8; pe_dst++) {
                for (int pe_src=0; pe_src<2; pe_src++){
                    PE_PEsrc[pe_dst][pe_src][i] = 0;
                }
            }
        }

        auto start_32 = fifo_inst_in.read();
        
    main:
        for (int i = 0; i < NUM_ITE; ++i) {
#pragma HLS loop_tripcount min=1 max=49
            
            // computation
        const auto end_32 = fifo_inst_in.read();

        computation:
            for (int j = start_32; j < end_32; ) {
#pragma HLS loop_tripcount min=1 max=200
#pragma HLS pipeline II=1
#pragma HLS dependence true variable=local_C distance=DEP_DIST_LOAD_STORE
#pragma HLS dependence true variable=PE_PEsrc distance=DEP_DIST_LOAD_STORE
                if (!fifo_aXvec.empty()) {
                    MultXVec raxv; 
                    fifo_aXvec.try_read(raxv);

                    computation_add:for (int p = 0; p < 8; ++p) {
                        auto a_row = raxv.row[p];
                        auto a_pvt = raxv.pvt[p];
                        int a_PE_src = raxv.PE_src[p];

                        if (a_row[14] == 0) { //Partial product belongs to private channel
                            if (a_pvt==1){
                                PUcore_Ymtx(a_row,
                                        raxv.axv[p],
                                        local_C[p]);
                            }

                            else { //Partial product belongs to shared channel
                                ap_uint<14>URAM_row  = a_row(14,1);
                                ap_uint<1>URAM_idx = a_row[0];
                                ap_uint<14> offset = num_v_init;

                                ext_PUcore_Ymtx(URAM_idx,
                                                URAM_row + (a_PE_src %4)*offset,
                                                a_pvt,
                                                a_PE_src,
                                                raxv.axv[p],
                                                PE_PEsrc[p]);                              
                            }
                        }
                    }
                    ++j;
                }
            }
            start_32 = end_32;
        }

        aggregate(PE_PEsrc, 
                  PE_PEsrc_agg,
                  num_v_init, 
                  num_v_out);

    write_C_outer:
        for (int i = 0, c_idx = 0; i < num_v_out; ++i) {
#pragma HLS loop_tripcount min=1 max=1800
#pragma HLS pipeline II=1
            float_v2 out_v;
            float_v2 p_out_v;
            
            ap_uint<64> u_64   = local_C[c_idx][i>>3];    // i>>3 division by 2^3 = 8 --> 8 PEs ki phle 0th location read ho gi, phir 8th PEs ki 1st location and so on. 
            ap_uint<64> p_u_64 = PE_PEsrc_agg[c_idx][i>>3]; 
            
            for (int d = 0; d < 2; ++d) {
                ap_uint<32> u_32_d   = u_64  (31 + 32 * d, 32 * d);
                ap_uint<32> p_u_32_d = p_u_64(31 + 32 * d, 32 * d);
                out_v[d]   = tapa::bit_cast<float>(u_32_d);
                p_out_v[d] = tapa::bit_cast<float>(p_u_32_d);
            }
            fifo_Y_out.write(out_v);
            fifo_Y_p_out.write(p_out_v);
            ++c_idx;
            if (c_idx == 8) {c_idx = 0;}
        }
    }
}
              


void Arbiter_PEsrc_agg(const int P_N,
               const int M,
               tapa::istreams<float_v2, NUM_CH_SPARSE_div_8> & fifo_in,
               tapa::ostream<float_v2> & fifo_out,
               tapa::ostream<int> & fifo_ch_idx
               ) {
    //std::cout<<"Arbiter_Y"<<std::endl;
    const int rp_time = (P_N == 0)? 1 : P_N;
    const int num_pe_output = ((M + NUM_CH_SPARSE_mult_2 - 1) / NUM_CH_SPARSE_mult_2) * NUM_CH_SPARSE_div_8; // *NUM_CH_SPARSE_div_8 because of two streams
    const int num_out = (M + 15) >> 4;
    const int num_ite_Y = num_pe_output * rp_time;
aby:
    for (int i = 0, c_idx = 1, o_idx = 0; i < num_ite_Y;) {
#pragma HLS loop_tripcount min=1 max=1800
#pragma HLS pipeline II=1
        if (!fifo_in[c_idx].empty() & !fifo_out.full()) {
            float_v2 tmp; fifo_in[c_idx].try_read(tmp);
            if (o_idx < num_out) {
                fifo_out.try_write(tmp);
                fifo_ch_idx.try_write(c_idx);
            }
            ++i;
            c_idx--;
            o_idx++;
            if (c_idx == -1) {c_idx = 1;}
            if (o_idx == num_pe_output) {o_idx = 0;}
        }
    }
}

void Merger_Y_PEsrc(tapa::istreams<float_v2, 8> & fifo_in,
                    tapa::istreams<int, 8> &fifo_ch_idx,
                    tapa::ostream<float_v16> & fifo_out) {

    for (;;) {
#pragma HLS pipeline II=1
        bool flag_nop = fifo_out.full();
        for (int i = 0; i < 8; ++i) {           // If any of the FIFO queues are empty, flag_nop will become true.
                                                // If all FIFO queues are non-empty, flag_nop will remain false.
                                                // Related to Re-order Unit
            flag_nop |= fifo_in[i].empty();
            flag_nop |= fifo_ch_idx[i].empty();
        }
        if (!flag_nop) {
            float_v16 tmpv16;
#pragma HLS aggregate variable=tmpv16
            
            int idx; fifo_ch_idx[0].try_read(idx);
            int idx1; fifo_ch_idx[1].try_read(idx1);
            int idx2; fifo_ch_idx[2].try_read(idx2);
            int idx3; fifo_ch_idx[3].try_read(idx3);
            int idx4; fifo_ch_idx[4].try_read(idx4);
            int idx5; fifo_ch_idx[5].try_read(idx5);
            int idx6; fifo_ch_idx[6].try_read(idx6);
            int idx7; fifo_ch_idx[7].try_read(idx7);
            if (idx==1){
            
                float_v2 tmp; fifo_in[7].try_read(tmp);
                for (int d = 0; d < 2; ++d) {
                    int a = d; //a =i * 2 + d;
                    tmpv16[a] = tmp[d];                    
                }
                for (int i = 0; i < 7; ++i) {
                    float_v2 tmp; fifo_in[i].try_read(tmp);
                    for (int d = 0; d < 2; ++d) {
                        int a =((i+1) * 2) + d;
                        tmpv16[a] = tmp[d];
                    }
                }
            }

            else { //idx=1 --> ch15 in the last pair.  This is confusing I was expectiong when idx=0 we will be on CH15 but its on ch15 when on idx=1
                for (int i = 0; i < 8; ++i) {
                    float_v2 tmp; fifo_in[i].try_read(tmp);
                    for (int d = 0; d < 2; ++d) {
                        int a =i * 2 + d;
                        tmpv16[a] = tmp[d];
                    }
                }
            }
            fifo_out.try_write(tmpv16);
        }
    }
}

void Arbiter_Y(const int P_N,
               const int M,
               tapa::istreams<float_v2, NUM_CH_SPARSE_div_8> & fifo_in,
               tapa::ostream<float_v2> & fifo_out
               ) {
    
    const int rp_time = (P_N == 0)? 1 : P_N;
    const int num_pe_output = ((M + NUM_CH_SPARSE_mult_2 - 1) / NUM_CH_SPARSE_mult_2) * NUM_CH_SPARSE_div_8; // *NUM_CH_SPARSE_div_8 because of two streams
    const int num_out = (M + 15) >> 4;
    const int num_ite_Y = num_pe_output * rp_time;
aby:
    for (int i = 0, c_idx = 0, o_idx = 0; i < num_ite_Y;) {
#pragma HLS loop_tripcount min=1 max=1800
#pragma HLS pipeline II=1
        if (!fifo_in[c_idx].empty() & !fifo_out.full()) {
            float_v2 tmp; fifo_in[c_idx].try_read(tmp);
            if (o_idx < num_out) {
                fifo_out.try_write(tmp);
            }
            ++i;
            c_idx++;
            o_idx++;
            if (c_idx == NUM_CH_SPARSE_div_8) {c_idx = 0;}
            if (o_idx == num_pe_output) {o_idx = 0;}
        }
    }
}

void Merger_Y(tapa::istreams<float_v2, 8> & fifo_in,
              tapa::ostream<float_v16> & fifo_out) {
    for (;;) {
#pragma HLS pipeline II=1
        bool flag_nop = fifo_out.full();
        for (int i = 0; i < 8; ++i) {
            flag_nop |= fifo_in[i].empty();
        }
        if (!flag_nop) {
            float_v16 tmpv16;
#pragma HLS aggregate variable=tmpv16
            for (int i = 0; i < 8; ++i) {
                float_v2 tmp; fifo_in[i].try_read(tmp);
                for (int d = 0; d < 2; ++d) {
                    int a =i * 2 + d;
                    tmpv16[a] = tmp[d];
                }
            }
            fifo_out.try_write(tmpv16);
        }
    } 
}


void FloatvMultConst(const int P_N,
                     const int M,
                     const int alpha_u,
                     tapa::istream<float_v16> & fifo_in,
                     tapa::ostream<float_v16> & fifo_out
                     ) {
    const float alpha_f = tapa::bit_cast<float>(alpha_u);
    const int rp_time = (P_N == 0)? 1 : P_N;
    const int num_ite_Y = ((M + 15) >> 4) * rp_time;
cc:
    for (int i = 0; i < num_ite_Y;) {
#pragma HLS pipeline II=1
        float_v16 tmp;
        bool read_ready = fifo_in.try_read(tmp);
        if (read_ready) {
            float_v16 c_out = tmp * alpha_f;
            fifo_out.write(c_out);
            ++i;
        }
    }
}

void read_Y(const int P_N,
            const int M,
            tapa::async_mmap<float_v16> & Y,
            tapa::ostream<float_v16> & fifo_Y
            ) {
    const int rp_time = (P_N == 0)? 1 : P_N;
    const int num_ite_Y = (M + 15) >> 4;
    
l_rp:
    for(int rp = 0; rp < rp_time; rp++) {
#pragma HLS loop_flatten off
#pragma HLS loop_tripcount min=1 max=16
    rd_Y:
        int iaa=0;
        for(int i_req = 0, i_resp = 0; i_resp < num_ite_Y;) {

#pragma HLS loop_tripcount min=1 max=500000
#pragma HLS pipeline II=1
            async_read(Y,
                       fifo_Y,
                       num_ite_Y,
                       i_req, i_resp);
            iaa++;
        }
    }
}

void FloatvAddFloatv(tapa::istream<float_v16> & fifo_in0,
                     tapa::istream<float_v16> & fifo_in1,
                     tapa::ostream<float_v16> & fifo_out
                     ) {
cc:
    for (;;) {
#pragma HLS pipeline II=1
        bool flag_nop = fifo_in0.empty() | fifo_in1.empty();
        if (!flag_nop) {
            float_v16 tmp0; fifo_in0.try_read(tmp0);
            float_v16 tmp1; fifo_in1.try_read(tmp1);
            float_v16 c_out = tmp0 + tmp1;
            fifo_out.write(c_out);
        }
    }
}

void write_Y(const int P_N,
             const int M,
             tapa::istream<float_v16> & fifo_Y,
             tapa::async_mmap<float_v16> & Y_out
             ) {
    const int rp_time = (P_N == 0)? 1 : P_N;
    const int num_ite_Y = (M + 15) >> 4;
    
l_rp:
    for(int rp = 0; rp < rp_time; rp++) {
#pragma HLS loop_flatten off
#pragma HLS loop_tripcount min=1 max=16
    wr_C:
        for(int i_req = 0, i_resp = 0; i_resp < num_ite_Y;) {
#pragma HLS loop_tripcount min=1 max=500000
#pragma HLS pipeline II=1
            if ((i_req < num_ite_Y) &
                !fifo_Y.empty() &
                !Y_out.write_addr.full() &
                !Y_out.write_data.full() ) {
                
                Y_out.write_addr.try_write(i_req);
                float_v16 tmpv16;
                fifo_Y.try_read(tmpv16);
                Y_out.write_data.try_write(tmpv16);
                ++i_req;
            }
            uint8_t n_resp;
            if (Y_out.write_resp.try_read(n_resp)) {
                i_resp += int(n_resp) + 1;
            }
        }
    }
}

void black_hole_int(tapa::istream<int> & fifo_in) {
    for (;;) {
#pragma HLS pipeline II=1
        int tmp; fifo_in.try_read(tmp);
    }
}

void black_hole_float_v2(tapa::istream<float_v2> & fifo_in) {
    for (;;) {
#pragma HLS pipeline II=1
        float_v2 tmp; fifo_in.try_read(tmp);
    }
}

void black_hole_float_v16(tapa::istream<float_v16> & fifo_in) {
    for (;;) {
#pragma HLS pipeline II=1
        float_v16 tmp; fifo_in.try_read(tmp);
    }
}

void Chason(tapa::mmap<int> edge_list_ptr,
             
             tapa::mmaps<ap_uint<512>, NUM_CH_SPARSE> edge_list_ch,
             
             tapa::mmap<float_v16> vec_X,
             
             tapa::mmap<float_v16> vec_Y,
             
             tapa::mmap<float_v16> vec_Y_out,
             
             const int NUM_ITE,
             const int NUM_A_LEN,
             const int M,
             const int K,
             const int P_N,
             const int alpha_u,
             const int beta_u
             ) {
    tapa::streams<int, NUM_CH_SPARSE + 1, FIFO_DEPTH> PE_inst("PE_inst"); //17
    
    tapa::streams<float_v16, NUM_CH_SPARSE + 1, FIFO_DEPTH> fifo_X_pe("fifo_X_pe"); //8*17
    
    tapa::streams<ap_uint<512>, NUM_CH_SPARSE, FIFO_DEPTH> fifo_A("fifo_A"); //8*16
    
    tapa::streams<int, NUM_CH_SPARSE, FIFO_DEPTH> Yvec_inst("Yvec_inst"); //16
    
    tapa::streams<MultXVec, NUM_CH_SPARSE, FIFO_DEPTH> fifo_aXvec("fifo_aXvec"); //32*16
    
    tapa::streams<float_v2, NUM_CH_SPARSE, FIFO_DEPTH> fifo_Y_pe("fifo_Y_pe"); //2*16

    tapa::streams<float_v2, 8, FIFO_DEPTH> fifo_Y_pe_abd("fifo_Y_pe_abd");//2*8    
    
    tapa::stream<float_v16, FIFO_DEPTH> fifo_Y_AX("fifo_Y_AX");//16  
       
    tapa::stream<float_v16, FIFO_DEPTH> fifo_Y_alpha_AX("fifo_Y_alpha_AX");//16
    
    tapa::stream<float_v16, FIFO_DEPTH> fifo_Y_in("fifo_Y_in");//16
    
    tapa::stream<float_v16, FIFO_DEPTH> fifo_Y_in_beta("fifo_Y_in_beta");//16
    
    tapa::stream<float_v16, FIFO_DEPTH> fifo_Y_out("fifo_Y_out");//16

    /* =========NEW STREAMS======= */

    tapa::streams<float_v2, NUM_CH_SPARSE, FIFO_DEPTH> fifo_PEsrc("fifo_PEsrc"); //2*16

    tapa::stream<float_v16, FIFO_DEPTH> fifo_P_AX("fifo_P_AX");//16

    tapa::streams<float_v2, 8, FIFO_DEPTH> fifo_Y_PEsrc_abd("fifo_Y_PEsrc_abd");

    tapa::stream<float_v16, FIFO_DEPTH> fifo_PY_out("fifo_PY_out");//16

    tapa::streams<int, 8, FIFO_DEPTH> fifo_ch_idx("fifo_ch_idx");
    


    /* =========deploy modules======= */
    
    tapa::task()
        .invoke(read_edge_list_ptr,
                NUM_ITE,
                M,
                P_N,
                K,
                edge_list_ptr,
                PE_inst
                )
    
        .invoke<tapa::join>(read_X,
                            P_N,
                            K,
                            vec_X,
                            fifo_X_pe
                            )
        
        .invoke<tapa::join, NUM_CH_SPARSE>(read_A,
                                           P_N,
                                           NUM_A_LEN,
                                           edge_list_ch,
                                           fifo_A
                                           )
    
        .invoke<tapa::join, NUM_CH_SPARSE>(PEG_Xvec,
                                           PE_inst,
                                           fifo_A,
                                           fifo_X_pe,
                                           PE_inst,
                                           fifo_X_pe,
                                           Yvec_inst,
                                           fifo_aXvec
                                           )

        .invoke<tapa::detach>(black_hole_int,
                              PE_inst)

        .invoke<tapa::detach>(black_hole_float_v16,
                              fifo_X_pe)

                     
        .invoke<tapa::join, NUM_CH_SPARSE>(PEG_Yvec,
                                           Yvec_inst,
                                           fifo_aXvec,
                                           fifo_Y_pe,
                                           fifo_PEsrc
                                           )

        .invoke<tapa::join, 8>(Arbiter_PEsrc_agg,
                              P_N,
                              M,
                              fifo_PEsrc,
                              fifo_Y_PEsrc_abd,
                              fifo_ch_idx
                              )

        .invoke<tapa::detach>(Merger_Y_PEsrc,
                              fifo_Y_PEsrc_abd, 
                              fifo_ch_idx,
                              fifo_P_AX
                              )
            
        .invoke<tapa::join, 8>(Arbiter_Y,
                               P_N,
                               M,
                               fifo_Y_pe,
                               fifo_Y_pe_abd
                               )

        .invoke<tapa::detach>(Merger_Y,
                              fifo_Y_pe_abd,
                              fifo_Y_AX
                              )

        .invoke<tapa::detach>(FloatvAddFloatv,
                              fifo_P_AX,
                              fifo_Y_AX,
                              fifo_PY_out
                              )
         
/*----------------------------------------------------------------------------------------------*/

        .invoke<tapa::join>(FloatvMultConst,
                            P_N,
                            M,
                            alpha_u,
                            fifo_PY_out,
                            fifo_Y_alpha_AX
                            )

        .invoke<tapa::join>(read_Y,
                            P_N,
                            M,
                            vec_Y,
                            fifo_Y_in
                            )
    
        .invoke<tapa::join>(FloatvMultConst,
                            P_N,
                            M,
                            beta_u,
                            fifo_Y_in,
                            fifo_Y_in_beta
                            )
    
        .invoke<tapa::detach>(FloatvAddFloatv,
                              fifo_Y_alpha_AX,
                              fifo_Y_in_beta,
                              fifo_Y_out
                              )
    
        .invoke<tapa::join>(write_Y,
                            P_N,
                            M,
                            fifo_Y_out,
                            vec_Y_out
                            )
    ;
}
