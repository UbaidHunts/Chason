#include <cmath>
#include <algorithm>
#include <vector>
#include <cstdlib>
#include <chrono>
#include <iostream>

#include <ap_int.h>
#include <tapa.h>

#include "mmio.h"
#include "chason.h"
#include "sparse_helper.h"

using std::cout;
using std::endl;
using std::ifstream;
using std::string;
using std::vector;
using std::min;
using std::max;

template <typename T>
using aligned_vector = std::vector<T, tapa::aligned_allocator<T>>;

int main(int argc, char **argv) {
    // cout<<"---------------------------- ---------------------------- -------------"<<argv[1]<<"--------------- ---------------------------- ----------------------------\n";
    printf("start host\n");

    float ALPHA = 0.85;
    float BETA = -2.06;
    int rp_time = 1;

    if (argc == 5) {
        rp_time = atoi(argv[2]);
        ALPHA = atof(argv[3]);
        BETA = atof(argv[4]);
    }
    else if (argc == 4) {
        ALPHA = atof(argv[2]);
        BETA = atof(argv[3]);
    }
    else if (argc == 3) {
        rp_time = atoi(argv[2]);
    }
    else if (argc != 2) {
        cout << "Usage: " << argv[0] << " [matrix A file] [rp_time] [alpha] [beta]" << std::endl;
        return EXIT_FAILURE;
    }

    char * filename_A = argv[1];

    cout << "rp_time = " << rp_time <<  "\n";
    cout << "alpha = "  << ALPHA << "\n";
    cout << "beta = "  << BETA << "\n";

    int M, K, nnz;
    vector<int> CSCColPtr;
    vector<int> CSCRowIndex;
    vector<float> CSCVal;
    vector<int> CSRRowPtr;
    vector<int> CSRColIndex;
    vector<float> CSRVal;

    cout << "Reading sparse A matrix ...";

    read_suitsparse_matrix(filename_A,
                           CSCColPtr,
                           CSCRowIndex,
                           CSCVal,
                           M,
                           K,
                           nnz,
                           CSC);
    
    CSC_2_CSR(M,
              K,
              nnz,
              CSCColPtr,
              CSCRowIndex,
              CSCVal,
              CSRRowPtr,
              CSRColIndex,
              CSRVal);
    
    cout <<  "done\n";

    cout << "Matrix size: \n";
    cout << "A: sparse matrix, " << M << " x " << K << ". NNZ = " << nnz <<  "\n";
    
    // initiate vec X and vec Y
    vector<float> vec_X_cpu, vec_Y_cpu;
    vec_X_cpu.resize(K, 0.0);
    vec_Y_cpu.resize(M, 0.0);
    
    cout << "Generating vector X ...";
    for (int kk = 0; kk < K; ++kk) {
        vec_X_cpu[kk] = 1.0 * (kk + 1);
    }
    
    cout << "Generating vector Y ...";
    for (int mm = 0; mm < M; ++mm) {
        vec_Y_cpu[mm] = -2.0 * (mm + 1);
    }
    
    cout << "done\n";
    cout << "Preparing sparse A for FPGA with " << NUM_CH_SPARSE << " HBM channels ...";

    vector<vector<edge> > edge_list_pes;
    vector<int> edge_list_ptr;

    generate_edge_list_for_all_PEs(CSCColPtr, //const vector<int> & CSCColPtr,
                                   CSCRowIndex, //const vector<int> & CSCRowIndex,
                                   CSCVal, //const vector<float> & CSCVal,
                                   NUM_CH_SPARSE*8, //const int NUM_PE,
                                   M, //const int NUM_ROW,
                                   K, //const int NUM_COLUMN,
                                   WINDOW_SIZE, //const int WINDOW_SIZE,
                                   edge_list_pes, //vector<vector<edge> > & edge_list_pes,
                                   edge_list_ptr, //vector<int> & edge_list_ptr,
                                   DEP_DIST_LOAD_STORE
                                   ); //const int DEP_DIST_LOAD_STORE = 10)

    aligned_vector<int> edge_list_ptr_fpga;
    int edge_list_ptr_fpga_size = ((edge_list_ptr.size() + 15) / 16) * 16;
    int edge_list_ptr_fpga_chunk_size = ((edge_list_ptr_fpga_size + 1023)/1024) * 1024;
    edge_list_ptr_fpga.resize(edge_list_ptr_fpga_chunk_size, 0);

    for (int i = 0; i < edge_list_ptr.size(); ++i) {
        edge_list_ptr_fpga[i] = edge_list_ptr[i];       
    }

    vector<aligned_vector<unsigned long> > sparse_A_fpga_vec(NUM_CH_SPARSE);
    vector<vector<edge> > sparse_A_fpga(NUM_CH_SPARSE);
    vector<vector<std::string> > hbm_ch(NUM_CH_SPARSE);
    int sparse_A_fpga_column_size = 8 * edge_list_ptr[edge_list_ptr.size()-1] * 4 / 4;
    int sparse_A_fpga_chunk_size = ((sparse_A_fpga_column_size + 511)/512) * 512;

    edge_list_64bit(edge_list_pes,
                    edge_list_ptr,
                    sparse_A_fpga_vec,
                    sparse_A_fpga,
                    hbm_ch,
                    NUM_CH_SPARSE);

    cout << "done\n";
    cout << "Preparing vector X for FPGA ...";
    
    int vec_X_fpga_column_size = ((K + 16 - 1) / 16) * 16;
    int vec_X_fpga_chunk_size = ((vec_X_fpga_column_size + 1023)/1024) * 1024;
    aligned_vector<float> vec_X_fpga(vec_X_fpga_chunk_size, 0.0);
    
    for (int kk = 0; kk < K; ++kk) {
        vec_X_fpga[kk] = vec_X_cpu[kk];
    }
    
    cout << "Preparing vector Y for FPGA ...";
    int vec_Y_fpga_column_size = ((M + 16 - 1) / 16) * 16;
    int vec_Y_fpga_chunk_size = ((vec_Y_fpga_column_size + 1023)/1024) * 1024;
    aligned_vector<float> vec_Y_fpga(vec_Y_fpga_chunk_size, 0.0);
    aligned_vector<float> vec_Y_out_fpga(vec_Y_fpga_chunk_size, 0.0);
    
    for (int mm = 0; mm < M; ++mm) {
        vec_Y_fpga[mm] = vec_Y_cpu[mm];
    }
    
    cout <<  "done\n";

    cout << "Run spmv on cpu...";
    auto start_cpu = std::chrono::steady_clock::now();
    cpu_spmv_CSR(M, K, nnz, ALPHA,
                 CSRRowPtr,
                 CSRColIndex,
                 CSRVal,
                 vec_X_cpu,
                 BETA,
                 vec_Y_cpu);
    auto end_cpu = std::chrono::steady_clock::now();
    double time_cpu = std::chrono::duration_cast<std::chrono::nanoseconds>(end_cpu - start_cpu).count();
    time_cpu *= 1e-9;
    cout << "done (" << time_cpu*1000 << " msec)\n";
    cout <<"CPU GFLOPS: " << 2.0 * (nnz + M) / 1e+9 / time_cpu << "\n";

    int MAX_SIZE_edge_LIST_PTR = edge_list_ptr.size() - 1;  //number of partitions wrt window size
    int MAX_LEN_edge_PTR = edge_list_ptr[MAX_SIZE_edge_LIST_PTR]; // size of all the ege_lists. They are all of same lengths.
    int * tmpPointer_v;
    tmpPointer_v = (int*) &ALPHA;
    int alpha_int = *tmpPointer_v;
    tmpPointer_v = (int*) &BETA;
    int beta_int = *tmpPointer_v;

    std::string bitstream;
    if (const auto bitstream_ptr = getenv("TAPAB")) {
        bitstream = bitstream_ptr;
    }

    vector<int> hbm_size;
    int NNZ_per_ch[NUM_CH_SPARSE];
    int stall_per_ch[NUM_CH_SPARSE];

    // /cout<<"---------------------------- Pre-Rescheduling ----------------------------\n\n";

    // hbm_ch_length(sparse_A_fpga, hbm_size);    
    // print_edges(sparse_A_fpga, NUM_CH_SPARSE, NNZ_per_ch, stall_per_ch,hbm_size);
    
    //cout<<"\n---------------------------- Post-Rescheduling ----------------------------\n\n";
    cout<<"\n---------------------------- Scheduling ----------------------------\n";
    vector<vector<edge>> sparse_A;
    vector<aligned_vector<unsigned long> > sparse_A_fpga_in(NUM_CH_SPARSE);


    phase_one_caller(sparse_A_fpga,edge_list_ptr,sparse_A);
    coalescing(sparse_A, sparse_A_fpga_in);

    
    //print_edges(sparse_A, NUM_CH_SPARSE, NNZ_per_ch, stall_per_ch, hbm_size);
    printf("Pre-Rescheduling  Channel Length: %d\n", sparse_A_fpga[0].size());
    printf("Post-Rescheduling Channel Length: %d\n", sparse_A[0].size());
    printf("Improvement  in   Channel Length: %f (more than 1 is better)\n", 
            double(sparse_A_fpga[0].size())/sparse_A[0].size());


    cout<<"---------------------------- Chason ----------------------------\n";
    
    
    aligned_vector<int> edge_list_ptr_fpga2;
    edge_list_ptr_fpga_size = ((edge_list_ptr.size() + 15) / 16) * 16;
    edge_list_ptr_fpga_chunk_size = ((edge_list_ptr_fpga_size + 1023)/1024) * 1024;
    
    edge_list_ptr_fpga2.resize(edge_list_ptr_fpga_chunk_size, 0);
    MAX_SIZE_edge_LIST_PTR = edge_list_ptr.size() - 1;  //number of partitions wrt window size
    MAX_LEN_edge_PTR = edge_list_ptr[MAX_SIZE_edge_LIST_PTR]; // size of all the ege_lists. They are all of same lengths.
    
    for (int i = 0; i < edge_list_ptr.size(); ++i) {
        edge_list_ptr_fpga2[i] = edge_list_ptr[i];       
    }

    cout << "launch kernel\n";
    double time_taken_ //= 10000;
    = tapa::invoke(Chason, bitstream,
                   tapa::read_only_mmap<int>(edge_list_ptr_fpga2), //edge_list_ptr
                   tapa::read_only_mmaps<unsigned long, NUM_CH_SPARSE>(sparse_A_fpga_in).reinterpret<ap_uint<512>>(),
                   tapa::read_only_mmap<float>(vec_X_fpga).reinterpret<float_v16>(),
                   tapa::read_only_mmap<float>(vec_Y_fpga).reinterpret<float_v16>(),
                   tapa::write_only_mmap<float>(vec_Y_out_fpga).reinterpret<float_v16>(),
                   MAX_SIZE_edge_LIST_PTR, //NUM_ITE
                   MAX_LEN_edge_PTR, //NUM_A_LEN
                   M,
                   K,
                   rp_time, //P_N
                   alpha_int,
                   beta_int
                   );
    time_taken_ *= (1e-9/rp_time); // total time in second
    printf("Kernel time is %f ms\n", time_taken_*1000);
    
    float gflops_ =
    2.0 * (nnz + M)
    / 1e+9
    / time_taken_
    ;
    
    printf("GFLOPS:%f \n", gflops_);
    int mismatch_cnt_ = 0;
    for (int mm = 0; mm < M; ++mm) {
        float v_cpu = vec_Y_cpu[mm];
        float v_fpga = vec_Y_out_fpga[mm];
        float dff = fabs(v_cpu - v_fpga);
        float x = min(fabs(v_cpu), fabs(v_fpga)) + 1e-4;
        mismatch_cnt_ += (dff/x > 1e-4);
    }

    float diffpercent_ = 100.0 * mismatch_cnt_ / M;
    bool pass_ = diffpercent_ < 2.0;

    if(pass_){
        cout << "Success!\n";
    } else{
        cout << "Failed.\n";
    }

    printf("num_mismatch = %d, percent = %.2f%%\n\n", mismatch_cnt_, diffpercent_);

    // cout<<"---------------------------- ---------------------------- ---------------------------- ---------------------------- ----------------------------\n";
    return EXIT_SUCCESS;
}
