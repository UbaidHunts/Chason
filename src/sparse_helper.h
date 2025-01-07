#include <vector>
#include <iostream>
#include "mmio.h"
#include <string>

using std::cout;
using std::endl;
using std::vector;
using std::min;
using std::max;
using std::string;

#ifndef SPARSE_HELPER
#define SPARSE_HELPER

struct rcv {
    int r;
    int c;
    float v;
};

enum MATRIX_FORMAT { CSR, CSC };

struct edge {
    int col;
    int row;
    float attr;
    int PE_src;

    edge(int d = -1, int s = -1, float v = 0.0): col(d), row(s), attr(v) {
    }


    edge &operator=(const edge &rhs) {
        col = rhs.col;
        row = rhs.row;
        attr = rhs.attr;
        PE_src = rhs.PE_src;
        return *this;
    }
};

int cmp_by_row_column(const void *aa,
                      const void *bb) {
    rcv *a = (rcv *) aa;
    rcv *b = (rcv *) bb;
    if (a->r > b->r) return +1;
    if (a->r < b->r) return -1;

    if (a->c > b->c) return +1;
    if (a->c < b->c) return -1;

    return 0;
}

int cmp_by_column_row(const void *aa,
                      const void *bb) {
    rcv *a = (rcv *) aa;
    rcv *b = (rcv *) bb;

    if (a->c > b->c) return +1;
    if (a->c < b->c) return -1;

    if (a->r > b->r) return +1;
    if (a->r < b->r) return -1;

    return 0;
}


void sort_by_fn(int nnz_s,
                vector<int> &cooRowIndex,
                vector<int> &cooColIndex,
                vector<float> &cooVal,
                int (*cmp_func)(const void *, const void *)) {
    rcv *rcv_arr = new rcv[nnz_s];

    for (int i = 0; i < nnz_s; ++i) {
        rcv_arr[i].r = cooRowIndex[i];
        rcv_arr[i].c = cooColIndex[i];
        rcv_arr[i].v = cooVal[i];
    }

    qsort(rcv_arr, nnz_s, sizeof(rcv), cmp_func);

    for (int i = 0; i < nnz_s; ++i) {
        cooRowIndex[i] = rcv_arr[i].r;
        cooColIndex[i] = rcv_arr[i].c;
        cooVal[i] = rcv_arr[i].v;
    }

    delete [] rcv_arr;
}

void mm_init_read(FILE *f,
                  char *filename,
                  MM_typecode &matcode,
                  int &m,
                  int &n,
                  int &nnz) {
    //if ((f = fopen(filename, "r")) == NULL) {
    //        cout << "Could not open " << filename << endl;
    //        return 1;
    //}

    if (mm_read_banner(f, &matcode) != 0) {
        cout << "Could not process Matrix Market banner for " << filename << endl;
        exit(1);
    }

    int ret_code;
    if ((ret_code = mm_read_mtx_crd_size(f, &m, &n, &nnz)) != 0) {
        cout << "Could not read Matrix Market format for " << filename << endl;
        exit(1);
    }
}

void load_S_matrix(FILE *f_A,
                   int nnz_mmio,
                   int &nnz,
                   vector<int> &cooRowIndex,
                   vector<int> &cooColIndex,
                   vector<float> &cooVal,
                   MM_typecode &matcode) {
    if (mm_is_complex(matcode)) {
        cout << "Redaing in a complex matrix, not supported yet!" << endl;
        exit(1);
    }

    if (!mm_is_symmetric(matcode)) {
        cout << "It's an NS matrix.\n";
    } else {
        cout << "It's an S matrix.\n";
    }

    int r_idx, c_idx;
    float value;
    int idx = 0;

    for (int i = 0; i < nnz_mmio; ++i) {
        if (mm_is_pattern(matcode)) {
            fscanf(f_A, "%d %d\n", &r_idx, &c_idx);
            value = 1.0;
        } else {
            fscanf(f_A, "%d %d %f\n", &r_idx, &c_idx, &value);
        }

        unsigned int *tmpPointer_v = reinterpret_cast<unsigned int *>(&value);;
        unsigned int uint_v = *tmpPointer_v;
        if (uint_v != 0) {
            if (r_idx < 1 || c_idx < 1) {
                // report error
                cout << "idx = " << idx << " [" << r_idx - 1 << ", " << c_idx - 1 << "] = " << value << endl;
                exit(1);
            }

            cooRowIndex[idx] = r_idx - 1;
            cooColIndex[idx] = c_idx - 1;
            cooVal[idx] = value;
            idx++;

            if (mm_is_symmetric(matcode)) {
                if (r_idx != c_idx) {
                    cooRowIndex[idx] = c_idx - 1;
                    cooColIndex[idx] = r_idx - 1;
                    cooVal[idx] = value;
                    idx++;
                }
            }
        }
    }
    nnz = idx;
}

void read_suitsparse_matrix(char *filename_A,
                            vector<int> &elePtr,
                            vector<int> &eleIndex,
                            vector<float> &eleVal,
                            int &M,
                            int &K,
                            int &nnz,
                            const MATRIX_FORMAT mf = CSR) {
    int nnz_mmio;
    MM_typecode matcode;
    FILE *f_A;

    if ((f_A = fopen(filename_A, "r")) == NULL) {
        cout << "Could not open " << filename_A << endl;
        exit(1);
    }

    mm_init_read(f_A, filename_A, matcode, M, K, nnz_mmio);

    if (!mm_is_coordinate(matcode)) {
        cout << "The input matrix file " << filename_A << "is not a coordinate file!" << endl;
        exit(1);
    }

    int nnz_alloc = (mm_is_symmetric(matcode)) ? (nnz_mmio * 2) : nnz_mmio;
    cout << "Matrix A -- #row: " << M << " #col: " << K << endl;

    vector<int> cooRowIndex(nnz_alloc);
    vector<int> cooColIndex(nnz_alloc);
    //eleIndex.resize(nnz_alloc);
    eleVal.resize(nnz_alloc);

    cout << "Loading input matrix A from " << filename_A << "\n";

    load_S_matrix(f_A, nnz_mmio, nnz, cooRowIndex, cooColIndex, eleVal, matcode);

    fclose(f_A);

    if (mf == CSR) {
        sort_by_fn(nnz, cooRowIndex, cooColIndex, eleVal, cmp_by_row_column);
    } else if (mf == CSC) {
        sort_by_fn(nnz, cooRowIndex, cooColIndex, eleVal, cmp_by_column_row);
    } else {
        cout << "Unknow format!\n";
        exit(1);
    }

    // convert to CSR/CSC format
    int M_K = (mf == CSR) ? M : K;
    elePtr.resize(M_K + 1);
    vector<int> counter(M_K, 0);

    if (mf == CSR) {
        for (int i = 0; i < nnz; i++) {
            counter[cooRowIndex[i]]++;
        }
    } else if (mf == CSC) {
        for (int i = 0; i < nnz; i++) {
            counter[cooColIndex[i]]++;
        }
    } else {
        cout << "Unknow format!\n";
        exit(1);
    }

    int t = 0;
    for (int i = 0; i < M_K; i++) {
        t += counter[i];
    }

    elePtr[0] = 0;
    for (int i = 1; i <= M_K; i++) {
        elePtr[i] = elePtr[i - 1] + counter[i - 1];
    }

    eleIndex.resize(nnz);
    if (mf == CSR) {
        for (int i = 0; i < nnz; ++i) {
            eleIndex[i] = cooColIndex[i];
        }
    } else if (mf == CSC) {
        for (int i = 0; i < nnz; ++i) {
            eleIndex[i] = cooRowIndex[i];
        }
    }

    if (mm_is_symmetric(matcode)) {
        //eleIndex.resize(nnz);
        eleVal.resize(nnz);
    }
}

void cpu_spmv_CSR(const int M,
                  const int K,
                  const int NNZ,
                  const float ALPHA,
                  const vector<int> &CSRRowPtr,
                  const vector<int> &CSRColIndex,
                  const vector<float> &CSRVal,
                  const vector<float> &vec_X,
                  const float BETA,
                  vector<float> &vec_Y) {
    // A: sparse matrix, M x K
    // X: dense vector, K x 1
    // Y: dense vecyor, M x 1
    // output vec_Y = ALPHA * mat_A * vec_X + BETA * vec_Y
    // dense matrices: column major

    for (int i = 0; i < M; ++i) {
        float psum = 0.0;
        for (int j = CSRRowPtr[i]; j < CSRRowPtr[i + 1]; ++j) {
            psum += CSRVal[j] * vec_X[CSRColIndex[j]];
        }
        vec_Y[i] = ALPHA * psum + BETA * vec_Y[i];
        //vec_Y[i] = psum ;
    }
    for (int i = 0; i < M; ++i) {
        //printf("vec_Y[%d]: %f\n",i, vec_Y[i]);
    }
}

void generate_edge_list_for_one_PE(const vector<edge> &tmp_edge_list,
                                   vector<edge> &edge_list,
                                   const int base_col_index,
                                   const int i_start,
                                   const int NUM_Row,
                                   const int NUM_PE,
                                   int PE_id,
                                   const int DEP_DIST_LOAD_STORE = 10) {
    edge e_empty = {-1, -1, 0.0};
    vector<edge> scheduled_edges;
    vector<int> cycles_rows(NUM_Row, -DEP_DIST_LOAD_STORE);
    int e_dst, e_src;
    float e_attr;
    for (unsigned int pp = 0; pp < tmp_edge_list.size(); ++pp) {
        e_src = tmp_edge_list[pp].col - base_col_index;
        e_dst = tmp_edge_list[pp].row / 2 / NUM_PE; // divided by two to consider a pair as one

        e_attr = tmp_edge_list[pp].attr;
        auto cycle = cycles_rows[e_dst] + DEP_DIST_LOAD_STORE;

        bool taken = true;
        while (taken) {
            if (cycle >= ((int) scheduled_edges.size())) {
                scheduled_edges.resize(cycle + 1, e_empty);
            }
            auto e = scheduled_edges[cycle];
            if (e.row != -1)
                cycle++;
            else
                taken = false;
        }
        scheduled_edges[cycle].col = e_src;
        scheduled_edges[cycle].row = e_dst * 2 + (tmp_edge_list[pp].row % 2);
        scheduled_edges[cycle].attr = e_attr;
        cycles_rows[e_dst] = cycle;
    }

    int scheduled_edges_size = scheduled_edges.size();
    if (scheduled_edges_size > 0) {
        edge_list.resize(i_start + scheduled_edges_size, e_empty);
        for (int i = 0; i < scheduled_edges_size; ++i) {
            edge_list[i + i_start] = scheduled_edges[i];
        }
    }
}


void generate_edge_list_for_all_PEs(const vector<int> &CSCColPtr,
                                    const vector<int> &CSCRowIndex,
                                    const vector<float> &CSCVal,
                                    const int NUM_PE,
                                    const int NUM_ROW,
                                    const int NUM_COLUMN,
                                    const int WINDOE_SIZE,
                                    vector<vector<edge> > &edge_list_pes,
                                    vector<int> &edge_list_ptr,
                                    const int DEP_DIST_LOAD_STORE = 10) {
    edge_list_pes.resize(NUM_PE); //outer
    edge_list_ptr.resize((NUM_COLUMN + WINDOE_SIZE - 1) / WINDOE_SIZE + 1, 0);
    int PE_id = 0;

    vector<vector<edge> > tmp_edge_list_pes(NUM_PE);
    for (int i = 0; i < (NUM_COLUMN + WINDOE_SIZE - 1) / WINDOE_SIZE; ++i) {
        for (int p = 0; p < NUM_PE; ++p) {
            tmp_edge_list_pes[p].resize(0);
        }
        for (int col = WINDOE_SIZE * i; col < min(WINDOE_SIZE * (i + 1), NUM_COLUMN); ++col) {
            for (int j = CSCColPtr[col]; j < CSCColPtr[col + 1]; ++j) {
                int p = (CSCRowIndex[j] / 2) % NUM_PE;
                int pos = tmp_edge_list_pes[p].size();
                tmp_edge_list_pes[p].resize(pos + 1);
                tmp_edge_list_pes[p][pos] = edge(col, CSCRowIndex[j], CSCVal[j]);
            }
        }

        //form the scheduled edge list for each PE
        for (int p = 0; p < NUM_PE; ++p) {
            int i_start = edge_list_pes[p].size();
            int base_col_index = i * WINDOE_SIZE;
            generate_edge_list_for_one_PE(tmp_edge_list_pes[p],
                                          edge_list_pes[p],
                                          base_col_index,
                                          i_start,
                                          NUM_ROW,
                                          NUM_PE,
                                          PE_id,
                                          DEP_DIST_LOAD_STORE);
            PE_id++;
        }

        //insert bubules to align edge list
        int max_len = 0;
        for (int p = 0; p < NUM_PE; ++p) {
            max_len = max((int) edge_list_pes[p].size(), max_len);
        }

        int sum_ = 0;
        for (int p = 0; p < NUM_PE; ++p) {
            edge_list_pes[p].resize(max_len, edge(-1, -1, 0.0));
            sum_ += edge_list_pes[p].size();
        }
        edge_list_ptr[i + 1] = max_len;
    }
}

void edge_list_64bit(const vector<vector<edge> > &edge_list_pes,
                     const vector<int> &edge_list_ptr,
                     vector<vector<unsigned long, tapa::aligned_allocator<unsigned long> > > &sparse_A_fpga_vec,
                     vector<vector<edge> > &sparse_A_fpga,
                     vector<vector<std::string> > &hbm_ch,
                     const int NUM_CH_SPARSE = 16) {
    
    int sparse_A_fpga_column_size = 8 * edge_list_ptr[edge_list_ptr.size() - 1] * 4 / 4;
    int sparse_A_fpga_chunk_size = ((sparse_A_fpga_column_size + 511) / 512) * 512;
    edge e_empty = {-1, -1, 0.0};
    edge e_ass = e_empty;

    for (int cc = 0; cc < NUM_CH_SPARSE; ++cc) {
        sparse_A_fpga_vec[cc].resize(sparse_A_fpga_chunk_size, 0);
        sparse_A_fpga[cc].resize(sparse_A_fpga_chunk_size);
        hbm_ch[cc].resize(sparse_A_fpga_chunk_size, "00");
        for (int i = 0; i < sparse_A_fpga[cc].size(); i++) {
            sparse_A_fpga[cc][i] = e_empty;
            sparse_A_fpga[cc][i].PE_src = -1;
        }
    }

    for (int i = 0; i < edge_list_ptr[edge_list_ptr.size() - 1]; ++i) {
        for (int cc = 0; cc < NUM_CH_SPARSE; ++cc) {
            for (int j = 0; j < 8; ++j) {
                edge e = edge_list_pes[j + cc * 8][i];
                unsigned long x = 0;
                if (e.row == -1) {
                    x = 0x3FFFF; //0xFFFFF; //x = 0x3FFFFF;
                    x = x << 32;
                    e_ass = e_empty;
                } else {
                    unsigned long x_col = e.col;
                    unsigned long x_row = e.row;
                    x_col = (x_col & 0x3FFF) << (32 + 18);
                    x_row = (x_row & 0x3FFFF) << 32;

                    float x_float = e.attr;
                    unsigned int x_float_in_int = *((unsigned int *) (&x_float));
                    unsigned long x_float_val_64 = ((unsigned long) x_float_in_int);
                    x_float_val_64 = x_float_val_64 & 0xFFFFFFFF;

                    x = x_col | x_row | x_float_val_64;
                    e_ass = {e.col, e.row, x_float};
                }
                if (NUM_CH_SPARSE == 16) {
                    int pe_idx = j + cc * 8;
                    int pix_m16 = pe_idx % 16;
                    int a = (pix_m16 % 8) * 2 + pix_m16 / 8;
                    int b = (pe_idx % 128) / 16 + i * 8;
                    //the spacing helps later in the merger stage.
                    //ch= 0: pe  0(  0,   1) pe 16( 32,  33) pe 32( 64,  65) pe 48( 96,  97) pe 64(128, 129) pe 80(160, 161) pe 96(192, 193) pe112(224, 225)
                    //ch= 1: pe  8( 16,  17) pe 24( 48,  49) pe 40( 80,  81) pe 56(112, 113) pe 72(144, 145) pe 88(176, 177) pe104(208, 209) pe120(240, 241)
                    //ch= 2: pe  1(  2,   3) pe 17( 34,  35) pe 33( 66,  67) pe 49( 98,  99) pe 65(130, 131) pe 81(162, 163) pe 97(194, 195) pe113(226, 227)
                    //ch= 3: pe  9( 18,  19) pe 25( 50,  51) pe 41( 82,  83) pe 57(114, 115) pe 73(146, 147) pe 89(178, 179) pe105(210, 211) pe121(242, 243)
                    //ch= 4: pe  2(  4,   5) pe 18( 36,  37) pe 34( 68,  69) pe 50(100, 101) pe 66(132, 133) pe 82(164, 165) pe 98(196, 197) pe114(228, 229)
                    //ch= 5: pe 10( 20,  21) pe 26( 52,  53) pe 42( 84,  85) pe 58(116, 117) pe 74(148, 149) pe 90(180, 181) pe106(212, 213) pe122(244, 245)
                    //ch= 6: pe  3(  6,   7) pe 19( 38,  39) pe 35( 70,  71) pe 51(102, 103) pe 67(134, 135) pe 83(166, 167) pe 99(198, 199) pe115(230, 231)
                    //ch= 7: pe 11( 22,  23) pe 27( 54,  55) pe 43( 86,  87) pe 59(118, 119) pe 75(150, 151) pe 91(182, 183) pe107(214, 215) pe123(246, 247)
                    //ch= 8: pe  4(  8,   9) pe 20( 40,  41) pe 36( 72,  73) pe 52(104, 105) pe 68(136, 137) pe 84(168, 169) pe100(200, 201) pe116(232, 233)
                    //ch= 9: pe 12( 24,  25) pe 28( 56,  57) pe 44( 88,  89) pe 60(120, 121) pe 76(152, 153) pe 92(184, 185) pe108(216, 217) pe124(248, 249)
                    //ch=10: pe  5( 10,  11) pe 21( 42,  43) pe 37( 74,  75) pe 53(106, 107) pe 69(138, 139) pe 85(170, 171) pe101(202, 203) pe117(234, 235)
                    //ch=11: pe 13( 26,  27) pe 29( 58,  59) pe 45( 90,  91) pe 61(122, 123) pe 77(154, 155) pe 93(186, 187) pe109(218, 219) pe125(250, 251)
                    //ch=12: pe  6( 12,  13) pe 22( 44,  45) pe 38( 76,  77) pe 54(108, 109) pe 70(140, 141) pe 86(172, 173) pe102(204, 205) pe118(236, 237)
                    //ch=13: pe 14( 28,  29) pe 30( 60,  61) pe 46( 92,  93) pe 62(124, 125) pe 78(156, 157) pe 94(188, 189) pe110(220, 221) pe126(252, 253)
                    //ch=14: pe  7( 14,  15) pe 23( 46,  47) pe 39( 78,  79) pe 55(110, 111) pe 71(142, 143) pe 87(174, 175) pe103(206, 207) pe119(238, 239)
                    //ch=15: pe 15( 30,  31) pe 31( 62,  63) pe 47( 94,  95) pe 63(126, 127) pe 79(158, 159) pe 95(190, 191) pe111(222, 223) pe127(254, 255)


                    sparse_A_fpga_vec[a][b] = x;
                    sparse_A_fpga[a][b] = e_ass;
                    sparse_A_fpga[a][b].PE_src = -1;
                }
                else {
                    cout<<"For now, Chason only supports 16 HBM Channels for Sparse Matrix A. Terminating !!"<<endl;
                    exit(1);
                }
            }
        }
    }
}

void CSC_2_CSR(int M,
               int K,
               int NNZ,
               const vector<int> &csc_col_Ptr,
               const vector<int> &csc_row_Index,
               const vector<float> &cscVal,
               vector<int> &csr_row_Ptr,
               vector<int> &csr_col_Index,
               vector<float> &csrVal) {
    csr_row_Ptr.resize(M + 1, 0);
    csrVal.resize(NNZ, 0.0);
    csr_col_Index.resize(NNZ, 0);

    for (int i = 0; i < NNZ; ++i) {
        csr_row_Ptr[csc_row_Index[i] + 1]++;
    }

    for (int i = 0; i < M; ++i) {
        csr_row_Ptr[i + 1] += csr_row_Ptr[i];
    }

    vector<int> row_nz(M, 0);
    for (int i = 0; i < K; ++i) {
        for (int j = csc_col_Ptr[i]; j < csc_col_Ptr[i + 1]; ++j) {
            int r = csc_row_Index[j];
            int c = i;
            float v = cscVal[j];

            int pos = csr_row_Ptr[r] + row_nz[r];
            csrVal[pos] = v;
            csr_col_Index[pos] = c;
            row_nz[r]++;
        }
    }
}

// ***************************************************************************************** */

void print_edges(vector<vector<edge> > hbm_ch, 
                int NUM_CH, 
                int *NNZ_per_ch, 
                int *stall_per_ch, 
                vector<int> &hbm_size ) {
    
    /* FOR DEBUGGING */
    cout << "HBM triplets: (Length, NNZ, Stalls)" << endl;
    int max_len = 0;
    for (int i = 0; i < 16; i++) {
        int stall = 0;
        cout<<"\n\n************************************Printing Channel: "<<i<<
              " ************************************"<<endl;
        cout<<"************************************Total Size      : "<<hbm_ch[i].size()<<
             " ************************************\n\n"<<endl;
        for (int j = 0; j < hbm_ch[i].size(); j++) {
              if (j%8==0) cout<<endl;
               printf("ch[%d][%d][%d]: (%d, %d, %f, %d)\n",i,j,j%8,
                       hbm_ch[i][j].col,hbm_ch[i][j].row,hbm_ch[i][j].attr, hbm_ch[i][j].PE_src);
            if (hbm_ch[i][j].row < 0)
                stall++;
        }
        NNZ_per_ch[i] = hbm_ch[i].size() - stall;
        stall_per_ch[i] = stall;
        int len = hbm_size[i] + 1;
        max_len = max(max_len, len);
        printf("In HBM[%d]: (%d,%d,%d)--[%]Stall= %f\n", i, hbm_ch[i].size(),
               NNZ_per_ch[i], stall_per_ch[i], (stall_per_ch[i] + 0.0001) * 100 / (hbm_ch[i].size()));
        if ((i + 1) % 4 == 0)cout << endl;
    }
}

void hbm_ch_length(vector<vector<edge> > hbm_ch, 
                  vector<int> &hbm_size) {
    hbm_size.resize(hbm_ch.size(), 0);
    for (int i = 0; i < hbm_ch.size(); i++) {
        int len = 0;
        for (int j = 0; j < hbm_ch[i].size(); j++) {
            if (hbm_ch[i][j].col > -1) {
                len = j;
            }
        }
        hbm_size[i] = len;
    }
}

void coalescing(vector<vector<edge>> &sparse_A, 
                vector<vector<unsigned long, 
                tapa::aligned_allocator<unsigned long>>>& sparse_A_fpga_vec ) {

    int pe_src=1;
    for (int i=0; i<sparse_A.size(); i++){
        sparse_A_fpga_vec[i].resize(sparse_A[i].size());
        for (int j=0; j<sparse_A[i].size();j++){
            edge e = sparse_A[i][j];
            unsigned long x =0;
            unsigned long pvt;
            if (e.row == -1) {   //STALL
                x = 0x7FFFF; //0xFFFFF; //x = 0x3FFFFF;
                x = x << 32;

            } else {
                
                unsigned long x_row = e.row;
                x_row = (x_row & 0x7FFFF) << 32; //x_row = (x_row & 0xFFFFF) << 32; //x_row = (x_row & 0x3FFFFF) << 32;
                
                unsigned long PE_src = e.PE_src;
                if (e.PE_src<0){ //private
                    PE_src= 0;
                    pvt = 1;
                    pvt = (pvt & 0x01) << (32+15);
                    int pvt_pe_idx = j % 8;
                    e.PE_src = pvt_pe_idx; //just to make lengthCheckMod() work
                    PE_src = pvt_pe_idx;
                    PE_src = (PE_src & 0x07) << (32+15+1);                    
                }
                else {
                    pvt=0;
                    pvt = (pvt & 0x00) << (32+15);
                    PE_src = (PE_src & 0x07) << (32+15+1);
                }

                unsigned long x_col = e.col;
                x_col = (x_col & 0x1FFF) << (32 + 15 + 1 +3); // x_col = (x_col & 0xFFF) << (32 + 20); //x_col = (x_col & 0x3FF) << (32 + 22);

                float x_float = e.attr;
                //float x_float = 1.0;
                unsigned int x_float_in_int = *((unsigned int*)(&x_float));
                unsigned long x_float_val_64 = ((unsigned long) x_float_in_int);
                x_float_val_64 = x_float_val_64 & 0xFFFFFFFF;
                
                x = x_col | PE_src | pvt | x_row | x_float_val_64;
            }
            sparse_A_fpga_vec[i][j] = x;
        }
    }

}

void phase_one(vector<vector<edge> > &hbm_ch) {
    int NUM_CH_SPARSE = 16; //or, 32, 40, 48, 56
    int URAM_DEPTH = ((NUM_CH_SPARSE == 16) ? 1 : 2) * 4096; // 16 -> 12,288, others -> 8,192
    edge e_empty = {-1, -1, 0.0};
    vector<vector<vector<vector<int> > > > trk_mem;
    trk_mem.resize(hbm_ch.size());

    for (int i = 0; i < trk_mem.size(); i++) {
        trk_mem[i].resize(8);
        for (int j = 0; j < 8; j++) {
            trk_mem[i][j].resize(8);
            for (int k = 0; k < 8; k++) {
                trk_mem[i][j][k].resize(URAM_DEPTH, -1);
            }
        }
    }

    // Preprocess channels to get non-zero indices
    vector<vector<int> > non_zero_indices(16);
    for (int ch = 0; ch < 16; ++ch) {
        for (int idx = 0; idx < hbm_ch[ch].size(); ++idx) {
            if (hbm_ch[ch][idx].col >= 0) {
                non_zero_indices[ch].push_back(idx);
            }
        }
    }

    for (int i = 0; i < 16; ++i) {
        int src_pe;
        int dst_pe ;
        int stall = 0;
        int dst_ch = i;
        int src_ch = (i + 1) % 16; // Wrap around
        bool last_ch = (i == 16 - 1);

        for (int j = 0; j < hbm_ch[dst_ch].size(); ++j) {
            int idx;
            int itr;
            itr = non_zero_indices[src_ch].size() - 1;

            if (hbm_ch[dst_ch][j].col < 0) {
                stall++;
                while (itr >= 0) {
                    idx = non_zero_indices[src_ch][itr];
                    edge &e = hbm_ch[src_ch][idx];
                    if (e.col > -1) {
                        if (last_ch && hbm_ch[src_ch][idx].PE_src != -1) {
                            itr--;
                            continue;
                        }

                        dst_pe = j % 8;
                        src_pe = idx % 8;
                        int src_row = e.row;
                        bool odd = src_row % 2;
                        int pair = (odd) ? -1 : 1;

                        int dst_cycle = trk_mem[dst_ch][dst_pe][src_pe][src_row];
                        int dst_cycle2 = trk_mem[dst_ch][dst_pe][src_pe][src_row + pair];

                        if (dst_cycle > -1 && dst_cycle2 > -1) {
                            if (j < dst_cycle || j < dst_cycle2) {
                                itr--;

                                continue;
                            }
                        }

                        // Update tracking memory
                        trk_mem[dst_ch][dst_pe][src_pe][src_row] = j + (8 * 10);
                        trk_mem[dst_ch][dst_pe][src_pe][src_row + pair] = j + (8 * 10);

                        // Replace stall with non-zero element
                        hbm_ch[dst_ch][j] = hbm_ch[src_ch][idx];
                        hbm_ch[dst_ch][j].PE_src = src_pe;

                        // Mark the source element as used
                        e = e_empty;
                        e.PE_src = -11;
                        break;
                    }
                    itr--;
                }
            }
        }
    }
}

void resize_ch0(vector<vector<edge> > &hbm_ch) {
    vector<edge> shadow;
    vector<vector<vector<int> > > trk_mem;
    edge e_empty = {-99, -99, -99};
    trk_mem.resize(8);

    for (int i = 0; i < trk_mem.size(); i++) {
        trk_mem[i].resize(8);
        for (int k = 0; k < 8; k++) {
            trk_mem[i][k].resize(URAM_DEPTH, -1);
        }
    }

    int first_nnz = 0;
    int PE_addr = -1;
    for (int i = 0; i < hbm_ch[0].size(); i++) {
        int src = hbm_ch[0][i + 1].row;
        if (hbm_ch[0][i].PE_src != -11) {
            first_nnz = i;
            PE_addr = i % 8;
            int size = hbm_ch[0].size();
            shadow.resize(hbm_ch[0].size() - first_nnz + PE_addr, e_empty);
            break;
        }
    }

    for (int i = first_nnz; i < hbm_ch[0].size(); i++) {
        shadow[PE_addr] = hbm_ch[0][i];
        PE_addr++;
    }

    vector<edge> shadow2;
    int size = shadow.size();
    shadow2.resize(size, e_empty);
    for (int i = 0; i < shadow2.size(); i++)shadow2[i].PE_src = -1;
    int sh2 = 0;
    int dst_pe;
    int src_pe;
    int src_row;

    for (int i = 0; i < shadow.size(); i++) {
        if (shadow[i].col > -1) {
        A:
            dst_pe = sh2 % 8;
            src_pe = shadow[i].PE_src; 
            src_row = shadow[i].row;
            
            bool odd = src_row % 2;
            int pair = (odd) ? -1 : 1;
            
            if (shadow[i].PE_src != -1) {
                int dst_cycle = trk_mem[dst_pe][src_pe][src_row];
                int dst_cycle2 = trk_mem[dst_pe][src_pe][src_row + pair];
                if (dst_cycle > -1 && dst_cycle2 > -1 ) {
                    if (sh2 < dst_cycle || sh2 < dst_cycle2) {
                        sh2++;
                        goto A;
                    }
                }
                trk_mem[dst_pe][src_pe][src_row] = sh2 + (8 * 10); //*DEP_DIST_LOAD_STORE;
                trk_mem[dst_pe][src_pe][src_row + pair] = sh2 + (8 * 10);
            }
            shadow2[sh2] = shadow[i];
            sh2++;
        }
    }
    shadow2.resize(sh2);
    hbm_ch[0].resize(shadow2.size());
    hbm_ch[0] = shadow2;
}

void phase_one_caller(vector<vector<edge> > &hbm_ch, vector<int> &edge_list_ptr, vector<vector<edge> > &hbm_ch_op) {
    hbm_ch_op.resize(16);
    vector<int> new_edge_list_ptr(edge_list_ptr.size());
    new_edge_list_ptr[0] = 0;

    for (int i = 0; i < edge_list_ptr.size() - 1; i++) {
        int start = edge_list_ptr[i] * 8;
        int end = edge_list_ptr[i + 1] * 8;
        vector<vector<edge> > hbm_ch_temp(16);
        for (int k = 0; k < 16; k++) {
            hbm_ch_temp[k].resize(end - start);
            int itr = 0;
            for (int j = start; j < end; j++) {
                hbm_ch_temp[k][itr] = hbm_ch[k][j];
                itr++;
            }
        }
        //remap the temp_ch
        phase_one(hbm_ch_temp);
        resize_ch0(hbm_ch_temp);

        //adjust the sizes to fpga chunk sizes
        int max_len = 0;
        vector<int> hbm_size(16);
        hbm_ch_length(hbm_ch_temp, hbm_size);
        for (int k = 0; k < 16; k++) {
            int siz = hbm_size[k] + 1;
            max_len = max(siz, max_len);
        }
        int max_len_MULT_512 = ((max_len + 511) / 512) * 512;
        for (int k = 0; k < 16; k++) {
            hbm_ch_temp[k].resize(max_len_MULT_512, 0);
        }
        start = new_edge_list_ptr[i] * 8;
        for (int k = 0; k < 16; k++) {
            if (hbm_ch_op[k].size() != 0) hbm_ch_op[k].resize(hbm_ch_op[k].size() + hbm_ch_temp[k].size());
            else hbm_ch_op[k].resize(hbm_ch_temp[k].size());
            for (int l = 0; l < hbm_ch_temp[k].size(); l++) {
                hbm_ch_op[k][l + start] = hbm_ch_temp[k][l];
            }
        }
        new_edge_list_ptr[i + 1] = new_edge_list_ptr[i] + max_len_MULT_512 / 8;
    }

    for (int ptr = 0; ptr < new_edge_list_ptr.size(); ptr++) {
        edge_list_ptr[ptr] = new_edge_list_ptr[ptr];
    }
}

#endif
