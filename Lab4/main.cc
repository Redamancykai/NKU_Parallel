#include <vector>
#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <set>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <sys/time.h>
#include <omp.h>
#include "hnswlib/hnswlib/hnswlib.h"
#include "flat_scan.h"
#include "flat_simd.h"
#include "sq_simd.h"
#include "pq_simd.h"
#include "flat_pthread.h"
#include "flat_openmp.h"
#include "pq_pthread.h"
#include "pq_openmp.h"
#include "ivf_simd.h"
#include "ivf_simd_openmp.h"
#include "ivf_pq_simd.h"
#include "ivf_pq_openmp.h"
#include "ivf_pq_pthread.h"
#include "ivf_pq_simd_other.h"
#include "hnsw_search.h"
#include "ivf_pq_mpi.h"
#include "ivf_pq_mpi_pthread.h"
#include "ivf_pq_mpi_openmp.h"
#include "ivf_hnsw_mpi.h"
#include "partition_hnsw_mpi.h"
#include "hnsw_hnsw_mpi.h"
// 可以自行添加需要的头文件

using namespace hnswlib;

template<typename T>
T *LoadData(std::string data_path, size_t& n, size_t& d)
{
    std::ifstream fin;
    fin.open(data_path, std::ios::in | std::ios::binary);
    fin.read((char*)&n,4);
    fin.read((char*)&d,4);
    T* data = new T[n*d];
    int sz = sizeof(T);
    for(int i = 0; i < n; ++i){
        fin.read(((char*)data + i*d*sz), d*sz);
    }
    fin.close();

    std::cerr<<"load data "<<data_path<<"\n";
    std::cerr<<"dimension: "<<d<<"  number:"<<n<<"  size_per_element:"<<sizeof(T)<<"\n";

    return data;
}

struct SearchResult
{
    float recall;
    int64_t latency; // 单位us
};

void build_index(float* base, size_t base_number, size_t vecdim)
{
    const int efConstruction = 150; // 为防止索引构建时间过长，efc建议设置200以下
    const int M = 16; // M建议设置为16以下

    HierarchicalNSW<float> *appr_alg;
    InnerProductSpace ipspace(vecdim);
    appr_alg = new HierarchicalNSW<float>(&ipspace, base_number, M, efConstruction);

    appr_alg->addPoint(base, 0);
    #pragma omp parallel for
    for(int i = 1; i < base_number; ++i) {
        appr_alg->addPoint(base + 1ll*vecdim*i, i);
    }

    char path_index[1024] = "files/hnsw.index";
    appr_alg->saveIndex(path_index);
}


int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    int mpi_rank = 0, mpi_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    size_t test_number = 0, base_number = 0;
    size_t test_gt_d = 0, vecdim = 0;

    std::string data_path = "/anndata/"; 
    auto test_query = LoadData<float>(data_path + "DEEP100K.query.fbin", test_number, vecdim);
    auto test_gt = LoadData<int>(data_path + "DEEP100K.gt.query.100k.top100.bin", test_number, test_gt_d);
    auto base = LoadData<float>(data_path + "DEEP100K.base.100k.fbin", base_number, vecdim);
    // 只测试前2000条查询
    test_number = 2000;

    const size_t k = 10;

    std::vector<SearchResult> results;
    results.resize(test_number);

    // 如果你需要保存索引，可以在这里添加你需要的函数，你可以将下面的注释删除来查看pbs是否将build.index返回到你的files目录中
    // 要保存的目录必须是files/*
    // 每个人的目录空间有限，不需要的索引请及时删除，避免占空间太大
    // 不建议在正式测试查询时同时构建索引，否则性能波动会较大
    // 下面是一个构建hnsw索引的示例
    // build_index(base, base_number, vecdim);

    // SQ 预处理代码
    // float min_val, max_val, step, inv_step;
    // std::vector<uint32_t> base_qsum;
    // std::vector<float> base_bias;

    // uint8_t* base_q = quantize_base_u8(
    //     base,
    //     base_number,
    //     vecdim,
    //     min_val,
    //     max_val,
    //     step,
    //     inv_step,
    //     base_qsum,
    //     base_bias
    // );

    // PQ 预处理代码
    // PQIndex pq(4, 256, 10);
    // pq.build_pq_index(base, base_number, vecdim);

    // PQPthreadIndex pq_pthread(4, 256, 10, 6);
    // pq_pthread.build_pq_index(base, base_number, vecdim);

    // PQOpenMPIndex pq_openmp(4, 256, 10, 2);
    // pq_openmp.build_pq_index(base, base_number, vecdim);

    // IVF 预处理代码
    // IVFIndex ivf;
    // ivf_build(ivf, base, base_number, vecdim, 2048, 10);

    // IVF-PQ 预处理代码
    // IVFPQIndex index;

    // ivfpq_build(index, base, base_number, vecdim,
    //     1024,   // nlist
    //     16,     // M
    //     256,    // Ks
    //     10,     // PQ train iters
    //     10      // IVF train iters
    // );

    // IVFFirstPQIndex ivffirst_pq;

    // ivffirstpq_build(ivffirst_pq, base, base_number, vecdim,
    //     1024,   // nlist
    //     16,      // M
    //     256,    // Ks
    //     10,     // pq_train_iters
    //     10      // ivf_train_iters
    // );

    // HNSW 预处理代码
    // HNSWIndex hnsw;
    // hnsw_load(hnsw, base_number, vecdim);

    // MPI 预处理代码
    IVFPQIndex mpi_ivfpq;

    IVFPQMPIShardInfo mpi_shard = ivfpq_mpi_build_local_index(
        mpi_ivfpq,
        base,
        base_number,
        vecdim,
        1024,   // nlist：每个 rank 的本地 IVF 簇数
        16,     // M
        256,    // Ks
        10,     // PQ train iters
        10,     // IVF train iters
        MPI_COMM_WORLD,
        "files/ivfpq_mpi_index"
    );

    // IVFHNSWMPIIndex ivf_hnsw;

    // ivf_hnsw_mpi_load_or_build(
    //     ivf_hnsw,
    //     base,
    //     base_number,
    //     vecdim,
    //     64,   // nlist
    //     16,     // HNSW M
    //     50,    // ef_construction
    //     10      // IVF train iters
    // );

    // PartitionHNSWMPIIndex part_hnsw;

    // partition_hnsw_mpi_load_or_build(
    //     part_hnsw,
    //     base,
    //     base_number,
    //     vecdim,
    //     2,                                      // P: 分片数，建议 2~8
    //     16,                                     // HNSW M
    //     150,                                    // ef_construction
    //     PART_HNSW_PARTITION_RANDOM_BALANCED     // 分片方式
    // );

    // HNSWHNSWMPIIndex index;

    // hnsw_hnsw_mpi_load_or_build(
    //     index,
    //     base,
    //     base_number,
    //     vecdim,
    //     64,     // subgraph_num
    //     4,      // top_M
    //     32,     // top_ef_construction
    //     16,     // bottom_M
    //     50,    // bottom_ef_construction
    //     100     // random_seed
    // );

    // ivfpq_mpi_reset_rank_balance_stats();
    // ivfpq_thread_balance_reset(2);
    // ivfpq_mpi_merge_profile_reset();
    // ivfpq_mpi_comm_profile_reset();

    // 查询测试代码
    for(int i = 0; i < test_number; ++i) {
        const unsigned long Converter = 1000 * 1000;
        struct timeval val;
        int ret = gettimeofday(&val, NULL);

        // 该文件已有代码中你只能修改该函数的调用方式
        // 可以任意修改函数名，函数参数或者改为调用成员函数，但是不能修改函数返回值。
        // SIMD 实验
        // auto res = flat_search(base, test_query + i*vecdim, base_number, vecdim, k);
        // auto res = flat_search_simd(base, test_query + i*vecdim, base_number, vecdim, k);
        // auto res = sq_search(base, test_query + i*vecdim, base_number, vecdim, k, 50);
        // auto res = pq.pq_search(test_query + i*vecdim, k, 1000);
        // auto res = sq_search(base, base_q, test_query + i*vecdim, base_qsum, base_bias, base_number, vecdim, k, 50, min_val, step, inv_step);

        // 多线程实验
        // flat
        // auto res = flat_search_pthread_simd_8threads(base, test_query + i * vecdim, base_number, vecdim, k, 4);
        // auto res = flat_search_openmp_simd(base, test_query + i * vecdim, base_number, vecdim, k, 4, 8);

        // pq
        // auto res = pq_pthread.pq_search_pthread(test_query + i * vecdim, k, 1000);
        // auto res = pq_openmp.pq_search_openmp(test_query + i * vecdim, k, 1000);

        //ivf
        // auto res = ivf_search_simd(ivf, test_query + i * vecdim, k, 32);
        // auto res = ivf_search_simd_pthread_refine_cluster(ivf, test_query + i * vecdim, k, 10, 128, 8);
        // auto res = ivf_search_simd_pthread_refine_dynamic(ivf, test_query + i * vecdim, k, 10, 128, 8);
        // auto res = ivf_search_batch_query_pthread_cached(ivf, test_query, i, test_number, vecdim, k, 32, 8);
        // auto res = ivf_search_simd_openmp_refine_cluster(ivf, test_query + i * vecdim, k, 10, 128, 8);
        // auto res = ivf_search_batch_query_openmp_cached(ivf, test_query, i, test_number, vecdim, k, 32, 8);

        //ivfpq
        // auto res = ivfpq_search_simd(index, test_query + i * vecdim, k, 75, 128);

        // auto res = ivfpq_search_simd_pthread_refine_cluster(index, test_query + i * vecdim, k, 200, 128, 2); 效果不如 dynamic
        // auto res = ivfpq_search_simd_pthread_refine_dynamic(index, test_query + i * vecdim, k, 200, 128, 4);
        // auto res = ivfpq_search_batch_query_pthread_cached_optimized(index, test_query, i, test_number, vecdim, k, 200, 128, 2, 1);

        // auto res = ivfpq_search_simd_openmp_refine_cluster(index, test_query + i * vecdim, k, 200, 128, 6); 不用了
        // auto res = ivfpq_search_simd_openmp_refine_dynamic(index, test_query + i * vecdim, k, 200, 128, 6);
        // auto res = ivfpq_search_batch_query_openmp_cached(index, test_query, i, test_number, vecdim, k, 100, 32 ,8); 不用了
        // auto res = ivfpq_search_simd_openmp_centroid_cluster(index, test_query + i * vecdim, k, 200, 128, 4); 不用了

        // auto res = ivfpq_search_simd_pthread_centroid_cluster(index, test_query + i * vecdim, k, 200, 128, 8);
        // auto res = ivfpq_search_simd_pthread_balanced(index, test_query + i * vecdim, k, 200, 128, 6); 不用了

        // auto res = ivfpq_search_simd_openmp_optimized(index, test_query + i * vecdim, k, 100, 32, 8, 1);
        // auto res = ivfpq_search_simd_openmp_optimized_guided(index, test_query + i * vecdim, k, 200, 128, 2, 8); 不用了
        // auto res = ivfpq_search_batch_query_openmp_cached_optimized(index, test_query, i, test_number, vecdim, k, 200, 128 ,2, 1);

        // auto res = ivffirstpq_search_simd(ivffirst_pq, test_query + i * vecdim, k, 200, 128);

        // HNSW
        // auto res = hnsw_search_layer0(hnsw, test_query + i * vecdim, k, 64);
        // auto res = hnsw_search_layer0_multi_entry_pthread(hnsw, test_query + i * vecdim, k, 2, 8, 8);
        // auto res = hnsw_search_layer0_multi_entry_openmp(hnsw, test_query + i * vecdim, k, 2, 4, 4);
        // auto res = hnsw_search_hierarchical(hnsw, test_query + i * vecdim, k, 16); // 原始串行版本

        // MPI
        // MPI baseline
        // auto res = ivfpq_search_simd_mpi(
        //     mpi_ivfpq,
        //     mpi_shard,
        //     test_query + i * vecdim,
        //     k,
        //     50,    // top_p
        //     64,    // nprobe
        //     MPI_COMM_WORLD,
        //     0
        // );

        // MPI+Pthread+簇级并行+动态负载
        auto res = ivfpq_search_simd_mpi_pthread_cluster_dynamic(
            mpi_ivfpq,
            mpi_shard,
            mpi_rank == 0 ? test_query + i * vecdim : nullptr,
            k,
            50,                        // top_p
            64,                        // nprobe
            2,                         // thread_num
            MPI_COMM_WORLD,
            0
        );

        // MPI+OpenMP+簇级并行+动态负载
        // auto res = ivfpq_search_simd_mpi_openmp_cluster_dynamic(
        //     mpi_ivfpq,
        //     mpi_shard,
        //     mpi_rank == 0 ? test_query + i * vecdim : nullptr,
        //     k,
        //     50,
        //     64,
        //     2,
        //     MPI_COMM_WORLD,
        //     0
        // );

        // MPI+Pthread+rerank并行+静态负载
        // auto res = ivfpq_search_simd_mpi_pthread_rerank_static(
        //     mpi_ivfpq,
        //     mpi_shard,
        //     mpi_rank == 0 ? test_query + i * vecdim : nullptr,
        //     k,
        //     50,
        //     64,
        //     2,
        //     MPI_COMM_WORLD,
        //     0
        // );

        // MPI+OpenMP+rerank并行+静态负载
        // auto res = ivfpq_search_simd_mpi_openmp_rerank_static(
        //     mpi_ivfpq,
        //     mpi_shard,
        //     mpi_rank == 0 ? test_query + i * vecdim : nullptr,
        //     k,
        //     50,    // top_p
        //     64,    // nprobe
        //     2,     // thread_num
        //     MPI_COMM_WORLD,
        //     0
        // );

        // 探索 MPI rank 间负载均衡
        // auto res = ivfpq_search_simd_mpi_rank_balance_profile(
        //     mpi_ivfpq,
        //     mpi_shard,
        //     mpi_rank == 0 ? test_query + i * vecdim : nullptr,
        //     k,
        //     50,    // top_p
        //     64,    // nprobe
        //     MPI_COMM_WORLD,
        //     0
        // );

        // auto res = ivfpq_search_simd_mpi_pthread_cluster_static(
        //     mpi_ivfpq,
        //     mpi_shard,
        //     mpi_rank == 0 ? test_query + i * vecdim : nullptr,
        //     k,
        //     50,    // top_p
        //     64,    // nprobe
        //     4,     // thread_num
        //     MPI_COMM_WORLD,
        //     0
        // );

        // auto res = ivfpq_search_simd_mpi_openmp_cluster_static(
        //     mpi_ivfpq,
        //     mpi_shard,
        //     mpi_rank == 0 ? test_query + i * vecdim : nullptr,
        //     k,
        //     50,    // top_p
        //     64,    // nprobe
        //     2,     // thread_num
        //     MPI_COMM_WORLD,
        //     0
        // );

        // 探索 MPI Merge 开销
        // auto res = ivfpq_search_simd_mpi_pthread_cluster_dynamic_merge_profile(
        //     mpi_ivfpq,
        //     mpi_shard,
        //     mpi_rank == 0 ? test_query + i * vecdim : nullptr,
        //     k,
        //     50,    // top_p
        //     64,    // nprobe
        //     2,
        //     MPI_COMM_WORLD,
        //     0
        // );

        // 探索 two-gather 通信方案开销
        // auto res = ivfpq_search_simd_mpi_pthread_cluster_dynamic_two_gather_comm_profile(
        //     mpi_ivfpq,
        //     mpi_shard,
        //     mpi_rank == 0 ? test_query + i * vecdim : nullptr,
        //     k,
        //     50,
        //     64,
        //     1,
        //     MPI_COMM_WORLD,
        //     0
        // );
        
        // 探索 single-gather 通信方案开销
        // auto res = ivfpq_search_simd_mpi_pthread_cluster_dynamic_single_gather_comm_profile(
        //     mpi_ivfpq,
        //     mpi_shard,
        //     mpi_rank == 0 ? test_query + i * vecdim : nullptr,
        //     k,
        //     50,
        //     64,
        //     1,
        //     MPI_COMM_WORLD,
        //     0
        // );

        // HNSW MPI baseline
        // auto res = ivfhnsw_search_mpi(
        //     ivf_hnsw,
        //     test_query + i * vecdim,
        //     k,
        //     4,                             // nprobe
        //     8,                             // ef
        //     IVFHNSW_SEARCH_LAYER0_OPENMP,    // 簇内 HNSW 搜索模式
        //     2,                              // entry_count，仅 layer0 多入口模式使用
        //     2                             // thread_num，MPI 多进程时建议先设 1
        // );

        // Partition-HNSW MPI
        // auto res = partition_hnsw_search_mpi(
        //     part_hnsw,
        //     test_query + i * vecdim,
        //     k,
        //     32,                                // ef_search
        //     PART_HNSW_SEARCH_HIERARCHICAL,     // 分片内 HNSW 搜索方式
        //     1,                                 // entry_count
        //     1                                  // thread_num
        // );

        // HNSW on HNSW MPI
        // auto res = hnsw_hnsw_search_mpi(
        //     index,
        //     test_query + i * vecdim,
        //     k,
        //     32,      // upper_probe
        //     32,     // upper_ef_search
        //     64      // bottom_ef_search
        // );

        struct timeval newVal;
        ret = gettimeofday(&newVal, NULL);
        int64_t diff = (newVal.tv_sec * Converter + newVal.tv_usec) - (val.tv_sec * Converter + val.tv_usec);

        std::set<uint32_t> gtset;
        for(int j = 0; j < k; ++j){
            int t = test_gt[j + i*test_gt_d];
            gtset.insert(t);
        }

        size_t acc = 0;
        while (res.size()) {   
            int x = res.top().second;
            if(gtset.find(x) != gtset.end()){
                ++acc;
            }
            res.pop();
        }
        float recall = (float)acc/k;

        results[i] = {recall, diff};
    }

    // float avg_recall = 0, avg_latency = 0;
    // for(int i = 0; i < test_number; ++i) {
    //     avg_recall += results[i].recall;
    //     avg_latency += results[i].latency;
    // }

    // // 浮点误差可能导致一些精确算法平均recall不是1
    // std::cout << "average recall: "<<avg_recall / test_number<<"\n";
    // std::cout << "average latency (us): "<<avg_latency / test_number<<"\n";

    if (mpi_rank == 0) {
        float avg_recall = 0, avg_latency = 0;
        for(int i = 0; i < test_number; ++i) {
            avg_recall += results[i].recall;
            avg_latency += results[i].latency;
        }

        // 浮点误差可能导致一些精确算法平均recall不是1
        std::cout << "average recall: "<<avg_recall / test_number<<"\n";
        std::cout << "average latency (us): "<<avg_latency / test_number<<"\n";
    }

    // delete[] base_q;

    // ivfpq_mpi_dump_rank_balance_stats(
    //     mpi_shard,
    //     "",
    //     MPI_COMM_WORLD,
    //     0
    // );

    // ivfpq_thread_balance_dump(
    //     2,
    //     "",
    //     MPI_COMM_WORLD,
    //     0
    // );

    // ivfpq_mpi_merge_profile_dump(
    //     MPI_COMM_WORLD,
    //     0
    // );

    // ivfpq_mpi_comm_profile_dump(
    //     "single-gather",
    //     MPI_COMM_WORLD,
    //     0
    // );

    MPI_Finalize();

    return 0;
}
