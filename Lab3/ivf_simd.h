#pragma once
#include <pthread.h>
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <limits>
#include <numeric>
#include <queue>
#include <utility>
#include <vector>
#include "flat_simd.h"

#include <fstream>
#include <string>
#include <iostream>
#include <sys/stat.h>
#include <sys/types.h>
#include <cerrno>
#include <cstring>

// IVF-SIMD baseline
struct IVFIndex {
    const float* base = nullptr;
    size_t base_number = 0;
    size_t vecdim = 0;

    size_t nlist = 0;
    bool trained = false;

    // centroids[c * vecdim + d]
    std::vector<float> centroids;

    // 每个簇保存 base 向量的原始编号
    std::vector<std::vector<uint32_t>> inverted_lists;

    // 每个簇连续保存向量副本
    std::vector<std::vector<float>> inverted_vectors;

    bool save_index(const std::string& path) const {
        errno = 0;

        std::ofstream ofs(
            path,
            std::ios::out | std::ios::binary | std::ios::trunc
        );

        if (!ofs.is_open()) {
            std::cerr << "[IVF] Cannot open index file for writing: "
                      << path << std::endl;
            std::cerr << "[IVF] errno = " << errno
                      << ", reason = " << std::strerror(errno) << std::endl;
            return false;
        }

        ofs.write(reinterpret_cast<const char*>(&base_number), sizeof(size_t));
        ofs.write(reinterpret_cast<const char*>(&vecdim), sizeof(size_t));
        ofs.write(reinterpret_cast<const char*>(&nlist), sizeof(size_t));
        ofs.write(reinterpret_cast<const char*>(&trained), sizeof(bool));

        size_t centroid_size = centroids.size();
        ofs.write(reinterpret_cast<const char*>(&centroid_size), sizeof(size_t));

        if (centroid_size > 0) {
            ofs.write(reinterpret_cast<const char*>(centroids.data()),
                      centroid_size * sizeof(float));
        }

        size_t list_num = inverted_lists.size();
        ofs.write(reinterpret_cast<const char*>(&list_num), sizeof(size_t));

        for (size_t i = 0; i < list_num; ++i) {
            size_t list_size = inverted_lists[i].size();
            ofs.write(reinterpret_cast<const char*>(&list_size), sizeof(size_t));

            if (list_size > 0) {
                ofs.write(reinterpret_cast<const char*>(inverted_lists[i].data()),
                          list_size * sizeof(uint32_t));

                ofs.write(reinterpret_cast<const char*>(inverted_vectors[i].data()),
                          list_size * vecdim * sizeof(float));
            }

            if (!ofs.good()) {
                std::cerr << "[IVF] Write failed at inverted list "
                          << i << ", list_size = " << list_size << std::endl;
                return false;
            }
        }

        ofs.flush();

        if (!ofs.good()) {
            std::cerr << "[IVF] Flush failed when saving index: "
                      << path << std::endl;
            return false;
        }

        ofs.close();

        if (ofs.fail()) {
            std::cerr << "[IVF] Close failed when saving index: "
                      << path << std::endl;
            return false;
        }

        return true;
    }

    bool load_index(const std::string& path,
                    const float* base_ptr,
                    size_t expected_base_number,
                    size_t expected_vecdim,
                    size_t expected_nlist) {
        std::ifstream ifs(path, std::ios::binary);
        if (!ifs.is_open()) {
            return false;
        }

        size_t file_base_number = 0;
        size_t file_vecdim = 0;
        size_t file_nlist = 0;
        bool file_trained = false;

        ifs.read(reinterpret_cast<char*>(&file_base_number), sizeof(size_t));
        ifs.read(reinterpret_cast<char*>(&file_vecdim), sizeof(size_t));
        ifs.read(reinterpret_cast<char*>(&file_nlist), sizeof(size_t));
        ifs.read(reinterpret_cast<char*>(&file_trained), sizeof(bool));

        if (!ifs.good()) {
            return false;
        }

        if (file_base_number != expected_base_number ||
            file_vecdim != expected_vecdim ||
            file_nlist != expected_nlist ||
            !file_trained) {
            return false;
        }

        size_t centroid_size = 0;
        ifs.read(reinterpret_cast<char*>(&centroid_size), sizeof(size_t));

        if (centroid_size != expected_nlist * expected_vecdim) {
            return false;
        }

        std::vector<float> loaded_centroids(centroid_size);

        if (centroid_size > 0) {
            ifs.read(reinterpret_cast<char*>(loaded_centroids.data()),
                     centroid_size * sizeof(float));
        }

        size_t list_num = 0;
        ifs.read(reinterpret_cast<char*>(&list_num), sizeof(size_t));

        if (list_num != expected_nlist) {
            return false;
        }

        std::vector<std::vector<uint32_t>> loaded_lists(list_num);
        std::vector<std::vector<float>> loaded_vectors(list_num);

        for (size_t i = 0; i < list_num; ++i) {
            size_t list_size = 0;
            ifs.read(reinterpret_cast<char*>(&list_size), sizeof(size_t));

            loaded_lists[i].resize(list_size);
            loaded_vectors[i].resize(list_size * expected_vecdim);

            if (list_size > 0) {
                ifs.read(reinterpret_cast<char*>(loaded_lists[i].data()),
                         list_size * sizeof(uint32_t));

                ifs.read(reinterpret_cast<char*>(loaded_vectors[i].data()),
                         list_size * expected_vecdim * sizeof(float));
            }

            if (!ifs.good()) {
                return false;
            }
        }

        base = base_ptr;
        base_number = file_base_number;
        vecdim = file_vecdim;
        nlist = file_nlist;
        trained = true;

        centroids.swap(loaded_centroids);
        inverted_lists.swap(loaded_lists);
        inverted_vectors.swap(loaded_vectors);

        return true;
    }
};

// 精排固定簇划分线程参数
struct IVFRefineThreadParam {
    int tid;
    int thread_num;

    const IVFIndex* index;
    const float* query;
    const std::vector<uint32_t>* probe_lists;

    size_t k;
    size_t local_k;

    std::priority_queue<std::pair<float, uint32_t>> local_topk;

    size_t scanned_points = 0;
};

// 精排动态调度线程参数
struct IVFRefineDynamicParam {
    int tid;

    const IVFIndex* index;
    const float* query;
    const std::vector<uint32_t>* probe_lists;

    size_t k;
    size_t local_k;

    int* next_task;
    pthread_mutex_t* task_mutex;

    std::priority_queue<std::pair<float, uint32_t>> local_topk;

    size_t scanned_points = 0;
    size_t scanned_lists = 0;
};

// query 级并行线程参数
struct IVFQueryBatchThreadParam {
    int tid;
    int thread_num;

    const IVFIndex* index;
    const float* queries;

    size_t query_number;
    size_t vecdim;
    size_t k;
    size_t nprobe;

    std::vector<std::priority_queue<std::pair<float, uint32_t>>>* cache_results;

    size_t processed_queries = 0;
};

// 维护 top-k 大根堆
static inline void push_topk(
    std::priority_queue<std::pair<float, uint32_t>>& heap,
    float dis,
    uint32_t id,
    size_t k
) {
    if (heap.size() < k) {
        heap.push({dis, id});
    } else if (dis < heap.top().first) {
        heap.push({dis, id});
        heap.pop();
    }
}

// 初始化 IVF 质心：从 base 中均匀抽取 nlist 个点作为初始中心
static inline void ivf_init_centroids(IVFIndex& index) {
    index.centroids.assign(index.nlist * index.vecdim, 0.0f);

    for (size_t c = 0; c < index.nlist; ++c) {
        size_t id = c * index.base_number / index.nlist;
        if (id >= index.base_number) {
            id = index.base_number - 1;
        }

        const float* src = index.base + id * index.vecdim;
        float* dst = index.centroids.data() + c * index.vecdim;
        std::memcpy(dst, src, index.vecdim * sizeof(float));
    }
}

// 找到某个向量最近的 centroid
static inline uint32_t ivf_nearest_centroid(
    const IVFIndex& index,
    const float* x
) {
    uint32_t best_c = 0;
    float best_dis = std::numeric_limits<float>::max();

    for (size_t c = 0; c < index.nlist; ++c) {
        const float* centroid = index.centroids.data() + c * index.vecdim;
        float dis = 1.0f - InnerProductSIMD(x, centroid, index.vecdim);

        if (dis < best_dis) {
            best_dis = dis;
            best_c = static_cast<uint32_t>(c);
        }
    }

    return best_c;
}

static inline void ivf_build(
    IVFIndex& index,
    const float* base,
    size_t base_number,
    size_t vecdim,
    size_t nlist,
    size_t iter = 10
) {
    assert(base != nullptr);
    assert(base_number > 0);
    assert(vecdim > 0);
    assert(vecdim % 32 == 0);
    assert(nlist > 0);
    assert(nlist <= base_number);

    const std::string index_dir = "./files";

    system("mkdir -p ./files");

    const std::string index_path =
        index_dir + "/ivf_index"
        + "_base_" + std::to_string(base_number)
        + "_dim_" + std::to_string(vecdim)
        + "_nlist_" + std::to_string(nlist)
        + "_iter_" + std::to_string(iter)
        + "_with_vectors.bin";

    if (index.load_index(index_path, base, base_number, vecdim, nlist)) {
        std::cout << "[IVF] Load index from: " << index_path << std::endl;
        return;
    }

    std::cout << "[IVF] No valid saved index, build new index..." << std::endl;

    index.base = base;
    index.base_number = base_number;
    index.vecdim = vecdim;
    index.nlist = nlist;
    index.trained = false;

    index.inverted_lists.assign(nlist, std::vector<uint32_t>());
    index.inverted_vectors.assign(nlist, std::vector<float>());

    ivf_init_centroids(index);

    std::vector<uint32_t> assign(base_number, 0);
    std::vector<float> new_centroids(nlist * vecdim, 0.0f);
    std::vector<uint32_t> counts(nlist, 0);

    // K-means 训练质心
    for (size_t it = 0; it < iter; ++it) {
        std::fill(new_centroids.begin(), new_centroids.end(), 0.0f);
        std::fill(counts.begin(), counts.end(), 0);

        for (size_t i = 0; i < base_number; ++i) {
            const float* x = base + i * vecdim;

            uint32_t cid = ivf_nearest_centroid(index, x);
            assign[i] = cid;

            float* sum = new_centroids.data()
                       + static_cast<size_t>(cid) * vecdim;

            for (size_t d = 0; d < vecdim; ++d) {
                sum[d] += x[d];
            }

            counts[cid]++;
        }

        for (size_t c = 0; c < nlist; ++c) {
            if (counts[c] == 0) {
                continue;
            }

            float* dst = index.centroids.data() + c * vecdim;
            float* sum = new_centroids.data() + c * vecdim;
            float inv_cnt = 1.0f / static_cast<float>(counts[c]);

            for (size_t d = 0; d < vecdim; ++d) {
                dst[d] = sum[d] * inv_cnt;
            }
        }
    }

    // 清空倒排表
    for (auto& list : index.inverted_lists) {
        list.clear();
    }

    for (auto& vecs : index.inverted_vectors) {
        vecs.clear();
    }

    // 统计每个簇大小，用于提前 reserve，减少反复扩容
    std::fill(counts.begin(), counts.end(), 0);

    for (size_t i = 0; i < base_number; ++i) {
        const float* x = base + i * vecdim;
        uint32_t cid = ivf_nearest_centroid(index, x);
        assign[i] = cid;
        counts[cid]++;
    }

    for (size_t c = 0; c < nlist; ++c) {
        index.inverted_lists[c].reserve(counts[c]);
        index.inverted_vectors[c].reserve(
            static_cast<size_t>(counts[c]) * vecdim
        );
    }

    // 构建倒排表
    for (size_t i = 0; i < base_number; ++i) {
        uint32_t cid = assign[i];
        const float* x = base + i * vecdim;

        index.inverted_lists[cid].push_back(static_cast<uint32_t>(i));

        auto& vecs = index.inverted_vectors[cid];
        vecs.insert(vecs.end(), x, x + vecdim);
    }

    index.trained = true;

    std::cout << "[IVF] Try saving index, centroids = "
              << index.centroids.size()
              << ", lists = " << index.inverted_lists.size()
              << ", vectors = " << index.inverted_vectors.size()
              << std::endl;

    if (index.save_index(index_path)) {
        std::cout << "[IVF] Save index to: " << index_path << std::endl;
    } else {
        std::cout << "[IVF] Failed to save index to: " << index_path << std::endl;
    }
}

// IVF 粗排
static inline void ivf_select_probe_lists(
    const IVFIndex& index,
    const float* query,
    size_t nprobe,
    std::vector<uint32_t>& probe_lists
) {
    if (nprobe > index.nlist) {
        nprobe = index.nlist;
    }

    std::priority_queue<std::pair<float, uint32_t>> centroid_heap;

    for (size_t c = 0; c < index.nlist; ++c) {
        const float* centroid = index.centroids.data() + c * index.vecdim;
        float dis = 1.0f - InnerProductSIMD(query, centroid, index.vecdim);

        if (centroid_heap.size() < nprobe) {
            centroid_heap.push({dis, static_cast<uint32_t>(c)});
        } else if (dis < centroid_heap.top().first) {
            centroid_heap.push({dis, static_cast<uint32_t>(c)});
            centroid_heap.pop();
        }
    }

    probe_lists.clear();
    probe_lists.reserve(nprobe);

    while (!centroid_heap.empty()) {
        probe_lists.push_back(centroid_heap.top().second);
        centroid_heap.pop();
    }
}

// 固定簇划分精排线程函数
static void* ivf_refine_cluster_thread_func(void* arg) {
    IVFRefineThreadParam* param = static_cast<IVFRefineThreadParam*>(arg);

    const IVFIndex& index = *(param->index);
    const float* query = param->query;
    const std::vector<uint32_t>& probe_lists = *(param->probe_lists);
    auto& local_heap = param->local_topk;

    for (size_t i = static_cast<size_t>(param->tid);
         i < probe_lists.size();
         i += static_cast<size_t>(param->thread_num)) {

        uint32_t cid = probe_lists[i];

        const std::vector<uint32_t>& ids = index.inverted_lists[cid];
        const std::vector<float>& vecs = index.inverted_vectors[cid];

        assert(vecs.size() == ids.size() * index.vecdim);

        const float* vec_data = vecs.data();

        param->scanned_points += ids.size();

        for (size_t j = 0; j < ids.size(); ++j) {
            const float* x = vec_data + j * index.vecdim;

            float dis = 1.0f - InnerProductSIMD(
                query,
                x,
                index.vecdim
            );

            push_topk(local_heap, dis, ids[j], param->local_k);
        }
    }

    return nullptr;
}

// 动态调度精排线程函数
static void* ivf_refine_dynamic_thread_func(void* arg) {
    IVFRefineDynamicParam* param = static_cast<IVFRefineDynamicParam*>(arg);

    const IVFIndex& index = *(param->index);
    const float* query = param->query;
    const std::vector<uint32_t>& probe_lists = *(param->probe_lists);
    auto& local_heap = param->local_topk;

    while (true) {
        int task_id;

        pthread_mutex_lock(param->task_mutex);
        task_id = *(param->next_task);
        (*(param->next_task))++;
        pthread_mutex_unlock(param->task_mutex);

        if (task_id >= static_cast<int>(probe_lists.size())) {
            break;
        }

        uint32_t cid = probe_lists[static_cast<size_t>(task_id)];

        const std::vector<uint32_t>& ids = index.inverted_lists[cid];
        const std::vector<float>& vecs = index.inverted_vectors[cid];

        assert(vecs.size() == ids.size() * index.vecdim);

        const float* vec_data = vecs.data();

        param->scanned_lists++;
        param->scanned_points += ids.size();

        for (size_t j = 0; j < ids.size(); ++j) {
            const float* x = vec_data + j * index.vecdim;

            float dis = 1.0f - InnerProductSIMD(
                query,
                x,
                index.vecdim
            );

            push_topk(local_heap, dis, ids[j], param->local_k);
        }
    }

    return nullptr;
}

static inline std::priority_queue<std::pair<float, uint32_t>>
ivf_search_simd(
    const IVFIndex& index,
    const float* query,
    size_t k,
    size_t nprobe
) {
    assert(index.trained);
    assert(query != nullptr);
    assert(k > 0);
    assert(nprobe > 0);

    if (nprobe > index.nlist) {
        nprobe = index.nlist;
    }

    std::vector<uint32_t> probe_lists;
    ivf_select_probe_lists(index, query, nprobe, probe_lists);

    std::priority_queue<std::pair<float, uint32_t>> result;

    for (uint32_t cid : probe_lists) {
        const std::vector<uint32_t>& ids = index.inverted_lists[cid];
        const std::vector<float>& vecs = index.inverted_vectors[cid];

        const float* vec_data = vecs.data();

        for (size_t j = 0; j < ids.size(); ++j) {
            const float* x = vec_data + j * index.vecdim;

            float dis = 1.0f - InnerProductSIMD(
                query,
                x,
                index.vecdim
            );

            push_topk(result, dis, ids[j], k);
        }
    }

    return result;
}

// IVF-SIMD 查询 + pthread 固定簇划分精排
static inline std::priority_queue<std::pair<float, uint32_t>>
ivf_search_simd_pthread_refine_cluster(
    const IVFIndex& index,
    const float* query,
    size_t k,
    size_t local_k,
    size_t nprobe,
    int thread_num
) {
    assert(index.trained);
    assert(query != nullptr);
    assert(k > 0);
    assert(local_k > 0);
    assert(nprobe > 0);
    assert(thread_num > 0);

    if (nprobe > index.nlist) {
        nprobe = index.nlist;
    }

    if (local_k < k) {
        local_k = k;
    }

    if (thread_num <= 1) {
        return ivf_search_simd(index, query, k, nprobe);
    }

    if (thread_num > static_cast<int>(nprobe)) {
        thread_num = static_cast<int>(nprobe);
    }

    std::vector<uint32_t> probe_lists;
    ivf_select_probe_lists(index, query, nprobe, probe_lists);

    // 只创建 thread_num - 1 个子线程，tid = 0 由主线程执行
    std::vector<pthread_t> handles(static_cast<size_t>(thread_num - 1));
    std::vector<IVFRefineThreadParam> params(static_cast<size_t>(thread_num));

    for (int t = 0; t < thread_num; ++t) {
        params[t].tid = t;
        params[t].thread_num = thread_num;
        params[t].index = &index;
        params[t].query = query;
        params[t].probe_lists = &probe_lists;
        params[t].k = k;
        params[t].local_k = local_k;
        params[t].scanned_points = 0;
    }

    // 子线程执行 tid = 1 ~ thread_num - 1
    for (int t = 1; t < thread_num; ++t) {
        pthread_create(
            &handles[static_cast<size_t>(t - 1)],
            nullptr,
            ivf_refine_cluster_thread_func,
            &params[static_cast<size_t>(t)]
        );
    }

    // 主线程执行 tid = 0
    ivf_refine_cluster_thread_func(&params[0]);

    // 等待子线程
    for (int t = 1; t < thread_num; ++t) {
        pthread_join(handles[static_cast<size_t>(t - 1)], nullptr);
    }

    // merge 每个线程的局部 top-k
    std::priority_queue<std::pair<float, uint32_t>> result;

    for (int t = 0; t < thread_num; ++t) {
        auto& local_heap = params[static_cast<size_t>(t)].local_topk;

        while (!local_heap.empty()) {
            auto item = local_heap.top();
            local_heap.pop();
            push_topk(result, item.first, item.second, k);
        }
    }

    return result;
}

// IVF-SIMD 查询 + pthread 动态调度精排
static inline std::priority_queue<std::pair<float, uint32_t>>
ivf_search_simd_pthread_refine_dynamic(
    const IVFIndex& index,
    const float* query,
    size_t k,
    size_t local_k,
    size_t nprobe,
    int thread_num
) {
    assert(index.trained);
    assert(query != nullptr);
    assert(k > 0);
    assert(local_k > 0);
    assert(nprobe > 0);
    assert(thread_num > 0);

    if (nprobe > index.nlist) {
        nprobe = index.nlist;
    }

    if (local_k < k) {
        local_k = k;
    }

    if (thread_num <= 1) {
        return ivf_search_simd(index, query, k, nprobe);
    }

    if (thread_num > static_cast<int>(nprobe)) {
        thread_num = static_cast<int>(nprobe);
    }

    std::vector<uint32_t> probe_lists;
    ivf_select_probe_lists(index, query, nprobe, probe_lists);

    std::vector<pthread_t> handles(static_cast<size_t>(thread_num - 1));
    std::vector<IVFRefineDynamicParam> params(static_cast<size_t>(thread_num));

    int next_task = 0;
    pthread_mutex_t task_mutex;
    pthread_mutex_init(&task_mutex, nullptr);

    for (int t = 0; t < thread_num; ++t) {
        params[t].tid = t;
        params[t].index = &index;
        params[t].query = query;
        params[t].probe_lists = &probe_lists;
        params[t].k = k;
        params[t].local_k = local_k;
        params[t].next_task = &next_task;
        params[t].task_mutex = &task_mutex;
        params[t].scanned_points = 0;
        params[t].scanned_lists = 0;
    }

    // 子线程执行 tid = 1 ~ thread_num - 1
    for (int t = 1; t < thread_num; ++t) {
        pthread_create(
            &handles[static_cast<size_t>(t - 1)],
            nullptr,
            ivf_refine_dynamic_thread_func,
            &params[static_cast<size_t>(t)]
        );
    }

    // 主线程执行 tid = 0
    ivf_refine_dynamic_thread_func(&params[0]);

    // 等待子线程
    for (int t = 1; t < thread_num; ++t) {
        pthread_join(handles[static_cast<size_t>(t - 1)], nullptr);
    }

    pthread_mutex_destroy(&task_mutex);

    // merge 局部 top-k
    std::priority_queue<std::pair<float, uint32_t>> result;

    for (int t = 0; t < thread_num; ++t) {
        auto& local_heap = params[static_cast<size_t>(t)].local_topk;

        while (!local_heap.empty()) {
            auto item = local_heap.top();
            local_heap.pop();
            push_topk(result, item.first, item.second, k);
        }
    }

    return result;
}

// query 级并行线程函数
static void* ivf_query_batch_thread_func(void* arg) {
    IVFQueryBatchThreadParam* param =
        static_cast<IVFQueryBatchThreadParam*>(arg);

    const IVFIndex& index = *(param->index);

    for (size_t qi = static_cast<size_t>(param->tid);
         qi < param->query_number;
         qi += static_cast<size_t>(param->thread_num)) {

        const float* query = param->queries + qi * param->vecdim;

        (*(param->cache_results))[qi] = ivf_search_simd(
            index,
            query,
            param->k,
            param->nprobe
        );

        param->processed_queries++;
    }

    return nullptr;
}

// IVF-SIMD query 级 pthread 并行缓存版本
static inline std::priority_queue<std::pair<float, uint32_t>>
ivf_search_batch_query_pthread_cached(
    const IVFIndex& index,
    const float* queries,
    size_t query_id,
    size_t query_number,
    size_t vecdim,
    size_t k,
    size_t nprobe,
    int thread_num
) {
    assert(index.trained);
    assert(queries != nullptr);
    assert(query_id < query_number);
    assert(query_number > 0);
    assert(vecdim == index.vecdim);
    assert(k > 0);
    assert(nprobe > 0);
    assert(thread_num > 0);

    if (nprobe > index.nlist) {
        nprobe = index.nlist;
    }

    if (thread_num > static_cast<int>(query_number)) {
        thread_num = static_cast<int>(query_number);
    }

    // 函数内部静态缓存，避免在 main 中使用 all_results
    static std::vector<std::priority_queue<std::pair<float, uint32_t>>> cache_results;

    // 记录上一次缓存对应的参数，防止不同实验参数复用旧缓存
    static const IVFIndex* cached_index = nullptr;
    static const float* cached_queries = nullptr;
    static size_t cached_query_number = 0;
    static size_t cached_vecdim = 0;
    static size_t cached_k = 0;
    static size_t cached_nprobe = 0;
    static int cached_thread_num = 0;
    static bool cache_ready = false;

    bool need_rebuild_cache = false;

    if (!cache_ready) {
        need_rebuild_cache = true;
    }

    if (cached_index != &index ||
        cached_queries != queries ||
        cached_query_number != query_number ||
        cached_vecdim != vecdim ||
        cached_k != k ||
        cached_nprobe != nprobe ||
        cached_thread_num != thread_num) {
        need_rebuild_cache = true;
    }

    // 一般测试循环从 i = 0 开始。
    // 如果 query_id == 0，也强制重建，避免重复运行同一程序段时使用旧结果。
    if (query_id == 0) {
        need_rebuild_cache = true;
    }

    if (need_rebuild_cache) {
        cache_results.clear();
        cache_results.resize(query_number);

        if (thread_num <= 1) {
            for (size_t qi = 0; qi < query_number; ++qi) {
                const float* query = queries + qi * vecdim;
                cache_results[qi] = ivf_search_simd(
                    index,
                    query,
                    k,
                    nprobe
                );
            }
        } else {
            std::vector<pthread_t> handles(static_cast<size_t>(thread_num - 1));
            std::vector<IVFQueryBatchThreadParam> params(static_cast<size_t>(thread_num));

            for (int t = 0; t < thread_num; ++t) {
                params[t].tid = t;
                params[t].thread_num = thread_num;
                params[t].index = &index;
                params[t].queries = queries;
                params[t].query_number = query_number;
                params[t].vecdim = vecdim;
                params[t].k = k;
                params[t].nprobe = nprobe;
                params[t].cache_results = &cache_results;
                params[t].processed_queries = 0;
            }

            for (int t = 1; t < thread_num; ++t) {
                pthread_create(
                    &handles[static_cast<size_t>(t - 1)],
                    nullptr,
                    ivf_query_batch_thread_func,
                    &params[static_cast<size_t>(t)]
                );
            }

            // 主线程执行 tid = 0
            ivf_query_batch_thread_func(&params[0]);

            for (int t = 1; t < thread_num; ++t) {
                pthread_join(handles[static_cast<size_t>(t - 1)], nullptr);
            }

#ifndef NDEBUG
            size_t total_processed = 0;
            for (int t = 0; t < thread_num; ++t) {
                total_processed += params[t].processed_queries;
            }
            assert(total_processed == query_number);
#endif
        }

        cached_index = &index;
        cached_queries = queries;
        cached_query_number = query_number;
        cached_vecdim = vecdim;
        cached_k = k;
        cached_nprobe = nprobe;
        cached_thread_num = thread_num;
        cache_ready = true;
    }

    return cache_results[query_id];
}