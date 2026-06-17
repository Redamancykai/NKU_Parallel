float ComputeRecallAtK(std::priority_queue<std::pair<float, uint32_t>> result,
                       const int* gt,
                       size_t gt_stride,
                       size_t query_id,
                       size_t k) {
    std::set<uint32_t> gtset;
    for (size_t j = 0; j < k; ++j) {
        gtset.insert(static_cast<uint32_t>(gt[query_id * gt_stride + j]));
    }

    size_t hit = 0;
    while (!result.empty()) {
        uint32_t id = result.top().second;
        if (gtset.find(id) != gtset.end()) {
            ++hit;
        }
        result.pop();
    }
    return static_cast<float>(hit) / static_cast<float>(k);
}
