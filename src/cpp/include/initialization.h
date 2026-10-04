#pragma once
#include <vector>
#include <cstdint>
#include <random>
#include <utility>
#include <cmath>
#include <unordered_set>
#include <string>
#include <map>
#include <fstream>
#include <algorithm>

struct SearchResult {
    uint32_t index;
    float distance;
};

enum class DistanceMetric { EUCLIDEAN, COSINE };

// Helper to get current Resident Set Size (RSS) in bytes
inline size_t get_current_rss_bytes() {
    std::ifstream file("/proc/self/status");
    std::string line;
    while (std::getline(file, line)) {
        if (line.substr(0, 6) == "VmRSS:") {
            size_t i = 7;
            while (i < line.size() && !std::isdigit(line[i])) i++;
            if (i < line.size()) {
                size_t value = std::stoul(line.substr(i));
                return value * 1024; // Convert kB to bytes
            }
        }
    }
    return 0;
}

// Helper to calculate vector of vectors memory
inline size_t calculate_dataset_memory(const std::vector<std::vector<float>>& dataset) {
    if (dataset.empty()) return 0;
    size_t total = dataset.size() * sizeof(std::vector<float>);
    total += dataset.size() * dataset[0].size() * sizeof(float);
    return total;
}

class InitializationApproach {
protected:
    mutable size_t distance_computations_ = 0;
    mutable size_t memory_usage_ = 0;
    mutable size_t index_size_ = 0;
    mutable size_t rss_baseline_ = 0;
    mutable size_t peak_rss_ = 0;
    std::vector<std::vector<float>> dataset_;
    DistanceMetric metric_ = DistanceMetric::EUCLIDEAN;

    void update_peak_rss() const {
        size_t current_rss = get_current_rss_bytes();
        if (current_rss > peak_rss_) {
            peak_rss_ = current_rss;
        }
    }

    float compute_l2_distance(const std::vector<float>& v1, const std::vector<float>& v2) const {
        distance_computations_++;
        float dist_sq = 0.0f;
        for (size_t i = 0; i < v1.size(); ++i) {
            float diff = v1[i] - v2[i];
            dist_sq += diff * diff;
        }
        return std::sqrt(dist_sq);
    }

    float compute_cosine_distance(const std::vector<float>& v1, const std::vector<float>& v2) const {
        distance_computations_++;
        float dot = 0.0f;
        for (size_t i = 0; i < v1.size(); ++i) {
            dot += v1[i] * v2[i];
        }
        float dist = 1.0f - dot;
        return (dist < 0.0f) ? 0.0f : dist;
    }

    float compute_distance(const std::vector<float>& v1, const std::vector<float>& v2) const {
        if (metric_ == DistanceMetric::COSINE) {
            return compute_cosine_distance(v1, v2);
        }
        return compute_l2_distance(v1, v2);
    }

public:
    virtual ~InitializationApproach() = default;
    
    virtual void build(const std::vector<std::vector<float>>& dataset) {
        if (rss_baseline_ == 0) rss_baseline_ = get_current_rss_bytes();
        dataset_ = dataset;
        update_peak_rss();
        if (metric_ == DistanceMetric::COSINE) {
            for (auto& vec : dataset_) {
                float sum_sq = 0.0f;
                for (float x : vec) sum_sq += x * x;
                if (sum_sq > 0.0f) {
                    float inv_norm = 1.0f / std::sqrt(sum_sq);
                    for (float& x : vec) x *= inv_norm;
                }
            }
        }
        build_index();
        update_peak_rss();
    }
    
    virtual void add_items(const std::vector<std::vector<float>>& items) {
        if (rss_baseline_ == 0) rss_baseline_ = get_current_rss_bytes();
        dataset_.insert(dataset_.end(), items.begin(), items.end());
        update_peak_rss();
    }
    
    virtual void build_index() = 0;
    
    virtual void set_query_time_params(const std::map<std::string, std::string>& params) {}
    
    virtual std::vector<SearchResult> search(const std::vector<float>& query, size_t k) = 0;

    virtual size_t get_memory_usage() const = 0;
    virtual size_t get_index_size() const = 0;

    size_t get_distance_computations() const { return distance_computations_; }
    void reset_distance_computations() { distance_computations_ = 0; }
};

class RandomPointsInit : public InitializationApproach {
private:
    std::mt19937 gen_;
    size_t sample_size_ = 0;
public:
    RandomPointsInit(uint32_t seed = 42, const std::string& metric = "l2");
    void build_index() override;
    void set_query_time_params(const std::map<std::string, std::string>& params) override;
    std::vector<SearchResult> search(const std::vector<float>& query, size_t k) override;
    size_t get_memory_usage() const override;
    size_t get_index_size() const override;
};

class MedoidInit : public InitializationApproach {
private:
    uint32_t medoid_index_;
public:
    MedoidInit(const std::string& metric = "l2");
    void build_index() override;
    std::vector<SearchResult> search(const std::vector<float>& query, size_t k) override;
    size_t get_memory_usage() const override;
    size_t get_index_size() const override;
};

struct FlannState;

class FlannKDTreeInit : public InitializationApproach {
private:
    int num_trees_;
    int checks_;
    FlannState* state_;

public:
    FlannKDTreeInit(int trees = 4, int checks = 32, const std::string& metric = "l2");
    ~FlannKDTreeInit() override;

    void build_index() override;
    void set_query_time_params(const std::map<std::string, std::string>& params) override;
    std::vector<SearchResult> search(const std::vector<float>& query, size_t k) override;
    size_t get_memory_usage() const override;
    size_t get_index_size() const override;
};

class FlannKMeansInit : public InitializationApproach {
private:
    int num_trees_;
    int branching_;
    int iterations_;
    int checks_;
    FlannState* state_;

public:
    FlannKMeansInit(int trees = 1, int branching = 32, int iterations = 11, int checks = 32, const std::string& metric = "l2");
    ~FlannKMeansInit() override;

    void build_index() override;
    void set_query_time_params(const std::map<std::string, std::string>& params) override;
    std::vector<SearchResult> search(const std::vector<float>& query, size_t k) override;
    size_t get_memory_usage() const override;
    size_t get_index_size() const override;
};

struct NmslibState;

class VPTreeInit : public InitializationApproach {
private:
    int max_leaves_to_visit_;
    float alpha_left_;
    float alpha_right_;
    NmslibState* state_;

public:
    VPTreeInit(int max_leaves_to_visit = 1000, float alpha_left = 1.0f, float alpha_right = 1.0f, const std::string& metric = "l2");
    ~VPTreeInit() override;

    void build_index() override;
    void set_query_time_params(const std::map<std::string, std::string>& params) override;
    std::vector<SearchResult> search(const std::vector<float>& query, size_t k) override;
    size_t get_memory_usage() const override;
    size_t get_index_size() const override;
};

class StackedNSWInit : public InitializationApproach {
private:
    int M_;
    int ef_construction_;
    int ef_;
    NmslibState* state_;

public:
    StackedNSWInit(int M = 16, int ef_construction = 200, int ef = 100, const std::string& metric = "l2");
    ~StackedNSWInit() override;

    void build_index() override;
    void set_query_time_params(const std::map<std::string, std::string>& params) override;
    std::vector<SearchResult> search(const std::vector<float>& query, size_t k) override;
    size_t get_memory_usage() const override;
    size_t get_index_size() const override;
};

struct LSHState;

class LSHInit : public InitializationApproach {
private:
    int num_hash_tables_;
    int num_hash_bits_;
    int num_probes_;
    LSHState* state_;

public:
    LSHInit(int num_hash_tables = 50, int num_hash_bits = 16, int num_probes = 100, const std::string& metric = "l2");
    ~LSHInit() override;

    void build_index() override;
    void set_query_time_params(const std::map<std::string, std::string>& params) override;
    std::vector<SearchResult> search(const std::vector<float>& query, size_t k) override;
    size_t get_memory_usage() const override;
    size_t get_index_size() const override;
};

namespace hvs { class HVSIndex; }

class HVSInit : public InitializationApproach {
private:
    int levels_;
    float delta_;
    int ef_search_;
    hvs::HVSIndex* index_ = nullptr;

public:
    HVSInit(int levels = 1, float delta = 0.5f, int ef_search = 1000, const std::string& metric = "l2");
    ~HVSInit() override;

    void build_index() override;
    void set_query_time_params(const std::map<std::string, std::string>& params) override;
    std::vector<SearchResult> search(const std::vector<float>& query, size_t k) override;
    size_t get_memory_usage() const override;
    size_t get_index_size() const override;
};

namespace lsb { class LSBTree; }

class LSBTreeInit : public InitializationApproach {
private:
    int L_;
    int K_;
    float W_;
    uint32_t max_candidates_ = 0;
    std::vector<float> flat_dataset_;
    lsb::LSBTree* tree_ = nullptr;

public:
    LSBTreeInit(int L = 10, int K = 10, float W = 1.0f, const std::string& metric = "l2");
    ~LSBTreeInit() override;

    void build_index() override;
    void set_query_time_params(const std::map<std::string, std::string>& params) override;
    std::vector<SearchResult> search(const std::vector<float>& query, size_t k) override;
    size_t get_memory_usage() const override;
    size_t get_index_size() const override;
};


