#pragma once
#include <vector>
#include <cstdint>
#include <random>
#include <utility>
#include <cmath>
#include <unordered_set>

struct SearchResult {
    uint32_t index;
    float distance;
};

enum class DistanceMetric { EUCLIDEAN, COSINE };

class InitializationApproach {
protected:
    mutable size_t distance_computations_ = 0;
    mutable size_t memory_usage_ = 0;
    mutable size_t index_size_ = 0;
    std::vector<std::vector<float>> dataset_;
    DistanceMetric metric_ = DistanceMetric::EUCLIDEAN;

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
        float norm1 = 0.0f;
        float norm2 = 0.0f;
        for (size_t i = 0; i < v1.size(); ++i) {
            dot += v1[i] * v2[i];
            norm1 += v1[i] * v1[i];
            norm2 += v2[i] * v2[i];
        }
        if (norm1 == 0.0f || norm2 == 0.0f) return 1.0f;
        return 1.0f - (dot / (std::sqrt(norm1) * std::sqrt(norm2)));
    }

    float compute_distance(const std::vector<float>& v1, const std::vector<float>& v2) const {
        if (metric_ == DistanceMetric::COSINE) {
            return compute_cosine_distance(v1, v2);
        }
        return compute_l2_distance(v1, v2);
    }

public:
    virtual ~InitializationApproach() = default;
    
    // Build offline index / initialize required components
    virtual void build(const std::vector<std::vector<float>>& dataset) = 0;
    
    // Search online, returning a sorted list of (index, distance)
    virtual std::vector<SearchResult> search(const std::vector<float>& query, size_t k) = 0;

    virtual size_t get_memory_usage() const = 0;
    virtual size_t get_index_size() const = 0;

    size_t get_distance_computations() const { return distance_computations_; }
    void reset_distance_computations() { distance_computations_ = 0; }
};

class RandomPointsInit : public InitializationApproach {
private:
    std::mt19937 gen_;
public:
    RandomPointsInit(uint32_t seed = 42, const std::string& metric = "l2");
    void build(const std::vector<std::vector<float>>& dataset) override;
    std::vector<SearchResult> search(const std::vector<float>& query, size_t k) override;
    size_t get_memory_usage() const override;
    size_t get_index_size() const override;
};

class MedoidInit : public InitializationApproach {
private:
    uint32_t medoid_index_;
public:
    MedoidInit(const std::string& metric = "l2");
    void build(const std::vector<std::vector<float>>& dataset) override;
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

    void build(const std::vector<std::vector<float>>& dataset) override;
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

    void build(const std::vector<std::vector<float>>& dataset) override;
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

    void build(const std::vector<std::vector<float>>& dataset) override;
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

    void build(const std::vector<std::vector<float>>& dataset) override;
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

    void build(const std::vector<std::vector<float>>& dataset) override;
    std::vector<SearchResult> search(const std::vector<float>& query, size_t k) override;
    size_t get_memory_usage() const override;
    size_t get_index_size() const override;
};
