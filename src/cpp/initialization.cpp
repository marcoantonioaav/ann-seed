#include "initialization.h"
#include <numeric>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <limits>
#include <fstream>
#include <string>
#include <unordered_set>

// Helper to get current Resident Set Size (RSS) in bytes
size_t get_current_rss_bytes() {
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
size_t calculate_dataset_memory(const std::vector<std::vector<float>>& dataset) {
    if (dataset.empty()) return 0;
    size_t total = dataset.size() * sizeof(std::vector<float>);
    total += dataset.size() * dataset[0].size() * sizeof(float);
    return total;
}

RandomPointsInit::RandomPointsInit(uint32_t seed, const std::string& metric) : gen_(seed) {
    if (metric == "cosine") metric_ = DistanceMetric::COSINE;
}

void RandomPointsInit::build(const std::vector<std::vector<float>>& dataset) {
    dataset_ = dataset;
}

size_t RandomPointsInit::get_memory_usage() const {
    return calculate_dataset_memory(dataset_);
}

size_t RandomPointsInit::get_index_size() const {
    return 0; // Random points approach has no persistent index structure
}

std::vector<SearchResult> RandomPointsInit::search(const std::vector<float>& query, size_t k) {
    size_t num_points = dataset_.size();
    if (k > num_points) {
        throw std::invalid_argument("k cannot be larger than the dataset size.");
    }
    
    // 1. Sample k random indices efficiently (O(k) expected time)
    std::unordered_set<uint32_t> sampled_indices;
    std::uniform_int_distribution<uint32_t> dist_gen(0, num_points - 1);
    while (sampled_indices.size() < k) {
        sampled_indices.insert(dist_gen(gen_));
    }
    
    // 2. Compute distances
    std::vector<SearchResult> results;
    results.reserve(k);
    for (uint32_t idx : sampled_indices) {
        float dist = compute_distance(dataset_[idx], query);
        results.push_back({idx, dist});
    }
    
    // 3. Sort by distance
    std::sort(results.begin(), results.end(), [](const SearchResult& a, const SearchResult& b) {
        return a.distance < b.distance;
    });
    
    return results;
}

MedoidInit::MedoidInit(const std::string& metric) : medoid_index_(0) {
    if (metric == "cosine") metric_ = DistanceMetric::COSINE;
}

void MedoidInit::build(const std::vector<std::vector<float>>& dataset) {
    dataset_ = dataset;
    if (dataset_.empty()) {
        throw std::invalid_argument("Dataset cannot be empty.");
    }
    size_t num_points = dataset_.size();
    size_t dims = dataset_[0].size();
    
    // Calculate centroid
    std::vector<float> centroid(dims, 0.0f);
    for (const auto& point : dataset_) {
        for (size_t i = 0; i < dims; ++i) {
            centroid[i] += point[i];
        }
    }
    for (size_t i = 0; i < dims; ++i) {
        centroid[i] /= num_points;
    }
    
    // Find point closest to centroid
    float min_dist_sq = std::numeric_limits<float>::max();
    medoid_index_ = 0;
    
    for (size_t i = 0; i < num_points; ++i) {
        float dist_sq = 0.0f;
        for (size_t j = 0; j < dims; ++j) {
            float diff = dataset_[i][j] - centroid[j];
            dist_sq += diff * diff;
        }
        if (dist_sq < min_dist_sq) {
            min_dist_sq = dist_sq;
            medoid_index_ = i;
        }
    }
}

size_t MedoidInit::get_memory_usage() const {
    return calculate_dataset_memory(dataset_) + sizeof(medoid_index_);
}

size_t MedoidInit::get_index_size() const {
    return sizeof(medoid_index_);
}

std::vector<SearchResult> MedoidInit::search(const std::vector<float>& query, size_t k) {
    float dist = compute_distance(dataset_[medoid_index_], query);
    return {{medoid_index_, dist}};
}

// ---------------------------------------------------------
// FLANN Implementation
// ---------------------------------------------------------
#include <flann/flann.hpp>

struct FlannState {
    std::unique_ptr<flann::Matrix<float>> flann_dataset;
    std::unique_ptr<flann::Index<flann::L2<float>>> index;
    std::vector<std::unique_ptr<flann::Index<flann::L2<float>>>> kmeans_indexes;
};

FlannKDTreeInit::FlannKDTreeInit(int trees, int checks, const std::string& metric) 
    : num_trees_(trees), checks_(checks), state_(new FlannState()) {
    if (metric == "cosine") metric_ = DistanceMetric::COSINE;
}

FlannKDTreeInit::~FlannKDTreeInit() {
    delete state_;
}

void FlannKDTreeInit::build(const std::vector<std::vector<float>>& dataset) {
    size_t rss_start = get_current_rss_bytes();
    dataset_ = dataset;
    
    if (dataset_.empty()) {
        throw std::invalid_argument("Dataset cannot be empty.");
    }

    size_t num_points = dataset_.size();
    size_t dims = dataset_[0].size();

    float* data_ptr = new float[num_points * dims];
    for (size_t i = 0; i < num_points; ++i) {
        for (size_t j = 0; j < dims; ++j) {
            data_ptr[i * dims + j] = dataset_[i][j];
        }
    }

    state_->flann_dataset = std::make_unique<flann::Matrix<float>>(data_ptr, num_points, dims);
    
    flann::KDTreeIndexParams params(num_trees_);
    state_->index = std::make_unique<flann::Index<flann::L2<float>>>(*state_->flann_dataset, params);
    
    size_t rss_before_index = get_current_rss_bytes();
    state_->index->buildIndex();
    size_t rss_after_index = get_current_rss_bytes();

    size_t rss_end = get_current_rss_bytes();
    
    // Total footprint during indexing
    memory_usage_ = (rss_end > rss_start) ? (rss_end - rss_start) : calculate_dataset_memory(dataset_);
    // Actual index structure size
    index_size_ = (rss_after_index > rss_before_index) ? (rss_after_index - rss_before_index) : 0;
}

std::vector<SearchResult> FlannKDTreeInit::search(const std::vector<float>& query, size_t k) {
    size_t dims = query.size();
    std::vector<float> query_copy = query;
    flann::Matrix<float> flann_query(query_copy.data(), 1, dims);

    std::vector<int> indices(k);
    std::vector<float> dists(k);
    flann::Matrix<int> flann_indices(indices.data(), 1, k);
    flann::Matrix<float> flann_dists(dists.data(), 1, k);

    flann::SearchParams params(checks_);
    state_->index->knnSearch(flann_query, flann_indices, flann_dists, k, params);
    
    // Since the searching process on KD-trees only make actual distance computations when 
    // leaf nodes are visited, the number of leaf nodes visited (checks_) is reported as the 
    // number of distance computations performed.
    distance_computations_ += checks_;

    std::vector<SearchResult> results;
    results.reserve(k);
    for (size_t i = 0; i < k; ++i) {
        if (indices[i] >= 0) {
            float final_dist = (metric_ == DistanceMetric::COSINE) ? (dists[i] / 2.0f) : std::sqrt(dists[i]);
            results.push_back({(uint32_t)indices[i], final_dist});
        }
    }

    return results;
}

size_t FlannKDTreeInit::get_memory_usage() const {
    return memory_usage_;
}

size_t FlannKDTreeInit::get_index_size() const {
    return index_size_;
}

// ---------------------------------------------------------
// FLANN KMeans Tree Implementation
// ---------------------------------------------------------

FlannKMeansInit::FlannKMeansInit(int trees, int branching, int iterations, int checks, const std::string& metric) 
    : num_trees_(trees), branching_(branching), iterations_(iterations), checks_(checks), state_(new FlannState()) {
    if (metric == "cosine") metric_ = DistanceMetric::COSINE;
}

FlannKMeansInit::~FlannKMeansInit() {
    delete state_;
}

void FlannKMeansInit::build(const std::vector<std::vector<float>>& dataset) {
    size_t rss_start = get_current_rss_bytes();
    dataset_ = dataset;
    
    if (dataset_.empty()) {
        throw std::invalid_argument("Dataset cannot be empty.");
    }

    size_t num_points = dataset_.size();
    size_t dims = dataset_[0].size();

    float* data_ptr = new float[num_points * dims];
    for (size_t i = 0; i < num_points; ++i) {
        for (size_t j = 0; j < dims; ++j) {
            data_ptr[i * dims + j] = dataset_[i][j];
        }
    }

    state_->flann_dataset = std::make_unique<flann::Matrix<float>>(data_ptr, num_points, dims);
    state_->kmeans_indexes.clear();
    
    size_t rss_before_index = get_current_rss_bytes();
    for (int t = 0; t < num_trees_; ++t) {
        flann::KMeansIndexParams params(branching_, iterations_, flann::FLANN_CENTERS_RANDOM, 0.2);
        auto index = std::make_unique<flann::Index<flann::L2<float>>>(*state_->flann_dataset, params);
        index->buildIndex();
        state_->kmeans_indexes.push_back(std::move(index));
    }
    size_t rss_after_index = get_current_rss_bytes();

    size_t rss_end = get_current_rss_bytes();
    
    memory_usage_ = (rss_end > rss_start) ? (rss_end - rss_start) : calculate_dataset_memory(dataset_);
    index_size_ = (rss_after_index > rss_before_index) ? (rss_after_index - rss_before_index) : 0;
}

std::vector<SearchResult> FlannKMeansInit::search(const std::vector<float>& query, size_t k) {
    size_t dims = query.size();
    std::vector<float> query_copy = query;
    flann::Matrix<float> flann_query(query_copy.data(), 1, dims);

    std::unordered_map<uint32_t, float> merged_candidates;
    flann::SearchParams params(checks_);

    for (auto& index : state_->kmeans_indexes) {
        std::vector<int> indices(k);
        std::vector<float> dists(k);
        flann::Matrix<int> flann_indices(indices.data(), 1, k);
        flann::Matrix<float> flann_dists(dists.data(), 1, k);

        index->knnSearch(flann_query, flann_indices, flann_dists, k, params);
        distance_computations_ += checks_;

        for (size_t i = 0; i < k; ++i) {
            if (indices[i] >= 0) {
                uint32_t idx = (uint32_t)indices[i];
                float d = (metric_ == DistanceMetric::COSINE) ? (dists[i] / 2.0f) : std::sqrt(dists[i]);
                if (merged_candidates.find(idx) == merged_candidates.end() || d < merged_candidates[idx]) {
                    merged_candidates[idx] = d;
                }
            }
        }
    }

    std::vector<SearchResult> results;
    results.reserve(merged_candidates.size());
    for (auto& pair : merged_candidates) {
        results.push_back({pair.first, pair.second});
    }

    std::sort(results.begin(), results.end(), [](const SearchResult& a, const SearchResult& b) {
        return a.distance < b.distance;
    });

    if (results.size() > k) {
        results.resize(k);
    }

    return results;
}

size_t FlannKMeansInit::get_memory_usage() const {
    return memory_usage_;
}

size_t FlannKMeansInit::get_index_size() const {
    return index_size_;
}

// ---------------------------------------------------------
// NMSLIB VPTree Implementation
// ---------------------------------------------------------
#include "init.h"
#include "index.h"
#include "params.h"
#include "knnquery.h"
#include "knnqueue.h"
#include "methodfactory.h"
#include "spacefactory.h"
#include "space.h"
#include "object.h"

struct NmslibState {
    similarity::Space<float>* space;
    similarity::ObjectVector data;
    similarity::Index<float>* index;
    bool library_initialized;

    NmslibState() : space(nullptr), index(nullptr), library_initialized(false) {}
};

VPTreeInit::VPTreeInit(int max_leaves_to_visit, float alpha_left, float alpha_right, const std::string& metric) 
    : max_leaves_to_visit_(max_leaves_to_visit), alpha_left_(alpha_left), alpha_right_(alpha_right), state_(new NmslibState()) {
    if (metric == "cosine") metric_ = DistanceMetric::COSINE;
    static bool global_init = false;
    if (!global_init) {
        similarity::initLibrary(0, LIB_LOGNONE, nullptr);
        global_init = true;
    }
    state_->library_initialized = true;
}

VPTreeInit::~VPTreeInit() {
    if (state_->index) delete state_->index;
    for (auto obj : state_->data) {
        delete obj;
    }
    if (state_->space) delete state_->space;
    delete state_;
}

void VPTreeInit::build(const std::vector<std::vector<float>>& dataset) {
    size_t rss_start = get_current_rss_bytes();
    dataset_ = dataset;

    if (dataset_.empty()) {
        throw std::invalid_argument("Dataset cannot be empty.");
    }

    size_t num_points = dataset_.size();
    size_t dims = dataset_[0].size();

    state_->space = similarity::SpaceFactoryRegistry<float>::Instance().CreateSpace("l2", similarity::AnyParams());

    for (size_t i = 0; i < num_points; ++i) {
        state_->data.push_back(new similarity::Object(i, -1, dims * sizeof(float), dataset_[i].data()));
    }

    state_->index = similarity::MethodFactoryRegistry<float>::Instance().CreateMethod(false, "vptree", "l2", *state_->space, state_->data);

    size_t rss_before_index = get_current_rss_bytes();
    state_->index->CreateIndex(similarity::AnyParams());
    
    // Set approximation parameters
    state_->index->SetQueryTimeParams(similarity::AnyParams({
        "maxLeavesToVisit=" + std::to_string(max_leaves_to_visit_),
        "alphaLeft=" + std::to_string(alpha_left_),
        "alphaRight=" + std::to_string(alpha_right_)
    }));
    
    size_t rss_after_index = get_current_rss_bytes();

    size_t rss_end = get_current_rss_bytes();

    memory_usage_ = (rss_end > rss_start) ? (rss_end - rss_start) : calculate_dataset_memory(dataset_);
    index_size_ = (rss_after_index > rss_before_index) ? (rss_after_index - rss_before_index) : 0;
}

std::vector<SearchResult> VPTreeInit::search(const std::vector<float>& query, size_t k) {
    similarity::Object* query_obj = new similarity::Object(-1, -1, query.size() * sizeof(float), query.data());
    similarity::KNNQuery<float> knn_query(*state_->space, query_obj, k);

    state_->index->Search(&knn_query);

    distance_computations_ += knn_query.DistanceComputations();

    std::vector<SearchResult> results;
    similarity::KNNQueue<float>* res_queue = knn_query.Result()->Clone();
    
    // The queue extracts in max-to-min order (largest distance first), so we read and reverse
    std::vector<SearchResult> temp_results;
    while (!res_queue->Empty()) {
        temp_results.push_back({(uint32_t)res_queue->TopObject()->id(), res_queue->TopDistance()});
        res_queue->Pop();
    }
    delete res_queue;
    delete query_obj;

    std::reverse(temp_results.begin(), temp_results.end());
    return temp_results;
}

size_t VPTreeInit::get_memory_usage() const {
    return memory_usage_;
}

size_t VPTreeInit::get_index_size() const {
    return index_size_;
}

// ---------------------------------------------------------
// NMSLIB Stacked NSW Implementation
// ---------------------------------------------------------
// Reuses NmslibState (defined above in the VP-Tree section).

StackedNSWInit::StackedNSWInit(int M, int ef_construction, int ef, const std::string& metric)
    : M_(M), ef_construction_(ef_construction), ef_(ef), state_(new NmslibState()) {
    if (metric == "cosine") metric_ = DistanceMetric::COSINE;
    static bool global_init = false;
    if (!global_init) {
        similarity::initLibrary(0, LIB_LOGNONE, nullptr);
        global_init = true;
    }
    state_->library_initialized = true;
}

StackedNSWInit::~StackedNSWInit() {
    if (state_->index) delete state_->index;
    for (auto obj : state_->data) {
        delete obj;
    }
    if (state_->space) delete state_->space;
    delete state_;
}

void StackedNSWInit::build(const std::vector<std::vector<float>>& dataset) {
    size_t rss_start = get_current_rss_bytes();
    dataset_ = dataset;

    if (dataset_.empty()) {
        throw std::invalid_argument("Dataset cannot be empty.");
    }

    size_t num_points = dataset_.size();
    size_t dims = dataset_[0].size();

    std::string space_name = (metric_ == DistanceMetric::COSINE) ? "cosinesimil" : "l2";
    state_->space = similarity::SpaceFactoryRegistry<float>::Instance().CreateSpace(space_name, similarity::AnyParams());

    for (size_t i = 0; i < num_points; ++i) {
        state_->data.push_back(new similarity::Object(i, -1, dims * sizeof(float), dataset_[i].data()));
    }

    state_->index = similarity::MethodFactoryRegistry<float>::Instance().CreateMethod(
        false, "hnsw", space_name, *state_->space, state_->data);

    size_t rss_before_index = get_current_rss_bytes();
    // skip_optimized_index=1 forces the unoptimized HnswNode structure, giving
    // correct access to per-layer links during the modified layer-1 search.
    state_->index->CreateIndex(similarity::AnyParams({
        "M=" + std::to_string(M_),
        "efConstruction=" + std::to_string(ef_construction_),
        "skip_optimized_index=1"
    }));
    size_t rss_after_index = get_current_rss_bytes();

    // Set search-time ef
    state_->index->SetQueryTimeParams(similarity::AnyParams({
        "ef=" + std::to_string(ef_)
    }));

    size_t rss_end = get_current_rss_bytes();

    memory_usage_ = (rss_end > rss_start) ? (rss_end - rss_start) : calculate_dataset_memory(dataset_);
    index_size_ = (rss_after_index > rss_before_index) ? (rss_after_index - rss_before_index) : 0;
}

std::vector<SearchResult> StackedNSWInit::search(const std::vector<float>& query, size_t k) {
    similarity::Object* query_obj = new similarity::Object(-1, -1, query.size() * sizeof(float), query.data());
    similarity::KNNQuery<float> knn_query(*state_->space, query_obj, k);

    state_->index->Search(&knn_query);

    distance_computations_ += knn_query.DistanceComputations();

    std::vector<SearchResult> temp_results;
    similarity::KNNQueue<float>* res_queue = knn_query.Result()->Clone();
    while (!res_queue->Empty()) {
        temp_results.push_back({(uint32_t)res_queue->TopObject()->id(), res_queue->TopDistance()});
        res_queue->Pop();
    }
    delete res_queue;
    delete query_obj;

    std::reverse(temp_results.begin(), temp_results.end());
    return temp_results;
}

size_t StackedNSWInit::get_memory_usage() const {
    return memory_usage_;
}

size_t StackedNSWInit::get_index_size() const {
    return index_size_;
}

// ---------------------------------------------------------
// FALCONN LSH Implementation
// ---------------------------------------------------------
#include <falconn/lsh_nn_table.h>

struct LSHState {
    std::vector<falconn::DenseVector<float>> falconn_data;
    std::unique_ptr<falconn::LSHNearestNeighborTable<falconn::DenseVector<float>>> table;
};

LSHInit::LSHInit(int num_hash_tables, int num_hash_bits, int num_probes, const std::string& metric)
    : num_hash_tables_(num_hash_tables), num_hash_bits_(num_hash_bits), num_probes_(num_probes), state_(new LSHState()) {
    if (metric == "cosine") metric_ = DistanceMetric::COSINE;
}

LSHInit::~LSHInit() {
    delete state_;
}

void LSHInit::build(const std::vector<std::vector<float>>& dataset) {
    size_t rss_start = get_current_rss_bytes();
    dataset_ = dataset;
    if (dataset_.empty()) {
        throw std::invalid_argument("Dataset cannot be empty.");
    }

    size_t num_points = dataset_.size();
    size_t dims = dataset_[0].size();

    state_->falconn_data.resize(num_points);
    for (size_t i = 0; i < num_points; ++i) {
        state_->falconn_data[i] = Eigen::Map<const falconn::DenseVector<float>>(dataset_[i].data(), dims);
    }

    falconn::LSHConstructionParameters params;
    params.dimension = dims;
    params.lsh_family = falconn::LSHFamily::CrossPolytope;
    if (metric_ == DistanceMetric::COSINE) {
        params.distance_function = falconn::DistanceFunction::NegativeInnerProduct;
    } else {
        params.distance_function = falconn::DistanceFunction::EuclideanSquared;
    }
    params.storage_hash_table = falconn::StorageHashTable::FlatHashTable;
    params.l = num_hash_tables_;
    params.num_setup_threads = 0;
    params.num_rotations = 1;
    
    falconn::compute_number_of_hash_functions<falconn::DenseVector<float>>(num_hash_bits_, &params);

    size_t rss_before_index = get_current_rss_bytes();
    state_->table = falconn::construct_table<falconn::DenseVector<float>>(state_->falconn_data, params);
    size_t rss_after_index = get_current_rss_bytes();

    size_t rss_end = get_current_rss_bytes();
    memory_usage_ = (rss_end > rss_start) ? (rss_end - rss_start) : calculate_dataset_memory(dataset_);
    index_size_ = (rss_after_index > rss_before_index) ? (rss_after_index - rss_before_index) : 0;
}

std::vector<SearchResult> LSHInit::search(const std::vector<float>& query, size_t k) {
    falconn::DenseVector<float> eigen_query = Eigen::Map<const falconn::DenseVector<float>>(const_cast<float*>(query.data()), query.size());
    
    auto query_obj = state_->table->construct_query_object(num_probes_);
    std::vector<int32_t> candidates;
    query_obj->find_k_nearest_neighbors(eigen_query, k, &candidates);
    
    falconn::QueryStatistics stats = query_obj->get_query_statistics();
    distance_computations_ += stats.average_num_unique_candidates * stats.num_queries;

    std::vector<SearchResult> results;
    results.reserve(candidates.size());
    for (int32_t idx : candidates) {
        float dist = compute_distance(dataset_[idx], query);
        results.push_back({(uint32_t)idx, dist});
    }

    std::sort(results.begin(), results.end(), [](const SearchResult& a, const SearchResult& b) {
        return a.distance < b.distance;
    });

    return results;
}

size_t LSHInit::get_memory_usage() const {
    return memory_usage_;
}

size_t LSHInit::get_index_size() const {
    return index_size_;
}
