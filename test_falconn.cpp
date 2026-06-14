#include <falconn/lsh_nn_table.h>
int main() {
    falconn::LSHNNQuery<falconn::DenseVector<float>, int32_t>* q = nullptr;
    q->reset_query_statistics();
    q->set_num_probes(10);
    return 0;
}
