#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "initialization.h"

namespace py = pybind11;

PYBIND11_MODULE(initialization_cpp, m) {
    m.doc() = "C++ module for initialization approaches with distance tracking";
    
    py::class_<SearchResult>(m, "SearchResult")
        .def_readonly("index", &SearchResult::index)
        .def_readonly("distance", &SearchResult::distance);

    py::class_<InitializationApproach>(m, "InitializationApproach")
        .def("get_distance_computations", &InitializationApproach::get_distance_computations)
        .def("reset_distance_computations", &InitializationApproach::reset_distance_computations)
        .def("get_memory_usage", &InitializationApproach::get_memory_usage)
        .def("get_index_size", &InitializationApproach::get_index_size);
    
    py::class_<RandomPointsInit, InitializationApproach>(m, "RandomPointsInit")
        .def(py::init<uint32_t>(), py::arg("seed") = 42)
        .def("build", &RandomPointsInit::build)
        .def("search", &RandomPointsInit::search);
        
    py::class_<MedoidInit, InitializationApproach>(m, "MedoidInit")
        .def(py::init<>())
        .def("build", &MedoidInit::build)
        .def("search", &MedoidInit::search);

    py::class_<FlannKDTreeInit, InitializationApproach>(m, "FlannKDTreeInit")
        .def(py::init<int, int>(), py::arg("trees") = 4, py::arg("checks") = 32)
        .def("build", &FlannKDTreeInit::build)
        .def("search", &FlannKDTreeInit::search);

    py::class_<FlannKMeansInit, InitializationApproach>(m, "FlannKMeansInit")
        .def(py::init<int, int, int, int>(), 
             py::arg("trees") = 1, 
             py::arg("branching") = 32, 
             py::arg("iterations") = 11, 
             py::arg("checks") = 32)
        .def("build", &FlannKMeansInit::build)
        .def("search", &FlannKMeansInit::search);

    py::class_<VPTreeInit, InitializationApproach>(m, "VPTreeInit")
        .def(py::init<int, float, float>(),
             py::arg("max_leaves_to_visit") = 1000,
             py::arg("alpha_left") = 1.0f,
             py::arg("alpha_right") = 1.0f)
        .def("build", &VPTreeInit::build)
        .def("search", &VPTreeInit::search);

    py::class_<StackedNSWInit, InitializationApproach>(m, "StackedNSWInit")
        .def(py::init<int, int, int>(),
             py::arg("M") = 16,
             py::arg("ef_construction") = 200,
             py::arg("ef") = 100)
        .def("build", &StackedNSWInit::build)
        .def("search", &StackedNSWInit::search);

    py::class_<LSHInit, InitializationApproach>(m, "LSHInit")
        .def(py::init<int, int, int>(),
             py::arg("num_hash_tables") = 50,
             py::arg("num_hash_bits") = 16,
             py::arg("num_probes") = 100)
        .def("build", &LSHInit::build)
        .def("search", &LSHInit::search);
}
