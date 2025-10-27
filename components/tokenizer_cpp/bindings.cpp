#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "bl_bpe_tokenizer.h"

namespace py = pybind11;

PYBIND11_MODULE(tokenizer_cpp, m) {
    py::class_<BlBPETokenizer>(m, "BlBPETokenizer")
        .def(py::init<int, const std::vector<std::string>&>(), py::arg("vocab_size") = 10000, py::arg("special_tokens") = std::vector<std::string>{})
        .def("train", &BlBPETokenizer::train, py::arg("texts"))
        .def("encode", &BlBPETokenizer::encode, py::arg("text"))
        .def("decode", &BlBPETokenizer::decode, py::arg("tokens"))
        .def("get_special_token_id", &BlBPETokenizer::getSpecialTokenId, py::arg("token"));
}