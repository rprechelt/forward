#include "forward/forward.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/complex.h>
#include <vector>
#include <complex>

namespace py = pybind11;
using namespace forward;

// Numpy real and complex arrays
using NumpyReal = py::array_t<double>;
using NumpyComplex = py::array_t<std::complex<double>>;


// convert a std::vector into a py::array
template <typename T> auto
vector_to_array(const std::vector<T>& vector) -> py::array_t<T> {
  return py::array_t<T>(vector.size(), vector.data());
}

// convert a py::array into an std::vector
template <typename T> auto
array_to_vector(const py::array_t<T>& array) -> std::vector<T> {

  // get a pointer to the buffer
  const auto buf{array.request()};

  // and create the vector pointing into the buffer
  return std::vector<T>(static_cast<T*>(buf.ptr),
                        static_cast<T*>(buf.ptr)+array.size());


}

// create our Python module
PYBIND11_MODULE(forward, m) {

  // set a docstring
  m.doc() = "Deconvolve 1D signals using ForWaRD.";

  // the supported wavelet types
  py::enum_<WaveletType>(m, "WaveletType")
    .value("Meyer", WaveletType::Meyer)
    .value("d8", WaveletType::d8)
    .value("d10", WaveletType::d10)
    .value("d12", WaveletType::d12)
    .value("d14", WaveletType::d14)
    .value("d16", WaveletType::d16)
    .value("d18", WaveletType::d18)
    .value("d20", WaveletType::d20);

  // the supported threshold rules
  py::enum_<ThresholdRule>(m, "ThresholdRule")
    .value("Soft", ThresholdRule::Soft)
    .value("Hard", ThresholdRule::Hard);

  m.def("fwt",
        [](const NumpyComplex& z, const unsigned int sdim,
           const NumpyComplex& util, const NumpyComplex& vtil) -> NumpyComplex {
          
          // compute the wavelet transform as an std::vector
          const auto wt{fwt(array_to_vector(z),
                            sdim,
                            array_to_vector(util),
                            array_to_vector(vtil))};

          // and return it back as a Python array
          return vector_to_array(wt);
        },
        py::arg("z"), py::arg("sdim"), py::arg("util"), py::arg("vtil"),
        "Perform the forward wavelet transform.");
  m.def("ifwt",
        [](const NumpyComplex& z, const unsigned int sdim,
           const NumpyComplex& u, const NumpyComplex& v) -> NumpyComplex {

          // compute the wavelet transform as an std::vector
          const auto wt{ifwt(array_to_vector(z),
                             sdim,
                             array_to_vector(u),
                             array_to_vector(v))};

          // and return it back as a Python array
          return vector_to_array(wt);
        },
        py::arg("z"), py::arg("sdim"), py::arg("util"), py::arg("vtil"),
        "Perform the inverse forward wavelet transform.");
  m.def("deconvolve",
        [](const NumpyReal& signal, const NumpyReal& response,
           const unsigned int p, const WaveletType type,
           const double noiseSd, const NumpyReal& scaling,
           const NumpyReal& rho, const ThresholdRule rule) -> NumpyReal {

          // compute the wavelet transform as an std::vector
          const auto deconvolved{deconvolve(array_to_vector(signal),
                                            array_to_vector(response),
                                            p,
                                            type,
                                            noiseSd,
                                            array_to_vector(scaling),
                                            array_to_vector(rho),
                                            rule)};

          // and return it back as a Python array
          return vector_to_array(deconvolved);
        },
        py::arg("signal"), py::arg("response"), py::arg("p"), py::arg("type"),
        py::arg("noiseSd"), py::arg("scaling"), py::arg("rho"), py::arg("rule"),
        "Perform FoRWarD wavelet deconvolution.");
  

} // PYBIND11-MODULE
