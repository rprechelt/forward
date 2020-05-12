#ifndef __WAVELET_DECONVOLUTION_H_
#define __WAVELET_DECONVOLUTION_H_

#include "AnalysisWaveform.h"
#include "SystemResponse.h"
#include "forward/forward.hpp"
#include <complex>
#include <vector>

namespace AnitaResponse {

  /**
   * Deconvolve using Fourier-regularized Wavelet Deconvolution (ForWaRD).
   *
   * This is a re-implementation (by R. Prechelt) of wavelet
   * deconvolution using the FoRWarD algorithm [2] first
   * implemented by D. Bhattacharya [2][3].
   *
   *
   * [1] https://ieeexplore.ieee.org/document/126132
   * [2] https://github.com/debdeepbh/libWTools
   * [3] https://debdeepbh.github.io/content/report-anita.pdf
   *
   */
  class WaveletDeconvolution final : public DeconvolutionMethod {

    // some aliases to make our life easier.
    using RealArray     = std::vector<double>;
    using ComplexArray  = std::vector<std::complex<double>>;
    using ComplexMatrix = std::vector<std::vector<std::complex<double>>>;

    public:
    /**
     * Construct a WaveletDeconvolution-er.
     *
     * @param p          The number of wavelet levels to use.
     * @param noiseSd    The standard deviation of the noise.
     * @param scaling    The Wiener scaling parameter.
     * @param rho        The Fourier shrinkage parameter.
     * @param type       The wavelet family to use.
     */
    WaveletDeconvolution(const unsigned int p,
                         const forward::WaveletType type,
                         const double noiseSd,
                         const RealArray& scaling,
                         const RealArray& rho);

    /**
     * Deconvolve `response` out of `wf`.`
     *
     * @param wf          The waveform to deconvolve-in-place
     * @param response    The impulse response to deconvolve.
     */
    auto
    deconvolve(AnalysisWaveform* wf, const AnalysisWaveform* response) const -> void;

    /**
     * A no-op destructor.
     */
    ~WaveletDeconvolution(){};

    private:
    /**
     * Update the internal state to deconvolve waveforms of length `N`.
     *
     * @param N    The length of waveforms that we want to deconvolve.
     */
    auto
    update_state(const unsigned int N) const -> void;

    /**
     * Return the next highest power of 2.
     */
    static auto
    next_pow2(const unsigned int N) -> unsigned int;

    /**
     * Convert an AnalysisWaveform into a RealArray of length N.
     *
     * @param N    The minimum length of the RealArray's
     * @param wf   The waveform to convert into a vector.
     *
     */
    static auto
    get_vector(const unsigned int N, const AnalysisWaveform& wf) -> RealArray;

    const unsigned int p_;      ///< The number of wavelet levels to use.
    forward::WaveletType type_; ///< The wavelet family to use.
    double noiseSd_;            ///< The noise standard deviation.
    RealArray scaling_;         ///< The Wiener scaling parameter.
    RealArray rho_;             ///< The fourier shrinkage parameter.
    forward::ThresholdRule rule_{forward::ThresholdRule::Soft}; ///< The threshold rule
    mutable ComplexArray u_;       ///< The 'u' basis vector.
    mutable ComplexArray v_;       ///< The 'v' basis vector.
    mutable ComplexMatrix matrix_; ///< The matrix of basis vectors.
    mutable unsigned int N_{0};    ///< The current length of basis vectors.

  }; // END: Wavelet Deconvolution

} // namespace AnitaResponse

#endif // __WAVELET_DECONVOLUTION_H_
