#include "WaveletDeconvolution.h"

using namespace AnitaResponse;

WaveletDeconvolution::WaveletDeconvolution(const unsigned int p,
                                           const forward::WaveletType type,
                                           const double noiseSd,
                                           const RealArray& scaling,
                                           const RealArray& rho)
    : p_{p}, type_(type), noiseSd_(noiseSd), scaling_(scaling), rho_(rho) {}

auto
WaveletDeconvolution::update_state(const unsigned int length) const -> void {

  // get the next power of 2 in length
  const unsigned int next_N{next_pow2(length)};

  // if this is different than what is currently stored,
  // then we need to update the internal state
  if (next_N != N_) {

    // save the new value of N
    this->N_ = next_N;

    // get the filters - u, v, util, vtil
    const auto filters{forward::filt(N_, this->type_)};

    // and save them
    this->u_ = std::get<0>(filters);
    this->v_ = std::get<1>(filters);

    // and create the matrix
    this->matrix_ = forward::getBasisMatrix(u_, v_, p_);
  }

  // we are good
}

auto
WaveletDeconvolution::get_vector(const unsigned int N, const AnalysisWaveform& wf)
    -> RealArray {

  // check that we have enough space
  if (N < wf.even()->GetN()) {
    throw std::invalid_argument("`N` is less than the length of AnalysisWaveform.");
  }

  // create the vector to store the waveform
  RealArray waveform(N, 0.);

  // and copy the even waveform into the array
  std::copy(wf.even()->GetY(), wf.even()->GetY() + wf.even()->GetN(), waveform.begin());

  // and return the waveform
  return waveform;
}

auto
WaveletDeconvolution::deconvolve(AnalysisWaveform* wf,
                                 const AnalysisWaveform* response) const -> void {

  // update the internal state for waveforms of this length
  this->update_state(wf->even()->GetN());

  // compute the size of the waveforms that we use
  const auto N{next_pow2(wf->even()->GetN())};

  // get the vectors for the waveform and response
  const auto waveform{get_vector(N, *wf)};
  const auto impulse{get_vector(N, *response)};

  // compute the deconvolved waveform
  const auto deconvolved{
      forward::deconvolve(waveform, impulse, p_, type_, noiseSd_, scaling_, rho_, rule_)};
}

auto
WaveletDeconvolution::next_pow2(const unsigned int N) -> unsigned int {
  return pow(2., ceil(log2(N)));
}
