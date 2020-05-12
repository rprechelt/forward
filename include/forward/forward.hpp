/**
 * This is an implementation of the Fourier-regularized Wavelet
 * Deconvolution (ForWaRD) algorithm for 1D signals.
 *
 * This was originally written in C by Debdeep Bhattacharya
 * in 2018. This is a modern C++ const-correct rewrite by
 * Remy Prechelt in April 2020 that closely follows
 * the original code and uses the same variable names and
 * code flow where possible.
 *
 * See: https://debdeepbh.github.io/content/report-anita.pdf
 * for a detailed report on the algorithm and implementation.
 *
 * This has *not* been optimized for performance; it is
 * intended as a close rewrite of the original algorithm.
 *
 *
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <complex>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include <vector>

#include <fftw3.h>

namespace forward {

  /* To match FFTW, all deconvolutions is performed as doubles. */
  using complex = std::complex<double>;

  /* Use std::vector's for now and template later. */
  template <class T> using Array = std::vector<T>;

  /* A matrix as a vector of vectors */
  template <class T> using Matrix = std::vector<std::vector<T>>;

  /**
   * The supported wavelet types.
   */
  enum class WaveletType { Meyer, d8, d10, d12, d14, d16, d18, d20 };

  /**
   * The thresholding that we use
   */
  enum class ThresholdRule { Soft, Hard };

  /**
   * Compute the FFT of `in`.
   *
   * @param in    The complex vector to transform.
   *
   * @returns    The FFT of `in`.`
   */
  auto
  fft(const Array<complex>& in) -> Array<complex> {

    // get the length of the input vector
    const auto N{in.size()};

    // create the output vector
    Array<complex> out(N, 0.);

    // create the FFTW plan
    auto plan{
        fftw_plan_dft_1d(N,
                         reinterpret_cast<fftw_complex*>(const_cast<complex*>(in.data())),
                         reinterpret_cast<fftw_complex*>(out.data()),
                         FFTW_FORWARD,
                         FFTW_ESTIMATE)};

    // perform the fft
    fftw_execute(plan);

    // and then destroy the plan
    fftw_destroy_plan(plan);

    // and return the FFT
    return out;
  }

  /**
   * Compute the FFT of `in` (where `in` is real).
   *
   * @param in    The real vector to transform.
   *
   * @returns    The FFT of `in`.`
   */
  auto
  fft(const Array<double>& in) -> Array<complex> {

    // get the length of the input vector
    const auto N{in.size()};

    // create the complex version of the input vector
    Array<complex> complex_in(N, 0.);

    // and copy `in` into `complex_in`
    std::transform(in.begin(),
                   in.end(),
                   complex_in.begin(),
                   [](const double& val) -> complex { return complex(val); });

    // and return the fft of the complex vector
    return fft(complex_in);
  }

  /**
   * Compute the IFFT of `in`.
   *
   * @param in    The complex vector to inverse transform.
   *
   * @returns    The IFFT of `in`.`
   */
  auto
  ifft(const Array<complex>& in) -> Array<complex> {

    // get the length of the input vector
    const auto N{in.size()};

    // create the output vector
    Array<complex> out(N, 0.);

    // create the plan
    fftw_plan plan_a =
        fftw_plan_dft_1d(N,
                         reinterpret_cast<fftw_complex*>(const_cast<complex*>(in.data())),
                         reinterpret_cast<fftw_complex*>(out.data()),
                         FFTW_BACKWARD,
                         FFTW_ESTIMATE);

    // execute the plan
    fftw_execute(plan_a);

    //  destroy the plan
    fftw_destroy_plan(plan_a);

    // and divide by N to renormalize the FFT
    std::transform(out.begin(), out.end(), out.begin(), [&N](complex& val) {
      return val / complex(N, 0);
    });

    // and return the inverse transform
    return out;
  }

  /**
   * Convolve two complex arrays A, B.
   *
   * This assumes that the size of A is the size of B.
   *
   * @param A    The first input array.
   * @param B    The second input array.
   *
   * @returns    The convolution of A & B.
   */
  auto
  convolve(const Array<complex>& A, const Array<complex>& B) -> Array<complex> {

    // check that these are the same length
    if (A.size() != B.size()) {
      throw std::invalid_argument("convolve: A is not the same size as B.");
    }

    // compute the fft of A and B
    const auto fA{fft(A)};
    const auto fB{fft(B)};

    // get the length of the input vector
    const auto N{A.size()};

    if (N == 0) {
      throw std::invalid_argument("N == 0 in wienForwd");
    }

    // create the output vector
    Array<complex> out(N);

    // multiply element-wise
    for (unsigned int i = 0; i < N; ++i) {
      out[i] = fA[i] * fB[i];
    }

    // and take the inverse FFT
    return ifft(out);
  }

  /**
   * Downsample a vector of even length.
   *
   * The length of the returned vector is len(A)/2
   *
   * @param A     The vector to downsample.
   *
   * @returns     The downsampled vector.
   *
   */
  template <typename T>
  auto
  downsample(const Array<T>& A) -> Array<T> {

    // get the length of the input vector
    const auto N{A.size()};

    if (N == 0) {
      throw std::invalid_argument("N == 0 in wienForwd");
    }

    // make sure that the vector is of even length
    if (N % 2 == 1) {
      throw std::invalid_argument("down: `A` is not of even length.");
    }

    // allocate the output vector
    Array<T> out(int(N / 2));

    // fill in the downsampled array
    for (unsigned int i = 0; i < N / 2; ++i) {
      out[i] = A[2 * i];
    }

    // and return the downsampled array
    return out;
  }

  /**
   * Upsample a vector.
   *
   * The length of the returned vector is 2*len(A)
   *
   * @param A     The vector to upsample.
   *
   * @returns     The downsampled vector.
   *
   */
  template <typename T>
  auto
  upsample(const Array<T> A) -> Array<T> {

    // get the length of the input vector
    const auto N{A.size()};

    if (N == 0) {
      throw std::invalid_argument("N == 0 in wienForwd");
    }

    // and allocate the output vector and fill it with zero
    Array<T> out(2 * N, 0.);

    // and now fill in every other value
    for (unsigned int i = 0; i < N; ++i) {
      out[2 * i] = A[i];
    }

    // and returned the output
    return out;
  }

  // Folds a vector A of length N in half and adds it to produce B of length N/2
  /**
   * Fold a vector of length N in half.
   *
   * This produces a vector of length N/2.
   *
   * @param A    The vector to fold.
   *
   * @returns    The input vector folded in half.
   */
  auto
  fold(const Array<complex>& A) -> Array<complex> {

    // get the length of the input vector
    const auto N{A.size()};

    if (N == 0) {
      throw std::invalid_argument("N == 0 in wienForwd");
    }

    // check that the vector is of even length
    if (N % 2 == 1) {
      throw std::invalid_argument("fold: `A` is not of even length.");
    }

    // and allocate the output vector and fill it with zeros
    Array<complex> out(N / 2, 0.);

    // and do the fold
    for (unsigned int i = 0; i < N / 2; ++i) {
      out[i] = A[i] + A[N / 2 + i];
    }

    // and return the output vector
    return out;
  }

  /**
   * Pecursively perform the Forward Wavelet Transform (FWT).
   *
   * This performs the wavelet transform of `z` w.r.t to
   * the parent wavelets `u` and `v` with the smallest possible
   * dimension `sdim`. e.g. for p-th stage wavelets, sdim = N/2**p.
   *
   *
   * @param z      The input signal
   * @param sdim   The smallest dimension
   * @param util   The first parent wavelet
   * @param ztil   The second parent wavelet.
   *
   */
  auto
  fwt(const Array<complex>& z,
      const unsigned int sdim,
      const Array<complex>& util,
      const Array<complex>& vtil) -> Array<complex> {

    // get the length of the input vectors
    const auto N{util.size()};

    // util, vtil and u must all be the same size.
    if ((util.size() != vtil.size()) || (util.size() != z.size())) {
      throw std::invalid_argument("fwt: input vectors are not the same size.");
    }

    // convolve z and vtil
    const auto convolved_first{convolve(z, vtil)};

    // and downsample the convolution
    const auto first{downsample(convolved_first)};

    // compute the convolution of z and util
    const auto convolved{convolve(z, util)};

    // and downsample convolved
    auto second{downsample(convolved)};

    // we assume that we are at the terminating stage

    // if we are *not* at the terminating stage
    if (N > 2 * sdim) {

      // fold util and vtil in half
      const auto util_folded{fold(util)};
      const auto vtil_folded{fold(vtil)};

      // and recursively compute the second part of the fwt
      second = fwt(second, sdim, util_folded, vtil_folded);
    }

    // allocate the output vector of length N
    Array<complex> w(N, 0.);

    // fill in the first half of the vector
    for (unsigned int i = 0; i < N / 2; ++i) {
      w[i] = first[i];
    }

    // and fill in the second half of the vector
    for (unsigned int i = N / 2; i < N; ++i) {
      w[i] = second[i - N / 2];
    }

    // and return the output
    return w;
  }

  // // recursive implementation of inverse wavelet transform
  auto
  ifwt(const Array<complex>& z,
       const unsigned int sdim,
       const Array<complex>& u,
       const Array<complex>& v) -> Array<complex> {

    // get the size of the input vector
    const auto N{z.size()};

    // we break z into two parts
    Array<complex> z_first(z.begin(), z.begin() + N / 2);
    Array<complex> z_second(z.begin() + N / 2, z.end());

    // upsample upped_first
    const auto upped_first{upsample(z_first)};

    // and convolve it with v
    const auto first{convolve(upped_first, v)};

    // upsamle the second half of z
    const auto upped_second{upsample(z_second)};

    // and convolve it with u to get the second half
    auto second{convolve(upped_second, u)};

    // if we are *not* at the terminating stage
    if (N > 2 * sdim) {

      // fold u and v
      const auto u_folded{fold(u)};
      const auto v_folded{fold(v)};

      // compute the ifwt of the second-half recursively
      const auto recur_store{ifwt(z_second, sdim, u_folded, v_folded)};

      // and upsamle the recursive second-half
      const auto upped_recur{upsample(recur_store)};

      // and convolve it with u to get the second vector
      second = convolve(upped_recur, u);
    }

    // allocate the output vector
    Array<complex> w(N, 0.);

    // and add the first and second parts together for the output
    for (unsigned int i = 0; i < N; ++i) {
      w[i] = first[i] + second[i];
    }

    // and return the transform
    return w;
  }

  /**
   * Given a parent wavelet `u`, get the other wavelet `v`.
   *
   * This works for all the Daubauchies wavelets from Wikipedia.
   *
   * @params u    The parent wavelet, u.
   *
   * @returns v   The other parent wavelet.
   */
  auto
  getother(const Array<complex>& u) -> Array<complex> {

    // get the length of the input vector
    const auto N{u.size()};

    // allocate the output array
    Array<complex> v(N, 0.);

    // and iterate over the wavelet coefficients
    for (unsigned int k = 0; k < N; ++k) {

      //  ((x % N + N ) % N produces positive number for x%N
      const unsigned int idx = ((1 - k) % N + N) % N;

      // and fill in the vector
      v[k] = pow(-1., k - 1) * conj(u[idx]);

    } // END: for loop

    // and return the output vector
    return v;
  }

  // takes a real vector and turns it into a std::complex vector
  auto
  realToComplex(const Array<double>& real) -> Array<complex> {

    // the size of the input array
    const auto N{real.size()};

    // allocate output array
    Array<complex> out(N, 0.);

    // and loop through the array
    for (unsigned int i = 0; i < N; ++i) {
      out[i] = std::complex<double>(real[i]);
    }

    // and return the complex array
    return out;
  }

  /**
   * Return the wavelet basis vectors.
   *
   * This returns u, v, u-tilde, and v-tilde.
   *
   * @param N    The length of the output arrays.
   * @param filterType    The wavelets to use.
   */
  auto
  filt(const int N, const WaveletType& filterType)
      -> std::tuple<Array<complex>, Array<complex>, Array<complex>, Array<complex>> {

    // allocate the output filters of the right length
    Array<complex> u(N, 0.);

    // meyer wavelets
    if (filterType == WaveletType::Meyer) {

      // the fixed coefficients for the Meyer basis
      const Array<double> meyer{-1.009999956941423e-12, 8.519459636796214e-09,
                                -1.111944952595278e-08, -1.0798819539621958e-08,
                                6.066975741351135e-08,  -1.0866516536735883e-07,
                                8.200680650386481e-08,  1.1783004497663934e-07,
                                -5.506340565252278e-07, 1.1307947017916706e-06,
                                -1.489549216497156e-06, 7.367572885903746e-07,
                                3.20544191334478e-06,   -1.6312699734552807e-05,
                                6.554305930575149e-05,  -0.0006011502343516092,
                                -0.002704672124643725,  0.002202534100911002,
                                0.006045814097323304,   -0.006387718318497156,
                                -0.011061496392513451,  0.015270015130934803,
                                0.017423434103729693,   -0.03213079399021176,
                                -0.024348745906078023,  0.0637390243228016,
                                0.030655091960824263,   -0.13284520043622938,
                                -0.035087555656258346,  0.44459300275757724,
                                0.7445855923188063,     0.44459300275757724,
                                -0.035087555656258346,  -0.13284520043622938,
                                0.030655091960824263,   0.0637390243228016,
                                -0.024348745906078023,  -0.03213079399021176,
                                0.017423434103729693,   0.015270015130934803,
                                -0.011061496392513451,  -0.006387718318497156,
                                0.006045814097323304,   0.002202534100911002,
                                -0.002704672124643725,  -0.0006011502343516092,
                                6.554305930575149e-05,  -1.6312699734552807e-05,
                                3.20544191334478e-06,   7.367572885903746e-07,
                                -1.489549216497156e-06, 1.1307947017916706e-06,
                                -5.506340565252278e-07, 1.1783004497663934e-07,
                                8.200680650386481e-08,  -1.0866516536735883e-07,
                                6.066975741351135e-08,  -1.0798819539621958e-08,
                                -1.111944952595278e-08, 8.519459636796214e-09,
                                -1.009999956941423e-12, 0.0};

      // and copy the coefficients into u and make them complex
      std::transform(meyer.begin(), meyer.end(), u.begin(), [](const double& val) {
        return complex(val);
      });

    } // END Meyer wavelets
    // d8
    else if (filterType == WaveletType::d8) {
      // the wavelet coefficients
      const Array<double> coeffs{0.32580343,
                                 1.01094572,
                                 0.89220014,
                                 -0.03957503,
                                 -0.26450717,
                                 0.0436163,
                                 0.0465036,
                                 -0.01498699};

      // and copy them into the u-vector and make them complex
      std::transform(coeffs.begin(), coeffs.end(), u.begin(), [](const double& val) {
        return complex(val) / sqrt(2.);
      });

    } // END: d8
    // d12
    else if (filterType == WaveletType::d10) {
      // the wavelet coefficients
      const Array<double> coeffs{0.22641898,
                                 0.85394354,
                                 1.02432694,
                                 0.19576696,
                                 -0.34265671,
                                 -0.04560113,
                                 0.10970265,
                                 -0.0088268,
                                 -0.01779187,
                                 0.00471742793};

      // and copy them into the u-vector and make them complex
      std::transform(coeffs.begin(), coeffs.end(), u.begin(), [](const double& val) {
        return complex(val) / sqrt(2.);
      });

    } // END: d110
    // d12
    else if (filterType == WaveletType::d12) {
      // the wavelet coefficients
      const Array<double> coeffs{0.15774243,
                                 0.69950381,
                                 1.06226376,
                                 0.44583132,
                                 -0.3199866,
                                 -0.18351806,
                                 0.13788809,
                                 0.03892321,
                                 -0.04466375,
                                 0.000783251152,
                                 0.00675606236,
                                 -0.00152353381};

      // and copy them into the u-vector and make them complex
      std::transform(coeffs.begin(), coeffs.end(), u.begin(), [](const double& val) {
        return complex(val) / sqrt(2.);
      });

    } // END: d12
    // d14
    else if (filterType == WaveletType::d14) {
      // the wavelet coefficients
      const Array<double> coeffs{0.11009943,
                                 0.56079128,
                                 1.03114849,
                                 0.66437248,
                                 -0.20351382,
                                 -0.31683501,
                                 0.1008467,
                                 0.11400345,
                                 -0.05378245,
                                 -0.02343994,
                                 0.01774979,
                                 0.000607514995,
                                 -0.00254790472,
                                 0.000500226853};

      // and copy them into the u-vector and make them complex
      std::transform(coeffs.begin(), coeffs.end(), u.begin(), [](const double& val) {
        return complex(val) / sqrt(2.);
      });

    } // END: d14
    // d16
    else if (filterType == WaveletType::d16) {
      // the wavelet coefficients
      const Array<double> coeffs{0.07695562,
                                 0.44246725,
                                 0.95548615,
                                 0.82781653,
                                 -0.02238574,
                                 -0.40165863,
                                 0.000668194092,
                                 0.18207636,
                                 -0.0245639,
                                 -0.06235021,
                                 0.01977216,
                                 0.01236884,
                                 -0.00688771926,
                                 -0.000554004549,
                                 0.000955229711,
                                 -0.000166137261};

      // and copy them into the u-vector and make them complex
      std::transform(coeffs.begin(), coeffs.end(), u.begin(), [](const double& val) {
        return complex(val) / sqrt(2.);
      });

    } // END: d16
    // d18
    else if (filterType == WaveletType::d18) {
      // the wavelet coefficients
      const Array<double> coeffs{0.05385035,
                                 0.3448343,
                                 0.85534906,
                                 0.92954571,
                                 0.18836955,
                                 -0.41475176,
                                 -0.13695355,
                                 0.21006834,
                                 0.043452675,
                                 -0.09564726,
                                 0.000354892813,
                                 0.03162417,
                                 -0.00667962023,
                                 -0.00605496058,
                                 0.00261296728,
                                 0.000325814671,
                                 -0.000356329759,
                                 5.5645514e-05};

      // and copy them into the u-vector and make them complex
      std::transform(coeffs.begin(), coeffs.end(), u.begin(), [](const double& val) {
        return complex(val) / sqrt(2.);
      });

    } // END: d18
      // d20
    else if (filterType == WaveletType::d20) {
      // the wavelet coefficients
      const Array<double> coeffs{
          0.03771716,    0.26612218,     0.74557507,      0.97362811,     0.39763774,
          -0.3533362,    -0.27710988,    0.18012745,      0.13160299,     -0.10096657,
          -0.04165925,   0.04696981,     0.00510043697,   -0.015179,      0.00197332536,
          0.00281768659, -0.00096994784, -0.000164709006, 0.000132354367, -1.875841e-05};

      // and copy them into the u-vector and make them complex
      std::transform(coeffs.begin(), coeffs.end(), u.begin(), [](const double& val) {
        return complex(val) / sqrt(2.);
      });

    } // END: d20

    // and get the v-vector from the u-vector
    const auto v{getother(u)};

    // compute u in the fourier domain
    auto fu{fft(u)};

    // conjugate fu
    std::transform(
        fu.begin(), fu.end(), fu.begin(), [](complex& val) { return std::conj(val); });

    // aind inverse transform to find util in the wavelet domain
    const auto util{ifft(fu)};

    // compute v in the fourier domain
    auto fv{fft(v)};

    // conjugate fv
    std::transform(
        fv.begin(), fv.end(), fv.begin(), [](complex& val) { return std::conj(val); });

    // aind inverse transform to find vtil in the wavelet domain
    const auto vtil{ifft(fv)};

    // and return them all as a tuple
    return std::make_tuple(u, v, util, vtil);

  } // END: filt

  /**
   * A scaled Wiener deconvolution in the Fourier domain
   *
   * We supply the fft of the observed signal (fSignal), fft of the impulse response
   * (fImpulse), the standard deviation of the noise (noiseSd) and a scaling constant
   *  (0<scaling<1) and it outputs the fft of deconvolved signal (fOutput) and the
   *  Fourier shrinkage multipler (multipler) which is needed for the Wavelet based
   *  deconvolution method. fImpulse and fSignal must be the same length.
   *
   *  @param fSignal    The frequency domain signal.
   *  @param fImpulse   The frequency domain impulse response.
   *  @param noiseSd    The standard deviation of the noise.
   *  @param scaling    The Wiener scaling parameter.
   *
   */
  auto
  fWienDec(const Array<complex>& fSignal,
           const Array<complex>& fImpulse,
           const double noiseSd,
           const double scaling) -> std::pair<Array<complex>, Array<complex>> {

    // the length of the input signal
    const auto N{fSignal.size()};

    // check that it is the same length as the impulse
    if (fSignal.size() != fImpulse.size()) {
      throw std::invalid_argument("fWienDec: fSignal and fImpulse are not the same size.");
    }

    // compute square of the L2 norm of fImpulse
    const double normSqImpulse{std::accumulate(
        fImpulse.begin(), fImpulse.end(), 0., [](const double& a, const complex& b) {
          return a + pow(std::abs(b), 2.);
        })};

    // the output vectors
    Array<complex> out(N, 0.);
    Array<complex> multiplier(N, 0.);

    // perform the deconvolution
    for (unsigned int i = 0; i < N; i++) {

      // perform a naive deconvolution
      const auto fNaive{fSignal[i] / fImpulse[i]};

      // a model for the original signal power to be supplied to the algorithm
      // Typically, fOriginal is same as the observed signal fSignal
      // We supply fOriginal = fSignal / ||fImpulse||_2
      // i.e. we normalize powOriginalSq by the mean of powImpulseSq
      const auto powerOriginalSq{pow(abs(fSignal[i]), 2) / normSqImpulse * N};

      // and the squared power
      const auto powerImpulseSq{pow(abs(fImpulse[i]), 2)};

      // Compute the Fourier multiplier
      multiplier[i] = powerImpulseSq /
                      (powerImpulseSq + scaling * N * pow(noiseSd, 2) / powerOriginalSq);

      // Perform the scaled Wiener Deconvolution
      out[i] = fNaive * multiplier[i];
    }

    // and return the pair
    return std::make_pair(out, multiplier);

  } // END: fWienDec

  /**
   * Compute the inner product of two vectors.
   *
   * @param A    The first input vector.
   * @param B    The second input vector.
   *
   * @returns    The complex dot product of A and B.
   */
  template <typename T>
  auto
  innerProduct(const Array<T>& A, const Array<T>& B) -> T {

    // check that A and B are the same size
    if (A.size() != B.size()) {
      throw std::invalid_argument("innerProduct: A and B are different sizes.");
    }

    // the variable that we use to accumulate the product
    T output = 0.;

    // the length of the input array
    const auto N{A.size()};

    // and loop over the array computing the dot product
    for (unsigned int i = 0; i < N; ++i) {
      output += conj(A[i]) * B[i];
    }

    // and return the output
    return output;
  }

  /**
   * Circularly shift a vector around an index.
   *
   * @param A        The array to shift.
   * @param index    The index to shift around.
   *
   */
  template <typename T>
  auto
  circShift(const Array<T>& A, const unsigned int index) -> Array<T> {

    // get the length of the input vector
    const auto N{A.size()};

    // check that the provided index is valid
    if (index > N) {
      throw std::invalid_argument("circShift: index is larger than array.");
    }

    // allocate the output vector
    Array<T> out(N, 0.);

    // now do the shift around the index
    for (unsigned int i = index; i < N; ++i) {
      out[i] = A[i - index];
    }

    for (unsigned int i = 0; i < index; ++i) {
      out[i] = A[N - index + i];
    }

    // and return the output vector
    return out;
  }

  /**
   * Apply thresholds to a wavelet transform.
   *
   * iven a p-th wavelet transform wt, rule and a
   * threshold vector of size (p+1), apply thresholds.
   *
   * @param wt                 The wavelet coefficients.
   * @param rule               The thresholding rule.
   * @param thresholdVector    The p+1 threshold.
   *
   * @returns The thresholded vector and the thresholded ratios.
   *
   */
  auto
  applyThreshold(const Array<complex>& wt,
                 const ThresholdRule rule,
                 const Array<double>& thresholdVector)
      -> std::pair<Array<complex>, Array<double>> {

    // get the length of the input vector
    const auto N{wt.size()};

    // the length of the thresholding vector
    const auto p{thresholdVector.size() - 1};

    // allocate the output vectors
    Array<complex> output(N, 0.);
    Array<double> ratioThresholded(N, 0.);

    // some loop parameters
    int starting = 0;
    int len      = N / 2;

    // for the first p finer levels
    for (unsigned int k = 0; k < (p + 1); ++k) {

      // the number of samples that we thresholded
      unsigned int thresholded{0};

      // where we end the loop
      const auto ending{k == p ? N : starting + len};

      // loop over the elements in the wavelet basis
      for (unsigned int l = starting; l < ending; ++l) {

        // if we want hard thresholding
        if (rule == ThresholdRule::Hard) {

          // if the weight is less than the threshold, set it to zero.
          if (abs(wt[l]) < thresholdVector[k]) {
            output[l]   = 0;
            thresholded = thresholded + 1;
          } else {
            output[l] = wt[l]; // save the wavelets
          }
        } // END: hard thresholding

        // we want soft thresholding
        else if (rule == ThresholdRule::Soft) {

          // if the wavelet is above threshold
          if (abs(wt[l]) >= thresholdVector[k]) {

            // works if the wavelet transform is real
            // for std::complex, multiply by the phase term instead of +1 or -1
            output[l] = (std::complex<double>((wt[l].real() > 0. ? 1. : -1.))) *
                        (std::abs(wt[l].real()) - thresholdVector[k]);

          }
          // if we are below threshold
          else {
            output[l]   = 0;
            thresholded = thresholded + 1;
          }
        }

      } // END: loop over l

      const auto Nj{k == p ? N - starting + 1 : len};

      // save the thresholded vector
      ratioThresholded[k] = (Nj - thresholded) / Nj;

      // for the next loop
      // lenght related to the next level wavelet coeffs
      len      = len / 2;
      starting = ending + 1;

    } // END: loop over k

    // and return the output vectors
    return std::make_pair(output, ratioThresholded);
  }

  // // get the basis matrix for p-th stage wavelet tansform
  // // level = 1, 2, ..., (p+1)
  // // (p+1)th level is the coarsest level
  // // u, v are the filters
  // // Structure of the basisMatrix ///////////
  // // basisMatrix is N*(p+1) length, but we are storing it as a linear array
  // // A sample basisMatrix: std::complex<double> sampleMat[(p+1)*N];
  // // The last row corresponds to the coarsest wavelet level
  // // To get the j-th row (j=0,...,p) of this matrix, print
  // // from (j*N) to (j*N + N) i.e. (for i=0; i<N; i++){ jRow[i] = sampleMat[j*N+i];}
  // // Use getRow(N, sampleMat, k, output) to get the k-th row
  auto
  getBasisMatrix(const Array<complex>& u, const Array<complex>& v, const unsigned int p)
      -> Matrix<complex> {

    // get the length of the input filters
    const auto N{u.size()};

    // allocate the U and V matrices
    Matrix<complex> U(p, Array<complex>(N, 0.));
    Matrix<complex> V(p, Array<complex>(N, 0.));

    // copy u and v into the first row of U and V
    std::copy(u.begin(), u.end(), U[0].begin());
    std::copy(v.begin(), v.end(), V[0].begin());

    // for U[k][], ,k=2,..,p
    for (unsigned int k = 2; k <= p; ++k) {

      // create copies of u and v
      Array<complex> u_folded(u);
      Array<complex> v_folded(v);

      // fold u, and v  k-1 times
      for (unsigned int j = 1; j <= (k - 1); ++j) {
        u_folded = fold(u_folded);
        v_folded = fold(v_folded);
      } // END: fold

      // Now we need to upsample k-1 times
      Array<complex> u_upped(u_folded);
      Array<complex> v_upped(v_folded);

      // ow we upsample k-1 times
      for (unsigned int j = 1; j <= (k - 1); ++j) {
        u_upped = upsample(u_upped);
        v_upped = upsample(v_upped);
      } // END: upsampling

      // copy folded and upped vector to matrix U, V
      std::copy(u_upped.begin(), u_upped.end(), U[k - 1].begin());
      std::copy(v_upped.begin(), v_upped.end(), V[k - 1].begin());

    } // END: for (unsigned int k...)

    // allocate the basic matrix
    Matrix<complex> basisMatrix(p + 1, Array<complex>(N, 0.));

    // the first row of Psi is v
    std::copy(v.begin(), v.end(), basisMatrix[0].begin());

    // mutable copies of u and v
    Array<complex> f_old(v.begin(), v.end());
    Array<complex> g_old(u.begin(), u.end());

    // fill in the other rows of Psi
    for (unsigned int j = 1; j < p; ++j) {

      // compute the convolution of u with the matrix
      const auto g_new{convolve(g_old, U[j])};
      const auto f_new{convolve(g_old, V[j])}; // this is from libWTools
      // IS THIS A BUG? RP. DB had both of these using g_old
      // const auto f_new{convolve(f_old, V[j])};

      // add this into the matrix
      std::copy(f_new.begin(), f_new.end(), basisMatrix[j].begin());

      // and update the contents of f_new and g_new
      f_old = f_new;
      g_old = g_new;

    } // END: loop over j

    // fill last row of the basisMatrix
    std::copy(g_old.begin(), g_old.end(), basisMatrix[p].begin());

    // and return the basis matrix
    return basisMatrix;

  } // END: getBasisMatrix

  /**
   * Get the k-th row of a matrix stored in a linear vector.
   *
   * @param rowsize     The size of each matrix row.
   * @param matrix      The matrix to access.
   * @param k           The index of the row to access.
   *
   * @returns           The k-th row of the matrix.
   *
   */
  template <typename T>
  auto
  getRow(const unsigned int rowsize, std::complex<double>* matrix, int k)
      -> Array<complex> {

    // allocate the output vector
    Array<complex> out(rowsize, 0.);

    // loop over the matrix
    for (unsigned int j = 0; j < rowsize; ++j) {
      out[j] = matrix[k * rowsize + j];
    }

    // and return the row
    return out;
  }

  // // feed the Fourier transform of signal fSignal,
  // // fft of impulse response fImpulse
  // // for p-th stage wavelet based deconvolution
  // // noise standard deviation
  // // scaling alpha_j, level dependent, j = 1, 2, ..., p+1
  // // wavelet threshold parameter vector rho_j, j= 1, ... , p+1 (usually, rho_j = 1 for
  // all j)
  // // thresholdRule - hard, soft
  // // store deconvolved signal in output
  auto
  wienForwd(const Array<complex>& fSignal,
            const Array<complex>& fImpulse,
            const Matrix<complex>& basisMatrix,
            const double noiseSd,
            const Array<double>& scaling,
            const Array<double>& rho,
            const ThresholdRule rule) -> std::pair<Array<complex>, Array<double>> {

    // the number of elements in the signal
    const auto N{fSignal.size()};

    // the number of stages in the wavelet decomposition
    const auto p{basisMatrix.size() - 1};

    // allocate the array for the wavelet coefficients
    Array<complex> wSignal(N, 0.);

    // and the thresholded noise standard deviations
    Array<complex> leakedNoiseSd(p + 1, 0.);

    // a loop index
    int startIndex = 0;

    // at j-th level
    for (unsigned int j = 1; j <= (p + 1); ++j) {

      // get the deconvolved and multiplier from a scaled Wiener
      const auto fDecMult{fWienDec(fSignal, fImpulse, noiseSd, scaling[j - 1])};
      const auto fDec{fDecMult.first}; // DANG: I miss C++17
      const auto multiplier{fDecMult.second};

      // prepare  beta(k), the estimate for the k-th wavelet coeff
      // this can also be computed using FWT and choosing the appropriate indices,
      // that would probably be faster
      const auto levelLength{j == (p + 1) ? N / pow(2, p) : N / pow(2, j)};

      // compute the stopping index of the current level
      const auto endIndex{j == (p + 1) ? N : startIndex + levelLength};

      // get the jth row of the basis matrix
      const auto jRow{basisMatrix[j - 1]};

      ///////// start segment ///////////////////////////
      Array<complex> beta(levelLength, 0.);

      // loop over the levels
      for (unsigned int k = 0; k < levelLength; ++k) {

        // compute the time-domain psi
        const auto Psi{j == (p + 1) ? circShift(jRow, pow(2., p) * k)
                                    : circShift(jRow, pow(2., j) * k)};

        // and go to the frequency domain
        const auto fPsi{fft(Psi)};

        // plancheral to get th j,k the coefficient of Decn
        // will it always be a real number?
        beta[k] = innerProduct(fPsi, fDec);

      } // END: loop over k

      // store beta in the wavelet transform
      // Here, we divide by N because
      // in Matlab, Plancheral holds with a factor of N
      // i.e. <a,b> = <ahat, bhat>/N
      // if we use fwt to get beta, we do not divide by N
      for (unsigned int l = startIndex; l < endIndex; ++l) {
        wSignal[l] = beta[l - startIndex] / std::complex<double>(N);
      }

      ////////// end segment /////////////////////////////

      //////////////////////////////////////////////////////
      //// Caution::
      //// need to redefine the arguments to
      //// involve util and vtil along with basisMatrix
      //// use fwt instead
      //// this whole for loop can be replaced by
      //// w = fwt(ifft(fDec))
      //// beta = w(starting:ending)
      //// without the division by N in the end
      //// but it needs util and vtil
      //////////////////////////////////////////////////////

      // update the the startIndex for the next loop
      startIndex = endIndex;

      // Now to compute the leaked noise variance
      Array<complex> firstPart(N, 0.);
      Array<complex> absMultiSq(N, 0.);

      // compute the FFT of the basis row
      const auto fBasis{fft(jRow)};

      // fill in firstPart and absMultiSq
      for (unsigned int l = 0; l < N; ++l) {

        // first part of the dot product
        // No need to work with fPsi
        // this quantity is independent of the
        // location parameter
        // fPsi won't probably be avilable here
        firstPart[l] = pow(abs(fBasis[l]) / abs(fImpulse[l]), 2.);

        // second part of the dot product
        absMultiSq[l] = pow(abs(multiplier[l]), 2.);

      } // END: loop over l
      // noise standard deviation is scalar for now
      // computing the leaked noise variance
      // at j-th level
      // in my matlab code: sigmal
      leakedNoiseSd[j - 1] =
          pow(pow(noiseSd, 2.) * real(innerProduct(firstPart, absMultiSq)) / N, 0.5);

    } // END: loop over j

    ////////////// Wavelet thresholding ////////////////

    // an array for the vector thresholds
    Array<double> thresholdVector(p + 1, 0.);

    // and fill in the thresholds
    for (unsigned int l = 0; l < (p + 1); ++l) {
      thresholdVector[l] = real(leakedNoiseSd[l]) * rho[l];
    }

    // return the wavelets and the thresholds
    return applyThreshold(wSignal, rule, thresholdVector);

  } // END: wienForwd

  /**
   * Deconvolve `response` out of `waveform`.
   *
   * Both waveforms must be the same-size and their length
   * must be a power of 2.
   *
   * @param waveform    The waveform to deconvolve.
   * @param response    The response to use.
   * @param p           The number of wavelet stages.
   * @param type        The wavelet family to use.
   * @param noisedSd    The standard deviation of the noise.
   * @param scaling     The (p+1)-length Wiener scaling.
   * @param rho         The (p+1) Fourier shrinkage parameters.
   * @param rule        The threshold rule to use.
   */
  auto
  deconvolve(const Array<double>& signal,
             const Array<double>& response,
             const unsigned int p,
             const WaveletType type,
             const double noiseSd,
             const Array<double>& scaling,
             const Array<double>& rho,
             const ThresholdRule rule) -> Array<double> {

    // check that scaling and rho are the right length
    if ((scaling.size() != p+1) || (rho.size() != p+1)) {
      throw std::invalid_argument("`scaling` and `rho` must be of length (p+1)");
    }

    // compute the FFT of the signal and the transform
    const auto fSignal{fft(signal)};
    const auto fResponse{fft(response)};

    // the length of the signals
    const auto N{signal.size()};

    // generate the filters
    const auto filters{filt(N, type)};

    // and get references to the u, v filters
    const auto u{std::get<0>(filters)};
    const auto v{std::get<1>(filters)};

    // generate the basis matrix
    const auto matrix{getBasisMatrix(u, v, p)};

    // do the deconvolution into the wavelet basis
    const auto wave_thresholds{
        wienForwd(fSignal, fResponse, matrix, noiseSd, scaling, rho, rule)};

    // extract references to the wavelet coefficients
    const auto wavelets{std::get<0>(wave_thresholds)};

    // the stopping dimension for the p-th stage IWT
    const unsigned int stoppingDim = N / pow(2., p);

    // do the inverse wavelet transform
    const auto wavelet_deconv{ifwt(wavelets, stoppingDim, u, v)};

    // and make deconvolved a real array
    Array<double> deconvolved(N, 0.);

    // fill it in with the transformed array
    std::transform(wavelet_deconv.cbegin(),
                   wavelet_deconv.cend(),
                   deconvolved.begin(),
                   [](const complex& val) -> double { return real(val); });

    // and return the deconvolved waveform
    return deconvolved;

  } // END: deconvolve

} // END namespace forward
