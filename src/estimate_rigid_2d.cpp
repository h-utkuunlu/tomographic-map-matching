/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include <tomographic_map_matching/estimate_rigid_2d.hpp>

using namespace cv;

namespace map_matcher {

int
RANSACUpdateNumIters(double p, double ep, int modelPoints, int maxIters)
{
  if (modelPoints <= 0)
    CV_Error(Error::StsOutOfRange, "the number of model points should be positive");

  p = MAX(p, 0.);
  p = MIN(p, 1.);
  ep = MAX(ep, 0.);
  ep = MIN(ep, 1.);

  // avoid inf's & nan's
  double num = MAX(1. - p, DBL_MIN);
  double denom = 1. - std::pow(1. - ep, modelPoints);
  if (denom < DBL_MIN)
    return 0;

  num = std::log(num);
  denom = std::log(denom);

  return denom >= 0 || -num >= maxIters * (-denom) ? maxIters : cvRound(num / denom);
}

class RANSACPointSetRegistrator : public PointSetRegistrator
{
public:
  RANSACPointSetRegistrator(const Ptr<PointSetRegistrator::Callback>& _cb =
                              Ptr<PointSetRegistrator::Callback>(),
                            int _modelPoints = 0,
                            double _threshold = 0,
                            double _confidence = 0.99,
                            int _maxIters = 1000)
    : cb(_cb)
    , modelPoints(_modelPoints)
    , threshold(_threshold)
    , confidence(_confidence)
    , maxIters(_maxIters)
  {
  }

  int findInliers(const Mat& m1,
                  const Mat& m2,
                  const Mat& model,
                  Mat& err,
                  Mat& mask,
                  double thresh) const
  {
    cb->computeError(m1, m2, model, err);
    mask.create(err.size(), CV_8U);

    CV_Assert(err.isContinuous() && err.type() == CV_32F && mask.isContinuous() &&
              mask.type() == CV_8U);
    const float* errptr = err.ptr<float>();
    uchar* maskptr = mask.ptr<uchar>();
    float t = (float)(thresh * thresh);
    int i, n = (int)err.total(), nz = 0;
    for (i = 0; i < n; i++) {
      int f = errptr[i] <= t;
      maskptr[i] = (uchar)f;
      nz += f;
    }
    return nz;
  }

  bool getSubset(const Mat& m1,
                 const Mat& m2,
                 Mat& ms1,
                 Mat& ms2,
                 RNG& rng,
                 int maxAttempts = 1000) const
  {
    cv::AutoBuffer<int> _idx(modelPoints);
    int* idx = _idx.data();

    const int d1 = m1.channels() > 1 ? m1.channels() : m1.cols;
    const int d2 = m2.channels() > 1 ? m2.channels() : m2.cols;

    int esz1 = (int)m1.elemSize1() * d1;
    int esz2 = (int)m2.elemSize1() * d2;
    CV_Assert((esz1 % sizeof(int)) == 0 && (esz2 % sizeof(int)) == 0);
    esz1 /= sizeof(int);
    esz2 /= sizeof(int);

    const int count = m1.checkVector(d1);
    const int count2 = m2.checkVector(d2);
    CV_Assert(count >= modelPoints && count == count2);

    const int* m1ptr = m1.ptr<int>();
    const int* m2ptr = m2.ptr<int>();

    ms1.create(modelPoints, 1, CV_MAKETYPE(m1.depth(), d1));
    ms2.create(modelPoints, 1, CV_MAKETYPE(m2.depth(), d2));

    int* ms1ptr = ms1.ptr<int>();
    int* ms2ptr = ms2.ptr<int>();

    for (int iters = 0; iters < maxAttempts; ++iters) {
      int i;

      for (i = 0; i < modelPoints; ++i) {
        int idx_i;

        for (idx_i = rng.uniform(0, count); std::find(idx, idx + i, idx_i) != idx + i;
             idx_i = rng.uniform(0, count)) {
        }

        idx[i] = idx_i;

        for (int k = 0; k < esz1; ++k)
          ms1ptr[i * esz1 + k] = m1ptr[idx_i * esz1 + k];

        for (int k = 0; k < esz2; ++k)
          ms2ptr[i * esz2 + k] = m2ptr[idx_i * esz2 + k];
      }

      if (cb->checkSubset(ms1, ms2, i))
        return true;
    }

    return false;
  }

  bool run(InputArray _m1,
           InputArray _m2,
           OutputArray _model,
           OutputArray _mask) const CV_OVERRIDE
  {
    bool result = false;
    Mat m1 = _m1.getMat(), m2 = _m2.getMat();
    Mat err, mask, model, bestModel, ms1, ms2;

    int iter, niters = MAX(maxIters, 1);
    int d1 = m1.channels() > 1 ? m1.channels() : m1.cols;
    int d2 = m2.channels() > 1 ? m2.channels() : m2.cols;
    int count = m1.checkVector(d1), count2 = m2.checkVector(d2), maxGoodCount = 0;

    RNG rng((uint64)-1);

    CV_Assert(cb);
    CV_Assert(confidence > 0 && confidence < 1);

    CV_Assert(count >= 0 && count2 == count);
    if (count < modelPoints)
      return false;

    Mat bestMask0, bestMask;

    if (_mask.needed()) {
      _mask.create(count, 1, CV_8U, -1, true);
      bestMask0 = bestMask = _mask.getMat();
      CV_Assert((bestMask.cols == 1 || bestMask.rows == 1) &&
                (int)bestMask.total() == count);
    } else {
      bestMask.create(count, 1, CV_8U);
      bestMask0 = bestMask;
    }

    if (count == modelPoints) {
      if (cb->runKernel(m1, m2, bestModel) <= 0)
        return false;
      bestModel.copyTo(_model);
      bestMask.setTo(Scalar::all(1));
      return true;
    }

    for (iter = 0; iter < niters; iter++) {
      int i, nmodels;
      if (count > modelPoints) {
        bool found = getSubset(m1, m2, ms1, ms2, rng, 10000);
        if (!found) {
          if (iter == 0)
            return false;
          break;
        }
      }

      nmodels = cb->runKernel(ms1, ms2, model);
      if (nmodels <= 0)
        continue;
      CV_Assert(model.rows % nmodels == 0);
      Size modelSize(model.cols, model.rows / nmodels);

      for (i = 0; i < nmodels; i++) {
        Mat model_i = model.rowRange(i * modelSize.height, (i + 1) * modelSize.height);
        int goodCount = findInliers(m1, m2, model_i, err, mask, threshold);

        if (goodCount > MAX(maxGoodCount, modelPoints - 1)) {
          std::swap(mask, bestMask);
          model_i.copyTo(bestModel);
          maxGoodCount = goodCount;
          niters = RANSACUpdateNumIters(
            confidence, (double)(count - goodCount) / count, modelPoints, niters);
        }
      }
    }

    if (maxGoodCount > 0) {
      if (bestMask.data != bestMask0.data) {
        if (bestMask.size() == bestMask0.size())
          bestMask.copyTo(bestMask0);
        else
          transpose(bestMask, bestMask0);
      }
      bestModel.copyTo(_model);
      result = true;
    } else
      _model.release();

    return result;
  }

  void setCallback(const Ptr<PointSetRegistrator::Callback>& _cb) CV_OVERRIDE
  {
    cb = _cb;
  }

  Ptr<PointSetRegistrator::Callback> cb;
  int modelPoints;
  double threshold;
  double confidence;
  int maxIters;
};

class LMeDSPointSetRegistrator : public RANSACPointSetRegistrator
{
public:
  LMeDSPointSetRegistrator(const Ptr<PointSetRegistrator::Callback>& _cb =
                             Ptr<PointSetRegistrator::Callback>(),
                           int _modelPoints = 0,
                           double _confidence = 0.99,
                           int _maxIters = 1000)
    : RANSACPointSetRegistrator(_cb, _modelPoints, 0, _confidence, _maxIters)
  {
  }

  bool run(InputArray _m1,
           InputArray _m2,
           OutputArray _model,
           OutputArray _mask) const CV_OVERRIDE
  {
    const double outlierRatio = 0.45;
    bool result = false;
    Mat m1 = _m1.getMat(), m2 = _m2.getMat();
    Mat ms1, ms2, err, errf, model, bestModel, mask, mask0;

    int d1 = m1.channels() > 1 ? m1.channels() : m1.cols;
    int d2 = m2.channels() > 1 ? m2.channels() : m2.cols;
    int count = m1.checkVector(d1), count2 = m2.checkVector(d2);
    double minMedian = DBL_MAX;

    RNG rng((uint64)-1);

    CV_Assert(cb);
    CV_Assert(confidence > 0 && confidence < 1);

    CV_Assert(count >= 0 && count2 == count);
    if (count < modelPoints)
      return false;

    if (_mask.needed()) {
      _mask.create(count, 1, CV_8U, -1, true);
      mask0 = mask = _mask.getMat();
      CV_Assert((mask.cols == 1 || mask.rows == 1) && (int)mask.total() == count);
    }

    if (count == modelPoints) {
      if (cb->runKernel(m1, m2, bestModel) <= 0)
        return false;
      bestModel.copyTo(_model);
      mask.setTo(Scalar::all(1));
      return true;
    }

    int iter,
      niters = RANSACUpdateNumIters(confidence, outlierRatio, modelPoints, maxIters);
    niters = MAX(niters, 3);

    for (iter = 0; iter < niters; iter++) {
      int i, nmodels;
      if (count > modelPoints) {
        bool found = getSubset(m1, m2, ms1, ms2, rng);
        if (!found) {
          if (iter == 0)
            return false;
          break;
        }
      }

      nmodels = cb->runKernel(ms1, ms2, model);
      if (nmodels <= 0)
        continue;

      CV_Assert(model.rows % nmodels == 0);
      Size modelSize(model.cols, model.rows / nmodels);

      for (i = 0; i < nmodels; i++) {
        Mat model_i = model.rowRange(i * modelSize.height, (i + 1) * modelSize.height);
        cb->computeError(m1, m2, model_i, err);
        if (err.depth() != CV_32F)
          err.convertTo(errf, CV_32F);
        else
          errf = err;
        CV_Assert(errf.isContinuous() && errf.type() == CV_32F &&
                  (int)errf.total() == count);
        std::nth_element(
          errf.ptr<int>(), errf.ptr<int>() + count / 2, errf.ptr<int>() + count);
        double median = errf.at<float>(count / 2);

        if (median < minMedian) {
          minMedian = median;
          model_i.copyTo(bestModel);
        }
      }
    }

    if (minMedian < DBL_MAX) {
      double sigma =
        2.5 * 1.4826 * (1 + 5. / (count - modelPoints)) * std::sqrt(minMedian);
      sigma = MAX(sigma, 0.001);

      count = findInliers(m1, m2, bestModel, err, mask, sigma);
      if (_mask.needed() && mask0.data != mask.data) {
        if (mask0.size() == mask.size())
          mask.copyTo(mask0);
        else
          transpose(mask, mask0);
      }
      bestModel.copyTo(_model);
      result = count >= modelPoints;
    } else
      _model.release();

    return result;
  }
};

Ptr<PointSetRegistrator>
createRANSACPointSetRegistrator(const Ptr<PointSetRegistrator::Callback>& _cb,
                                int _modelPoints,
                                double _threshold,
                                double _confidence,
                                int _maxIters)
{
  return Ptr<PointSetRegistrator>(new RANSACPointSetRegistrator(
    _cb, _modelPoints, _threshold, _confidence, _maxIters));
}

Ptr<PointSetRegistrator>
createLMeDSPointSetRegistrator(const Ptr<PointSetRegistrator::Callback>& _cb,
                               int _modelPoints,
                               double _confidence,
                               int _maxIters)
{
  return Ptr<PointSetRegistrator>(
    new LMeDSPointSetRegistrator(_cb, _modelPoints, _confidence, _maxIters));
}

class Rigid2DEstimatorCallback : public PointSetRegistrator::Callback
{
public:
  int runKernel(InputArray _m1, InputArray _m2, OutputArray _model) const CV_OVERRIDE
  {
    Mat m1 = _m1.getMat(), m2 = _m2.getMat();
    const Point2f* from = m1.ptr<Point2f>();
    const Point2f* to = m2.ptr<Point2f>();
    _model.create(2, 3, CV_64F);
    Mat M_mat = _model.getMat();
    double* M = M_mat.ptr<double>();

    // we need only 2 points to estimate transform
    double x1 = from[0].x;
    double y1 = from[0].y;
    double x2 = from[1].x;
    double y2 = from[1].y;

    double X1 = to[0].x;
    double Y1 = to[0].y;
    double X2 = to[1].x;
    double Y2 = to[1].y;

    /*
    we are solving AS = B
        | x1 -y1 1 0 |
        | y1  x1 0 1 |
    A = | x2 -y2 1 0 |
        | y2  x2 0 1 |
    B = (X1, Y1, X2, Y2).t()
    we solve that analytically
    */
    double d = 1. / ((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));

    // solution vector
    double S0 = d * ((X1 - X2) * (x1 - x2) + (Y1 - Y2) * (y1 - y2));
    double S1 = d * ((Y1 - Y2) * (x1 - x2) - (X1 - X2) * (y1 - y2));
    double S2 = d * ((Y1 - Y2) * (x1 * y2 - x2 * y1) - (X1 * y2 - X2 * y1) * (y1 - y2) -
                     (X1 * x2 - X2 * x1) * (x1 - x2));
    double S3 = d * (-(X1 - X2) * (x1 * y2 - x2 * y1) -
                     (Y1 * x2 - Y2 * x1) * (x1 - x2) - (Y1 * y2 - Y2 * y1) * (y1 - y2));

    // remove scale parameter from S0 and S1
    double theta = std::atan2(S1, S0);

    // set model, rotation part is antisymmetric
    M[0] = M[4] = std::cos(theta);
    M[1] = -std::sin(theta);
    M[2] = S2;
    M[3] = std::sin(theta);
    M[5] = S3;
    return 1;
  }
  void computeError(InputArray _m1,
                    InputArray _m2,
                    InputArray _model,
                    OutputArray _err) const CV_OVERRIDE
  {
    Mat m1 = _m1.getMat(), m2 = _m2.getMat(), model = _model.getMat();
    const Point2f* from = m1.ptr<Point2f>();
    const Point2f* to = m2.ptr<Point2f>();
    const double* F = model.ptr<double>();

    int count = m1.checkVector(2);
    CV_Assert(count > 0);

    _err.create(count, 1, CV_32F);
    Mat err = _err.getMat();
    float* errptr = err.ptr<float>();
    // transform matrix to floats
    float F0 = (float)F[0], F1 = (float)F[1], F2 = (float)F[2];
    float F3 = (float)F[3], F4 = (float)F[4], F5 = (float)F[5];

    for (int i = 0; i < count; i++) {
      const Point2f& f = from[i];
      const Point2f& t = to[i];

      float a = F0 * f.x + F1 * f.y + F2 - t.x;
      float b = F3 * f.x + F4 * f.y + F5 - t.y;

      errptr[i] = a * a + b * b;
    }
  }

  bool checkSubset(InputArray _ms1, InputArray _ms2, int count) const CV_OVERRIDE
  {
    Mat ms1 = _ms1.getMat();
    Mat ms2 = _ms2.getMat();
    // check collinearity and also check that points are too close
    return !haveCollinearPoints(ms1, count) && !haveCollinearPoints(ms2, count);
  }
};

class Rigid2DRefineCallback : public LMSolver::Callback
{
public:
  Rigid2DRefineCallback(InputArray _src, InputArray _dst)
  {
    src = _src.getMat();
    dst = _dst.getMat();
  }

  bool compute(InputArray _param, OutputArray _err, OutputArray _Jac) const CV_OVERRIDE
  {
    int i, count = src.checkVector(2);
    Mat param = _param.getMat();
    _err.create(count * 2, 1, CV_64F);
    Mat err = _err.getMat(), J;
    if (_Jac.needed()) {
      _Jac.create(count * 2, param.rows, CV_64F);
      J = _Jac.getMat();
      CV_Assert(J.isContinuous() && J.cols == 3);
    }

    const Point2f* M = src.ptr<Point2f>();
    const Point2f* m = dst.ptr<Point2f>();
    const double* h = param.ptr<double>();
    double* errptr = err.ptr<double>();
    double* Jptr = J.data ? J.ptr<double>() : 0;

    for (i = 0; i < count; i++) {
      double Mx = M[i].x, My = M[i].y;
      double xi = std::cos(h[0]) * Mx - std::sin(h[0]) * My + h[1];
      double yi = std::sin(h[0]) * Mx + std::cos(h[0]) * My + h[2];
      errptr[i * 2] = xi - m[i].x;
      errptr[i * 2 + 1] = yi - m[i].y;

      /*
      Jacobian should be:
          {-x.sin(t) -y.cos(t), 1, 0}
          {x.cos(t) - y.sin(t), 0, 1}
      */
      if (Jptr) {
        Jptr[0] = -Mx * std::sin(h[0]) - My * std::cos(h[0]);
        Jptr[1] = 1.;
        Jptr[2] = 0.;
        Jptr[3] = Mx * std::cos(h[0]) - My * std::sin(h[0]);
        Jptr[4] = 0.;
        Jptr[5] = 1.;
        Jptr += 3 * 2;
      }
    }

    return true;
  }

  Mat src, dst;
};

Mat
estimateRigid2D(InputArray _from,
                InputArray _to,
                OutputArray _inliers,
                const int method,
                const double ransacReprojThreshold,
                const size_t maxIters,
                const double confidence,
                const size_t refineIters)
{
  Mat from = _from.getMat(), to = _to.getMat();
  const int count = from.checkVector(2);
  bool result = false;
  Mat H;

  CV_Assert(count >= 0 && to.checkVector(2) == count);

  if (from.type() != CV_32FC2 || to.type() != CV_32FC2) {
    Mat tmp1, tmp2;
    from.convertTo(tmp1, CV_32FC2);
    from = tmp1;
    to.convertTo(tmp2, CV_32FC2);
    to = tmp2;
  } else {
    // avoid changing of inputs in compressElems() call
    from = from.clone();
    to = to.clone();
  }

  // convert to N x 1 vectors
  from = from.reshape(2, count);
  to = to.reshape(2, count);

  Mat inliers;
  if (_inliers.needed()) {
    _inliers.create(count, 1, CV_8U, -1, true);
    inliers = _inliers.getMat();
  }

  // run robust estimation
  Ptr<PointSetRegistrator::Callback> cb = makePtr<Rigid2DEstimatorCallback>();
  if (method == RANSAC)
    result = createRANSACPointSetRegistrator(
               cb, 2, ransacReprojThreshold, confidence, static_cast<int>(maxIters))
               ->run(from, to, H, inliers);
  else if (method == LMEDS)
    result =
      createLMeDSPointSetRegistrator(cb, 2, confidence, static_cast<int>(maxIters))
        ->run(from, to, H, inliers);
  else
    CV_Error(Error::StsBadArg, "Unknown or unsupported robust estimation method");

  if (result && count > 2 && refineIters) {
    // reorder to start with inliers
    compressElems(from.ptr<Point2f>(), inliers.ptr<uchar>(), 1, count);
    int inliers_count =
      compressElems(to.ptr<Point2f>(), inliers.ptr<uchar>(), 1, count);
    if (inliers_count > 0) {
      Mat src = from.rowRange(0, inliers_count);
      Mat dst = to.rowRange(0, inliers_count);
      // H is
      //     cos(theta) -sin(theta) tx
      //     sin(theta)  cos(theta) ty
      // Hvec model for LevMarq is
      //     (theta, tx, ty)
      double* Hptr = H.ptr<double>();
      double angle = std::atan2(Hptr[3], Hptr[0]);
      double Hvec_buf[3] = { angle, Hptr[2], Hptr[5] };
      Mat Hvec(3, 1, CV_64F, Hvec_buf);
      LMSolver::create(makePtr<Rigid2DRefineCallback>(src, dst),
                       static_cast<int>(refineIters))
        ->run(Hvec);
      // update H with refined parameters
      Hptr[0] = Hptr[4] = std::cos(Hvec_buf[0]);
      Hptr[1] = -std::sin(Hvec_buf[0]);
      Hptr[2] = Hvec_buf[1];
      Hptr[3] = std::sin(Hvec_buf[0]);
      Hptr[5] = Hvec_buf[2];
    }
  }

  if (!result) {
    H.release();
    if (_inliers.needed()) {
      inliers = Mat::zeros(count, 1, CV_8U);
      inliers.copyTo(_inliers);
    }
  }

  return H;
}
}
