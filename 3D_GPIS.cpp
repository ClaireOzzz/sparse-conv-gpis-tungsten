#include "SparseConvolutionNoise.hpp"
#include "sampling/UniformPathSampler.hpp"

#include <sampling/Gaussian.hpp>

#define _USE_MATH_DEFINES
#include <cmath>
#include "MathUtil.hpp"


namespace Tungsten {

float sparseConvNoiseLateralScale(const Vec3f& p) const {
    return getKernelScale(p);
}

float nonStationarySplattingKernelScale(const Vec3f& p) const {
    if (_multiResolutionGrid)
        return 1.;
    else {
        return sparseConvNoiseLateralScale(p) / sparseConvNoiseMaxLateralScale();
    }
}

void getNonstationaryAniso3D(const Vec3f& p_world, std::shared_ptr<Eigen::Matrix3f>& aniso) const {
    if (_anisoField) {
        // The anisotropy is hard-coded here.
        // (Should be simple) future work: specify non-stationary anisotropy matrix similarly as SGGX
        float angle = (*_anisoField)(Vec3d(p_world)) * PI_HALF;
        Vec3f anisotropyAxis = Vec3f(_anisotropyOnAxis, 1.f / _anisotropyOnAxis, 1.f);
        aniso = std::make_shared<Eigen::Matrix3f>(compute_ansio_full<Eigen::Matrix3f>(angle, vec_conv<Eigen::Vector3f>(anisotropyAxis)));
    }
    else
        aniso = nullptr;
    return;
}

Eigen::Matrix3f getInvCovMtx(Vec3f ab, bool isCov, bool isIsotropic, float globalScale, float localScale, std::shared_ptr<Eigen::Matrix3f> aniso_inv) const {
    Eigen::Matrix3f invCovMtx;
    if (aniso_inv) {
        Eigen::Matrix3f invSigma = Eigen::Matrix3f::Identity();
        if (!isIsotropic) {
            if (!_use_aniso_mtx)
                invSigma.diagonal() << to_eigen3f(_l_aniso_inv);
            else
                invSigma = _world_to_local;
            invSigma /= globalScale;
        }
        invCovMtx = invSigma.transpose() * (*aniso_inv) * invSigma;
    }
    else {
        invCovMtx = Eigen::Matrix3f::Identity();
        if (!isIsotropic) {
            if (!_use_aniso_mtx)
                invCovMtx.diagonal() << to_eigen3f(_l_aniso_inv.cwiseProduct(_l_aniso_inv));
            else
                invCovMtx = _cov_mtx_inv; // = _world_to_local.transpose() * _world_to_local;
            invCovMtx /= sqr(globalScale);
        }
    }
    if (isCov)
        invCovMtx *= 0.5f;
    invCovMtx /= sqr(localScale);
    invCovMtx *= 0.5f;
    return invCovMtx;
}

float splattingKernel3DVal(Vec3f ab, bool isCov, bool isIsotropic, float globalScale, float localScale, std::shared_ptr<Eigen::Matrix3f> aniso_inv) const {
    Eigen::Matrix3f invCovMtx = getInvCovMtx(ab, isCov, isIsotropic, globalScale, localScale, aniso_inv);
    float absq = dist2_ab(to_eigen3f(ab), invCovMtx);
    return exp(-absq);
}

Vec3f splattingKernel3D1stGrad(Vec3f ab, bool isCov, bool isIsotropic, float globalScale, float localScale, std::shared_ptr<Eigen::Matrix3f> aniso_inv) const {
    float f = splattingKernel3DVal(ab, isCov, isIsotropic, globalScale, localScale, aniso_inv);
    Eigen::Matrix3f invCovMtx = getInvCovMtx(ab, isCov, isIsotropic, globalScale, localScale, aniso_inv);
    float dfdx = -2.f * to_eigen3f(ab).dot(invCovMtx.col(0)) * f;
    float dfdy = -2.f * to_eigen3f(ab).dot(invCovMtx.col(1)) * f;
    float dfdz = -2.f * to_eigen3f(ab).dot(invCovMtx.col(2)) * f;
    return Vec3f(dfdx, dfdy, dfdz);
}

Vec4f splattingKernel3D(Vec3f pa, Vec3f pb, bool isCov, bool isIsotropic, float globalScale, const Vec3f& p_world) const {
    // Put the logic here to prevent computing aniso_inv twice
    float localScale = nonStationarySplattingKernelScale(p_world);
    std::shared_ptr<Eigen::Matrix3f> aniso_inv;
    getNonstationaryAniso3D(p_world, aniso_inv);
    if (aniso_inv)
        *aniso_inv = aniso_inv->inverse();
    float val = splattingKernel3DVal(pa, pb, isCov, isIsotropic, globalScale, localScale, aniso_inv);
    Vec3f grad = splattingKernel3D1stGrad(pa, pb, isCov, isIsotropic, globalScale, localScale, aniso_inv);
    return Vec4f(val, grad.x(), grad.y(), grad.z());
}

float SquaredExponentialCovariance::splattingKernel3DVal(Vec3f ab, bool isCov, bool isIsotropic, float globalScale, float localScale, std::shared_ptr<Eigen::Matrix3f> aniso_inv) const {
    Eigen::Matrix3f invCovMtx = getInvCovMtx(ab, isCov, isIsotropic, globalScale, localScale, aniso_inv);
    float absq = dist2_ab(to_eigen3f(ab), invCovMtx);
    return exp(-absq);
}

Eigen::Matrix3f SquaredExponentialCovariance::getInvCovMtx(Vec3f ab, bool isCov, bool isIsotropic, float globalScale, float localScale, std::shared_ptr<Eigen::Matrix3f> aniso_inv) const {
    Eigen::Matrix3f invCovMtx;
    if (aniso_inv) {
        Eigen::Matrix3f invSigma = Eigen::Matrix3f::Identity();
        if (!isIsotropic) {
            if (!_use_aniso_mtx)
                invSigma.diagonal() << to_eigen3f(_l_aniso_inv);
            else
                invSigma = _world_to_local;
            invSigma /= globalScale;
        }
        invCovMtx = invSigma.transpose() * (*aniso_inv) * invSigma;
    }
    else {
        invCovMtx = Eigen::Matrix3f::Identity();
        if (!isIsotropic) {
            if (!_use_aniso_mtx)
                invCovMtx.diagonal() << to_eigen3f(_l_aniso_inv.cwiseProduct(_l_aniso_inv));
            else
                invCovMtx = _cov_mtx_inv; // = _world_to_local.transpose() * _world_to_local;
            invCovMtx /= sqr(globalScale);
        }
    }
    if (isCov)
        invCovMtx *= 0.5f;
    invCovMtx /= sqr(localScale);
    invCovMtx *= 0.5f;
    return invCovMtx;
}


Vec4f cell3D(const Vec3f& p_world, const Vec3u& ijk, const Vec3f& p, 
    const uint seed, UniformSampler& sampler, float impulseDensity, 
    float kernelRadius, float kernelSpatialScale) 
{
    sampler.set_state(MathUtil::xxhash32(Vec4u(ijk.z(), ijk.y(), ijk.x(), seed)) + 1u);

    // Replace the Poisson point distribution with fixed number of splats per cell for efficiency ([Tavernier et al. 2019])
    // uint number_of_impulses = poisson(impulse_density_per_kernel, sampler);
    uint number_of_impulses = uint(impulseDensity);

    Vec4f sum = Vec4f(0.f);
    for (uint k = 0u; k < number_of_impulses; ++k) {
        Vec3f p_i = Vec3f(sampler.next3D());
        float w_i = MathUtil::Bernoulli(sampler.next1D(), -1.f, 1.f, 0.5f); // Bernoulli distribution
        // float w_i = (sampler.next1f() * 2.0 - 1.0) * sqrt(3.0); // Uniform distribution
        Vec3f to_point = p - p_i;

        if (to_point.lengthSq() < 1.0) {
            // splattingKernel3D(Vec3f pa, Vec3f pb, bool isCov, bool isIsotropic, float globalScale, const Vec3f& p_world) {
            sum += w_i * _gp->_cov->splattingKernel3D(kernelRadius * p, kernelRadius * p_i, false, _isotropicSpace3DSampling, kernelSpatialScale, p_world);
        }
    }

    return sum;
}


Vec4f noise3D(const Vec3f& p_world, const Vec3f& p, const uint seed, UniformSampler& sampler, float impulseDensity, float kernelRadius, float kernelSpatialScale) const {
    Vec3f p_grid = p / kernelRadius;
    // Component-wise floor on Vec3f
    Vec3f floor_grid(std::floor(p_grid.x()), std::floor(p_grid.y()), std::floor(p_grid.z()));
    
    Vec3f frac = p_grid - floor_grid;
    Vec3i ijk = Vec3i(floor_grid);
    Vec4f sum = Vec4f(0.0f);

    for (int dx = -1; dx <= 1; ++dx)
        for (int dy = -1; dy <= 1; ++dy)
            for (int dz = -1; dz <= 1; ++dz)
                sum += cell3D(p_world, Vec3u(ijk + Vec3i(dx, dy, dz)), frac - Vec3f((float)dx, (float)dy, (float)dz), seed, sampler, impulseDensity, kernelRadius, kernelSpatialScale);
    return sum;
}

Vec3f filterWithZero(Vec3f input) {
    Vec3f output = input;
    if (std::isinf(output.x()) || std::isnan(output.x()))
        output.x() = 0;
    if (std::isinf(output.y()) || std::isnan(output.y()))
        output.y() = 0;
    if (std::isinf(output.z()) || std::isnan(output.z()))
        output.z() = 0;
    return output;
}

void fromJson(JsonPtr value, const Scene& scene) {
    StationaryCovariance::fromJson(value, scene);
    value.getField("sigma", _sigma);
    value.getField("lengthScale", _l);
    _l_conv = _l * sqrt(2.f) / 2;
    value.getField("aniso", _aniso);
    value.getField("useAnisoMtx", _use_aniso_mtx);
    if (!_use_aniso_mtx) {
        _l_aniso = Vec3f(_l_conv) * _aniso;
        _l_aniso_inv = Vec3f(1.0) / _l_aniso;
        _l_aniso_inv = filterWithZero(_l_aniso_inv);
        _local_to_world.diagonal() << to_eigen3f(_l_aniso);
        _world_to_local.diagonal() << to_eigen3f(_l_aniso_inv);
    }
    else {
        value.getField("anisoMtx", _aniso_mtx);
        _local_to_world = _l_conv * _aniso_mtx;
        _world_to_local = _local_to_world.inverse();
        _cov_mtx_inv = _world_to_local.transpose() * _world_to_local; // Σ^TΣ -> X^T(Σ^TΣ)X = (ΣX)^T(ΣX)
        // _cov_mtx =  _cov_mtx_inv.inverse(); // _local_to_world * _world_to_local.transpose()
        _cov_mtx_inv_determinant = _cov_mtx_inv.determinant();
    }

    _local_to_world_transpose = _local_to_world.transpose();
    _world_to_local_transpose = _world_to_local.transpose();
}

// TODO:: what is _gp? 
// std::shared_ptr<GaussianProcess> _gp;
// TODO:: what is _base? 
// float _base;
// TODO: what is coeff_3d?
// SparseConvConditioningCoefficients3D coeff_3D;
// TODO: what is isIdentity for?
Vec3f StationaryCovariance::transformPosDirWorldtoLocal(const Vec3f& posOrDir, const float localScale) const {
    return to_vec3f(_world_to_local * to_eigen3f(posOrDir) / localScale);
}

float SquaredExponentialCovariance::sparseConvNoiseVariance3D(float impulseDensity, float kernelRadius, bool isIdentity, float globalScale, float localScale) const {
    double impulseDensityUnitArea = impulseDensity / (kernelRadius * kernelRadius * kernelRadius);
    double covDeterminantSqrt = 1.0;
    if (!isIdentity) {
        if (!_use_aniso_mtx)
            covDeterminantSqrt = _l_aniso.x() * _l_aniso.y() * _l_aniso.z();
        else
            covDeterminantSqrt = 1.0 / sqrt(_cov_mtx_inv_determinant);
        covDeterminantSqrt *= pow(globalScale, 3);
    }
    covDeterminantSqrt *= pow(localScale, 3);
    double integralKernelSquared = pow(M_PI, 1.5) * covDeterminantSqrt;
    return impulseDensityUnitArea * integralKernelSquared;
}

// bool _activateConditioning;
Vec4f evaluateNoise3DIsotropicRayNormalized(const Vec3f& p, const Vec3f& rayDir, const uint seed, UniformSampler& sampler, float impulseDensity, float kernelRadius, float kernelSpatialScale, bool conditioning) {
    // Note: Only isotropic space supports global model, but not isotropic ray space. We could probably reconcile this with more spatial transform.
    // So the current code only achieves global model through the world space implementation (evaluateNoise3DNormalized)

    // Transform the point from world space into isotropic ray space
    Vec3f ray_dir_iso = _gp->_cov->transformPosDirWorldtoLocal(rayDir, 1.0).normalized();
    TangentFrame coord = TangentFrame(ray_dir_iso);
    Vec3f p_iso = _gp->_cov->transformPosDirWorldtoLocal(p, kernelSpatialScale);
    Vec3f p_iso_ray = coord.toLocal(p_iso);
    int additional_seed = floor(log(kernelSpatialScale) / log(_base)); // For multi-resolution noise
    // Evaluate the noise in isotropic ray space
    Vec4f noise_iso_ray = noise3D(p, p_iso_ray, seed + additional_seed, sampler, impulseDensity, kernelRadius, 1.0);

    Vec3f grad_iso_ray = noise_iso_ray.yzw();
    // Transform the gradient from isotropic ray space back to world space
    Vec3f grad_iso = Vec3f(coord.toGlobal(Vec3f(grad_iso_ray)));
    Vec3f grad_world = _gp->_cov->transformGradLocaltoWorld(grad_iso, kernelSpatialScale);
    Vec4f noise_world = Vec4f(noise_iso_ray.x(), grad_world.x(), grad_world.y(), grad_world.z());
    float normalization_factor = sqrt(_gp->_cov->sparseConvNoiseVariance3D(p, impulseDensity, kernelRadius, true, 1.0));    
    noise_world /= normalization_factor;
    if (_activateConditioning && conditioning) {
        Vec3f origin_iso = _gp->_cov->transformPosDirWorldtoLocal(coeff_3D.ray_origin, kernelSpatialScale);
        Vec3f origin_iso_ray = coord.toLocal(origin_iso);
        Vec4f noise_delta_iso_ray = _gp->_cov->splattingKernel3D(p_iso_ray, origin_iso_ray, true, true, 1.0, p) * coeff_3D.value_scale + _gp->_cov->splattingKernel3DGrad(p_iso_ray, origin_iso_ray, coeff_3D.gradient_scale, true, true, 1.0, p);
        Vec3f grad_delta_iso_ray = noise_delta_iso_ray.yzw();
        Vec3f grad_delta_iso = Vec3f(coord.toGlobal(Vec3f(grad_delta_iso_ray)));
        Vec3f grad_delta_world = _gp->_cov->transformGradLocaltoWorld(grad_delta_iso, kernelSpatialScale);
        noise_world += Vec4f(noise_delta_iso_ray.x(), grad_delta_world.x(), grad_delta_world.y(), grad_delta_world.z());
    }
    return noise_world;
}

Vec4f SparseConvolutionNoiseRealization::evaluateNoise3DIsotropicNormalized(const Vec3f& p, const Vec3f& rayDir, const uint seed, UniformSampler& sampler, float impulseDensity, float kernelRadius, float kernelSpatialScale, bool conditioning) {
    // Transform the point from world space into isotropic space
    Vec3f p_iso = _gp->_cov->transformPosDirWorldtoLocal(p, kernelSpatialScale);
    int additional_seed = floor(log(kernelSpatialScale) / log(_base)); // For multi-resolution noise
    // Evaluate the noise in isotropic space
    Vec4f noise_iso = noise3D(p, p_iso, seed + additional_seed, sampler, impulseDensity, kernelRadius, 1.0);
    Vec3f grad_iso = noise_iso.yzw();
    // Transform the gradient from isotropic space back to world space
    Vec3f grad_world = _gp->_cov->transformGradLocaltoWorld(grad_iso, kernelSpatialScale);
    Vec4f noise_world = Vec4f(noise_iso.x(), grad_world.x(), grad_world.y(), grad_world.z());
    float normalization_factor = sqrt(_gp->_cov->sparseConvNoiseVariance3D(p, impulseDensity, kernelRadius, true, 1.0));
    noise_world /= normalization_factor;
    if (_activateConditioning && conditioning) {
        Vec3f origin_iso = _gp->_cov->transformPosDirWorldtoLocal(coeff_3D.ray_origin, kernelSpatialScale);
        Vec4f noise_delta_iso = _gp->_cov->splattingKernel3D(p_iso, origin_iso, true, true, 1.0, p) * coeff_3D.value_scale + _gp->_cov->splattingKernel3DGrad(p_iso, origin_iso, coeff_3D.gradient_scale, true, true, 1.0, p);
        Vec3f grad_delta_iso = noise_delta_iso.yzw();
        Vec3f grad_delta_world = _gp->_cov->transformGradLocaltoWorld(grad_delta_iso, kernelSpatialScale);
        noise_world += Vec4f(noise_delta_iso.x(), grad_delta_world.x(), grad_delta_world.y(), grad_delta_world.z());
    }
    return noise_world;
}


inline Vec4f SparseConvolutionNoiseRealization::evaluateNoise3DIsotropicNormalizedSelect(const Vec3f& p, const Vec3f& rayDir, const uint seed, UniformSampler& sampler, float impulseDensity, float kernelRadius, float kernelSpatialScale, bool conditioning) {
    if (_isotropicRaySpace3DSampling) // Evaluate 3D noise in isotropic ray space
        return evaluateNoise3DIsotropicRayNormalized(p, rayDir, seed, sampler, impulseDensity, kernelRadius, kernelSpatialScale, conditioning);
    else // Evaluate 3D noise in isotropic space
        return evaluateNoise3DIsotropicNormalized(p, rayDir, seed, sampler, impulseDensity, kernelRadius, kernelSpatialScale, conditioning);
}

Vec4f SparseConvolutionNoiseRealization::evaluateNoise3D(const Vec3f& p, const Vec3f& rayDir, const uint seed, UniformSampler& sampler, bool conditioning) {
    if (!_isotropicSpace3DSampling) {
        if (!_multiResolutionGrid) {
            float kernelSpatialScale = _gp->_cov->worldSamplingSpatialScale();
            return evaluateNoise3DNormalized(p, seed, sampler, _impulseDensity, _gp->_cov->splattingKernelRadius(false, 1.0), kernelSpatialScale, conditioning);
        }
        else {
            Vec4f info = kernelScaleLevelRatio(p);
            Vec4f noise_low = evaluateNoise3DNormalized(p, seed, sampler, _impulseDensity, _gp->_cov->splattingKernelRadius(false, info.x()), info.x(), conditioning);
            Vec4f noise_high = evaluateNoise3DNormalized(p, seed, sampler, _impulseDensity, _gp->_cov->splattingKernelRadius(false, info.y()), info.y(), conditioning);
            return info.z() * noise_low + info.w() * noise_high;
        }
    }
    else {
        if (!_multiResolutionGrid)
        // this is the one we want, and that prints
            return evaluateNoise3DIsotropicNormalizedSelect(p, rayDir, seed, sampler, _impulseDensity, _gp->_cov->splattingKernelRadius(true, 1.0), 1.0, conditioning);
        else {
            Vec4f info = kernelScaleLevelRatio(p);
            Vec4f noise_low = evaluateNoise3DIsotropicNormalizedSelect(p, rayDir, seed, sampler, _impulseDensity, _gp->_cov->splattingKernelRadius(true, 1.0), info.x(), conditioning);
            Vec4f noise_high = evaluateNoise3DIsotropicNormalizedSelect(p, rayDir, seed, sampler, _impulseDensity, _gp->_cov->splattingKernelRadius(true, 1.0), info.y(), conditioning);
            return info.z() * noise_low + info.w() * noise_high;
        }
    }
}


}