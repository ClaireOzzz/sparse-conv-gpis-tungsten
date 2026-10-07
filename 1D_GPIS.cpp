

Vec2f SparseConvolutionNoiseRealization::cell1D(const Vec3f& p_world, const Vec3f& ray_dir_world, const uint i, const float t, const uint seed, UniformSampler& sampler, float impulseDensity, float kernelRadius) const {
    sampler.set_state(MathUtil::xxhash32(Vec2u(i, seed)) + 1u);

    // Replace the Poisson point distribution with fixed number of splats per cell for efficiency ([Tavernier et al. 2019])
    // uint number_of_impulses = poisson(impulse_density_per_kernel, sampler);
    uint number_of_impulses = uint(impulseDensity);

    Vec2f sum = Vec2f(0.f);
    for (uint k = 0u; k < number_of_impulses; ++k) {
        float t_i = sampler.next1D();
        float w_i = MathUtil::Bernoulli(sampler.next1D(), -1.f, 1.f, 0.5f); // Bernoulli distribution
        // float w_i = (sampler.next1f() * 2.0 - 1.0) * sqrt(3.0); // Uniform distribution
        float to_point = t - t_i;

        if (to_point * to_point < 1.0) {
            // Note: take the length scale parameter from the query location (not the kernel center)
            sum += w_i * _gp->_cov->splattingKernel1D(kernelRadius * t, kernelRadius * t_i, p_world, ray_dir_world);
        }
    }
    return sum;
}

Vec2f SparseConvolutionNoiseRealization::noise1D(const Vec3f& p_world, const Vec3f& ray_dir_world, const float t, const uint seed, UniformSampler& sampler, float impulseDensity, float kernelRadius) const {
    float t_grid = t / kernelRadius;
    float frac = t_grid - floor(t_grid);
    int i = int(floor(t_grid));
    Vec2f sum = Vec2f(0.0);

    for (int dx = -1; dx <= 1; ++dx)
        sum += cell1D(p_world, ray_dir_world, uint(i + dx), frac - dx, seed, sampler, impulseDensity, kernelRadius);
    return sum;
}

Vec4f SparseConvolutionNoiseRealization::evaluateNoise1DNormalized(const Vec3f& p, const float t, const Vec3f& rayDir, const uint seed, UniformSampler& sampler, float impulseDensity, float kernelRadius, float kernelSpatialScale, bool conditioning, bool multiResLowLevel) {
    // The logic is similar to evaluateNoise3DIsotropicRayNormalized, but evaluating 1D noise along the ray in isotropic ray space
    Vec3f ray_dir_iso = _gp->_cov->transformPosDirWorldtoLocal(rayDir, 1.0).normalized();
    TangentFrame coord = TangentFrame(ray_dir_iso);
    Vec3f p_iso = _gp->_cov->transformPosDirWorldtoLocal(p, kernelSpatialScale);
    Vec3f p_iso_ray = coord.toLocal(p_iso);
    int additional_seed = floor(log(kernelSpatialScale) / log(_base)); // For multi-resolution noise

    Vec2f noise = noise1D(p, rayDir, p_iso_ray.z(), seed + additional_seed, sampler, impulseDensity, kernelRadius);
    float normalization_factor = sqrt(_gp->_cov->sparseConvNoiseVariance1D(p, impulseDensity, kernelRadius));
    noise /= normalization_factor;

    // Only need to account for non-zero xy components when considering the correlation with the previous vertex (see comments in evaluateGradientNoise1D(..))
    float grad_condition_splat_x = 0.f;
    float grad_condition_splat_y = 0.f;
    if (_activateConditioning && conditioning) {
        float origin_scale_factor = 1.0;
        if (_multiResolutionGrid) {
            Vec4f origin_info = kernelScaleLevelRatio(coeff_1D.ray_origin);
            origin_scale_factor = multiResLowLevel ? origin_info.z() : origin_info.w();
        }

        Vec3f origin_iso = _gp->_cov->transformPosDirWorldtoLocal(coeff_1D.ray_origin, kernelSpatialScale);
        Vec3f origin_iso_ray = coord.toLocal(origin_iso);
        Vec2f value_condition_splat = coeff_1D.value_scale *
                _gp->_cov->covarianceKernel1D(p_iso_ray.z(), origin_iso_ray.z(), p, coeff_1D.ray_origin, ray_dir_iso);
        Vec2f grad_condition_splat_z = kernelSpatialScale * coeff_1D.gradient_scale.z() *
                _gp->_cov->covarianceKernel1DGrad(p_iso_ray.z(), origin_iso_ray.z(), p, coeff_1D.ray_origin, ray_dir_iso);
        noise += origin_scale_factor * (value_condition_splat + grad_condition_splat_z);
        if (_correlationXY) {
            // Consider the effect of the conditioning kernels placed at the start of the ray
            grad_condition_splat_x = origin_scale_factor * kernelSpatialScale * coeff_1D.gradient_scale.x() * _gp->_cov->covarianceKernel1DGradFor3DNormal(p_iso_ray.z(), origin_iso_ray.z(), p, coeff_1D.ray_origin, coord.tangent);
            grad_condition_splat_y = origin_scale_factor * kernelSpatialScale * coeff_1D.gradient_scale.y() * _gp->_cov->covarianceKernel1DGradFor3DNormal(p_iso_ray.z(), origin_iso_ray.z(), p, coeff_1D.ray_origin, coord.bitangent);
        }
    }
    Vec4f noise_full = Vec4f(noise.x(), grad_condition_splat_x, grad_condition_splat_y, noise.y());
    // The gradient is not transformed to world space here, but in evaluateGradientNoise1D()
    return noise_full;
}