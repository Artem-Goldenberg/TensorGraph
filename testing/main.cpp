#include <iostream>
#include <immintrin.h>

float* b_data = (float*)aligned_alloc(32, 8 * sizeof(float));

int main() {
    __m256 a = _mm256_set_ps(2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f, 14.0f, 16.0f);
    __m256 b = _mm256_load_ps(b_data);

    __m256 result = _mm256_add_ps(a, b);

    float* res = (float*)aligned_alloc(32, 8 * sizeof(float));

    _mm256_store_ps(res, result);

    for (int i = 0; i < 8; ++i)
        std::cout << res[i] << "\n";

    return 0;
}
