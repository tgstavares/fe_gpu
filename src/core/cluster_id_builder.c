#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define K_MAX_CLUSTER_DIMS 4

typedef struct {
    int32_t values[K_MAX_CLUSTER_DIMS];
    int len;
} Key;

static size_t linear_index(int dim, int ld, long long obs) {
    return (size_t)dim + (size_t)obs * (size_t)ld;
}

static int radix_sort_pairs(uint64_t* keys,
                            int32_t* order,
                            uint64_t* tmp_keys,
                            int32_t* tmp_order,
                            size_t n) {
    const int kRadixBits = 11;
    const size_t kBucketCount = 1u << kRadixBits;
    const size_t kMask = kBucketCount - 1;
    const int kPasses = (64 + kRadixBits - 1) / kRadixBits;
    if (n == 0) return 0;

    uint64_t* src_keys = keys;
    uint64_t* dst_keys = tmp_keys;
    int32_t* src_idx = order;
    int32_t* dst_idx = tmp_order;
    int swapped = 0;

#ifdef _OPENMP
    int nthreads = omp_get_max_threads();
#else
    int nthreads = 1;
#endif

    /* Use a simple serial path for small arrays or single-thread runs. */
    if (nthreads <= 1 || n < 1 << 16) {
        size_t* counts = (size_t*)malloc(kBucketCount * sizeof(size_t));
        if (!counts) return 6;
        for (int pass = 0; pass < kPasses; ++pass) {
            memset(counts, 0, kBucketCount * sizeof(size_t));
            unsigned shift = (unsigned)(pass * kRadixBits);
            for (size_t i = 0; i < n; ++i) {
                size_t bucket = (size_t)((src_keys[i] >> shift) & kMask);
                counts[bucket]++;
            }
            size_t sum = 0;
            for (size_t bucket = 0; bucket < kBucketCount; ++bucket) {
                size_t c = counts[bucket];
                counts[bucket] = sum;
                sum += c;
            }
            for (size_t i = 0; i < n; ++i) {
                size_t bucket = (size_t)((src_keys[i] >> shift) & kMask);
                size_t pos = counts[bucket]++;
                dst_keys[pos] = src_keys[i];
                dst_idx[pos] = src_idx[i];
            }
            uint64_t* tmpk = src_keys;
            src_keys = dst_keys;
            dst_keys = tmpk;
            int32_t* tmpi = src_idx;
            src_idx = dst_idx;
            dst_idx = tmpi;
            swapped = !swapped;
        }
        free(counts);
    } else {
        size_t* counts = (size_t*)calloc((size_t)nthreads * kBucketCount, sizeof(size_t));
        size_t* offsets = (size_t*)malloc((size_t)nthreads * kBucketCount * sizeof(size_t));
        if (!counts || !offsets) {
            free(counts);
            free(offsets);
            return 6;
        }
        for (int pass = 0; pass < kPasses; ++pass) {
            unsigned shift = (unsigned)(pass * kRadixBits);
            memset(counts, 0, (size_t)nthreads * kBucketCount * sizeof(size_t));

#pragma omp parallel default(shared)
            {
                int tid = 0;
#ifdef _OPENMP
                tid = omp_get_thread_num();
#endif
                size_t* local_counts = counts + (size_t)tid * kBucketCount;
#pragma omp for schedule(static)
                for (size_t i = 0; i < n; ++i) {
                    size_t bucket = (size_t)((src_keys[i] >> shift) & kMask);
                    local_counts[bucket]++;
                }
            }

            size_t running = 0;
            for (size_t bucket = 0; bucket < kBucketCount; ++bucket) {
                size_t cumulative = 0;
                for (int t = 0; t < nthreads; ++t) {
                    size_t c = counts[(size_t)t * kBucketCount + bucket];
                    offsets[(size_t)t * kBucketCount + bucket] = running + cumulative;
                    cumulative += c;
                }
                running += cumulative;
            }

#pragma omp parallel default(shared)
            {
                int tid = 0;
#ifdef _OPENMP
                tid = omp_get_thread_num();
#endif
                size_t* local_offsets = offsets + (size_t)tid * kBucketCount;
#pragma omp for schedule(static)
                for (size_t i = 0; i < n; ++i) {
                    size_t bucket = (size_t)((src_keys[i] >> shift) & kMask);
                    size_t pos = local_offsets[bucket]++;
                    dst_keys[pos] = src_keys[i];
                    dst_idx[pos] = src_idx[i];
                }
            }

            uint64_t* tmpk = src_keys;
            src_keys = dst_keys;
            dst_keys = tmpk;
            int32_t* tmpi = src_idx;
            src_idx = dst_idx;
            dst_idx = tmpi;
            swapped = !swapped;
        }
        free(counts);
        free(offsets);
    }

    if (swapped) {
        memcpy(keys, src_keys, n * sizeof(uint64_t));
        memcpy(order, src_idx, n * sizeof(int32_t));
    }
    return 0;
}

static int assign_clusters_from_keys(uint64_t* keys,
                                     int32_t* out_ids,
                                     int32_t* out_n_clusters,
                                     size_t n) {
    if (n == 0) {
        *out_n_clusters = 0;
        return 0;
    }
    int32_t* order = (int32_t*)malloc(n * sizeof(int32_t));
    uint64_t* tmp_keys = (uint64_t*)malloc(n * sizeof(uint64_t));
    int32_t* tmp_order = (int32_t*)malloc(n * sizeof(int32_t));
    if (!order || !tmp_keys || !tmp_order) {
        free(order);
        free(tmp_keys);
        free(tmp_order);
        return 6;
    }
    for (size_t i = 0; i < n; ++i) order[i] = (int32_t)i;

    int sort_status = radix_sort_pairs(keys, order, tmp_keys, tmp_order, n);
    if (sort_status != 0) {
        free(order);
        free(tmp_keys);
        free(tmp_order);
        return sort_status;
    }

    uint64_t prev_key = UINT64_MAX;
    int32_t current_id = 0;
    for (size_t i = 0; i < n; ++i) {
        uint64_t key = keys[i];
        if (key != prev_key) {
            if (current_id == INT32_MAX) {
                free(order);
                free(tmp_keys);
                free(tmp_order);
                return 5;
            }
            ++current_id;
            prev_key = key;
        }
        out_ids[(size_t)order[i]] = current_id;
    }
    *out_n_clusters = current_id;
    free(order);
    free(tmp_keys);
    free(tmp_order);
    return 0;
}

typedef struct {
    Key key;
    int32_t id;
    int used;
} HashEntry;

static int build_with_hash(const int32_t* fe_ids,
                           int ld_fe,
                           long long n_obs,
                           const int32_t* subset_dims,
                           int subset_len,
                           int32_t* out_ids,
                           int32_t* out_n_clusters) {
    size_t capacity = (size_t)n_obs * 2 + 1;
    HashEntry* table = (HashEntry*)calloc(capacity, sizeof(HashEntry));
    if (!table) return 6;
    int32_t next_id = 0;

    for (long long obs = 0; obs < n_obs; ++obs) {
        Key key;
        memset(&key, 0, sizeof(Key));
        key.len = subset_len;
        for (int j = 0; j < subset_len; ++j) {
            int dim = subset_dims[j];
            if (dim < 1 || dim > ld_fe) {
                free(table);
                return 4;
            }
            size_t idx = linear_index(dim - 1, ld_fe, obs);
            key.values[j] = fe_ids[idx];
        }
        // simple open addressing
        uint64_t h = 1469598103934665603ULL;
        for (int j = 0; j < key.len; ++j) {
            uint64_t v = (uint32_t)key.values[j];
            h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        }
        size_t pos = (size_t)(h % capacity);
        while (1) {
            if (!table[pos].used) {
                table[pos].used = 1;
                table[pos].key = key;
                if (next_id == INT32_MAX) {
                    free(table);
                    return 5;
                }
                table[pos].id = ++next_id;
                out_ids[obs] = table[pos].id;
                break;
            }
            if (table[pos].key.len == key.len &&
                memcmp(table[pos].key.values, key.values, sizeof(int32_t) * key.len) == 0) {
                out_ids[obs] = table[pos].id;
                break;
            }
            pos = (pos + 1) % capacity;
        }
    }
    *out_n_clusters = next_id;
    free(table);
    return 0;
}

static int use_sort_encoding(const int32_t* subset_sizes, int subset_len, uint64_t* total_span) {
    *total_span = 1;
    for (int i = 0; i < subset_len; ++i) {
        uint64_t size = (uint64_t)subset_sizes[i];
        if (size == 0) return 0;
        if (size > UINT64_MAX / *total_span) return 0;
        *total_span *= size;
        if (*total_span > (1ULL << 62)) return 0;
    }
    return 1;
}

int fe_build_cluster_ids(const int32_t* fe_ids,
                         int ld_fe,
                         long long n_obs,
                         const int32_t* subset_dims,
                         const int32_t* subset_sizes,
                         int subset_len,
                         int32_t* out_ids,
                         int32_t* out_n_clusters) {
    if (!fe_ids || !subset_dims || !subset_sizes || !out_ids || !out_n_clusters) return 1;
    if (ld_fe <= 0 || n_obs <= 0) return 2;
    if (subset_len <= 0 || subset_len > K_MAX_CLUSTER_DIMS) return 3;

    uint64_t total_span = 0;
    int sort_ok = use_sort_encoding(subset_sizes, subset_len, &total_span);
    if (sort_ok) {
        uint64_t* keys = (uint64_t*)malloc((size_t)n_obs * sizeof(uint64_t));
        if (!keys) return 6;
        uint64_t multipliers[K_MAX_CLUSTER_DIMS];
        uint64_t stride = 1;
        for (int i = 0; i < subset_len; ++i) {
            multipliers[i] = stride;
            stride *= (uint64_t)subset_sizes[i];
        }
#pragma omp parallel for schedule(static)
        for (long long obs = 0; obs < n_obs; ++obs) {
            uint64_t key = 0;
            for (int j = 0; j < subset_len; ++j) {
                int dim = subset_dims[j];
                if (dim < 1 || dim > ld_fe) {
                    continue;
                }
                size_t idx = linear_index(dim - 1, ld_fe, obs);
                int32_t val = fe_ids[idx];
                key += multipliers[j] * (uint64_t)(val - 1);
            }
            keys[(size_t)obs] = key;
        }
        int status = assign_clusters_from_keys(keys, out_ids, out_n_clusters, (size_t)n_obs);
        free(keys);
        return status;
    }

    return build_with_hash(fe_ids, ld_fe, n_obs, subset_dims, subset_len, out_ids, out_n_clusters);
}
