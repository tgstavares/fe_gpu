#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <new>
#include <numeric>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

constexpr int kMaxClusterDims = 4;

struct Key {
    std::array<std::int32_t, kMaxClusterDims> values{};
    int len = 0;
};

struct KeyHash {
    std::size_t operator()(const Key& key) const noexcept {
        std::size_t h = 1469598103934665603ULL;
        for (int i = 0; i < key.len; ++i) {
            std::size_t v = static_cast<std::uint32_t>(key.values[i]);
            h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        }
        return h;
    }
};

struct KeyEqual {
    bool operator()(const Key& a, const Key& b) const noexcept {
        if (a.len != b.len) {
            return false;
        }
        for (int i = 0; i < a.len; ++i) {
            if (a.values[i] != b.values[i]) {
                return false;
            }
        }
        return true;
    }
};

inline std::size_t linear_index(int dim, int ld, long long obs) {
    return static_cast<std::size_t>(dim) + static_cast<std::size_t>(obs) * static_cast<std::size_t>(ld);
}

int build_with_hash(const std::int32_t* fe_ids,
                    int ld_fe,
                    long long n_obs,
                    const std::int32_t* subset_dims,
                    int subset_len,
                    std::int32_t* out_ids,
                    std::int32_t* out_n_clusters) {
    std::unordered_map<Key, std::int32_t, KeyHash, KeyEqual> map;
    map.reserve(static_cast<std::size_t>(n_obs / 4 + 1));

    std::int32_t next_id = 0;
    for (long long obs = 0; obs < n_obs; ++obs) {
        Key key;
        key.len = subset_len;
        for (int j = 0; j < subset_len; ++j) {
            int dim = subset_dims[j];
            if (dim < 1 || dim > ld_fe) {
                return 4;
            }
            std::size_t idx = linear_index(dim - 1, ld_fe, obs);
            key.values[j] = fe_ids[idx];
        }

        auto it = map.find(key);
        if (it == map.end()) {
            if (next_id == std::numeric_limits<std::int32_t>::max()) {
                return 5;
            }
            it = map.emplace(key, next_id + 1).first;
            ++next_id;
        }
        out_ids[obs] = it->second;
    }

    *out_n_clusters = next_id;
    return 0;
}

bool should_use_sort_encoding(const std::int32_t* subset_sizes, int subset_len, std::uint64_t& total_span) {
    total_span = 1;
    for (int i = 0; i < subset_len; ++i) {
        std::uint64_t size = static_cast<std::uint64_t>(subset_sizes[i]);
        if (size == 0) {
            return false;
        }
        if (size > std::numeric_limits<std::uint64_t>::max() / total_span) {
            return false;
        }
        total_span *= size;
        if (total_span > (1ULL << 62)) {
            return false;
        }
    }
    return true;
}

}  // namespace

extern "C" int fe_build_cluster_ids(const std::int32_t* fe_ids,
                                    int ld_fe,
                                    long long n_obs,
                                    const std::int32_t* subset_dims,
                                    const std::int32_t* subset_sizes,
                                    int subset_len,
                                    std::int32_t* out_ids,
                                    std::int32_t* out_n_clusters) {
    if (!fe_ids || !subset_dims || !subset_sizes || !out_ids || !out_n_clusters) {
        return 1;
    }
    if (ld_fe <= 0 || n_obs <= 0) {
        return 2;
    }
    if (subset_len <= 0 || subset_len > kMaxClusterDims) {
        return 3;
    }

    try {
        std::uint64_t total_span = 0;
        bool use_sort_encoding = should_use_sort_encoding(subset_sizes, subset_len, total_span);
        if (use_sort_encoding) {
            std::vector<std::uint64_t> multipliers(subset_len);
            std::uint64_t stride = 1;
            for (int i = 0; i < subset_len; ++i) {
                multipliers[i] = stride;
                stride *= static_cast<std::uint64_t>(subset_sizes[i]);
            }

            std::vector<std::uint64_t> keys(static_cast<std::size_t>(n_obs));
            for (long long obs = 0; obs < n_obs; ++obs) {
                std::uint64_t key = 0;
                for (int j = 0; j < subset_len; ++j) {
                    int dim = subset_dims[j];
                    if (dim < 1 || dim > ld_fe) {
                        return 4;
                    }
                    std::size_t idx = linear_index(dim - 1, ld_fe, obs);
                    std::int32_t val = fe_ids[idx];
                    key += multipliers[j] * static_cast<std::uint64_t>(val - 1);
                }
                keys[static_cast<std::size_t>(obs)] = key;
            }

            std::vector<std::int32_t> order(static_cast<std::size_t>(n_obs));
            std::iota(order.begin(), order.end(), 0);
            std::stable_sort(order.begin(), order.end(), [&](std::int32_t a, std::int32_t b) {
                return keys[static_cast<std::size_t>(a)] < keys[static_cast<std::size_t>(b)];
            });

            std::int32_t current_id = 0;
            std::uint64_t prev_key = std::numeric_limits<std::uint64_t>::max();
            for (std::int32_t idx : order) {
                std::uint64_t key = keys[static_cast<std::size_t>(idx)];
                if (key != prev_key) {
                    if (current_id == std::numeric_limits<std::int32_t>::max()) {
                        return 5;
                    }
                    ++current_id;
                    prev_key = key;
                }
                out_ids[idx] = current_id;
            }
            *out_n_clusters = current_id;
            return 0;
        }

        return build_with_hash(fe_ids, ld_fe, n_obs, subset_dims, subset_len, out_ids, out_n_clusters);
    } catch (const std::bad_alloc&) {
        return 6;
    } catch (...) {
        return 7;
    }
}
