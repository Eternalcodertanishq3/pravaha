use pyo3::prelude::*;

/// Real-time memory allocator statistics.
///
/// Tracks allocation patterns, hit rates, and utilization
/// for monitoring via Prometheus/Grafana.
#[pyclass]
#[derive(Clone)]
pub struct AllocatorStats {
    #[pyo3(get)]
    pub total_allocations: u64,
    #[pyo3(get)]
    pub total_frees: u64,
    #[pyo3(get)]
    pub total_evictions: u64,
    #[pyo3(get)]
    pub cache_hits: u64,
    #[pyo3(get)]
    pub cache_misses: u64,
    #[pyo3(get)]
    pub peak_blocks_used: usize,
    #[pyo3(get)]
    pub current_blocks_used: usize,
    #[pyo3(get)]
    pub total_blocks: usize,
}

#[pymethods]
impl AllocatorStats {
    #[new]
    pub fn new(total_blocks: usize) -> Self {
        Self {
            total_allocations: 0,
            total_frees: 0,
            total_evictions: 0,
            cache_hits: 0,
            cache_misses: 0,
            peak_blocks_used: 0,
            current_blocks_used: 0,
            total_blocks,
        }
    }

    /// Cache hit rate as a percentage.
    pub fn hit_rate(&self) -> f64 {
        let total = self.cache_hits + self.cache_misses;
        if total == 0 {
            return 0.0;
        }
        (self.cache_hits as f64 / total as f64) * 100.0
    }

    /// Memory utilization as a percentage.
    pub fn utilization(&self) -> f64 {
        if self.total_blocks == 0 {
            return 0.0;
        }
        (self.current_blocks_used as f64 / self.total_blocks as f64) * 100.0
    }

    /// Allocation rate (allocs per free).
    pub fn alloc_free_ratio(&self) -> f64 {
        if self.total_frees == 0 {
            return self.total_allocations as f64;
        }
        self.total_allocations as f64 / self.total_frees as f64
    }

    pub fn record_allocation(&mut self, count: usize) {
        self.total_allocations += count as u64;
        self.current_blocks_used += count;
        if self.current_blocks_used > self.peak_blocks_used {
            self.peak_blocks_used = self.current_blocks_used;
        }
    }

    pub fn record_free(&mut self, count: usize) {
        self.total_frees += count as u64;
        if self.current_blocks_used >= count {
            self.current_blocks_used -= count;
        }
    }

    pub fn record_eviction(&mut self) {
        self.total_evictions += 1;
    }

    pub fn record_cache_hit(&mut self) {
        self.cache_hits += 1;
    }

    pub fn record_cache_miss(&mut self) {
        self.cache_misses += 1;
    }
}
