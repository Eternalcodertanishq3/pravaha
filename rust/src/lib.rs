mod allocator;
use allocator::BlockAllocator;
use pyo3::prelude::*;

#[pymodule]
fn pravaha_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<BlockAllocator>()?;
    Ok(())
}
