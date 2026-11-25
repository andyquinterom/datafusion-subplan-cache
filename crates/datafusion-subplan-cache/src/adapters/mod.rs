//! Ready-made [`crate::ImmutableFileTable`] implementations.

#[cfg(feature = "delta")]
pub mod delta;
pub mod parquet;
