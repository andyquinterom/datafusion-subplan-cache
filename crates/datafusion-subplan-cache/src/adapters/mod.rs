//! Ready-made [`crate::ImmutableFileTable`] implementations.

#[cfg(feature = "delta")]
pub mod delta;
#[cfg(feature = "ducklake")]
pub mod ducklake;
pub mod parquet;
