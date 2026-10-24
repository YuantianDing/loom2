//! Graph rewrites applied to traced functions before lowering.

pub mod fusion;

pub use fusion::fuse;
