//! Storage for dense and jagged tensor metadata.
//!
//! JaggedTensor deliberately owns a flat buffer. Keeping the payload flat
//! makes it suitable for packed i64 lowering while the dimension metadata
//! retains the logical layout produced by Python tracing.

/// A tensor whose innermost dimensions may have different lengths.
///
/// Data is stored in row-major order. The tensor dimensions describe the
/// regular prefix of the shape. Each entry in the jagged dimensions contains
/// the starting offsets for one jagged layer. Offset arrays always start at
/// zero, are nondecreasing, and end at the number of elements in the layer
/// they index.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct JaggedTensor<T> {
    pub data: Vec<T>,
    pub tensor_dims: Vec<usize>,
    pub jagged_dims: Vec<Vec<usize>>,
}

impl<T> JaggedTensor<T> {
    /// Construct a tensor from its flat payload and shape metadata.
    ///
    /// This low-level constructor assumes its metadata has already been
    /// validated. Use [`JaggedTensor::try_new`] at input boundaries.
    pub fn new(data: Vec<T>, tensor_dims: Vec<usize>, jagged_dims: Vec<Vec<usize>>) -> Self {
        Self {
            data,
            tensor_dims,
            jagged_dims,
        }
    }

    /// Validate shape metadata and construct a tensor.
    pub fn try_new(
        data: Vec<T>,
        tensor_dims: Vec<usize>,
        jagged_dims: Vec<Vec<usize>>,
    ) -> Result<Self, String> {
        if tensor_dims.contains(&0) {
            return Err("regular tensor dimensions must be nonzero".into());
        }
        let regular_elements = tensor_dims
            .iter()
            .try_fold(1usize, |size, &dimension| size.checked_mul(dimension))
            .ok_or_else(|| "tensor dimensions overflow usize".to_string())?;

        if jagged_dims.is_empty() {
            if !tensor_dims.is_empty() && regular_elements != data.len() {
                return Err(format!(
                    "dense tensor shape contains {regular_elements} elements but data contains {}",
                    data.len()
                ));
            }
        } else {
            let mut expected_rows = regular_elements;
            for (layer, offsets) in jagged_dims.iter().enumerate() {
                if offsets.len() != expected_rows.saturating_add(1) {
                    return Err(format!(
                        "jagged layer {layer} has {} offsets; expected {}",
                        offsets.len(),
                        expected_rows + 1
                    ));
                }
                if offsets.first() != Some(&0) {
                    return Err(format!("jagged layer {layer} must start at offset zero"));
                }
                if offsets.windows(2).any(|pair| pair[0] > pair[1]) {
                    return Err(format!(
                        "jagged layer {layer} offsets must be nondecreasing"
                    ));
                }
                expected_rows = *offsets.last().unwrap_or(&0);
            }
            if expected_rows != data.len() {
                return Err(format!(
                    "final jagged offset is {expected_rows} but data contains {} elements",
                    data.len()
                ));
            }
        }
        Ok(Self::new(data, tensor_dims, jagged_dims))
    }

    /// Construct a one-dimensional, non-jagged tensor.
    pub fn from_flat(data: Vec<T>) -> Self {
        let len = data.len();
        Self::new(data, vec![len], Vec::new())
    }

    /// Return the flat payload in row-major order.
    pub fn data(&self) -> &[T] {
        &self.data
    }

    /// Return the regular prefix of the logical shape.
    pub fn tensor_dims(&self) -> &[usize] {
        &self.tensor_dims
    }

    /// Return the start offsets for each jagged layer.
    pub fn jagged_dims(&self) -> &[Vec<usize>] {
        &self.jagged_dims
    }

    /// Number of scalar values in the flat payload.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Number of outer rows: the first jagged layer's row count, or the
    /// number of values when the tensor is not jagged.
    pub fn row_count(&self) -> usize {
        match self.jagged_dims.first() {
            Some(offsets) => offsets.len().saturating_sub(1),
            None => self.data.len(),
        }
    }

    /// Values of outer row `row` (one value per row when not jagged).
    pub fn row(&self, row: usize) -> &[T] {
        match self.jagged_dims.first() {
            Some(offsets) => &self.data[offsets[row]..offsets[row + 1]],
            None => std::slice::from_ref(&self.data[row]),
        }
    }

    /// Whether the tensor has no scalar values.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Consume the tensor and return its three storage components.
    pub fn into_parts(self) -> (Vec<T>, Vec<usize>, Vec<Vec<usize>>) {
        (self.data, self.tensor_dims, self.jagged_dims)
    }
}

impl<T> FromIterator<T> for JaggedTensor<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self::from_flat(iter.into_iter().collect())
    }
}

impl<T> IntoIterator for JaggedTensor<T> {
    type Item = T;
    type IntoIter = std::vec::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter()
    }
}

impl<'a, T> IntoIterator for &'a JaggedTensor<T> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::JaggedTensor;

    #[test]
    fn flat_collection_preserves_values_and_shape() {
        let tensor = [2_u64, 3, 5].into_iter().collect::<JaggedTensor<_>>();

        assert_eq!(tensor.data(), &[2, 3, 5]);
        assert_eq!(tensor.tensor_dims(), &[3]);
        assert!(tensor.jagged_dims().is_empty());
    }

    #[test]
    fn validates_dense_and_jagged_layouts() {
        assert!(JaggedTensor::try_new(vec![1, 2, 3], vec![3], vec![]).is_ok());
        assert!(JaggedTensor::try_new(vec![1, 2, 3], vec![2], vec![vec![0, 1, 3]],).is_ok());
        assert!(JaggedTensor::try_new(vec![1, 2], vec![2], vec![vec![1, 2, 2]]).is_err());
        assert!(JaggedTensor::try_new(vec![1, 2], vec![2], vec![vec![0, 1, 3]]).is_err());
    }
}
