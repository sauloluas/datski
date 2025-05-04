use std::iter::IntoIterator;
use lin2::Vector;

pub struct DataPoint {
    xs: Vector<f64>,  // Dynamic number of inputs (x1, x2, ... xn)
    y: f64,
}

impl DataPoint {
    pub fn new(xs: Vector<f64>, y: f64) -> Self {
        Self { xs, y }
    }

    /// Get input values as a slice
    pub fn xs(&self) -> &Vector<f64> {
        &self.xs
    }

    /// Get output value (y)
    pub fn y(&self) -> f64 {
        self.y
    }

    /// Number of input dimensions
    pub fn dim(&self) -> usize {
        self.xs.len()
    }
}

pub struct DataSet {
	pub points: Vec<DataPoint>
}

impl DataSet {
	pub fn new(points: Vec<DataPoint>) -> DataSet {
		DataSet { points }
	}

    pub fn mse(&self, predictor: impl Fn(&Vector<f64>) -> f64) -> f64 {
    	let mut sqe = 0.0;

    	for point in &self.points {
    		let y_exp = point.y();
    		let y_pred = predictor(point.xs());
    		sqe += (y_exp - y_pred).powi(2);
    	}

    	sqe / self.points.len() as f64

    }

}

impl IntoIterator for DataSet {
	type Item = DataPoint;
	type IntoIter = std::vec::IntoIter<Self::Item>;

	fn into_iter(self) -> Self::IntoIter {
		self.points.into_iter()
	}
}


