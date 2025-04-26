use std::ops::Range;

pub fn sigmoid(x: f64) -> f64 {
	1.0 / ( 1.0 + (-x).exp() )
}

pub fn id(x: f64) -> f64 {
	x
}

pub fn relu(x: f64) -> f64 {
	if x <= 0.0 {
		0.0
	} else {
		x
	}
}

pub fn tanh(x: f64) -> f64 {
	x.tanh()
}

pub fn swish(x: f64) -> f64 {
	x * sigmoid(x)
}



pub fn make_sig(range: Range<f64>) -> impl Fn(f64) -> f64 {
	move |x| { 
		let ampl = range.start - range.end;
		ampl*sigmoid(x) + range.start
	}
}

#[derive(Debug, Clone, Copy)]
pub enum Activator {
	Linear,
	Sigmoid,
	Relu,
	Tanh,
	Swish,
	Custom(fn(f64) -> f64),
}

impl Activator {
	pub fn apply(self, x: f64) -> f64 {
		use Activator::*;
		match self {
			Linear => x,
			Sigmoid => sigmoid(x),
			Relu => relu(x),
			Tanh => x.tanh(),
			Swish => swish(x),
			Custom(f) => f(x), 
		}
	}
}