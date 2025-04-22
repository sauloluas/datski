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



pub fn make_sig(range: Range<f64>) -> impl Fn(f64) -> f64 {
	move |x| { 
		let ampl = range.start - range.end;
		ampl*sigmoid(x) + range.start
	}
}
