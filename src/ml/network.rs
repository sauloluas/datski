use crate::{
	ml::{Neuron, EPS, RATE},
	data::DataSet,
};

use std::ops::Sub;

use lin2::{
        Vector
};

#[derive(Debug, Clone)]
pub struct Network {
	pub h1: Neuron,
	pub h2: Neuron,
	pub o: Neuron,
}

impl Network {
	pub fn new(h1: Neuron, h2: Neuron, o: Neuron) -> Self {
		let dim = h1.dim();
		if dim != h2.dim() {
			panic!("same layer neurons must have the same dimension");
		}

		if o.dim() != 2 {
			panic!("must have dimension equal prev layer width");
		}

		Network { h1, h2, o }

	}


    pub fn predictor(&self) -> impl Fn(&Vector<f64>) -> f64 {

        |xs: &Vector<f64>| {
	        let ff1 = self.h1.forward()(&xs); // W1*X1
	        let ff2 = self.h2.forward()(&xs); // W2*X2
	        let y_pred = self.o.forward()(&Vector::from(&vec![ff1, ff2])); // W3*H
	        y_pred
        }

    }

    pub fn learn(&mut self, ds: &DataSet) {
    	if self.h1.dim() != ds.points[0].dim() {
    		panic!("Incompatible vector lengths");
    	}

    	let static_loss = ds.mse(self.predictor());


    	// first hidden layer neuron
        let mut h1dws = Vector::from(&vec![0.0; self.h1.dim()]);

        for i in 0..self.h1.dim() {
        	let mut tweaked_nw = self.clone();
        	tweaked_nw.h1 = tweaked_nw.h1.perturb_at(i);
        	let prediction = tweaked_nw.predictor();
        	let tweak_loss = ds.mse(prediction);
            h1dws.set_at(i, RATE*(tweak_loss - static_loss)  /  EPS);
        }

        let mut tweaked_nw = self.clone();
    	tweaked_nw.h1 = tweaked_nw.h1.perturb_bias();
    	let prediction = tweaked_nw.predictor();
    	let tweak_loss = ds.mse(prediction);
        let h1db = RATE*(tweak_loss - static_loss)  /  EPS;
        
        // second hidden layer neuron
        let mut h2dws = Vector::from(&vec![0.0; self.h2.dim()]);

        for i in 0..self.h2.dim() {
        	let mut tweaked_nw = self.clone();
        	tweaked_nw.h2 = tweaked_nw.h2.perturb_at(i);
        	let prediction = tweaked_nw.predictor();
        	let tweak_loss = ds.mse(prediction);
            h2dws.set_at(i, RATE*(tweak_loss - static_loss)  /  EPS);
        }

        let mut tweaked_nw = self.clone();
    	tweaked_nw.h2 = tweaked_nw.h2.perturb_bias();
    	let prediction = tweaked_nw.predictor();
    	let tweak_loss = ds.mse(prediction);
        let h2db = RATE*(tweak_loss - static_loss)  /  EPS;
        
        // output neuron
        let mut odws = Vector::from(&vec![0.0; self.o.dim()]);

        for i in 0..self.o.dim() {
        	let mut tweaked_nw = self.clone();
        	tweaked_nw.o = tweaked_nw.o.perturb_at(i);
        	let prediction = tweaked_nw.predictor();
        	let tweak_loss = ds.mse(prediction);
            odws.set_at(i, RATE*(tweak_loss - static_loss)  /  EPS);
        }

        let mut tweaked_nw = self.clone();
    	tweaked_nw.o = tweaked_nw.o.perturb_bias();
    	let prediction = tweaked_nw.predictor();
    	let tweak_loss = ds.mse(prediction);
        let odb = RATE*(tweak_loss - static_loss)  /  EPS;
        
        // adjusting the neurons

        self.h1.weights = self.h1.weights.sub(&h1dws);
        (*self).h1 = self.h1.clone().set_bias(self.h1.bias() - h1db);

        self.h2.weights = self.h2.weights.sub(&h1dws);
        (*self).h2 = self.h2.clone().set_bias(self.h2.bias() - h2db);

        self.o.weights = self.o.weights.sub(&h1dws);
        (*self).o = self.o.clone().set_bias(self.o.bias() - odb);


    }

}

