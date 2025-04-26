use crate::{
    data::DataSet,
    ml::{
        Activator::{self, Linear},
        EPS,
        RATE,
    },
    la::{
        vecmul,
        vecsub,
    },
};
use rand::random;
   

pub type Reductor = Box<dyn Fn(&[f64]) -> f64>;

#[derive(Clone, Debug)]
pub struct Neuron {
    pub weights: Vec<f64>,
    pub bias: f64,
    pub activation: Activator
}

impl Neuron {
    pub fn from_dim(n: usize) -> Neuron {
        let mut weights: Vec<f64> = vec![0.0; n];

        for i in 0..n {
            weights[i] = 2.0 * random::<f64>() - 1.0;
        }

        let bias = 2.0 * random::<f64>() - 1.0;
        // let bias = 0.0;

        let activation = Linear;

        Neuron { weights, bias, activation }
    }

    pub fn bias(&self) -> f64 {
        self.bias
    }

    pub fn set_weight_at(mut self, n: usize, new_weight: f64) -> Self {
        self.weights[n] = new_weight;
        self
    }

    pub fn perturb_at(&self, n: usize) -> Neuron {
        let mut other = self.clone();
        let w = other.weights[n];
        other.set_weight_at(n, w + EPS).clone()
    }

    pub fn perturb_bias(&mut self) -> Neuron {
        let mut other = self.clone();
        let b = other.bias;
        other.set_bias(b + EPS).clone() // clone it until it works!
    }

    pub fn set_bias(mut self, new_bias: f64) -> Self {
        self.bias = new_bias;
        self
    }

    pub fn set_activation(mut self, new_actv: Activator) -> Self {
        self.activation = new_actv;
        self
    }

    pub fn loss(&self, ds: &DataSet) -> f64 {
        ds.mse(self.forward())
    }

    pub fn dim(&self) -> usize {
        self.weights.len()
    }

    pub fn forward(&self) -> impl Fn(&[f64]) -> f64 {

        |xs: &[f64]| {
            self.activation.apply(vecmul(&xs, &self.weights.clone()) + self.bias)
        }

    }

    pub fn fdiff(&self, ds: &DataSet, sl: f64) -> f64 {
        // println!("perturbed loss: {}", self.loss(&ds));
        ( self.loss(&ds) - sl ) / EPS
    }

    /// adjusting the neuron weights seeking the local minimum
    pub fn learn(&mut self, ds: &DataSet) {
        let dim = self.dim();
        if dim != ds.points[0].dim() {
            panic!("Incompatible vector lengths");
        }

        let static_loss = self.loss(&ds);

        let mut dw: Vec<f64> = vec![0.0; dim];

        for i in 0..dim {
            dw[i] = self.perturb_at(i).fdiff(&ds, static_loss)*RATE;
        }

        let db = self.perturb_bias().fdiff(&ds, static_loss)*RATE;
        // println!("db: {db}");

        self.weights = vecsub(&self.weights, &dw);

        *self = self.clone().set_bias(self.bias() - db);

    }

}

