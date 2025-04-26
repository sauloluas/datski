pub mod neuron;
pub mod activation_fn;
pub mod network;

const EPS: f64 = 1e-6;
const RATE: f64 = 1e-2; 

pub use neuron::Neuron;
pub use network::Network;
pub use activation_fn as act_fn;
pub use act_fn::Activator;

