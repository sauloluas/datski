#[allow(unused_imports)]
use datski::{
    ml::{
        Neuron,
        Network,
        Activator::{
            Sigmoid,
            Relu,
            Linear,
            Tanh,
            Swish
        },
    },
    data::{
        DataPoint, 
        DataSet,
    },
};

use lin2::Vector;

use termplot::*;


fn main() {
    // let x = [(2.0, 10.0), (2.0, 20.0), (-1.0, 3.0), (4.0, -2.0)];
    // let f: fn(f64, f64) -> f64 = |x1, x2| x2 * x1;
    // let sample: Vec<(f64, f64, f64)> = x.into_iter()
    //     .map(|(x1, x2)| (x1, x2, f(x1, x2)))
    //     .collect();

    // let x = [-10.0, -4.0, -1.0, 0.0, 1.0, 3.0, 7.0, 8.0];
    // let f: fn(f64) -> f64 = |x| x.powi(2);
    // let sample: Vec<(f64, f64)> = x.into_iter()
    //     .map(|x| (x, f(x)))
    //     .collect();
    

    let sample = [
        (0.0, 0.0, 0.0),
        (0.0, 1.0, 1.0),
        (1.0, 0.0, 1.0),
        (1.0, 1.0, 0.0),
    ];

    // let sample = [
    //     (-1.0, -1.0, -1.0),
    //     (-1.0, 0.0, 0.0),
    //     (-1.0, 1.0, 0.0),
    //     (0.0, -1.0, 0.0),
    //     (0.0, 0.0, 0.0),
    //     (0.0, 1.0, 0.0),
    //     (1.0, -1.0, 0.0),
    //     (1.0, 0.0, 0.0),
    //     (1.0, 1.0, 1.0)
    // ];

    // let sample = [
    //     (140.0, 2.7, 157.0, 6.2),
    //     (157.0, 6.2, 157.0, 3.9),
    //     (157.0, 3.9, 160.0, 7.0),
    //     (160.0, 7.0, 157.0, 6.9),
    //     (157.0, 6.9, 156.0, 5.7),
    //     (156.0, 5.7, 159.0, 5.9)
    // ];

    let ds = DataSet::new(
        sample.into_iter()
            .map(|(x1, x2, y)| DataPoint::new(Vector::from(&vec![x1, x2]), y))
            .collect()
    );

    // let ds = DataSet::new(
    //     sample.into_iter()
    //         .map(|(x, y)| DataPoint::new(vec![x], y))
    //         .collect()
    // );

    let mut n1 = Neuron::from_dim(2).set_activation(Sigmoid);
    let mut n2 = Neuron::from_dim(2).set_activation(Sigmoid);
    let mut n3 = Neuron::from_dim(2).set_activation(Sigmoid);
    let mut nw = Network::new(n1, n2, n3);

    (0..1000).for_each(|i| {
        println!("{i}.");
        println!("{nw:#?}");

        let prediction = nw.predictor();

        for point in &ds.points {
            // let ff1 = nw.h1.predictor()(point.xs()); // W1*X1
            // let ff2 = nw.h2.predictor()(point.xs()); // W2*X2
            // let y_pred = nw.o.predictor()(&[ff1, ff2]); // W3*H

            let y_pred = prediction(point.xs());
            
            println!("y: {} y pred: {y_pred:.3}", point.y()); // y1
        }

        println!("loss: {}", ds.mse(prediction));

        nw.learn(&ds);
        
        // println!("loss: {}", n1.loss(&ds));

        // n1.learn(&ds);
    });

    // for point in &ds.points {
    //     println!("y: {} ffgate1: {}", point.y(), nw.h1.forward()(point.xs()));     
    // }

    // for point in &ds.points {
    //     println!("y: {} ffgate2: {}", point.y(), nw.h2.forward()(point.xs()));       
    // }



    // n.set_activation(id);
    // n.set_weight_at(0, 0.1);
    // n.set_weight_at(1, 1.0);
    // n.set_weight_at(2, 0.1);

    // for point in &ds.points {
    //     let y_pred = n.predictor()(point.xs()); // w1*x1
    //     println!("y: {} y pred: {y_pred}", point.y()); // y1
    // }


    // (0..100).for_each(|i| {
    //     println!("{i}.");
    //     println!("{n1:#?}");

    //     for point in &ds.points {
    //         let y_pred = n1.predictor()(point.xs()); // w1*x1
    //         println!("y: {} y pred: {y_pred:.3}", point.y()); // y1
    //     }
        
    //     println!("loss: {}", n1.loss(&ds));

    //     n1.learn(&ds);
    // })


    // let mut plot = Plot::default();
    // plot.set_domain(Domain(-40.0..40.0))
    //     .set_codomain(Domain(-60.0..60.0))
    //     .set_title("f(x)")
    //     .set_size(Size::new(200, 100))
    //     .add_plot(Box::new(plot::Graph::new(f)));

    // println!("{plot}");


    // let mut plot = Plot::default();
    // plot.set_domain(Domain(-40.0..40.0))
    //     .set_codomain(Domain(-60.0..60.0))
    //     .set_title("pred f(x)")
    //     .set_size(Size::new(200, 100))
    //     .add_plot(Box::new(plot::Graph::new(move |x| nw.predictor()(&[x]))));

    // println!("{plot}");


}


