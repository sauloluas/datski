
pub struct Vector {
    data: Vec<f64>
}

impl Vector {
    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn iter(&self) -> impl Iterator<Item = &f64> {
        self.data.iter()
    }

    pub fn dot(&self, other: &Self) -> f64 {
        if self.len() != other.len() {
            panic!("Incompatible vector lengths");
        }

        self.iter()
            .zip(other.iter())
            .map(|(a, b)| a * b)
            .sum()
    }
}

pub fn vecmul(xs: &[f64], ws: &[f64]) -> f64 {
    if xs.len() != ws.len() {
        panic!("Incompatible vector lengths");
    }

    xs.iter()
        .zip(ws.iter())
        .map(|(x, w)| x * w)
        .sum()      // x1*w1 + x2*w2 + ... + xn*wn;

}

// pub fn vecscale(ms: &[f64], scalar: f64) -> Vec<f64> {
//     let ns = ms.clone();

//     ns.iter().map(|n| n * scalar).collect()
// }

pub fn vecsub(ms: &[f64], ns: &[f64]) -> Vec<f64> {
    let len = ms.len();
    if len != ns.len() {
        panic!("Incompatible vector lengths");
    }

    let mut os = ms.to_vec();

    for i in 0..len {
        os[i] -= ns[i]
    }

    os

}







