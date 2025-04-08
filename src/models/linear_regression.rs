use ndarray::{Array1, Array2};

#[allow(dead_code)]
pub struct LinearRegression {
  lr: f64,
  n_iters: usize,
  weights: Option<Array1<f64>>,
  bias: f64,
}

#[allow(non_snake_case)]
impl LinearRegression {
  pub fn new(lr: f64, n_iters: usize) -> Self {
    Self { 
      lr, 
      n_iters, 
      weights: None,
      bias: 0.0 
    }
  }

  pub fn fit(&mut self, X: &Array2<f64>, y: &Array1<f64>) {
    let (n_samples, n_features) = X.dim();
    let mut weights = Array1::<f64>::zeros(n_features);
    let mut bias = 0.0;

    for _ in 0..self.n_iters {
      let y_pred = X.dot(&weights) + bias;

      let error = &y_pred - y;
      let dw = X.t().dot(&error) / n_samples as f64;
      let db = error.sum() / n_samples as f64;

      weights = &weights - &(self.lr * &dw);
      bias -= self.lr * db;
    }

    self.weights = Some(weights);
    self.bias = bias;
  }

  pub fn predict(&self, X: &Array2<f64>) -> Array1<f64> {
    match &self.weights {
      Some(w) => X.dot(w) + self.bias,
      None => panic!("Model has not been trained yet!"),
    }
  }
}