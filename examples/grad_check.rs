//! Gradient-flow check: does backward() through cross_entropy_onehot_loss
//! actually populate parameter grads? Zero nets in the race would show here.
use gras::utils::score::{cross_entropy_onehot_loss};
use flodl::{Device, Tensor, Variable};
use flodl::nn::{Linear, Module};

fn main() {
    let cpu = Device::CPU;
    let layer = Linear::new(4, 3).unwrap();
    let w = layer.parameters();
    // forward via Variable ops (what the net does)
    let x = Variable::new(Tensor::from_f32(&[1.0,0.5,-0.2,0.8, 0.3,-1.0,0.4,0.1], &[2,4], cpu).unwrap(), false);
    let pred = layer.forward(&x).unwrap();
    let y = Variable::new(Tensor::from_f32(&[0.0,1.0,0.0, 1.0,0.0,0.0], &[2,3], cpu).unwrap(), false);
    let loss = cross_entropy_onehot_loss(&pred, &y).unwrap();
    println!("loss = {:.4}", loss.item().unwrap());
    loss.set_requires_grad(true).unwrap();
    // zero grads, backward, check
    for p in &w { p.variable.zero_grad(); }
    loss.backward().unwrap();
    let mut any_grad = false;
    let mut total_norm = 0.0f64;
    for (i, p) in w.iter().enumerate() {
        if let Some(g) = p.variable.grad() {
            let n: f64 = g.to_f64_vec().unwrap().iter().map(|v| v*v).sum();
            total_norm += n;
            if n > 0.0 { any_grad = true; }
        }
        println!("param {i}: grad_present={}", p.variable.grad().is_some());
    }
    println!("RESULT: gradients {} (‖g‖² = {:.6})", if any_grad {"FLOW ✅"} else {"ABSENT ❌"}, total_norm);
}
