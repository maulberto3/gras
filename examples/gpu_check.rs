use flodl::{DType, Device, Tensor, TensorOptions};

fn main() {
    let device = Device::CUDA(0);
    let opts = TensorOptions {
        dtype: DType::Float32,
        device,
    };
    let a = Tensor::randn(&[4, 4], opts).unwrap();
    let b = Tensor::randn(&[4, 4], opts).unwrap();
    let c = a.matmul(&b).unwrap();
    println!(
        "CUDA matmul OK — result device: {:?}, shape: {:?}",
        c.device(),
        c.shape()
    );
}
