use pyo3_stub_gen::Result;

fn main() -> Result<()> {
    let stub = glycocr_rs::stub_info()?;
    stub.generate()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_stub_gen() {
        let stub = glycocr_rs::stub_info().expect("Failed to get stub info");
        stub.generate().expect("Failed to generate stub");
    }
}
