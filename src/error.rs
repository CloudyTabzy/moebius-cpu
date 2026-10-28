#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("weights: {0}")]
    Weights(String),
    #[error("i/o: {0}")]
    Io(#[from] std::io::Error),
    #[error("unsupported CPU: {0}")]
    UnsupportedCpu(String),
    #[error("cancelled")]
    Cancelled,
}

pub type Result<T> = std::result::Result<T, Error>;
