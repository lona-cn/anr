use anr_core::Error;
use anr_dfn3::Model;

#[test]
fn bytes_outside_the_supported_model_are_rejected() {
    assert!(matches!(
        Model::from_bytes(b"not the supported DFN3 export"),
        Err(Error::InvalidModel(_))
    ));
}
