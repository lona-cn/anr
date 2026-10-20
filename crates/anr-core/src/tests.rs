use std::error::Error as _;
use std::io;

use crate::Error;

#[test]
fn backend_error_preserves_downcastable_source() {
    let error = Error::backend(io::Error::from_raw_os_error(123));

    let source = error
        .source()
        .expect("backend errors must expose their original source");
    let original = source
        .downcast_ref::<io::Error>()
        .expect("backend errors must retain the concrete source type");

    assert_eq!(original.raw_os_error(), Some(123));
}
