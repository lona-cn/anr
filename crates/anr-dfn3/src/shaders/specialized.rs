use crate::Dispatch;
use std::fmt::Write;

const PARAMETER_ACCESSOR: &str =
    "fn p(i: u32) -> u32 { return parameters.values[i / 4u][i % 4u]; }";

pub(super) fn wgsl(dispatch: &Dispatch) -> String {
    source(super::wgsl(dispatch.shader), &dispatch.params)
}

pub(super) fn source(source: &str, parameters: &[u32; 32]) -> String {
    source_with_dynamic(source, parameters, (1 << 22) | (1 << 23))
}

pub(super) fn source_with_dynamic(source: &str, parameters: &[u32; 32], dynamic: u32) -> String {
    let (before, after) = source
        .split_once(PARAMETER_ACCESSOR)
        .expect("canonical kernels share the parameter accessor");
    let mut specialized = String::with_capacity(source.len() + 640);
    specialized.push_str(before);
    specialized.push_str("const fixed_parameters: array<u32, 32> = array<u32, 32>(");
    for (index, &value) in parameters.iter().enumerate() {
        // Runtime words must not affect the compiled source or its reuse key.
        let value = if dynamic & (1 << index) != 0 {
            0
        } else {
            value
        };
        write!(specialized, "{value}u,").unwrap();
    }
    specialized.push_str(");\n");
    write!(
        specialized,
        "fn p(i: u32) -> u32 {{\n\
         if ({dynamic}u & (1u << i)) != 0u {{ return parameters.values[i / 4u][i % 4u]; }}\n\
         return fixed_parameters[i];\n\
         }}"
    )
    .unwrap();
    specialized.push_str(after);
    specialized
}
