use crate::Dispatch;
use std::fmt::Write;

const PARAMETER_ACCESSOR: &str =
    "fn p(i: u32) -> u32 { return parameters.values[i / 4u][i % 4u]; }";

pub(super) fn wgsl(dispatch: &Dispatch) -> String {
    let source = super::wgsl(dispatch.shader);
    let (before, after) = source
        .split_once(PARAMETER_ACCESSOR)
        .expect("canonical kernels share the parameter accessor");
    let mut specialized = String::with_capacity(source.len() + 640);
    specialized.push_str(before);
    specialized.push_str("const fixed_parameters: array<u32, 32> = array<u32, 32>(");
    for (index, &value) in dispatch.params.iter().enumerate() {
        // Strength changes never require shader recompilation. Do not bake its
        // initial bits into the source, even though these array slots are unused.
        let value = if index == 22 || index == 23 { 0 } else { value };
        write!(specialized, "{value}u,").unwrap();
    }
    specialized.push_str(");\n");
    specialized.push_str(
        "fn p(i: u32) -> u32 {\n\
         if i == 22u || i == 23u { return parameters.values[i / 4u][i % 4u]; }\n\
         return fixed_parameters[i];\n\
         }",
    );
    specialized.push_str(after);
    specialized
}
