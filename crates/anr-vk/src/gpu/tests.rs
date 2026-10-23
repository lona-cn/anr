mod gru;
mod pointwise;

use super::*;
use anr_dfn3::Dispatch;

fn pointwise(spatial: u32) -> Dispatch {
    let mut params = [0; 32];
    params[1] = spatial * 64;
    params[5] = 0;
    params[6] = 64;
    params[7] = 1;
    params[8] = spatial;
    params[9] = 64;
    params[10] = 1;
    params[11] = spatial;
    params[12] = 1;
    params[13] = 1;
    params[14] = 1;
    params[15] = 1;
    params[18] = 1;
    params[19] = 1;
    Dispatch {
        shader: ShaderKind::Network,
        groups: spatial,
        params,
    }
}

#[test]
fn pointwise_tiles_cover_short_and_full_spatial_extents() {
    for (spatial, groups32, groups64) in
        [(8, 2, 4), (16, 2, 4), (32, 4, 8), (48, 6, 12), (96, 12, 24)]
    {
        assert_eq!(pointwise_groups(&pointwise(spatial), 32), Some(groups32));
        assert_eq!(pointwise_groups(&pointwise(spatial), 64), Some(groups64));
    }
}

#[test]
fn other_convolutions_and_activations_keep_the_scalar_kernel() {
    for (parameter, value) in [
        (0, 1),
        (6, 32),
        (9, 1),
        (12, 3),
        (14, 2),
        (16, 1),
        (18, 64),
        (19, 2),
    ] {
        let mut dispatch = pointwise(16);
        dispatch.params[parameter] = value;
        assert_eq!(
            pointwise_groups(&dispatch, 32),
            None,
            "parameter {parameter}"
        );
    }
    let mut dsp = pointwise(16);
    dsp.shader = ShaderKind::Dsp;
    assert_eq!(pointwise_groups(&dsp, 32), None);
}

#[test]
fn kernel_selection_requires_a_consistent_win_over_the_default() {
    assert_eq!(tuning::fastest(&mut [[10; 5], [10; 5]]), 0);
    assert_eq!(tuning::fastest(&mut [[4, 4, 4, 4, 100], [5; 5]]), 0);
    assert_eq!(tuning::fastest(&mut [[10; 5], [8, 8, 8, 8, 80], [9; 5]]), 1);
}

fn executor(cooperative: bool) -> Gpu {
    let path = std::env::var_os("ANR_DFN3_MODEL").expect("set ANR_DFN3_MODEL");
    let model = Model::load(std::path::Path::new(&path)).unwrap();
    let mut context = Context::new(&model).unwrap();
    if cooperative {
        assert!(
            context.cooperative_subgroup_size.is_some(),
            "requires a compatible KHR cooperative-matrix device"
        );
    } else {
        // Exercise exactly the resource/pipeline branch used by unsupported devices.
        context.cooperative_subgroup_size = None;
        context.timestamp_valid_bits = 0;
    }
    let policy = if cooperative {
        PointwisePolicy::Cooperative
    } else {
        PointwisePolicy::Fastest
    };
    Gpu::with_context(model, context, policy).unwrap()
}

#[test]
#[ignore = "requires ANR_DFN3_MODEL and a compatible KHR cooperative-matrix device"]
fn cooperative_pointwise_matches_scalar_through_attenuation_and_state_resets() {
    let input: Vec<f32> = include_bytes!("../../tests/fixtures/reference_input.f32le")
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| f32::from_le_bytes(*bytes))
        .collect();
    let mut scalar = executor(false);
    let mut cooperative = executor(true);
    let mut scalar_output = [0.0; FRAME_SIZE];
    let mut cooperative_output = [0.0; FRAME_SIZE];
    let mut maximum = 0.0_f32;
    let mut squared = 0.0_f64;
    let mut samples = 0;
    for index in 0..1100 {
        let mut frame = [0.0; FRAME_SIZE];
        let start = (index % input.len().div_ceil(FRAME_SIZE)) * FRAME_SIZE;
        let chunk = &input[start..(start + FRAME_SIZE).min(input.len())];
        frame[..chunk.len()].copy_from_slice(chunk);
        let attenuation = [48.0, 0.0, 3.0, 60.0][index / 275];
        let reset = index == 0 || index == 550;
        scalar
            .frame(&frame, &mut scalar_output, attenuation, reset)
            .unwrap();
        cooperative
            .frame(&frame, &mut cooperative_output, attenuation, reset)
            .unwrap();
        for (&actual, &expected) in cooperative_output.iter().zip(&scalar_output) {
            assert!(actual.is_finite() && expected.is_finite());
            let error = (actual - expected).abs();
            assert!(error < 2.0e-5, "frame {index}: error {error}");
            maximum = maximum.max(error);
            squared += f64::from(error).powi(2);
            samples += 1;
        }
    }
    let rms = (squared / samples as f64).sqrt();
    assert!(rms < 2.0e-6, "maximum {maximum:e}, RMS {rms:e}");
    eprintln!("pointwise scalar/cooperative: maximum={maximum:e}, RMS={rms:e}");
}
