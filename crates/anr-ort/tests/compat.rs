use anr_ort::{ModelTarget, ort, prepare_model};
use onnx_protobuf::{
    AttributeProto, GraphProto, Message, ModelProto, NodeProto, OperatorSetIdProto, TensorProto,
    TensorShapeProto, TypeProto, ValueInfoProto, attribute_proto::AttributeType,
    tensor_shape_proto::Dimension,
};
use protobuf::MessageField;
use std::sync::LazyLock;

#[path = "compat/nonzero.rs"]
mod nonzero;
#[path = "compat/simplify.rs"]
mod simplify;

static ENVIRONMENT: LazyLock<ort::environment::Environment> = LazyLock::new(|| {
    anr_ort::initialize_runtime().unwrap();
    ort::init().build().unwrap()
});

fn port(name: &str, dimensions: &[i64], element_type: i32) -> ValueInfoProto {
    let mut dtype = TypeProto::new();
    let tensor = dtype.mut_tensor_type();
    tensor.elem_type = element_type;
    tensor.shape = MessageField::some(TensorShapeProto {
        dim: dimensions
            .iter()
            .map(|&length| {
                let mut dimension = Dimension::new();
                if length >= 0 {
                    dimension.set_dim_value(length);
                }
                dimension
            })
            .collect(),
        ..Default::default()
    });
    ValueInfoProto {
        name: name.into(),
        type_: MessageField::some(dtype),
        ..Default::default()
    }
}

fn node(op: &str, inputs: &[&str], outputs: &[&str]) -> NodeProto {
    NodeProto {
        op_type: op.into(),
        input: inputs.iter().map(|s| (*s).into()).collect(),
        output: outputs.iter().map(|s| (*s).into()).collect(),
        ..Default::default()
    }
}

fn floats(name: &str, shape: &[i64], data: &[f32]) -> TensorProto {
    TensorProto {
        name: name.into(),
        dims: shape.into(),
        data_type: 1,
        float_data: data.into(),
        ..Default::default()
    }
}

fn integers(name: &str, shape: &[i64], data: &[i64]) -> TensorProto {
    TensorProto {
        name: name.into(),
        dims: shape.into(),
        data_type: 7,
        int64_data: data.into(),
        ..Default::default()
    }
}

fn model(graph: GraphProto) -> Vec<u8> {
    ModelProto {
        ir_version: 8,
        opset_import: vec![
            OperatorSetIdProto {
                version: 17,
                ..Default::default()
            },
            OperatorSetIdProto {
                domain: "com.microsoft".into(),
                version: 1,
                ..Default::default()
            },
        ],
        graph: MessageField::some(graph),
        ..Default::default()
    }
    .write_to_bytes()
    .unwrap()
}

fn session(bytes: &[u8], target: ModelTarget) -> ort::session::Session {
    let prepared = prepare_model(bytes, target).unwrap();
    ort::session::Session::builder(&ENVIRONMENT)
        .unwrap()
        .with_no_environment_execution_providers()
        .unwrap()
        .with_intra_threads(1)
        .unwrap()
        .commit_from_memory(&prepared)
        .unwrap()
}

fn gpu_targets() -> [ModelTarget; 3] {
    [
        ModelTarget::Cuda,
        ModelTarget::TensorRT,
        ModelTarget::DirectML,
    ]
}

fn prepared_graph(bytes: &[u8], target: ModelTarget) -> GraphProto {
    ModelProto::parse_from_bytes(&prepare_model(bytes, target).unwrap())
        .unwrap()
        .graph
        .unwrap()
}

fn boolean(name: &str, shape: &[i64], data: &[bool]) -> TensorProto {
    TensorProto {
        name: name.into(),
        dims: shape.into(),
        data_type: 9,
        raw_data: data.iter().map(|&value| u8::from(value)).collect(),
        ..Default::default()
    }
}

#[test]
fn sigmoid_convolution_preserves_bias_and_optional_residual() {
    for residual in [false, true] {
        let inputs = if residual {
            vec!["x", "w", "bias", "z"]
        } else {
            vec!["x", "w", "bias"]
        };
        let mut conv = node("FusedConv", &inputs, &["y"]);
        conv.domain = "com.microsoft".into();
        conv.attribute.push(AttributeProto {
            name: "activation".into(),
            type_: AttributeType::STRING.into(),
            s: b"Sigmoid".into(),
            ..Default::default()
        });
        let bytes = model(GraphProto {
            name: "sigmoid-convolution".into(),
            node: vec![conv],
            input: vec![port("x", &[1, 1, 3], 1)],
            output: vec![port("y", &[1, 1, 3], 1)],
            initializer: vec![
                floats("w", &[1, 1, 1], &[2.0]),
                floats("bias", &[1], &[1.0]),
                floats("z", &[1, 1, 3], &[1.0, -1.0, 0.0]),
            ],
            ..Default::default()
        });
        let mut session = session(&bytes, ModelTarget::Cuda);
        let x = ort::value::Tensor::from_array(([1, 1, 3], vec![-1.0_f32, 0.0, 1.0])).unwrap();
        let outputs = session.run(ort::inputs!["x" => x]).unwrap();
        let actual = outputs["y"].try_extract_tensor::<f32>().unwrap().1;
        let expected = if residual {
            [0.5, 0.5, 0.952_574_13]
        } else {
            [0.268_941_43, 0.731_058_6, 0.952_574_13]
        };
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-6, "{actual} != {expected}");
        }
    }
}

#[test]
fn static_sequence_indices_preserve_unequal_lengths_and_negative_index() {
    let bytes = model(GraphProto {
        name: "unequal-state-splits".into(),
        node: vec![
            node("SplitToSequence", &["x", "lengths"], &["sequence"]),
            node("SequenceAt", &["sequence", "last_index"], &["last"]),
            node("SequenceAt", &["sequence", "first_index"], &["first"]),
            node("SequenceAt", &["sequence", "middle_index"], &["middle"]),
        ],
        input: vec![port("x", &[6], 1)],
        output: vec![
            port("last", &[3], 1),
            port("first", &[2], 1),
            port("middle", &[1], 1),
        ],
        initializer: vec![
            integers("lengths", &[3], &[2, 1, 3]),
            integers("last_index", &[], &[-1]),
            integers("first_index", &[], &[0]),
            integers("middle_index", &[], &[1]),
        ],
        ..Default::default()
    });
    let mut session = session(&bytes, ModelTarget::TensorRT);
    let x = ort::value::Tensor::from_array(([6], vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0])).unwrap();
    let output = session.run(ort::inputs!["x" => x]).unwrap();
    assert_eq!(
        output["last"].try_extract_tensor::<f32>().unwrap().1,
        &[4.0, 5.0, 6.0]
    );
    assert_eq!(
        output["first"].try_extract_tensor::<f32>().unwrap().1,
        &[1.0, 2.0]
    );
    assert_eq!(
        output["middle"].try_extract_tensor::<f32>().unwrap().1,
        &[3.0]
    );
}

fn branch_attribute(name: &str, graph: GraphProto) -> AttributeProto {
    AttributeProto {
        name: name.into(),
        type_: AttributeType::GRAPH.into(),
        g: MessageField::some(graph),
        ..Default::default()
    }
}

#[test]
fn conditional_preserves_live_condition_and_branch_local_names() {
    let mut conditional = node("If", &["condition"], &["selected"]);
    conditional.attribute = vec![
        branch_attribute(
            "then_branch",
            GraphProto {
                name: "then".into(),
                node: vec![node("Add", &["x", "bias"], &["branch_y"])],
                initializer: vec![floats("bias", &[2], &[1.0, 2.0])],
                output: vec![port("branch_y", &[2], 1)],
                ..Default::default()
            },
        ),
        branch_attribute(
            "else_branch",
            GraphProto {
                name: "else".into(),
                node: vec![node("Sub", &["x", "bias"], &["branch_y"])],
                initializer: vec![floats("bias", &[2], &[10.0, 20.0])],
                output: vec![port("branch_y", &[2], 1)],
                ..Default::default()
            },
        ),
    ];
    let bytes = model(GraphProto {
        name: "dynamic-condition".into(),
        node: vec![conditional, node("Add", &["selected", "offset"], &["y"])],
        input: vec![port("condition", &[1, 1], 9), port("x", &[2], 1)],
        output: vec![port("y", &[2], 1)],
        initializer: vec![floats("offset", &[2], &[100.0, 200.0])],
        ..Default::default()
    });
    let mut session = session(&bytes, ModelTarget::TensorRT);
    for (condition, expected) in [
        (true, [104.0, 209.0]),
        (false, [93.0, 187.0]),
        (true, [104.0, 209.0]),
    ] {
        let condition = ort::value::Tensor::from_array(([1, 1], vec![condition])).unwrap();
        let x = ort::value::Tensor::from_array(([2], vec![3.0_f32, 7.0])).unwrap();
        let output = session
            .run(ort::inputs!["condition" => condition, "x" => x])
            .unwrap();
        let (shape, values) = output["y"].try_extract_tensor::<f32>().unwrap();
        assert_eq!(&shape[..], &[2]);
        assert_eq!(values, &expected);
    }
}

#[test]
fn runtime_sequence_index_is_not_frozen_during_preparation() {
    let bytes = model(GraphProto {
        name: "dynamic-sequence-index".into(),
        node: vec![
            node("SplitToSequence", &["x", "lengths"], &["sequence"]),
            node("SequenceAt", &["sequence", "index"], &["y"]),
        ],
        input: vec![port("x", &[6], 1), port("index", &[], 7)],
        output: vec![port("y", &[-1], 1)],
        initializer: vec![integers("lengths", &[3], &[2, 1, 3])],
        ..Default::default()
    });
    let mut session = session(&bytes, ModelTarget::TensorRT);
    for (index, expected) in [(0_i64, &[1.0_f32, 2.0][..]), (2, &[4.0, 5.0, 6.0][..])] {
        let x =
            ort::value::Tensor::from_array(([6], vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0])).unwrap();
        let index = ort::value::Tensor::from_array((Vec::<usize>::new(), vec![index])).unwrap();
        let output = session
            .run(ort::inputs!["x" => x, "index" => index])
            .unwrap();
        assert_eq!(output["y"].try_extract_tensor::<f32>().unwrap().1, expected);
    }
}

#[test]
fn unequal_branch_shapes_keep_conditional_execution() {
    let mut conditional = node("If", &["condition"], &["y"]);
    conditional.attribute = vec![
        branch_attribute(
            "then_branch",
            GraphProto {
                name: "short".into(),
                node: vec![node("Identity", &["value"], &["result"])],
                initializer: vec![floats("value", &[2], &[1.0, 2.0])],
                output: vec![port("result", &[2], 1)],
                ..Default::default()
            },
        ),
        branch_attribute(
            "else_branch",
            GraphProto {
                name: "long".into(),
                node: vec![node("Identity", &["value"], &["result"])],
                initializer: vec![floats("value", &[3], &[3.0, 4.0, 5.0])],
                output: vec![port("result", &[3], 1)],
                ..Default::default()
            },
        ),
    ];
    let bytes = model(GraphProto {
        name: "variable-output-length".into(),
        node: vec![conditional],
        input: vec![port("condition", &[], 9)],
        output: vec![port("y", &[-1], 1)],
        ..Default::default()
    });
    let mut session = session(&bytes, ModelTarget::TensorRT);
    for (condition, expected) in [(true, &[1.0_f32, 2.0][..]), (false, &[3.0, 4.0, 5.0][..])] {
        let condition =
            ort::value::Tensor::from_array((Vec::<usize>::new(), vec![condition])).unwrap();
        let output = session.run(ort::inputs!["condition" => condition]).unwrap();
        assert_eq!(output["y"].try_extract_tensor::<f32>().unwrap().1, expected);
    }
}

#[cfg(any(
    feature = "cuda",
    feature = "tensorrt",
    all(windows, feature = "directml")
))]
#[test]
#[ignore = "requires the supplied DFN3 model and GPU runtimes; set ANR_DFN3_MODEL"]
fn gpu_model_matches_original_cpu_while_strength_changes_and_stream_resets() {
    use anr_core::NoiseReduction;
    use anr_ort::DFN3;

    let path = std::env::var_os("ANR_DFN3_MODEL").expect("set ANR_DFN3_MODEL");
    let providers = [
        #[cfg(feature = "cuda")]
        ("cuda", ort::ep::CUDA::default().build()),
        #[cfg(feature = "tensorrt")]
        ("tensorrt", ort::ep::TensorRT::default().build()),
        #[cfg(all(windows, feature = "directml"))]
        ("directml", ort::ep::DirectML::default().build()),
    ];
    for (name, provider) in providers {
        let mut builder = ort::session::Session::builder(&ENVIRONMENT)
            .unwrap()
            .with_no_environment_execution_providers()
            .unwrap()
            .with_intra_threads(1)
            .unwrap()
            .with_parallel_execution(false)
            .unwrap();
        if name == "directml" {
            builder = builder.with_memory_pattern(false).unwrap();
        }
        let session = anr_ort::session_from_file(builder, &path, &[provider.error_on_failure()])
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        let mut actual = DFN3::new(session).unwrap();
        let mut reference = DFN3::from_file(&ENVIRONMENT, &path).unwrap();
        let mut max_error = 0.0_f32;
        for frame in 0..65 {
            if frame == 32 {
                actual.reset();
                reference.reset();
            }
            let db = [0.0, 3.0, 48.0, 60.0][(frame / 8) % 4];
            actual.set_atten_lim_db(db);
            reference.set_atten_lim_db(db);
            let count = if frame == 64 { 13 } else { 480 };
            let input: Vec<f32> = (0..count)
                .map(|i| {
                    let sample = (frame * 480 + i) as f32;
                    if frame < 3 || (29..32).contains(&frame) {
                        0.0
                    } else {
                        (sample * 0.057_595_864).sin() * 0.08 + (sample * 0.183_259_58).cos() * 0.02
                    }
                })
                .collect();
            let mut want = vec![0.0; count];
            let mut got = vec![0.0; count];
            reference.run(48_000, &input, &mut want).unwrap();
            actual.run(48_000, &input, &mut got).unwrap();
            for (i, (want, got)) in want.iter().zip(&got).enumerate() {
                let error = (want - got).abs();
                assert!(
                    got.is_finite() && error < 2e-5,
                    "{name}, frame {frame}, sample {i}, {db} dB: {got} != {want}"
                );
                max_error = max_error.max(error);
            }
        }
        eprintln!(
            "{name}: 65 frames, live 0/3/48/60 dB, reset and tail; max abs error {max_error:e}"
        );
    }
}

#[test]
fn an_unselected_out_of_bounds_gather_does_not_become_eager() {
    let mut conditional = node("If", &["condition"], &["y"]);
    conditional.attribute = vec![
        branch_attribute(
            "then_branch",
            GraphProto {
                name: "guarded-indexing".into(),
                node: vec![node("Gather", &["x", "indices"], &["result"])],
                output: vec![port("result", &[2], 1)],
                ..Default::default()
            },
        ),
        branch_attribute(
            "else_branch",
            GraphProto {
                name: "safe-identity".into(),
                node: vec![node("Identity", &["x"], &["result"])],
                output: vec![port("result", &[2], 1)],
                ..Default::default()
            },
        ),
    ];
    let bytes = model(GraphProto {
        name: "conditional-indexing".into(),
        node: vec![conditional],
        input: vec![port("condition", &[], 9), port("x", &[2], 1)],
        output: vec![port("y", &[2], 1)],
        initializer: vec![integers("indices", &[2], &[0, 5])],
        ..Default::default()
    });
    let mut session = session(&bytes, ModelTarget::TensorRT);
    let condition = ort::value::Tensor::from_array((Vec::<usize>::new(), vec![false])).unwrap();
    let x = ort::value::Tensor::from_array(([2], vec![3.0_f32, 7.0])).unwrap();
    let output = session
        .run(ort::inputs!["condition" => condition, "x" => x])
        .unwrap();
    assert_eq!(
        output["y"].try_extract_tensor::<f32>().unwrap().1,
        &[3.0, 7.0]
    );
}

#[test]
fn an_unselected_incompatible_broadcast_does_not_become_eager() {
    let mut conditional = node("If", &["condition"], &["y"]);
    conditional.attribute = vec![
        branch_attribute(
            "then_branch",
            GraphProto {
                name: "guarded-broadcast".into(),
                node: vec![node("Add", &["x", "bias"], &["result"])],
                initializer: vec![floats("bias", &[2], &[1.0, 2.0])],
                output: vec![port("result", &[2], 1)],
                ..Default::default()
            },
        ),
        branch_attribute(
            "else_branch",
            GraphProto {
                name: "constant-result".into(),
                node: vec![node("Identity", &["value"], &["result"])],
                initializer: vec![floats("value", &[2], &[8.0, 9.0])],
                output: vec![port("result", &[2], 1)],
                ..Default::default()
            },
        ),
    ];
    let bytes = model(GraphProto {
        name: "conditional-broadcast".into(),
        node: vec![conditional],
        input: vec![port("condition", &[], 9), port("x", &[-1], 1)],
        output: vec![port("y", &[2], 1)],
        ..Default::default()
    });
    let mut session = session(&bytes, ModelTarget::TensorRT);
    let condition = ort::value::Tensor::from_array((Vec::<usize>::new(), vec![false])).unwrap();
    let x = ort::value::Tensor::from_array(([3], vec![3.0_f32, 7.0, 11.0])).unwrap();
    let output = session
        .run(ort::inputs!["condition" => condition, "x" => x])
        .unwrap();
    assert_eq!(
        output["y"].try_extract_tensor::<f32>().unwrap().1,
        &[8.0, 9.0]
    );
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a CUDA runtime and GPU"]
fn gpu_external_weights_keep_the_callers_directory() {
    use onnx_protobuf::{StringStringEntryProto, tensor_proto::DataLocation};

    let directory = tempfile::tempdir().unwrap();
    let models = directory.path().join("models");
    let weights = directory.path().join("separate-weights");
    std::fs::create_dir(&models).unwrap();
    std::fs::create_dir(&weights).unwrap();
    std::fs::write(
        weights.join("weights.bin"),
        [2.0_f32, 3.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let bytes = model(GraphProto {
        name: "external-initializer".into(),
        node: vec![node("Mul", &["x", "weight"], &["y"])],
        input: vec![port("x", &[2], 1)],
        output: vec![port("y", &[2], 1)],
        initializer: vec![TensorProto {
            name: "weight".into(),
            dims: vec![2],
            data_type: 1,
            data_location: DataLocation::EXTERNAL.into(),
            external_data: vec![
                StringStringEntryProto {
                    key: "location".into(),
                    value: "weights.bin".into(),
                    ..Default::default()
                },
                StringStringEntryProto {
                    key: "offset".into(),
                    value: "0".into(),
                    ..Default::default()
                },
                StringStringEntryProto {
                    key: "length".into(),
                    value: "8".into(),
                    ..Default::default()
                },
            ],
            ..Default::default()
        }],
        ..Default::default()
    });
    let path = models.join("external.onnx");
    std::fs::write(&path, bytes).unwrap();
    let builder = ort::session::Session::builder(&ENVIRONMENT)
        .unwrap()
        .with_no_environment_execution_providers()
        .unwrap()
        .with_config_entry(
            "session.model_external_initializers_file_folder_path",
            weights.to_str().unwrap(),
        )
        .unwrap();
    let mut session =
        anr_ort::session_from_file(builder, &path, &[ort::ep::CUDA::default().build()]).unwrap();
    let x = ort::value::Tensor::from_array(([2], vec![1.5_f32, -2.0])).unwrap();
    let output = session.run(ort::inputs!["x" => x]).unwrap();
    assert_eq!(
        output["y"].try_extract_tensor::<f32>().unwrap().1,
        &[3.0, -6.0]
    );
}

#[test]
fn relu_convolution_is_portable_without_losing_bias_or_residual() {
    for residual in [false, true] {
        let inputs = if residual {
            vec!["x", "w", "bias", "z"]
        } else {
            vec!["x", "w", "bias"]
        };
        let mut convolution = node("FusedConv", &inputs, &["y"]);
        convolution.domain = "com.microsoft".into();
        convolution.attribute.push(AttributeProto {
            name: "activation".into(),
            type_: AttributeType::STRING.into(),
            s: b"Relu".into(),
            ..Default::default()
        });
        let bytes = model(GraphProto {
            name: "portable-relu-convolution".into(),
            node: vec![convolution],
            input: vec![port("x", &[1, 1, 3], 1)],
            output: vec![port("y", &[1, 1, 3], 1)],
            initializer: vec![
                floats("w", &[1, 1, 1], &[2.0]),
                floats("bias", &[1], &[1.0]),
                floats("z", &[1, 1, 3], &[1.0, -1.0, 0.0]),
            ],
            ..Default::default()
        });
        for target in [ModelTarget::TensorRT, ModelTarget::DirectML] {
            let graph = prepared_graph(&bytes, target);
            assert!(
                graph
                    .node
                    .iter()
                    .all(|node| { node.domain != "com.microsoft" || node.op_type != "FusedConv" })
            );
            let mut session = session(&bytes, target);
            let input =
                ort::value::Tensor::from_array(([1, 1, 3], vec![-1.0_f32, 0.0, 1.0])).unwrap();
            let output = session.run(ort::inputs!["x" => input]).unwrap();
            let expected = if residual {
                [0.0, 0.0, 3.0]
            } else {
                [0.0, 1.0, 3.0]
            };
            assert_eq!(
                output["y"].try_extract_tensor::<f32>().unwrap().1,
                &expected
            );
        }
    }
}

#[test]
fn portable_scaled_matmul_preserves_batched_vector_products() {
    let mut multiplication = node("FusedMatMul", &["x", "w"], &["y"]);
    multiplication.domain = "com.microsoft".into();
    multiplication.attribute.push(AttributeProto {
        name: "alpha".into(),
        type_: AttributeType::FLOAT.into(),
        f: 0.25,
        ..Default::default()
    });
    let bytes = model(GraphProto {
        name: "portable-batched-scaled-matmul".into(),
        node: vec![multiplication],
        input: vec![port("x", &[3], 1)],
        output: vec![port("y", &[2, 2], 1)],
        initializer: vec![floats(
            "w",
            &[2, 3, 2],
            &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0, -1.0, 0.0, 0.0, 2.0, 2.0, -1.0],
        )],
        ..Default::default()
    });
    for target in [ModelTarget::TensorRT, ModelTarget::DirectML] {
        let graph = prepared_graph(&bytes, target);
        assert!(
            graph
                .node
                .iter()
                .all(|node| { node.domain != "com.microsoft" || node.op_type != "FusedMatMul" })
        );
        let mut session = session(&bytes, target);
        let input = ort::value::Tensor::from_array(([3], vec![1.0_f32, 2.0, 3.0])).unwrap();
        let output = session.run(ort::inputs!["x" => input]).unwrap();
        let (shape, values) = output["y"].try_extract_tensor::<f32>().unwrap();
        assert_eq!(&shape[..], &[2, 2]);
        assert_eq!(values, &[1.0, 1.25, 1.25, 0.25]);
    }
}

#[test]
fn fixed_state_selection_stays_tensor_only_and_keeps_the_live_condition() {
    let mut conditional = node("If", &["condition"], &["y"]);
    conditional.attribute = vec![
        branch_attribute(
            "then_branch",
            GraphProto {
                name: "first-state".into(),
                node: vec![node("Identity", &["first"], &["result"])],
                output: vec![port("result", &[2], 1)],
                ..Default::default()
            },
        ),
        branch_attribute(
            "else_branch",
            GraphProto {
                name: "second-state".into(),
                node: vec![node("Identity", &["second"], &["result"])],
                output: vec![port("result", &[2], 1)],
                ..Default::default()
            },
        ),
    ];
    let bytes = model(GraphProto {
        name: "fixed-state-conditional".into(),
        node: vec![
            node("SplitToSequence", &["x", "lengths"], &["sequence"]),
            node("SequenceAt", &["sequence", "zero"], &["first"]),
            node("SequenceAt", &["sequence", "one"], &["second"]),
            conditional,
        ],
        input: vec![port("x", &[4], 1), port("condition", &[], 9)],
        output: vec![port("y", &[2], 1)],
        initializer: vec![
            integers("lengths", &[2], &[2, 2]),
            integers("zero", &[], &[0]),
            integers("one", &[], &[1]),
        ],
        ..Default::default()
    });
    for target in gpu_targets() {
        let graph = prepared_graph(&bytes, target);
        assert!(graph.node.iter().all(|node| !matches!(
            node.op_type.as_str(),
            "If" | "SplitToSequence" | "SequenceAt"
        )));
        let mut session = session(&bytes, target);
        for (condition, expected) in [(false, [3.0, 4.0]), (true, [1.0, 2.0])] {
            let input =
                ort::value::Tensor::from_array(([4], vec![1.0_f32, 2.0, 3.0, 4.0])).unwrap();
            let condition =
                ort::value::Tensor::from_array((Vec::<usize>::new(), vec![condition])).unwrap();
            let output = session
                .run(ort::inputs!["x" => input, "condition" => condition])
                .unwrap();
            assert_eq!(
                output["y"].try_extract_tensor::<f32>().unwrap().1,
                &expected
            );
        }
    }
}

#[test]
fn legacy_bfloat16_product_does_not_become_an_invalid_standard_schema() {
    let mut multiplication = node("FusedMatMul", &["x", "w"], &["y"]);
    multiplication.domain = "com.microsoft".into();
    let bytes = model(GraphProto {
        name: "legacy-bfloat16-matmul".into(),
        node: vec![multiplication],
        input: vec![port("x", &[2, 2], 16), port("w", &[2, 2], 16)],
        output: vec![port("y", &[2, 2], 16)],
        ..Default::default()
    });
    let mut parsed = ModelProto::parse_from_bytes(&bytes).unwrap();
    parsed.opset_import[0].version = 12;
    let bytes = parsed.write_to_bytes().unwrap();
    for target in [ModelTarget::TensorRT, ModelTarget::DirectML] {
        let prepared = prepare_model(&bytes, target).unwrap();
        let result = ort::session::Session::builder(&ENVIRONMENT)
            .unwrap()
            .with_no_environment_execution_providers()
            .unwrap()
            .with_intra_threads(1)
            .unwrap()
            .commit_from_memory(&prepared);
        if let Err(error) = result {
            // CPU may lack a BF16 kernel, but preparation must not invalidate
            // the schema of a graph that another provider can execute.
            assert_ne!(error.code(), ort::ErrorCode::InvalidGraph, "{error}");
        }
    }
}
