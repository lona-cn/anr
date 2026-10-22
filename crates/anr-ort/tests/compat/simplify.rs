use super::*;

#[test]
fn a_dominating_constant_removes_dead_selection_without_changing_public_ports() {
    let bytes = model(GraphProto {
        name: "constant-state-selection".into(),
        node: vec![
            node("Greater", &["flag", "zero"], &["positive"]),
            node("Or", &["positive", "always"], &["condition"]),
            node("Neg", &["x"], &["unused"]),
            node("Where", &["condition", "x", "unused"], &["y"]),
        ],
        input: vec![port("x", &[2], 1), port("flag", &[], 1)],
        output: vec![port("y", &[2], 1)],
        initializer: vec![floats("zero", &[], &[0.0]), boolean("always", &[], &[true])],
        ..Default::default()
    });
    for target in gpu_targets() {
        let graph = prepared_graph(&bytes, target);
        assert!(
            graph
                .node
                .iter()
                .all(|node| !matches!(node.op_type.as_str(), "Where" | "Or" | "Greater" | "Neg")),
            "constant selection must not retain its discarded data dependency"
        );
        assert_eq!(graph.input[0].name, "x");
        assert_eq!(graph.output[0].name, "y");
        let mut session = session(&bytes, target);
        let input = ort::value::Tensor::from_array(([2], vec![-3.0_f32, 7.0])).unwrap();
        let flag = ort::value::Tensor::from_array((Vec::<usize>::new(), vec![-1.0_f32])).unwrap();
        let output = session
            .run(ort::inputs!["x" => input, "flag" => flag])
            .unwrap();
        assert_eq!(
            output["y"].try_extract_tensor::<f32>().unwrap().1,
            &[-3.0, 7.0]
        );
    }
}

#[test]
fn constant_where_preserves_broadcast_of_a_selected_scalar() {
    let bytes = model(GraphProto {
        name: "scalar-selection-broadcast".into(),
        node: vec![node("Where", &["condition", "selected", "other"], &["y"])],
        input: vec![port("selected", &[], 1), port("other", &[2], 1)],
        output: vec![port("y", &[2], 1)],
        initializer: vec![boolean("condition", &[], &[true])],
        ..Default::default()
    });
    for target in gpu_targets() {
        let graph = prepared_graph(&bytes, target);
        assert!(graph.node.iter().all(|node| node.op_type != "Where"));
        let mut session = session(&bytes, target);
        let selected =
            ort::value::Tensor::from_array((Vec::<usize>::new(), vec![7.0_f32])).unwrap();
        let other = ort::value::Tensor::from_array(([2], vec![1.0_f32, 2.0])).unwrap();
        let output = session
            .run(ort::inputs!["selected" => selected, "other" => other])
            .unwrap();
        let (shape, values) = output["y"].try_extract_tensor::<f32>().unwrap();
        assert_eq!(&shape[..], &[2]);
        assert_eq!(values, &[7.0, 7.0]);
    }
}

#[test]
fn overridable_boolean_initializer_stays_live() {
    let bytes = model(GraphProto {
        name: "overridable-selection".into(),
        node: vec![node("Where", &["condition", "x", "other"], &["y"])],
        input: vec![port("condition", &[], 9), port("x", &[2], 1)],
        output: vec![port("y", &[2], 1)],
        initializer: vec![
            boolean("condition", &[], &[true]),
            floats("other", &[2], &[-1.0, -2.0]),
        ],
        ..Default::default()
    });
    for target in gpu_targets() {
        let mut session = session(&bytes, target);
        for (condition, expected) in [(false, [-1.0, -2.0]), (true, [3.0, 4.0])] {
            let condition =
                ort::value::Tensor::from_array((Vec::<usize>::new(), vec![condition])).unwrap();
            let input = ort::value::Tensor::from_array(([2], vec![3.0_f32, 4.0])).unwrap();
            let output = session
                .run(ort::inputs!["condition" => condition, "x" => input])
                .unwrap();
            assert_eq!(
                output["y"].try_extract_tensor::<f32>().unwrap().1,
                &expected
            );
        }
    }
}

#[test]
fn dead_branch_cleanup_keeps_values_captured_by_a_remaining_conditional() {
    let mut conditional = node("If", &["condition"], &["captured"]);
    conditional.attribute = vec![
        branch_attribute(
            "then_branch",
            GraphProto {
                name: "captured-long".into(),
                node: vec![node("Identity", &["sum"], &["result"])],
                output: vec![port("result", &[2], 1)],
                ..Default::default()
            },
        ),
        branch_attribute(
            "else_branch",
            GraphProto {
                name: "captured-short".into(),
                node: vec![node("Identity", &["short"], &["result"])],
                initializer: vec![floats("short", &[1], &[9.0])],
                output: vec![port("result", &[1], 1)],
                ..Default::default()
            },
        ),
    ];
    let bytes = model(GraphProto {
        name: "implicit-branch-capture".into(),
        node: vec![
            node("Add", &["x", "offset"], &["sum"]),
            node("Where", &["always", "x", "sum"], &["selected"]),
            conditional,
        ],
        input: vec![port("x", &[2], 1), port("condition", &[], 9)],
        output: vec![port("selected", &[2], 1), port("captured", &[-1], 1)],
        initializer: vec![
            floats("offset", &[2], &[10.0, 20.0]),
            boolean("always", &[], &[true]),
        ],
        ..Default::default()
    });
    for target in gpu_targets() {
        let graph = prepared_graph(&bytes, target);
        assert!(graph.node.iter().all(|node| node.op_type != "Where"));
        let mut session = session(&bytes, target);
        let condition = ort::value::Tensor::from_array((Vec::<usize>::new(), vec![true])).unwrap();
        let input = ort::value::Tensor::from_array(([2], vec![3.0_f32, 4.0])).unwrap();
        let output = session
            .run(ort::inputs!["x" => input, "condition" => condition])
            .unwrap();
        assert_eq!(
            output["selected"].try_extract_tensor::<f32>().unwrap().1,
            &[3.0, 4.0]
        );
        assert_eq!(
            output["captured"].try_extract_tensor::<f32>().unwrap().1,
            &[13.0, 24.0]
        );
    }
}

#[test]
fn a_false_constant_of_shape_folds_and_without_collapsing_the_boolean_output() {
    let mut fill = node("ConstantOfShape", &["shape"], &["never"]);
    fill.attribute.push(AttributeProto {
        name: "value".into(),
        type_: AttributeType::TENSOR.into(),
        t: MessageField::some(boolean("", &[1], &[false])),
        ..Default::default()
    });
    let bytes = model(GraphProto {
        name: "boolean-output-broadcast".into(),
        node: vec![fill, node("And", &["flag", "never"], &["y"])],
        input: vec![port("flag", &[2], 9)],
        output: vec![port("y", &[2], 9)],
        initializer: vec![integers("shape", &[1], &[1])],
        ..Default::default()
    });
    for target in gpu_targets() {
        let graph = prepared_graph(&bytes, target);
        assert!(graph.node.iter().all(|node| node.op_type != "And"));
        assert_eq!(graph.input[0].name, "flag");
        assert_eq!(graph.output[0].name, "y");
        let mut session = session(&bytes, target);
        let flag = ort::value::Tensor::from_array(([2], vec![true, true])).unwrap();
        let output = session.run(ort::inputs!["flag" => flag]).unwrap();
        let (shape, values) = output["y"].try_extract_tensor::<bool>().unwrap();
        assert_eq!(&shape[..], &[2]);
        assert_eq!(values, &[false, false]);
    }
}

#[test]
fn independent_unknown_dimensions_do_not_prove_a_where_identity() {
    let bytes = model(GraphProto {
        name: "dynamic-selection-broadcast".into(),
        node: vec![node("Where", &["condition", "selected", "other"], &["y"])],
        input: vec![port("selected", &[-1], 1), port("other", &[-1], 1)],
        output: vec![port("y", &[-1], 1)],
        initializer: vec![boolean("condition", &[], &[true])],
        ..Default::default()
    });
    for target in gpu_targets() {
        let graph = prepared_graph(&bytes, target);
        assert!(graph.node.iter().any(|node| node.op_type == "Where"));
        let mut session = session(&bytes, target);
        let selected = ort::value::Tensor::from_array(([1], vec![7.0_f32])).unwrap();
        let other = ort::value::Tensor::from_array(([2], vec![1.0_f32, 2.0])).unwrap();
        let output = session
            .run(ort::inputs!["selected" => selected, "other" => other])
            .unwrap();
        let (shape, values) = output["y"].try_extract_tensor::<f32>().unwrap();
        assert_eq!(&shape[..], &[2]);
        assert_eq!(values, &[7.0, 7.0]);
    }
}

#[test]
fn a_uniform_boolean_constant_keeps_the_dimensions_of_its_condition() {
    let mut condition = node("Constant", &[], &["condition"]);
    condition.attribute.push(AttributeProto {
        name: "value".into(),
        type_: AttributeType::TENSOR.into(),
        t: MessageField::some(TensorProto {
            data_type: 9,
            dims: vec![2],
            int32_data: vec![1, 1],
            ..Default::default()
        }),
        ..Default::default()
    });
    let bytes = model(GraphProto {
        name: "constant-condition-broadcast".into(),
        node: vec![
            condition,
            node("Where", &["condition", "selected", "other"], &["y"]),
        ],
        input: vec![port("selected", &[], 1), port("other", &[], 1)],
        output: vec![port("y", &[2], 1)],
        ..Default::default()
    });
    for target in gpu_targets() {
        let graph = prepared_graph(&bytes, target);
        assert!(graph.node.iter().all(|node| node.op_type != "Where"));
        let mut session = session(&bytes, target);
        let selected =
            ort::value::Tensor::from_array((Vec::<usize>::new(), vec![7.0_f32])).unwrap();
        let other = ort::value::Tensor::from_array((Vec::<usize>::new(), vec![2.0_f32])).unwrap();
        let output = session
            .run(ort::inputs!["selected" => selected, "other" => other])
            .unwrap();
        let (shape, values) = output["y"].try_extract_tensor::<f32>().unwrap();
        assert_eq!(&shape[..], &[2]);
        assert_eq!(values, &[7.0, 7.0]);
    }
}

#[test]
fn a_loop_body_input_is_not_replaced_by_a_same_named_outer_constant() {
    let mut loop_node = node("Loop", &["trip", "start", "x"], &["y"]);
    loop_node.attribute = vec![branch_attribute(
        "body",
        GraphProto {
            name: "local-condition".into(),
            node: vec![node("Where", &["condition", "state", "other"], &["result"])],
            input: vec![
                port("iteration", &[], 7),
                port("condition", &[], 9),
                port("state", &[2], 1),
            ],
            initializer: vec![
                boolean("continue", &[], &[false]),
                floats("other", &[2], &[3.0, 4.0]),
            ],
            output: vec![port("continue", &[], 9), port("result", &[2], 1)],
            ..Default::default()
        },
    )];
    let bytes = model(GraphProto {
        name: "shadowed-constant".into(),
        node: vec![loop_node],
        initializer: vec![
            integers("trip", &[], &[1]),
            boolean("start", &[], &[true]),
            boolean("condition", &[], &[false]),
        ],
        input: vec![port("x", &[2], 1)],
        output: vec![port("y", &[2], 1)],
        ..Default::default()
    });
    for target in gpu_targets() {
        let graph = prepared_graph(&bytes, target);
        let loop_node = graph
            .node
            .iter()
            .find(|node| node.op_type == "Loop")
            .unwrap();
        let body = loop_node
            .attribute
            .iter()
            .find(|attribute| attribute.name == "body")
            .unwrap();
        assert!(
            body.g
                .as_ref()
                .unwrap()
                .node
                .iter()
                .any(|node| node.op_type == "Where")
        );
        let mut session = session(&bytes, target);
        let input = ort::value::Tensor::from_array(([2], vec![1.0_f32, 2.0])).unwrap();
        let output = session.run(ort::inputs!["x" => input]).unwrap();
        assert_eq!(
            output["y"].try_extract_tensor::<f32>().unwrap().1,
            &[1.0, 2.0]
        );
    }
}

#[test]
fn a_local_function_named_or_is_not_treated_as_the_builtin_predicate() {
    let bytes = model(GraphProto {
        name: "function-predicate".into(),
        node: vec![
            node("Or", &["flag", "always"], &["condition"]),
            node("Where", &["condition", "x", "other"], &["y"]),
        ],
        input: vec![port("flag", &[], 9), port("x", &[2], 1)],
        output: vec![port("y", &[2], 1)],
        initializer: vec![
            boolean("always", &[], &[true]),
            floats("other", &[2], &[3.0, 4.0]),
        ],
        ..Default::default()
    });
    let mut parsed = ModelProto::parse_from_bytes(&bytes).unwrap();
    parsed.functions.push(onnx_protobuf::FunctionProto {
        name: "Or".into(),
        input: vec!["a".into(), "b".into()],
        output: vec!["result".into()],
        node: vec![node("And", &["a", "b"], &["result"])],
        opset_import: vec![OperatorSetIdProto {
            version: 17,
            ..Default::default()
        }],
        ..Default::default()
    });
    let bytes = parsed.write_to_bytes().unwrap();
    for target in gpu_targets() {
        let graph = prepared_graph(&bytes, target);
        assert!(graph.node.iter().any(|node| node.op_type == "Or"));
        assert!(graph.node.iter().any(|node| node.op_type == "Where"));
    }
}

#[test]
fn constant_selection_does_not_hide_an_unknown_operator_or_its_dependency() {
    let mut opaque = node("Stateful", &["x"], &["other"]);
    opaque.domain = "test.stateful".into();
    let bytes = model(GraphProto {
        name: "opaque-discarded-dependency".into(),
        node: vec![opaque, node("Where", &["always", "x", "other"], &["y"])],
        initializer: vec![boolean("always", &[], &[true])],
        input: vec![port("x", &[2], 1)],
        output: vec![port("y", &[2], 1)],
        value_info: vec![port("other", &[2], 1)],
        ..Default::default()
    });
    let mut parsed = ModelProto::parse_from_bytes(&bytes).unwrap();
    parsed.opset_import.push(OperatorSetIdProto {
        domain: "test.stateful".into(),
        version: 1,
        ..Default::default()
    });
    let bytes = parsed.write_to_bytes().unwrap();
    let reference_error = ort::session::Session::builder(&ENVIRONMENT)
        .unwrap()
        .with_no_environment_execution_providers()
        .unwrap()
        .with_intra_threads(1)
        .unwrap()
        .commit_from_memory(&bytes)
        .err()
        .expect("the original opaque operator is unsupported");
    for target in gpu_targets() {
        let prepared = prepare_model(&bytes, target).unwrap();
        let result = ort::session::Session::builder(&ENVIRONMENT)
            .unwrap()
            .with_no_environment_execution_providers()
            .unwrap()
            .with_intra_threads(1)
            .unwrap()
            .commit_from_memory(&prepared);
        let error = result
            .err()
            .expect("preparation must not hide an unsupported operator");
        assert_eq!(error.code(), reference_error.code());
        let graph = ModelProto::parse_from_bytes(&prepared)
            .unwrap()
            .graph
            .unwrap();
        assert!(graph.node.iter().any(|node| node.op_type == "Where"));
    }
}
