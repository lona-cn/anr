use super::*;

fn zero_check(shape: &[i64], coordinates_are_output: bool) -> Vec<u8> {
    let mut outputs = vec![port("is_zero", &[], 9)];
    if coordinates_are_output {
        outputs.push(port("coordinates", &[shape.len() as i64, -1], 7));
    }
    let mut coordinate_shape = node("Shape", &["coordinates"], &["coordinate_shape"]);
    coordinate_shape.attribute.push(AttributeProto {
        name: "start".into(),
        type_: AttributeType::INT.into(),
        i: 0,
        ..Default::default()
    });
    model(GraphProto {
        name: "state-zero-predicate".into(),
        node: vec![
            node("NonZero", &["x"], &["coordinates"]),
            coordinate_shape,
            node(
                "Gather",
                &["coordinate_shape", "permutation"],
                &["transposed_shape"],
            ),
            node("Gather", &["transposed_shape", "first"], &["count"]),
            node("Equal", &["count", "zero"], &["is_zero"]),
        ],
        input: vec![port("x", shape, 1)],
        output: outputs,
        initializer: vec![
            integers("permutation", &[2], &[1, 0]),
            integers("first", &[], &[0]),
            integers("zero", &[], &[0]),
        ],
        ..Default::default()
    })
}

#[test]
fn zero_predicate_avoids_dynamic_coordinates_and_tracks_reset_and_nonfinite_values() {
    for target in gpu_targets() {
        for shape in [vec![4], vec![1, 4, 1]] {
            let bytes = zero_check(&shape, false);
            let prepared = prepared_graph(&bytes, target);
            assert!(
                prepared.node.iter().all(|node| node.op_type != "NonZero"),
                "a zero-only predicate must not request dynamic coordinates"
            );
            let mut session = session(&bytes, target);
            for (values, expected) in [
                ([0.0, -0.0, 0.0, 0.0], true),
                ([0.0, 2.0, 0.0, 0.0], false),
                ([0.0, 0.0, 0.0, 0.0], true),
                ([0.0, f32::NAN, 0.0, 0.0], false),
                ([f32::INFINITY, 0.0, 0.0, 0.0], false),
                ([0.0, 0.0, f32::NEG_INFINITY, 0.0], false),
            ] {
                let dims: Vec<usize> = shape.iter().map(|&size| size as usize).collect();
                let input = ort::value::Tensor::from_array((dims, values.to_vec())).unwrap();
                let output = session.run(ort::inputs!["x" => input]).unwrap();
                let (dims, actual) = output["is_zero"].try_extract_tensor::<bool>().unwrap();
                assert!(dims.is_empty());
                assert_eq!(actual, &[expected]);
            }
        }
    }
}

#[test]
fn observable_coordinates_are_preserved_instead_of_replacing_nonzero() {
    let bytes = zero_check(&[3], true);
    for target in gpu_targets() {
        let mut session = session(&bytes, target);
        let input = ort::value::Tensor::from_array(([3], vec![0.0_f32, -2.0, 4.0])).unwrap();
        let output = session.run(ort::inputs!["x" => input]).unwrap();
        let (shape, coordinates) = output["coordinates"].try_extract_tensor::<i64>().unwrap();
        assert_eq!(&shape[..], &[1, 2]);
        assert_eq!(coordinates, &[1, 2]);
        assert_eq!(
            output["is_zero"].try_extract_tensor::<bool>().unwrap().1,
            &[false]
        );
    }
}

#[test]
fn direct_shape_count_check_is_lowered_through_local_views() {
    let mut graph = ModelProto::parse_from_bytes(&zero_check(&[4], false))
        .unwrap()
        .graph
        .unwrap();
    graph.node.remove(2);
    graph.node[2].input[0] = "coordinate_shape".into();
    graph.initializer[1].int64_data = vec![1];
    graph.node[0].input[0] = "view".into();
    graph.node.insert(0, node("Identity", &["x"], &["view"]));
    let bytes = model(graph);
    for target in gpu_targets() {
        let prepared = prepared_graph(&bytes, target);
        assert!(prepared.node.iter().all(|node| node.op_type != "NonZero"));
        let mut session = session(&bytes, target);
        for (values, expected) in [
            ([0.0_f32, -0.0, 0.0, 0.0], true),
            ([0.0, 0.0, f32::NAN, 0.0], false),
        ] {
            let input = ort::value::Tensor::from_array(([4], values.to_vec())).unwrap();
            let outputs = session.run(ort::inputs!["x" => input]).unwrap();
            assert_eq!(
                outputs["is_zero"].try_extract_tensor::<bool>().unwrap().1,
                &[expected]
            );
        }
    }
}

#[test]
fn observable_intermediate_counts_and_shapes_keep_the_coordinate_chain() {
    for (name, dims, expected) in [
        ("coordinate_shape", vec![2], vec![1_i64, 2]),
        ("transposed_shape", vec![2], vec![2, 1]),
        ("count", vec![], vec![2]),
    ] {
        let mut graph = ModelProto::parse_from_bytes(&zero_check(&[3], false))
            .unwrap()
            .graph
            .unwrap();
        graph.output.push(port(name, &dims, 7));
        let bytes = model(graph);
        for target in gpu_targets() {
            let prepared = prepared_graph(&bytes, target);
            assert!(prepared.node.iter().any(|node| node.op_type == "NonZero"));
            let mut session = session(&bytes, target);
            let input = ort::value::Tensor::from_array(([3], vec![0.0_f32, -2.0, 4.0])).unwrap();
            let outputs = session.run(ort::inputs!["x" => input]).unwrap();
            assert_eq!(
                outputs[name].try_extract_tensor::<i64>().unwrap().1,
                expected.as_slice()
            );
        }
    }
}

#[test]
fn count_captured_only_inside_graph_attributes_is_not_removed() {
    let mut graph = ModelProto::parse_from_bytes(&zero_check(&[3], false))
        .unwrap()
        .graph
        .unwrap();
    graph.input.push(port("condition", &[], 9));
    graph.output.push(port("captured_count", &[], 7));
    let mut conditional = node("If", &["condition"], &["captured_count"]);
    for branch_name in ["then_branch", "else_branch"] {
        conditional.attribute.push(AttributeProto {
            name: branch_name.into(),
            type_: AttributeType::GRAPH.into(),
            g: MessageField::some(GraphProto {
                name: branch_name.into(),
                node: vec![node("Identity", &["count"], &["branch_count"])],
                output: vec![port("branch_count", &[], 7)],
                ..Default::default()
            }),
            ..Default::default()
        });
    }
    graph.node.push(conditional);
    let bytes = model(graph);
    for target in gpu_targets() {
        let prepared = prepared_graph(&bytes, target);
        assert!(prepared.node.iter().any(|node| node.op_type == "NonZero"));
        let mut session = session(&bytes, target);
        for condition in [false, true] {
            let input = ort::value::Tensor::from_array(([3], vec![0.0_f32, -2.0, 4.0])).unwrap();
            let condition =
                ort::value::Tensor::from_array((Vec::<usize>::new(), vec![condition])).unwrap();
            let outputs = session
                .run(ort::inputs!["x" => input, "condition" => condition])
                .unwrap();
            assert_eq!(
                outputs["captured_count"]
                    .try_extract_tensor::<i64>()
                    .unwrap()
                    .1,
                &[2]
            );
        }
    }
}
