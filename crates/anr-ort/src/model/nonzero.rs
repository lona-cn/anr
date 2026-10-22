//! Replace private coordinate-count checks with a fixed-shape zero predicate.

use std::collections::{HashMap, HashSet};

use onnx_protobuf::{GraphProto, NodeProto, TensorProto};

use super::{
    CaptureShapes, Functions, Names, builtin, constant_tensor, input_tensor, int_attribute,
    integer_attribute, integers, operator, shape_dimensions, tensor_info,
};

struct Predicate {
    input: String,
    shape: Vec<i64>,
    nodes: Vec<usize>,
    equal: usize,
}

// References in graph attributes are uses too. Counting even locally shadowed
// references is conservative: a missed optimization is preferable to a lost capture.
fn captured_names<'a>(graph: &'a GraphProto, used: &mut HashSet<&'a str>) {
    used.extend(graph.output.iter().map(|output| output.name.as_str()));
    for node in &graph.node {
        used.extend(node.input.iter().map(String::as_str));
        for attribute in &node.attribute {
            if let Some(child) = attribute.g.as_ref() {
                captured_names(child, used);
            }
            for child in &attribute.graphs {
                captured_names(child, used);
            }
        }
    }
}

fn producer_map(graph: &GraphProto) -> HashMap<&str, usize> {
    let mut producers = HashMap::new();
    for (index, node) in graph.node.iter().enumerate() {
        for output in &node.output {
            if !output.is_empty() {
                producers.insert(output.as_str(), index);
            }
        }
    }
    producers
}

fn scalar_integer(graph: &GraphProto, name: &str, expected: i64, functions: &Functions) -> bool {
    constant_tensor(graph, name, functions).is_some_and(|tensor| {
        tensor.dims.is_empty() && integers(tensor).as_deref() == Some(&[expected])
    })
}

fn gather(node: &NodeProto, functions: &Functions) -> bool {
    builtin(node, "Gather", functions)
        && node.input.len() == 2
        && node.output.len() == 1
        && node
            .attribute
            .iter()
            .all(|attribute| attribute.name == "axis")
        && matches!(integer_attribute(node, "axis", 0), Some(0 | -1))
}

// Do not infer the type of a captured outer value from a child graph's output
// annotation. A local binding and local metadata (or a typed local source) are required.
fn local_float(
    graph: &GraphProto,
    name: &str,
    producers: &HashMap<&str, usize>,
    functions: &Functions,
    visiting: &mut HashSet<usize>,
) -> bool {
    if !graph.input.iter().any(|input| input.name == name)
        && !graph.initializer.iter().any(|tensor| tensor.name == name)
        && !producers.contains_key(name)
    {
        return false;
    }
    if let Some((dtype, _)) = input_tensor(graph, name) {
        return dtype == 1;
    }
    let Some(&index) = producers.get(name) else {
        return false;
    };
    if !visiting.insert(index) {
        return false;
    }
    let node = &graph.node[index];
    let preserves_type = match node.op_type.as_str() {
        "Identity" | "Transpose" | "Flatten" => node.input.len() == 1,
        "Reshape" | "Gather" => node.input.len() == 2,
        "Slice" => (3..=5).contains(&node.input.len()),
        "Squeeze" | "Unsqueeze" => (1..=2).contains(&node.input.len()),
        _ => false,
    };
    let result = preserves_type
        && builtin(node, &node.op_type, functions)
        && local_float(graph, &node.input[0], producers, functions, visiting);
    visiting.remove(&index);
    result
}

fn predicate(
    graph: &GraphProto,
    equal: usize,
    opset: i64,
    producers: &HashMap<&str, usize>,
    functions: &Functions,
) -> Option<Predicate> {
    let node = &graph.node[equal];
    if !builtin(node, "Equal", functions)
        || node.input.len() != 2
        || node.output.len() != 1
        || !node.attribute.is_empty()
    {
        return None;
    }
    let count = if scalar_integer(graph, &node.input[1], 0, functions) {
        &node.input[0]
    } else if scalar_integer(graph, &node.input[0], 0, functions) {
        &node.input[1]
    } else {
        return None;
    };
    let count_index = *producers.get(count.as_str())?;
    let count_node = &graph.node[count_index];
    if !gather(count_node, functions) {
        return None;
    }
    let mut nodes = vec![count_index];
    let mut source = count_node.input[0].as_str();
    if scalar_integer(graph, &count_node.input[1], 0, functions) {
        let permutation_index = *producers.get(source)?;
        let permutation = &graph.node[permutation_index];
        if !gather(permutation, functions) {
            return None;
        }
        let indices = constant_tensor(graph, &permutation.input[1], functions)?;
        if indices.dims != [2] || integers(indices).as_deref() != Some(&[1, 0]) {
            return None;
        }
        nodes.push(permutation_index);
        source = &permutation.input[0];
    } else if !scalar_integer(graph, &count_node.input[1], 1, functions) {
        return None;
    }
    let shape_index = *producers.get(source)?;
    let shape = &graph.node[shape_index];
    // NonZero coordinates have rank two; Shape must retain both dimensions.
    if !builtin(shape, "Shape", functions)
        || shape.input.len() != 1
        || shape.output.len() != 1
        || (!shape.attribute.is_empty() && opset < 15)
        || !shape_dimensions(shape, &[0, 0]).is_some_and(|dimensions| dimensions.len() == 2)
    {
        return None;
    }
    nodes.push(shape_index);
    let nonzero_index = *producers.get(shape.input[0].as_str())?;
    let nonzero = &graph.node[nonzero_index];
    if !builtin(nonzero, "NonZero", functions)
        || nonzero.input.len() != 1
        || nonzero.output.len() != 1
        || !nonzero.attribute.is_empty()
        || !local_float(
            graph,
            &nonzero.input[0],
            producers,
            functions,
            &mut HashSet::new(),
        )
    {
        return None;
    }
    let dimensions = CaptureShapes::new(graph, functions).shape(&nonzero.input[0])?;
    if dimensions.iter().any(|&dimension| dimension < 0) {
        return None;
    }
    nodes.push(nonzero_index);
    Some(Predicate {
        input: nonzero.input[0].clone(),
        shape: dimensions,
        nodes,
        equal,
    })
}

pub(super) fn lower_zero_predicates(
    graph: &mut GraphProto,
    opset: i64,
    names: &mut Names,
    functions: &Functions,
) -> bool {
    // Equal supports floating point here; ReduceSum uses the modern optional
    // axes input. Do not reinterpret an older schema or a locally overridden op.
    if opset < 13
        || ["Equal", "Not", "Cast", "ReduceSum"]
            .iter()
            .any(|op| !builtin(&operator(op, Vec::new(), Vec::new()), op, functions))
    {
        return false;
    }
    let mut changed = false;
    for node in &mut graph.node {
        for attribute in &mut node.attribute {
            if let Some(child) = attribute.g.as_mut() {
                changed |= lower_zero_predicates(child, opset, names, functions);
            }
            for child in &mut attribute.graphs {
                changed |= lower_zero_predicates(child, opset, names, functions);
            }
        }
    }

    let predicates = {
        let producers = producer_map(graph);
        let mut consumers = HashMap::<&str, (usize, usize)>::new();
        let mut observable = HashSet::new();
        observable.extend(
            graph
                .input
                .iter()
                .chain(&graph.output)
                .map(|port| port.name.as_str()),
        );
        for (index, node) in graph.node.iter().enumerate() {
            for input in &node.input {
                let usage = consumers.entry(input).or_insert((index, 0));
                usage.1 += 1;
            }
            for attribute in &node.attribute {
                if let Some(child) = attribute.g.as_ref() {
                    captured_names(child, &mut observable);
                }
                for child in &attribute.graphs {
                    captured_names(child, &mut observable);
                }
            }
        }
        let mut selected = HashSet::new();
        let mut predicates = Vec::new();
        for equal in 0..graph.node.len() {
            let Some(plan) = predicate(graph, equal, opset, &producers, functions) else {
                continue;
            };
            // The plan is in reverse data-flow order, starting at the count.
            // Every deleted output must have exactly its next chain node as a use.
            let mut consumer = equal;
            let private = plan.nodes.iter().all(|&index| {
                let output = graph.node[index].output[0].as_str();
                let safe = !selected.contains(&index)
                    && !observable.contains(output)
                    && consumers.get(output) == Some(&(consumer, 1));
                consumer = index;
                safe
            });
            if private {
                selected.extend(plan.nodes.iter().copied());
                selected.insert(equal);
                predicates.push(plan);
            }
        }
        predicates
    };
    if predicates.is_empty() {
        return changed;
    }

    let mut removed = HashSet::new();
    let mut removed_values = HashSet::new();
    let mut replacements = HashMap::new();
    for plan in predicates {
        for index in plan.nodes {
            removed.insert(index);
            removed_values.extend(graph.node[index].output.iter().cloned());
        }
        let zero = names.fresh();
        graph.initializer.push(TensorProto {
            name: zero.clone(),
            data_type: 1,
            float_data: vec![0.0],
            ..Default::default()
        });
        let is_zero = names.fresh();
        let is_nonzero = names.fresh();
        let indicator = names.fresh();
        let sum = names.fresh();
        let mut cast = operator("Cast", vec![is_nonzero.clone()], vec![indicator.clone()]);
        cast.attribute.push(int_attribute("to", 1));
        let mut reduce = operator("ReduceSum", vec![indicator.clone()], vec![sum.clone()]);
        reduce.attribute.push(int_attribute("keepdims", 0));
        graph.value_info.extend([
            tensor_info(&is_zero, 9, &plan.shape),
            tensor_info(&is_nonzero, 9, &plan.shape),
            tensor_info(&indicator, 1, &plan.shape),
            tensor_info(&sum, 1, &[]),
        ]);
        // Equal(x, 0) treats both signed zeros as zero and NaN/Inf as
        // nonzero. Summing nonnegative 0/1 indicators cannot cancel to zero.
        replacements.insert(
            plan.equal,
            vec![
                operator(
                    "Equal",
                    vec![plan.input, zero.clone()],
                    vec![is_zero.clone()],
                ),
                operator("Not", vec![is_zero], vec![is_nonzero]),
                cast,
                reduce,
                operator(
                    "Equal",
                    vec![sum, zero],
                    graph.node[plan.equal].output.clone(),
                ),
            ],
        );
    }
    graph
        .value_info
        .retain(|info| !removed_values.contains(&info.name));
    let mut nodes = Vec::with_capacity(graph.node.len() + replacements.len());
    for (index, node) in std::mem::take(&mut graph.node).into_iter().enumerate() {
        if let Some(replacement) = replacements.remove(&index) {
            nodes.extend(replacement);
        } else if !removed.contains(&index) {
            nodes.push(node);
        }
    }
    graph.node = nodes;
    true
}
