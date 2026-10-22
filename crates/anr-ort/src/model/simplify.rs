use std::collections::{HashMap, HashSet};

use onnx_protobuf::{GraphProto, NodeProto, TensorProto, attribute_proto::AttributeType};

use super::{
    CaptureShapes, Functions, Names, broadcast_shapes, builtin, constant_tensor, operator,
    pure_tensor_operator,
};

/// Fold only uniform boolean predicates and selections whose broadcast shape is
/// proven. Every nested graph gets its own facts: an outer constant must never
/// be mistaken for a shadowing local input or producer.
pub(super) fn simplify(
    graph: &mut GraphProto,
    opset: i64,
    names: &mut Names,
    functions: &Functions,
) -> bool {
    // Where and ConstantOfShape acquired their current broadcast semantics in 9.
    if opset < 9 {
        return false;
    }
    let mut changed = false;
    for node in &mut graph.node {
        for attribute in &mut node.attribute {
            if let Some(child) = attribute.g.as_mut() {
                changed |= simplify(child, opset, names, functions);
            }
            for child in &mut attribute.graphs {
                changed |= simplify(child, opset, names, functions);
            }
        }
    }

    // Preserve the shape proofs for Expand nodes introduced by an earlier
    // iteration, even when the original intermediate had no value_info.
    let mut expanded_shapes = HashMap::new();
    loop {
        let pure = pure_values(graph, functions);
        let mut booleans = Booleans::new(graph, functions);
        let mut shapes = CaptureShapes::new(graph, functions);
        shapes.known.extend(
            expanded_shapes
                .iter()
                .map(|(name, shape): (&String, &Vec<i64>)| (name.clone(), Some(shape.clone()))),
        );
        let mut replacements = Vec::new();
        let mut initializers = Vec::new();
        for (index, node) in graph.node.iter().enumerate() {
            if node.output.len() != 1
                || node.output[0].is_empty()
                || node.input.iter().any(String::is_empty)
                || !node.attribute.is_empty()
            {
                continue;
            }
            let selected = if builtin(node, "Where", functions) && node.input.len() == 3 {
                booleans
                    .value(&node.input[0])
                    .map(|condition| &node.input[if condition { 1 } else { 2 }])
            } else if (builtin(node, "Or", functions) || builtin(node, "And", functions))
                && node.input.len() == 2
            {
                booleans.value(&node.output[0]).and_then(|value| {
                    node.input
                        .iter()
                        .find(|input| booleans.value(input) == Some(value))
                })
            } else {
                None
            };
            let Some(selected) = selected else {
                continue;
            };
            // Keep ordering through discarded RNG, control-flow, or opaque
            // computations; their values being unselected does not erase effects.
            if node
                .input
                .iter()
                .any(|input| input != selected && !pure.contains(input.as_str()))
            {
                continue;
            }
            if let Some((replacement, shape)) = select(
                node,
                selected,
                &mut shapes,
                names,
                functions,
                &mut initializers,
            ) {
                if let Some(shape) = shape {
                    expanded_shapes.insert(node.output[0].clone(), shape);
                }
                replacements.push((index, replacement));
            }
        }
        if replacements.is_empty() {
            break;
        }
        for (index, replacement) in replacements {
            graph.node[index] = replacement;
        }
        graph.initializer.extend(initializers);
        changed = true;
    }
    changed | remove_dead_nodes(graph, functions)
}

/// Return the selected value without losing dimensions contributed by either
/// the predicate or the discarded data input. Unknown dimensions with the same
/// -1 marker are not evidence that two independently broadcast values match.
fn select(
    node: &NodeProto,
    selected: &str,
    shapes: &mut CaptureShapes<'_>,
    names: &mut Names,
    functions: &Functions,
    initializers: &mut Vec<TensorProto>,
) -> Option<(NodeProto, Option<Vec<i64>>)> {
    let selected_shape = shapes.shape(selected)?;
    let result_shape = shapes.shape(&node.output[0]);
    let fixed_match = result_shape.as_ref().is_some_and(|shape| {
        shape.iter().all(|&dimension| dimension >= 0) && *shape == selected_shape
    });
    let preserves_shape = node.input.iter().all(|input| {
        if input == selected {
            return true;
        }
        let Some(shape) = shapes.shape(input) else {
            return false;
        };
        shape.len() <= selected_shape.len()
            && shape
                .iter()
                .rev()
                .zip(selected_shape.iter().rev())
                .all(|(&other, &selected)| other == 1 || (other >= 0 && other == selected))
    });
    if (fixed_match || preserves_shape) && available("Identity", functions) {
        return Some((
            operator("Identity", vec![selected.into()], node.output.clone()),
            None,
        ));
    }
    let result_shape = result_shape?;
    if result_shape.iter().any(|&dimension| dimension < 0)
        || broadcast_shapes(&selected_shape, &result_shape)? != result_shape
        || !available("Expand", functions)
    {
        return None;
    }
    let shape_name = names.fresh();
    initializers.push(TensorProto {
        name: shape_name.clone(),
        data_type: 7,
        dims: vec![i64::try_from(result_shape.len()).ok()?],
        int64_data: result_shape.clone(),
        ..Default::default()
    });
    Some((
        operator(
            "Expand",
            vec![selected.into(), shape_name],
            node.output.clone(),
        ),
        Some(result_shape),
    ))
}

fn available(op: &str, functions: &Functions) -> bool {
    !functions.contains(&(String::new(), op.into(), String::new()))
}

struct Booleans<'a> {
    graph: &'a GraphProto,
    functions: &'a Functions,
    known: HashMap<String, Option<bool>>,
    visiting: HashSet<String>,
}

impl<'a> Booleans<'a> {
    fn new(graph: &'a GraphProto, functions: &'a Functions) -> Self {
        Self {
            graph,
            functions,
            known: HashMap::new(),
            visiting: HashSet::new(),
        }
    }

    fn value(&mut self, name: &str) -> Option<bool> {
        if let Some(&value) = self.known.get(name) {
            return value;
        }
        if !self.visiting.insert(name.into()) {
            return None;
        }
        let value = self.infer(name);
        self.visiting.remove(name);
        self.known.insert(name.into(), value);
        value
    }

    fn producer(&self, name: &str) -> Option<&'a NodeProto> {
        self.graph
            .node
            .iter()
            .find(|node| node.output.iter().any(|output| output == name))
    }

    fn infer(&mut self, name: &str) -> Option<bool> {
        // Initializers declared as inputs are caller-overridable, including in
        // Loop bodies and other nested graph scopes.
        if self.graph.input.iter().any(|input| input.name == name) {
            return None;
        }
        if let Some(tensor) = constant_tensor(self.graph, name, self.functions) {
            return uniform_boolean(tensor);
        }
        let node = self.producer(name)?;
        if node.output.len() != 1
            || !builtin(node, &node.op_type, self.functions)
            || node.input.iter().any(String::is_empty)
        {
            return None;
        }
        if node.op_type == "ConstantOfShape" && node.input.len() == 1 {
            let [value] = node.attribute.as_slice() else {
                return None;
            };
            if value.name != "value" || value.type_.value() != AttributeType::TENSOR as i32 {
                return None;
            }
            let tensor = value.t.as_ref()?;
            return (tensor.dims == [1])
                .then(|| uniform_boolean(tensor))
                .flatten();
        }
        if !node.attribute.is_empty() {
            return None;
        }
        match node.op_type.as_str() {
            "Identity" if node.input.len() == 1 => self.value(&node.input[0]),
            "Expand" if node.input.len() == 2 => self.value(&node.input[0]),
            "Or" if node.input.len() == 2 => {
                let left = self.value(&node.input[0]);
                if left == Some(true) {
                    return Some(true);
                }
                match (left, self.value(&node.input[1])) {
                    (_, Some(true)) => Some(true),
                    (Some(false), Some(false)) => Some(false),
                    _ => None,
                }
            }
            "And" if node.input.len() == 2 => {
                let left = self.value(&node.input[0]);
                if left == Some(false) {
                    return Some(false);
                }
                match (left, self.value(&node.input[1])) {
                    (_, Some(false)) => Some(false),
                    (Some(true), Some(true)) => Some(true),
                    _ => None,
                }
            }
            "Where" if node.input.len() == 3 => {
                let condition = self.value(&node.input[0])?;
                self.value(&node.input[if condition { 1 } else { 2 }])
            }
            _ => None,
        }
    }
}

fn uniform_boolean(tensor: &TensorProto) -> Option<bool> {
    if tensor.data_type != 9
        || !tensor.external_data.is_empty()
        || tensor.segment.is_some()
        || tensor.data_location.value() != 0
    {
        return None;
    }
    let count = tensor.dims.iter().try_fold(1_usize, |count, &dimension| {
        count.checked_mul(usize::try_from(dimension).ok()?)
    })?;
    if count == 0 {
        return None;
    }
    if !tensor.raw_data.is_empty() {
        let &first = tensor.raw_data.first()?;
        (first <= 1
            && tensor.raw_data.len() == count
            && tensor.raw_data.iter().all(|&value| value == first))
        .then_some(first != 0)
    } else {
        let &first = tensor.int32_data.first()?;
        ((first == 0 || first == 1)
            && tensor.int32_data.len() == count
            && tensor.int32_data.iter().all(|&value| value == first))
        .then_some(first != 0)
    }
}

/// Include implicit graph-attribute captures, not just the inputs of If/Loop.
/// A local name can conservatively retain an outer producer with the same name;
/// it must never hide a possibly captured outer dependency during dead cleanup.
fn captured_names<'a>(graph: &'a GraphProto, pending: &mut Vec<&'a str>) {
    pending.extend(graph.output.iter().map(|output| output.name.as_str()));
    for node in &graph.node {
        dependencies(node, pending);
    }
}

fn dependencies<'a>(node: &'a NodeProto, pending: &mut Vec<&'a str>) {
    pending.extend(
        node.input
            .iter()
            .map(String::as_str)
            .filter(|name| !name.is_empty()),
    );
    for attribute in &node.attribute {
        if let Some(child) = attribute.g.as_ref() {
            captured_names(child, pending);
        }
        for child in &attribute.graphs {
            captured_names(child, pending);
        }
    }
}

fn stateless(node: &NodeProto, functions: &Functions) -> bool {
    if pure_tensor_operator(node, functions) {
        return true;
    }
    let standard = builtin(node, &node.op_type, functions)
        && matches!(
            node.op_type.as_str(),
            "Cast"
                | "Concat"
                | "ConstantOfShape"
                | "Conv"
                | "ConvTranspose"
                | "Einsum"
                | "Expand"
                | "GRU"
                | "MatMul"
                | "ReduceSum"
                | "Relu"
                | "Reshape"
                | "ScatterND"
                | "Shape"
                | "Sigmoid"
                | "Slice"
                | "Split"
                | "Squeeze"
                | "Tanh"
                | "Transpose"
                | "Unsqueeze"
        );
    let fusion = node.domain == "com.microsoft"
        && matches!(node.op_type.as_str(), "FusedConv" | "FusedMatMul")
        && node.overload.is_empty()
        && !functions.contains(&(
            node.domain.clone(),
            node.op_type.clone(),
            node.overload.clone(),
        ));
    (standard || fusion)
        && node.attribute.iter().all(|attribute| {
            attribute.g.is_none()
                && attribute.graphs.is_empty()
                && attribute.ref_attr_name.is_empty()
        })
}

// ONNX nodes are topologically ordered. A captured outer value is deliberately
// absent: without its producer, this scope cannot prove that dropping it is safe.
fn pure_values<'a>(graph: &'a GraphProto, functions: &Functions) -> HashSet<&'a str> {
    let mut pure: HashSet<_> = graph
        .input
        .iter()
        .map(|input| input.name.as_str())
        .chain(graph.initializer.iter().map(|tensor| tensor.name.as_str()))
        .chain(
            graph
                .sparse_initializer
                .iter()
                .map(|tensor| tensor.values.name.as_str()),
        )
        .collect();
    for node in &graph.node {
        if stateless(node, functions)
            && node
                .input
                .iter()
                .all(|input| input.is_empty() || pure.contains(input.as_str()))
        {
            pure.extend(
                node.output
                    .iter()
                    .map(String::as_str)
                    .filter(|name| !name.is_empty()),
            );
        }
    }
    pure
}

fn remove_dead_nodes(graph: &mut GraphProto, functions: &Functions) -> bool {
    let live = {
        let mut producers: HashMap<&str, Vec<usize>> = HashMap::new();
        for (index, node) in graph.node.iter().enumerate() {
            for output in &node.output {
                if !output.is_empty() {
                    producers.entry(output).or_default().push(index);
                }
            }
        }
        let mut live = vec![false; graph.node.len()];
        let mut visited = HashSet::new();
        let mut pending: Vec<_> = graph
            .output
            .iter()
            .map(|output| output.name.as_str())
            .collect();
        for (index, node) in graph.node.iter().enumerate() {
            if !stateless(node, functions) {
                live[index] = true;
                dependencies(node, &mut pending);
            }
        }
        while let Some(name) = pending.pop() {
            if !visited.insert(name) {
                continue;
            }
            if let Some(indices) = producers.get(name) {
                for &index in indices {
                    if !live[index] {
                        live[index] = true;
                        dependencies(&graph.node[index], &mut pending);
                    }
                }
            }
        }
        live
    };
    if live.iter().all(|&live| live) {
        return false;
    }
    let mut removed = HashSet::new();
    graph.node = std::mem::take(&mut graph.node)
        .into_iter()
        .zip(live)
        .filter_map(|(node, live)| {
            if live {
                Some(node)
            } else {
                removed.extend(node.output);
                None
            }
        })
        .collect();
    // Keep public metadata, all initializers, and any surviving definition even
    // if an unusual graph contains multiple definitions of the same value.
    for name in graph
        .input
        .iter()
        .chain(&graph.output)
        .map(|port| &port.name)
        .chain(graph.initializer.iter().map(|tensor| &tensor.name))
        .chain(
            graph
                .sparse_initializer
                .iter()
                .map(|tensor| &tensor.values.name),
        )
        .chain(graph.node.iter().flat_map(|node| &node.output))
    {
        removed.remove(name);
    }
    graph
        .value_info
        .retain(|info| !removed.contains(&info.name));
    true
}
