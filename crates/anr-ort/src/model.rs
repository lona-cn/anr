//! Targeted, lossless protobuf rewrites applied only to the in-memory model.

use std::borrow::Cow;
use std::collections::{HashMap, HashSet};

use anr_core::{Error, Result};
use onnx_protobuf::{
    AttributeProto, GraphProto, Message, ModelProto, NodeProto, TensorProto, TensorShapeProto,
    TypeProto, ValueInfoProto, attribute_proto::AttributeType, tensor_shape_proto::Dimension,
};
use protobuf::MessageField;

mod nonzero;
mod simplify;

/// The GPU provider whose unsupported exported operators should be expanded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelTarget {
    /// Retain supported CUDA fusions while lowering avoidable host-side work.
    Cuda,
    /// Expand Microsoft fusions and lower fixed-shape state and control flow.
    TensorRT,
    /// Use standard tensor operators instead of Microsoft fusions and sequences.
    DirectML,
}

/// Prepare a model without changing its weights, public ports, or source file.
///
/// Unsupported patterns are left intact, rather than guessed or evaluated with
/// a particular set of runtime inputs. Unchanged models borrow the input bytes.
pub fn prepare_model(model: &[u8], target: ModelTarget) -> Result<Cow<'_, [u8]>> {
    let mut parsed = ModelProto::parse_from_bytes(model)
        .map_err(|error| Error::InvalidModel(format!("cannot decode ONNX model: {error}")))?;
    let opset = parsed
        .opset_import
        .iter()
        .find(|opset| standard_domain(&opset.domain))
        .map(|opset| opset.version)
        .unwrap_or(0);
    let functions: HashSet<_> = parsed
        .functions
        .iter()
        .map(|function| {
            (
                function.domain.clone(),
                function.name.clone(),
                function.overload.clone(),
            )
        })
        .collect();
    let graph = parsed
        .graph
        .as_mut()
        .ok_or_else(|| Error::InvalidModel("ONNX model has no graph".into()))?;
    let mut names = Names::default();
    names.collect(graph);
    let mut changed = expand_convolutions(graph, target, &mut names, &functions);
    if target != ModelTarget::Cuda {
        changed |= expand_scaled_matmuls(graph, opset, &mut names, &functions);
    }
    changed |= lower_sequences(graph, opset, &mut names, &functions);
    changed |= nonzero::lower_zero_predicates(graph, opset, &mut names, &functions);
    changed |= simplify::simplify(graph, opset, &mut names, &functions);
    changed |= lower_conditionals(graph, opset, &mut names, &functions);
    changed |= simplify::simplify(graph, opset, &mut names, &functions);
    if !changed {
        return Ok(Cow::Borrowed(model));
    }
    parsed
        .write_to_bytes()
        .map(Cow::Owned)
        .map_err(|error| Error::InvalidModel(format!("cannot encode prepared ONNX model: {error}")))
}

type Functions = HashSet<(String, String, String)>;

fn standard_domain(domain: &str) -> bool {
    domain.is_empty() || domain == "ai.onnx"
}

fn builtin(node: &NodeProto, op: &str, functions: &Functions) -> bool {
    standard_domain(&node.domain)
        && node.op_type == op
        && node.overload.is_empty()
        && !functions.contains(&(
            node.domain.clone(),
            node.op_type.clone(),
            node.overload.clone(),
        ))
}

#[derive(Default)]
struct Names {
    used: HashSet<String>,
    next: u64,
}

impl Names {
    fn collect(&mut self, graph: &GraphProto) {
        for port in graph
            .input
            .iter()
            .chain(&graph.output)
            .chain(&graph.value_info)
        {
            self.used.insert(port.name.clone());
        }
        for tensor in &graph.initializer {
            self.used.insert(tensor.name.clone());
        }
        for tensor in &graph.sparse_initializer {
            self.used.insert(tensor.values.name.clone());
            self.used.insert(tensor.indices.name.clone());
        }
        for node in &graph.node {
            self.used.insert(node.name.clone());
            self.used
                .extend(node.input.iter().chain(&node.output).cloned());
            for attribute in &node.attribute {
                if let Some(child) = attribute.g.as_ref() {
                    self.collect(child);
                }
                for child in &attribute.graphs {
                    self.collect(child);
                }
            }
        }
    }

    fn fresh(&mut self) -> String {
        loop {
            let name = format!("__anr_prepared_{}", self.next);
            self.next += 1;
            if self.used.insert(name.clone()) {
                return name;
            }
        }
    }
}

fn operator(op: &str, input: Vec<String>, output: Vec<String>) -> NodeProto {
    NodeProto {
        op_type: op.into(),
        input,
        output,
        ..Default::default()
    }
}

fn int_attribute(name: &str, value: i64) -> AttributeProto {
    AttributeProto {
        name: name.into(),
        type_: AttributeType::INT.into(),
        i: value,
        ..Default::default()
    }
}

fn integer_attribute(node: &NodeProto, name: &str, default: i64) -> Option<i64> {
    let mut attributes = node
        .attribute
        .iter()
        .filter(|attribute| attribute.name == name);
    match (attributes.next(), attributes.next()) {
        (None, _) => Some(default),
        (Some(attribute), None) if attribute.type_.value() == AttributeType::INT as i32 => {
            Some(attribute.i)
        }
        _ => None,
    }
}

fn expand_convolutions(
    graph: &mut GraphProto,
    target: ModelTarget,
    names: &mut Names,
    functions: &Functions,
) -> bool {
    let mut changed = false;
    let mut nodes = Vec::with_capacity(graph.node.len());
    for mut node in std::mem::take(&mut graph.node) {
        for attribute in &mut node.attribute {
            if let Some(child) = attribute.g.as_mut() {
                changed |= expand_convolutions(child, target, names, functions);
            }
            for child in &mut attribute.graphs {
                changed |= expand_convolutions(child, target, names, functions);
            }
        }
        let mut activations = node
            .attribute
            .iter()
            .filter(|attribute| attribute.name == "activation");
        let activation = match (activations.next(), activations.next()) {
            (Some(attribute), None) if attribute.type_.value() == AttributeType::STRING as i32 => {
                match attribute.s.as_slice() {
                    b"Sigmoid" => Some("Sigmoid"),
                    b"Relu" if target != ModelTarget::Cuda => Some("Relu"),
                    _ => None,
                }
            }
            _ => None,
        };
        let mut parameters = node
            .attribute
            .iter()
            .filter(|attribute| attribute.name == "activation_params");
        let parameterless = match (parameters.next(), parameters.next()) {
            (None, _) => true,
            (Some(attribute), None) => {
                attribute.type_.value() == AttributeType::FLOATS as i32
                    && attribute.floats.is_empty()
            }
            _ => false,
        };
        if node.domain != "com.microsoft"
            || node.op_type != "FusedConv"
            || activation.is_none()
            || !parameterless
            || !node.overload.is_empty()
            || !(2..=4).contains(&node.input.len())
            || node.output.len() != 1
            || node.output[0].is_empty()
            || functions.contains(&(
                node.domain.clone(),
                node.op_type.clone(),
                node.overload.clone(),
            ))
        {
            nodes.push(node);
            continue;
        }
        let output = node.output[0].clone();
        let residual = node.input.get(3).filter(|name| !name.is_empty()).cloned();
        node.input.truncate(3);
        node.domain.clear();
        node.op_type = "Conv".into();
        node.attribute.retain(|attribute| {
            attribute.name != "activation" && attribute.name != "activation_params"
        });
        let convolution = names.fresh();
        node.output[0] = convolution.clone();
        nodes.push(node);
        let activation_input = if let Some(residual) = residual {
            let sum = names.fresh();
            nodes.push(operator(
                "Add",
                vec![convolution, residual],
                vec![sum.clone()],
            ));
            sum
        } else {
            convolution
        };
        nodes.push(operator(
            activation.unwrap(),
            vec![activation_input],
            vec![output],
        ));
        changed = true;
    }
    graph.node = nodes;
    changed
}

fn expand_scaled_matmuls(
    graph: &mut GraphProto,
    opset: i64,
    names: &mut Names,
    functions: &Functions,
) -> bool {
    // FusedMatMul accepts BF16 even when the imported MatMul schema does not.
    if opset < 13 {
        return false;
    }
    let mut changed = false;
    let mut nodes = Vec::with_capacity(graph.node.len());
    for mut node in std::mem::take(&mut graph.node) {
        for attribute in &mut node.attribute {
            if let Some(child) = attribute.g.as_mut() {
                changed |= expand_scaled_matmuls(child, opset, names, functions);
            }
            for child in &mut attribute.graphs {
                changed |= expand_scaled_matmuls(child, opset, names, functions);
            }
        }
        if node.domain != "com.microsoft"
            || node.op_type != "FusedMatMul"
            || !node.overload.is_empty()
            || node.input.len() != 2
            || node.input.iter().any(String::is_empty)
            || node.output.len() != 1
            || node.output[0].is_empty()
            || functions.contains(&(
                node.domain.clone(),
                node.op_type.clone(),
                node.overload.clone(),
            ))
            || node.attribute.iter().any(|attribute| {
                !matches!(
                    attribute.name.as_str(),
                    "alpha" | "transA" | "transB" | "transBatchA" | "transBatchB"
                )
            })
            || ["transA", "transB", "transBatchA", "transBatchB"]
                .iter()
                .any(|name| integer_attribute(&node, name, 0) != Some(0))
        {
            nodes.push(node);
            continue;
        }
        let mut scales = node
            .attribute
            .iter()
            .filter(|attribute| attribute.name == "alpha");
        let alpha = match (scales.next(), scales.next()) {
            (None, _) => Some(1.0),
            (Some(attribute), None) if attribute.type_.value() == AttributeType::FLOAT as i32 => {
                Some(attribute.f)
            }
            _ => None,
        };
        let Some(alpha) = alpha else {
            nodes.push(node);
            continue;
        };
        // FusedMatMul's operands share a type. A known FLOAT operand proves
        // that the scalar scale will not introduce a mixed-type Mul.
        if alpha != 1.0
            && !node.input.iter().any(|name| {
                input_tensor(graph, name).is_some_and(|(element_type, _)| element_type == 1)
            })
        {
            nodes.push(node);
            continue;
        }
        node.domain.clear();
        node.op_type = "MatMul".into();
        node.attribute.clear();
        if alpha == 1.0 {
            nodes.push(node);
        } else {
            let output = node.output[0].clone();
            let product = names.fresh();
            let scale = names.fresh();
            node.output[0] = product.clone();
            nodes.push(node);
            graph.initializer.push(TensorProto {
                name: scale.clone(),
                data_type: 1,
                float_data: vec![alpha],
                ..Default::default()
            });
            nodes.push(operator("Mul", vec![product, scale], vec![output]));
        }
        changed = true;
    }
    graph.node = nodes;
    changed
}

/// Initializers that are also graph inputs may be overridden at runtime.
fn initializer<'a>(graph: &'a GraphProto, name: &str) -> Option<&'a TensorProto> {
    if graph.input.iter().any(|input| input.name == name) {
        return None;
    }
    let mut candidates = graph
        .initializer
        .iter()
        .filter(|tensor| tensor.name == name);
    match (candidates.next(), candidates.next()) {
        (Some(tensor), None) => Some(tensor),
        _ => None,
    }
}

fn constant_tensor<'a>(
    graph: &'a GraphProto,
    name: &str,
    functions: &Functions,
) -> Option<&'a TensorProto> {
    if graph.input.iter().any(|input| input.name == name) {
        return None;
    }
    if let Some(tensor) = initializer(graph, name) {
        return Some(tensor);
    }
    let node = graph
        .node
        .iter()
        .find(|node| node.output.iter().any(|output| output == name))?;
    if !builtin(node, "Constant", functions)
        || !node.input.is_empty()
        || node.output.len() != 1
        || node.attribute.len() != 1
    {
        return None;
    }
    let value = &node.attribute[0];
    if value.name != "value" || value.type_.value() != AttributeType::TENSOR as i32 {
        return None;
    }
    value.t.as_ref()
}

fn integers(tensor: &TensorProto) -> Option<Vec<i64>> {
    if tensor.data_type != 7
        || !tensor.external_data.is_empty()
        || tensor.segment.is_some()
        || tensor.data_location.value() != 0
    {
        return None;
    }
    let count = tensor.dims.iter().try_fold(1_usize, |count, &dimension| {
        count.checked_mul(usize::try_from(dimension).ok()?)
    })?;
    if !tensor.raw_data.is_empty() {
        if tensor.raw_data.len() != count.checked_mul(8)? {
            return None;
        }
        Some(
            tensor
                .raw_data
                .as_chunks::<8>()
                .0
                .iter()
                .map(|&chunk| i64::from_le_bytes(chunk))
                .collect(),
        )
    } else if tensor.int64_data.len() == count {
        Some(tensor.int64_data.clone())
    } else {
        None
    }
}

fn tensor_info(name: &str, element_type: i32, dimensions: &[i64]) -> ValueInfoProto {
    let mut dtype = TypeProto::new();
    let tensor = dtype.mut_tensor_type();
    tensor.elem_type = element_type;
    tensor.shape = MessageField::some(TensorShapeProto {
        dim: dimensions
            .iter()
            .map(|&length| {
                let mut dimension = Dimension::new();
                dimension.set_dim_value(length);
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

fn fixed_tensor(info: &ValueInfoProto) -> Option<(i32, Vec<i64>)> {
    let dtype = info.type_.as_ref()?;
    if !dtype.has_tensor_type() {
        return None;
    }
    let tensor = dtype.tensor_type();
    if tensor.elem_type == 0 {
        return None;
    }
    let dimensions = tensor
        .shape
        .as_ref()?
        .dim
        .iter()
        .map(|dimension| {
            if dimension.has_dim_value() && dimension.dim_value() >= 0 {
                Some(dimension.dim_value())
            } else {
                None
            }
        })
        .collect::<Option<Vec<_>>>()?;
    Some((tensor.elem_type, dimensions))
}

fn input_tensor(graph: &GraphProto, name: &str) -> Option<(i32, Vec<i64>)> {
    if let Some(info) = graph
        .input
        .iter()
        .chain(&graph.value_info)
        .chain(&graph.output)
        .find(|info| info.name == name)
    {
        return fixed_tensor(info);
    }
    let tensor = initializer(graph, name)?;
    if tensor.data_type == 0 || tensor.dims.iter().any(|&dimension| dimension < 0) {
        return None;
    }
    Some((tensor.data_type, tensor.dims.clone()))
}

// A nested graph can refer to its parent's values without listing them as node
// inputs on the control-flow operator. Conservatively reject even shadowed uses.
fn mentions(graph: &GraphProto, name: &str) -> bool {
    graph.output.iter().any(|output| output.name == name)
        || graph.node.iter().any(|node| {
            node.input.iter().any(|input| input == name)
                || node.attribute.iter().any(|attribute| {
                    attribute
                        .g
                        .as_ref()
                        .is_some_and(|child| mentions(child, name))
                        || attribute.graphs.iter().any(|child| mentions(child, name))
                })
        })
}

struct SequencePlan {
    lengths: Vec<i64>,
    axis: usize,
    element_type: i32,
    dimensions: Vec<i64>,
    consumers: Vec<(usize, usize)>,
}

fn sequence_plan(graph: &GraphProto, index: usize, functions: &Functions) -> Option<SequencePlan> {
    let node = &graph.node[index];
    if !builtin(node, "SplitToSequence", functions)
        || node.input.len() != 2
        || node.output.len() != 1
        || node.output[0].is_empty()
        || integer_attribute(node, "keepdims", 1)? != 1
        || node
            .attribute
            .iter()
            .any(|attribute| !matches!(attribute.name.as_str(), "axis" | "keepdims"))
    {
        return None;
    }
    let split = initializer(graph, &node.input[1])?;
    if split.dims.len() != 1 {
        return None;
    }
    let lengths = integers(split)?;
    if lengths.is_empty() || lengths.iter().any(|&length| length < 0) {
        return None;
    }
    let (element_type, dimensions) = input_tensor(graph, &node.input[0])?;
    let rank = i64::try_from(dimensions.len()).ok()?;
    let axis = integer_attribute(node, "axis", 0)?;
    let axis = if axis < 0 {
        axis.checked_add(rank)?
    } else {
        axis
    };
    let axis = usize::try_from(axis).ok()?;
    let total = lengths
        .iter()
        .try_fold(0_i64, |total, &length| total.checked_add(length))?;
    if dimensions.get(axis).copied()? != total {
        return None;
    }
    let sequence = &node.output[0];
    if graph.output.iter().any(|output| &output.name == sequence) {
        return None;
    }
    let mut consumers = Vec::new();
    for (consumer_index, consumer) in graph.node.iter().enumerate() {
        if consumer.attribute.iter().any(|attribute| {
            attribute
                .g
                .as_ref()
                .is_some_and(|child| mentions(child, sequence))
                || attribute
                    .graphs
                    .iter()
                    .any(|child| mentions(child, sequence))
        }) {
            return None;
        }
        if !consumer.input.iter().any(|input| input == sequence) {
            continue;
        }
        if consumer_index <= index
            || !builtin(consumer, "SequenceAt", functions)
            || consumer.input.len() != 2
            || &consumer.input[0] != sequence
            || consumer.output.len() != 1
            || consumer.output[0].is_empty()
            || !consumer.attribute.is_empty()
        {
            return None;
        }
        let position = constant_tensor(graph, &consumer.input[1], functions)?;
        if !position.dims.is_empty() {
            return None;
        }
        let position = *integers(position)?.first()?;
        let count = i64::try_from(lengths.len()).ok()?;
        let position = if position < 0 {
            position.checked_add(count)?
        } else {
            position
        };
        let position = usize::try_from(position).ok()?;
        if position >= lengths.len() {
            return None;
        }
        consumers.push((consumer_index, position));
    }
    if consumers.is_empty() {
        return None;
    }
    Some(SequencePlan {
        lengths,
        axis,
        element_type,
        dimensions,
        consumers,
    })
}

fn lower_sequences(
    graph: &mut GraphProto,
    opset: i64,
    names: &mut Names,
    functions: &Functions,
) -> bool {
    if opset < 11 {
        return false;
    }
    let mut changed = false;
    for index in 0..graph.node.len() {
        let Some(plan) = sequence_plan(graph, index, functions) else {
            continue;
        };
        let sequence = graph.node[index].output[0].clone();
        let outputs: Vec<_> = plan.lengths.iter().map(|_| names.fresh()).collect();
        let split = &mut graph.node[index];
        split.op_type = "Split".into();
        split.domain.clear();
        split.output = outputs.clone();
        split.attribute.clear();
        split
            .attribute
            .push(int_attribute("axis", plan.axis as i64));
        if opset < 13 {
            split.input.truncate(1);
            split.attribute.push(AttributeProto {
                name: "split".into(),
                type_: AttributeType::INTS.into(),
                ints: plan.lengths.clone(),
                ..Default::default()
            });
        }
        graph.value_info.retain(|info| info.name != sequence);
        for ((output, &length), position) in outputs.iter().zip(&plan.lengths).zip(0..) {
            let mut dimensions = plan.dimensions.clone();
            dimensions[plan.axis] = length;
            graph
                .value_info
                .push(tensor_info(output, plan.element_type, &dimensions));
            // Keep SequenceAt's output names, including multiple reads of one
            // position, instead of aliasing public outputs or downstream uses.
            for &(consumer, _) in plan
                .consumers
                .iter()
                .filter(|(_, selected)| *selected == position)
            {
                let node = &mut graph.node[consumer];
                node.op_type = "Identity".into();
                node.domain.clear();
                node.input = vec![output.clone()];
                if !graph
                    .value_info
                    .iter()
                    .chain(&graph.output)
                    .any(|info| info.name == node.output[0])
                {
                    graph.value_info.push(tensor_info(
                        &node.output[0],
                        plan.element_type,
                        &dimensions,
                    ));
                }
            }
        }
        changed = true;
    }
    changed
}

// Deliberately limited to deterministic tensor computations. No custom domains,
// functions, graph attributes, sequences, random operators, or control flow can
// become eagerly evaluated through this path.
fn pure_tensor_operator(node: &NodeProto, functions: &Functions) -> bool {
    matches!(
        node.op_type.as_str(),
        "Identity"
            | "Constant"
            | "Add"
            | "Sub"
            | "Mul"
            | "Div"
            | "Pow"
            | "Neg"
            | "Abs"
            | "Exp"
            | "Log"
            | "Sqrt"
            | "Reciprocal"
            | "Min"
            | "Max"
            | "Clip"
            | "Where"
            | "Gather"
            | "Equal"
            | "Greater"
            | "GreaterOrEqual"
            | "Less"
            | "LessOrEqual"
            | "And"
            | "Or"
            | "Xor"
            | "Not"
    ) && builtin(node, &node.op_type, functions)
        && node.attribute.iter().all(|attribute| {
            attribute.g.is_none()
                && attribute.graphs.is_empty()
                && attribute.ref_attr_name.is_empty()
        })
}

fn pure_branch(graph: &GraphProto, functions: &Functions) -> bool {
    graph.input.is_empty()
        && graph.sparse_initializer.is_empty()
        && graph.quantization_annotation.is_empty()
        && graph
            .node
            .iter()
            .all(|node| pure_tensor_operator(node, functions))
}

fn branches<'a>(
    node: &'a NodeProto,
    functions: &Functions,
) -> Option<(&'a GraphProto, &'a GraphProto)> {
    if !builtin(node, "If", functions)
        || node.input.len() != 1
        || node.input[0].is_empty()
        || node.output.is_empty()
        || node.attribute.len() != 2
    {
        return None;
    }
    let then_branch = node
        .attribute
        .iter()
        .find(|attribute| attribute.name == "then_branch")?;
    let else_branch = node
        .attribute
        .iter()
        .find(|attribute| attribute.name == "else_branch")?;
    if then_branch.type_.value() != AttributeType::GRAPH as i32
        || else_branch.type_.value() != AttributeType::GRAPH as i32
    {
        return None;
    }
    let then_branch = then_branch.g.as_ref()?;
    let else_branch = else_branch.g.as_ref()?;
    if !pure_branch(then_branch, functions)
        || !pure_branch(else_branch, functions)
        || then_branch.output.len() != node.output.len()
        || else_branch.output.len() != node.output.len()
    {
        return None;
    }
    for (then_output, else_output) in then_branch.output.iter().zip(&else_branch.output) {
        let then_type = fixed_tensor(then_output)?;
        let else_type = fixed_tensor(else_output)?;
        // Limit this lowering to the floating tensor types supported by every
        // relevant Where schema.
        if then_type != else_type || !matches!(then_type.0, 1 | 10 | 11) {
            return None;
        }
    }
    Some((then_branch, else_branch))
}

fn local_value(graph: &GraphProto, name: &str) -> bool {
    graph.initializer.iter().any(|tensor| tensor.name == name)
        || graph
            .node
            .iter()
            .any(|node| node.output.iter().any(|output| output == name))
}

fn scoped_initializer<'a>(
    branch: &'a GraphProto,
    outer: &'a GraphProto,
    name: &str,
) -> Option<&'a TensorProto> {
    if local_value(branch, name) {
        initializer(branch, name)
    } else {
        initializer(outer, name)
    }
}

fn normalized_axis(axis: i64, rank: usize) -> Option<usize> {
    let rank = i64::try_from(rank).ok()?;
    let axis = if axis < 0 {
        axis.checked_add(rank)?
    } else {
        axis
    };
    let axis = usize::try_from(axis).ok()?;
    (axis < rank as usize).then_some(axis)
}

fn broadcast_shapes(left: &[i64], right: &[i64]) -> Option<Vec<i64>> {
    let rank = left.len().max(right.len());
    let mut result = vec![1; rank];
    for (shape, offset) in [(left, rank - left.len()), (right, rank - right.len())] {
        for (index, &dimension) in shape.iter().enumerate() {
            let current = &mut result[offset + index];
            *current = match (*current, dimension) {
                (a, b) if a == b => a,
                (1, b) => b,
                (a, 1) => a,
                // Unknown dimensions are permitted only in the already-eager
                // outer graph. Successful execution constrains them to either
                // one or the known non-one dimension. Branch inputs are checked
                // to be fully fixed before this helper is used.
                (-1, b) => b,
                (a, -1) => a,
                _ => return None,
            };
        }
    }
    Some(result)
}

fn gathered_shape(data: &[i64], indices: &[i64], axis: usize) -> Vec<i64> {
    let mut result = data[..axis].to_vec();
    result.extend_from_slice(indices);
    result.extend_from_slice(&data[axis + 1..]);
    result
}

fn shape_dimensions(node: &NodeProto, dimensions: &[i64]) -> Option<Vec<i64>> {
    if node
        .attribute
        .iter()
        .any(|attribute| !matches!(attribute.name.as_str(), "start" | "end"))
    {
        return None;
    }
    let rank = i64::try_from(dimensions.len()).ok()?;
    let bound = |index: i64| {
        if index < 0 {
            index.saturating_add(rank)
        } else {
            index
        }
        .clamp(0, rank) as usize
    };
    let start = bound(integer_attribute(node, "start", 0)?);
    let end = bound(integer_attribute(node, "end", rank)?);
    Some(dimensions[start..end.max(start)].to_vec())
}

// This proof follows only shape metadata and constant shape tensors. Unknown
// outer dimensions use -1; they are never accepted as eager branch inputs.
// Unsupported producers simply make the conditional ineligible.
struct CaptureShapes<'a> {
    graph: &'a GraphProto,
    functions: &'a Functions,
    known: HashMap<String, Option<Vec<i64>>>,
    visiting: HashSet<String>,
}

impl<'a> CaptureShapes<'a> {
    fn new(graph: &'a GraphProto, functions: &'a Functions) -> Self {
        Self {
            graph,
            functions,
            known: HashMap::new(),
            visiting: HashSet::new(),
        }
    }

    fn shape(&mut self, name: &str) -> Option<Vec<i64>> {
        if let Some(shape) = self.known.get(name) {
            return shape.clone();
        }
        if !self.visiting.insert(name.into()) {
            return None;
        }
        let shape = self.infer(name);
        self.visiting.remove(name);
        self.known.insert(name.into(), shape.clone());
        shape
    }

    fn producer(&self, name: &str) -> Option<&'a NodeProto> {
        self.graph
            .node
            .iter()
            .find(|node| node.output.iter().any(|output| output == name))
    }

    fn values(&mut self, name: &str) -> Option<Vec<i64>> {
        if let Some(tensor) = constant_tensor(self.graph, name, self.functions) {
            return integers(tensor);
        }
        let node = self.producer(name)?;
        if !standard_domain(&node.domain) || !builtin(node, &node.op_type, self.functions) {
            return None;
        }
        match node.op_type.as_str() {
            "Shape" if node.input.len() == 1 => {
                let dimensions = shape_dimensions(node, &self.shape(&node.input[0])?)?;
                if dimensions.iter().any(|&dimension| dimension < 0) {
                    return None;
                }
                Some(dimensions)
            }
            "Concat" if integer_attribute(node, "axis", 0)? == 0 => {
                let mut result = Vec::new();
                for input in &node.input {
                    result.extend(self.values(input)?);
                }
                Some(result)
            }
            "Slice" if node.input.len() >= 3 => {
                let starts = self.values(&node.input[1])?;
                let ends = self.values(&node.input[2])?;
                let axes = if node.input.len() >= 4 {
                    self.values(&node.input[3])?
                } else {
                    vec![0]
                };
                let steps = if node.input.len() >= 5 {
                    self.values(&node.input[4])?
                } else {
                    vec![1]
                };
                if axes != [0] || steps != [1] || starts.len() != 1 || ends.len() != 1 {
                    return None;
                }
                // A statically empty shape slice does not require knowing the
                // dimensions of the tensor from which Shape was obtained.
                if starts[0] == ends[0] {
                    return Some(Vec::new());
                }
                let values = self.values(&node.input[0])?;
                let length = i64::try_from(values.len()).ok()?;
                let bound = |index: i64| {
                    if index < 0 {
                        index.saturating_add(length)
                    } else {
                        index
                    }
                    .clamp(0, length)
                };
                let start = bound(starts[0]) as usize;
                let end = bound(ends[0]) as usize;
                Some(values[start..end.max(start)].to_vec())
            }
            _ => None,
        }
    }

    fn infer(&mut self, name: &str) -> Option<Vec<i64>> {
        if let Some((_, shape)) = input_tensor(self.graph, name) {
            return Some(shape);
        }
        // Outer ports/value_info can establish rank without fixing every
        // dimension. These facts are never taken from a branch's output claim.
        if let Some(info) = self
            .graph
            .input
            .iter()
            .chain(&self.graph.output)
            .chain(&self.graph.value_info)
            .find(|info| info.name == name)
        {
            let dtype = info.type_.as_ref()?;
            if !dtype.has_tensor_type() {
                return None;
            }
            return Some(
                dtype
                    .tensor_type()
                    .shape
                    .as_ref()?
                    .dim
                    .iter()
                    .map(|dimension| {
                        if dimension.has_dim_value() && dimension.dim_value() >= 0 {
                            dimension.dim_value()
                        } else {
                            -1
                        }
                    })
                    .collect(),
            );
        }
        let node = self.producer(name)?;
        let fused_matmul = node.domain == "com.microsoft"
            && node.op_type == "FusedMatMul"
            && node.overload.is_empty()
            && !self.functions.contains(&(
                node.domain.clone(),
                node.op_type.clone(),
                node.overload.clone(),
            ));
        if !fused_matmul && !builtin(node, &node.op_type, self.functions) {
            return None;
        }
        match node.op_type.as_str() {
            "Reshape" if node.input.len() == 2 => {
                let mut shape = self.values(&node.input[1])?;
                if shape.iter().any(|&dimension| dimension < -1)
                    || shape.iter().filter(|&&dimension| dimension == -1).count() > 1
                {
                    return None;
                }
                if shape.contains(&0) && integer_attribute(node, "allowzero", 0)? == 0 {
                    let source = self.shape(&node.input[0])?;
                    for (index, dimension) in shape.iter_mut().enumerate() {
                        if *dimension == 0 {
                            *dimension = *source.get(index)?;
                        }
                    }
                }
                Some(shape)
            }
            "Identity" | "Abs" | "Neg" | "Sigmoid" | "Relu" | "ScatterND"
                if !node.input.is_empty() =>
            {
                self.shape(&node.input[0])
            }
            "ConstantOfShape" if node.input.len() == 1 => {
                if let Some(producer) = self.producer(&node.input[0])
                    && builtin(producer, "Shape", self.functions)
                    && producer.input.len() == 1
                {
                    return shape_dimensions(producer, &self.shape(&producer.input[0])?);
                }
                let shape = self.values(&node.input[0])?;
                shape
                    .iter()
                    .all(|&dimension| dimension >= 0)
                    .then_some(shape)
            }
            "Add" | "Sub" | "Mul" | "Div" | "Pow" | "Min" | "Max" | "Where" | "Equal"
            | "Greater" | "Less" | "GreaterOrEqual" | "LessOrEqual" | "And" | "Or" | "Xor" => {
                let mut shape = Vec::new();
                for input in &node.input {
                    shape = broadcast_shapes(&shape, &self.shape(input)?)?;
                }
                Some(shape)
            }
            "Gather" if node.input.len() == 2 => {
                let data = self.shape(&node.input[0])?;
                let indices = self.shape(&node.input[1])?;
                let axis = normalized_axis(integer_attribute(node, "axis", 0)?, data.len())?;
                Some(gathered_shape(&data, &indices, axis))
            }
            "Unsqueeze" if node.input.len() == 2 => {
                let mut shape = self.shape(&node.input[0])?;
                let axes = self.values(&node.input[1])?;
                let rank = shape.len().checked_add(axes.len())?;
                let mut axes = axes
                    .iter()
                    .map(|&axis| normalized_axis(axis, rank))
                    .collect::<Option<Vec<_>>>()?;
                axes.sort_unstable();
                if axes.windows(2).any(|pair| pair[0] == pair[1]) {
                    return None;
                }
                for axis in axes {
                    shape.insert(axis, 1);
                }
                Some(shape)
            }
            "Concat" if !node.input.is_empty() => {
                let mut shape = self.shape(&node.input[0])?;
                let axis = normalized_axis(integer_attribute(node, "axis", 0)?, shape.len())?;
                for input in &node.input[1..] {
                    let next = self.shape(input)?;
                    if next.len() != shape.len()
                        || next
                            .iter()
                            .enumerate()
                            .any(|(index, &dimension)| index != axis && dimension != shape[index])
                    {
                        return None;
                    }
                    shape[axis] = if shape[axis] < 0 || next[axis] < 0 {
                        -1
                    } else {
                        shape[axis].checked_add(next[axis])?
                    };
                }
                Some(shape)
            }
            "Slice" if node.input.len() >= 3 => {
                let mut shape = self.shape(&node.input[0])?;
                let starts = self.values(&node.input[1])?;
                let ends = self.values(&node.input[2])?;
                let axes = if node.input.len() >= 4 {
                    self.values(&node.input[3])?
                } else {
                    (0..i64::try_from(starts.len()).ok()?).collect()
                };
                let steps = if node.input.len() >= 5 {
                    self.values(&node.input[4])?
                } else {
                    vec![1; starts.len()]
                };
                if starts.len() != ends.len()
                    || starts.len() != axes.len()
                    || starts.len() != steps.len()
                {
                    return None;
                }
                let mut seen = HashSet::new();
                for (((start, end), axis), step) in
                    starts.into_iter().zip(ends).zip(axes).zip(steps)
                {
                    let axis = normalized_axis(axis, shape.len())?;
                    let length = shape[axis];
                    if length < 0 || step <= 0 || !seen.insert(axis) {
                        return None;
                    }
                    let bound = |index: i64| {
                        if index < 0 {
                            index.saturating_add(length)
                        } else {
                            index
                        }
                        .clamp(0, length)
                    };
                    let span = (bound(end) - bound(start)).max(0);
                    shape[axis] = span / step + i64::from(span % step != 0);
                }
                Some(shape)
            }
            "MatMul" | "FusedMatMul" if node.input.len() == 2 => {
                if fused_matmul
                    && ["transA", "transB", "transBatchA", "transBatchB"]
                        .iter()
                        .any(|name| integer_attribute(node, name, 0) != Some(0))
                {
                    return None;
                }
                let mut left = self.shape(&node.input[0])?;
                let mut right = self.shape(&node.input[1])?;
                if left.is_empty() || right.is_empty() {
                    return None;
                }
                let left_vector = left.len() == 1;
                let right_vector = right.len() == 1;
                if left_vector {
                    left.insert(0, 1);
                }
                if right_vector {
                    right.push(1);
                }
                let mut shape =
                    broadcast_shapes(&left[..left.len() - 2], &right[..right.len() - 2])?;
                if !left_vector {
                    shape.push(left[left.len() - 2]);
                }
                if !right_vector {
                    shape.push(right[right.len() - 1]);
                }
                Some(shape)
            }
            _ => None,
        }
    }
}

fn safe_eager_branch(branch: &GraphProto, outer: &GraphProto, functions: &Functions) -> bool {
    let mut captures = CaptureShapes::new(outer, functions);
    let mut locals = HashMap::<String, Vec<i64>>::new();
    for tensor in &branch.initializer {
        if tensor.dims.iter().any(|&dimension| dimension < 0) {
            return false;
        }
        locals.insert(tensor.name.clone(), tensor.dims.clone());
    }
    for node in &branch.node {
        if node.output.len() != 1 || node.output[0].is_empty() {
            return false;
        }
        let input_shapes = node
            .input
            .iter()
            .filter(|input| !input.is_empty())
            .map(|input| {
                let shape = if local_value(branch, input) {
                    locals.get(input).cloned()
                } else {
                    captures.shape(input)
                }?;
                shape
                    .iter()
                    .all(|&dimension| dimension >= 0)
                    .then_some(shape)
            })
            .collect::<Option<Vec<_>>>();
        let Some(inputs) = input_shapes else {
            return false;
        };
        let shape = match node.op_type.as_str() {
            "Constant" => constant_tensor(branch, &node.output[0], functions)
                .map(|tensor| tensor.dims.clone()),
            "Identity" | "Neg" | "Abs" | "Exp" | "Log" | "Sqrt" | "Reciprocal" | "Not"
                if inputs.len() == 1 =>
            {
                Some(inputs[0].clone())
            }
            "Clip" if !inputs.is_empty() && inputs[1..].iter().all(Vec::is_empty) => {
                Some(inputs[0].clone())
            }
            "Gather" if inputs.len() == 2 => {
                let axis = integer_attribute(node, "axis", 0)
                    .and_then(|axis| normalized_axis(axis, inputs[0].len()));
                let indices = scoped_initializer(branch, outer, &node.input[1]).and_then(integers);
                match (axis, indices) {
                    (Some(axis), Some(indices))
                        if indices
                            .iter()
                            .all(|&index| index >= -inputs[0][axis] && index < inputs[0][axis]) =>
                    {
                        Some(gathered_shape(&inputs[0], &inputs[1], axis))
                    }
                    _ => None,
                }
            }
            "Add" | "Sub" | "Mul" | "Pow" | "Equal" | "Greater" | "GreaterOrEqual" | "Less"
            | "LessOrEqual" | "And" | "Or" | "Xor"
                if inputs.len() == 2 =>
            {
                broadcast_shapes(&inputs[0], &inputs[1])
            }
            "Div" if inputs.len() == 2 => {
                let divisor = scoped_initializer(branch, outer, &node.input[1])
                    .map(|tensor| tensor.data_type)
                    .or_else(|| {
                        if local_value(branch, &node.input[1]) {
                            None
                        } else {
                            input_tensor(outer, &node.input[1])
                                .map(|(element_type, _)| element_type)
                        }
                    });
                if divisor.is_some_and(|element_type| matches!(element_type, 1 | 10 | 11)) {
                    broadcast_shapes(&inputs[0], &inputs[1])
                } else {
                    None
                }
            }
            "Where" if inputs.len() == 3 => broadcast_shapes(&inputs[0], &inputs[1])
                .and_then(|shape| broadcast_shapes(&shape, &inputs[2])),
            "Min" | "Max" if !inputs.is_empty() => inputs
                .iter()
                .try_fold(Vec::new(), |shape, next| broadcast_shapes(&shape, next)),
            _ => None,
        };
        let Some(shape) = shape else { return false };
        if shape.iter().any(|&dimension| dimension < 0) {
            return false;
        }
        locals.insert(node.output[0].clone(), shape);
    }
    branch.output.iter().all(|output| {
        let actual = if local_value(branch, &output.name) {
            locals.get(&output.name).cloned()
        } else {
            captures.shape(&output.name)
        };
        fixed_tensor(output).is_some_and(|(_, declared)| actual == Some(declared))
    })
}

fn inline_branch(
    mut branch: GraphProto,
    graph: &mut GraphProto,
    nodes: &mut Vec<NodeProto>,
    names: &mut Names,
) -> Vec<String> {
    let mut renamed = HashMap::new();
    for tensor in &branch.initializer {
        renamed
            .entry(tensor.name.clone())
            .or_insert_with(|| names.fresh());
    }
    for node in &branch.node {
        for output in node.output.iter().filter(|output| !output.is_empty()) {
            renamed
                .entry(output.clone())
                .or_insert_with(|| names.fresh());
        }
    }
    for mut tensor in branch.initializer.drain(..) {
        tensor.name = renamed[&tensor.name].clone();
        graph.initializer.push(tensor);
    }
    for mut node in branch.node.drain(..) {
        if !node.name.is_empty() {
            node.name = names.fresh();
        }
        for name in node.input.iter_mut().chain(&mut node.output) {
            if let Some(replacement) = renamed.get(name) {
                *name = replacement.clone();
            }
        }
        nodes.push(node);
    }
    for mut info in branch
        .value_info
        .drain(..)
        .chain(branch.output.iter().cloned())
    {
        if let Some(replacement) = renamed.get(&info.name) {
            info.name = replacement.clone();
            if !graph
                .value_info
                .iter()
                .any(|existing| existing.name == info.name)
            {
                graph.value_info.push(info);
            }
        }
    }
    branch
        .output
        .iter()
        .map(|output| renamed.get(&output.name).unwrap_or(&output.name).clone())
        .collect()
}

fn lower_conditionals(
    graph: &mut GraphProto,
    opset: i64,
    names: &mut Names,
    functions: &Functions,
) -> bool {
    if opset < 9 {
        return false;
    }
    // Analyze while the outer producer list is still available. In particular,
    // never eagerly run an unselected Gather with an unproven dynamic index.
    let eligible: Vec<_> = graph
        .node
        .iter()
        .map(|node| {
            branches(node, functions).is_some_and(|(then_branch, else_branch)| {
                safe_eager_branch(then_branch, graph, functions)
                    && safe_eager_branch(else_branch, graph, functions)
            })
        })
        .collect();
    let mut changed = false;
    let mut nodes = Vec::with_capacity(graph.node.len());
    for (index, mut node) in std::mem::take(&mut graph.node).into_iter().enumerate() {
        if !eligible[index] {
            nodes.push(node);
            continue;
        }
        let then_branch = node
            .attribute
            .iter_mut()
            .find(|attribute| attribute.name == "then_branch")
            .and_then(|attribute| attribute.g.take())
            .expect("eligible If has a then branch");
        let else_branch = node
            .attribute
            .iter_mut()
            .find(|attribute| attribute.name == "else_branch")
            .and_then(|attribute| attribute.g.take())
            .expect("eligible If has an else branch");
        node.attribute.clear();
        let then_outputs = inline_branch(then_branch, graph, &mut nodes, names);
        let else_outputs = inline_branch(else_branch, graph, &mut nodes, names);
        // If permits any one-element condition tensor. Normalize its rank to a
        // scalar so Where cannot accidentally broadcast a leading dimension.
        let shape = names.fresh();
        graph.initializer.push(TensorProto {
            name: shape.clone(),
            data_type: 7,
            dims: vec![0],
            ..Default::default()
        });
        let condition = names.fresh();
        nodes.push(operator(
            "Reshape",
            vec![node.input[0].clone(), shape],
            vec![condition.clone()],
        ));
        graph.value_info.push(tensor_info(&condition, 9, &[]));
        for ((output, then_output), else_output) in
            node.output.iter().zip(then_outputs).zip(else_outputs)
        {
            let mut select = node.clone();
            select.op_type = "Where".into();
            select.domain.clear();
            select.name = names.fresh();
            select.attribute.clear();
            select.input = vec![condition.clone(), then_output, else_output];
            select.output = vec![output.clone()];
            nodes.push(select);
        }
        changed = true;
    }
    graph.node = nodes;
    changed
}
