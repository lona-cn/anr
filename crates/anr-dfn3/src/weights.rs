use std::collections::HashMap;
use std::path::Path;

use anr_core::{Error, Result};
use prost::Message;
use sha2::{Digest, Sha256};

const MODEL_SHA: [u8; 32] = [
    0xfe, 0x5e, 0xb6, 0x4f, 0xa2, 0xe4, 0x15, 0x4c, 0x83, 0xf8, 0xe4, 0x93, 0x5e, 0x82, 0x87, 0x1c,
    0x85, 0x0c, 0x15, 0x43, 0x87, 0xee, 0x89, 0x2e, 0x0a, 0xb6, 0x5f, 0xe1, 0x79, 0xe7, 0xd8, 0xc9,
];

fn validate_model_bytes(bytes: &[u8]) -> Result<()> {
    if Sha256::digest(bytes)[..] != MODEL_SHA {
        return Err(Error::InvalidModel(
            "只支持 SHA-256 fe5eb64fa2e4154c83f8e4935e82871c850c154387ee892e0ab65fe179e7d8c9 对应的 DFN3 模型".into(),
        ));
    }
    Ok(())
}

/// Reads the supported model without constructing a compute dispatch plan.
pub fn model_bytes(path: &Path) -> Result<Vec<u8>> {
    let bytes = std::fs::read(path).map_err(Error::backend)?;
    validate_model_bytes(&bytes)?;
    Ok(bytes)
}

// This is a parameter-container decoder, not a graph interpreter. The digest
// fixes every node, attribute and tensor shape before any initializer is used.
#[derive(Clone, PartialEq, Message)]
struct ModelProto {
    #[prost(message, optional, tag = "7")]
    graph: Option<GraphProto>,
}
#[derive(Clone, PartialEq, Message)]
struct GraphProto {
    #[prost(message, repeated, tag = "5")]
    initializer: Vec<TensorProto>,
}
#[derive(Clone, PartialEq, Message)]
struct TensorProto {
    #[prost(int32, tag = "2")]
    data_type: i32,
    #[prost(string, tag = "8")]
    name: String,
    #[prost(bytes = "vec", tag = "9")]
    raw_data: Vec<u8>,
}

pub(crate) struct Weights {
    tensors: HashMap<String, Vec<f32>>,
    uploaded: HashMap<String, u32>,
    pub(crate) data: Vec<f32>,
}

impl Weights {
    pub(crate) fn load(path: &Path) -> Result<Self> {
        let bytes = model_bytes(path)?;
        let decoded =
            ModelProto::decode(bytes.as_slice()).map_err(|e| Error::InvalidModel(e.to_string()))?;
        let graph = decoded
            .graph
            .ok_or_else(|| Error::InvalidModel("模型没有计算图".into()))?;
        let tensors = graph
            .initializer
            .into_iter()
            .filter(|t| t.data_type == 1)
            .filter(|t| t.name != "rfft_matrix" && t.name != "irfft_matrix")
            .map(|t| {
                let values = t
                    .raw_data
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| f32::from_le_bytes(*b))
                    .collect();
                (t.name, values)
            })
            .collect();
        Ok(Self {
            tensors,
            uploaded: HashMap::new(),
            data: Vec::new(),
        })
    }

    pub(crate) fn get(&self, name: &str) -> Result<&[f32]> {
        self.tensors
            .get(name)
            .map(Vec::as_slice)
            .ok_or_else(|| Error::InvalidModel(format!("模型缺少权重 {name}")))
    }

    pub(crate) fn add(&mut self, name: &str) -> Result<u32> {
        if let Some(&offset) = self.uploaded.get(name) {
            return Ok(offset);
        }
        let offset = self.data.len() as u32;
        let values = self
            .tensors
            .get(name)
            .ok_or_else(|| Error::InvalidModel(format!("模型缺少权重 {name}")))?;
        self.data.extend_from_slice(values);
        self.uploaded.insert(name.to_owned(), offset);
        Ok(offset)
    }

    pub(crate) fn push(&mut self, values: &[f32]) -> u32 {
        let offset = self.data.len() as u32;
        self.data.extend_from_slice(values);
        offset
    }
}

pub(crate) struct Arena {
    pub(crate) len: u32,
    alignment: u32,
}
impl Arena {
    pub(crate) fn new(alignment: u32) -> Result<Self> {
        if !alignment.is_power_of_two() || alignment > 64 {
            return Err(crate::backend(
                "scratch alignment must be a power of two from 1 to 64 f32 elements",
            ));
        }
        Ok(Self { len: 0, alignment })
    }
    pub(crate) fn alloc(&mut self, len: u32) -> u32 {
        let start = self.len.next_multiple_of(self.alignment);
        self.len = start
            .checked_add(len)
            .expect("fixed DFN3 scratch size overflow");
        start
    }
}
