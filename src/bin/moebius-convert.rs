//! Converts upstream Moebius checkpoints (PyTorch `.bin`) to the engine's safetensors,
//! so that installing needs neither Python nor torch. It is the Rust twin of
//! `tools/convert_weights.py`, which documents the format; both write the same tensors.
//!
//!     moebius-convert --unet diffusion_pytorch_model.bin --vae vae-dir --out model-dir

use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::rc::Rc;

use clap::Parser;
use half::f16;
use memmap2::Mmap;
use safetensors::Dtype;
use safetensors::tensor::TensorView;
use sha2::{Digest, Sha256};

const FORMAT_VERSION: &str = "moebius-cpu/1";
const MANIFEST: &str = include_str!("../../docs/weights-manifest.txt");
const BN_EPS: f64 = 1e-5;
const F16_MAX: f32 = 65504.0;
/// BatchNorm module leaf and the layer it folds into, under the same parent.
const BN_TARGETS: [(&str, &str); 4] = [("bn1", "conv_dw"), ("bn2", "conv_pw"), ("norm_q", "to_q"), ("norm_v", "to_v")];
const BN_KEYS: [&str; 5] = ["weight", "bias", "running_mean", "running_var", "num_batches_tracked"];

type Res<T> = Result<T, String>;

#[derive(Parser)]
#[command(about = "Converts upstream Moebius checkpoints for moebius-cpu")]
struct Args {
    /// UNet diffusion_pytorch_model.bin.
    #[arg(long)]
    unet: PathBuf,
    /// Refuse the UNet file unless it has this SHA-256.
    #[arg(long)]
    unet_sha256: Option<String>,
    /// VAE directory: config.json and diffusion_pytorch_model.bin.
    #[arg(long)]
    vae: PathBuf,
    /// Refuse the VAE file unless it has this SHA-256.
    #[arg(long)]
    vae_sha256: Option<String>,
    /// Output directory for unet.safetensors and vae.safetensors.
    #[arg(long)]
    out: PathBuf,
}

// ---------------------------------------------------------------- zip container

/// Name to byte range of each stored entry. PyTorch never compresses, and it
/// writes zip64 records, so those are read when a 32-bit field is saturated.
fn read_zip(data: &[u8]) -> Res<HashMap<String, (usize, usize)>> {
    const EOCD: [u8; 4] = [0x50, 0x4b, 5, 6];
    const EOCD64_LOCATOR: [u8; 4] = [0x50, 0x4b, 6, 7];
    const ENTRY: [u8; 4] = [0x50, 0x4b, 1, 2];
    let u16_at = |p: usize| data.get(p..p + 2).map(|b| u16::from_le_bytes([b[0], b[1]]) as usize).ok_or("truncated zip");
    let u32_at = |p: usize| data.get(p..p + 4).map(|b| u32::from_le_bytes(b.try_into().unwrap()) as usize).ok_or("truncated zip");
    let u64_at = |p: usize| data.get(p..p + 8).map(|b| u64::from_le_bytes(b.try_into().unwrap()) as usize).ok_or("truncated zip");

    let tail = data.len().saturating_sub(65_557);
    let eocd = (tail..data.len().saturating_sub(21))
        .rev()
        .find(|&p| data[p..p + 4] == EOCD)
        .ok_or("not a PyTorch checkpoint: no zip directory")?;
    let mut count = u16_at(eocd + 10)?;
    let mut pos = u32_at(eocd + 16)?;
    if eocd >= 20 && data[eocd - 20..eocd - 16] == EOCD64_LOCATOR {
        let record = u64_at(eocd - 12)?;
        count = u64_at(record + 32)?;
        pos = u64_at(record + 48)?;
    }

    let mut entries = HashMap::with_capacity(count);
    for _ in 0..count {
        if data.get(pos..pos + 4) != Some(&ENTRY[..]) {
            return Err("corrupt zip directory".into());
        }
        let method = u16_at(pos + 10)?;
        let mut size = u32_at(pos + 24)?;
        let (name_len, extra_len, comment_len) = (u16_at(pos + 28)?, u16_at(pos + 30)?, u16_at(pos + 32)?);
        let mut local = u32_at(pos + 42)?;
        let name = data.get(pos + 46..pos + 46 + name_len).ok_or("truncated zip")?;
        let name = String::from_utf8_lossy(name).into_owned();

        // zip64 extra field: only saturated values appear, in this order.
        let mut extra = pos + 46 + name_len;
        let extra_end = extra + extra_len;
        while extra + 4 <= extra_end {
            let (id, len) = (u16_at(extra)?, u16_at(extra + 2)?);
            if id == 1 {
                let mut p = extra + 4;
                if size == u32::MAX as usize {
                    size = u64_at(p)?;
                    p += 8;
                }
                if u32_at(pos + 20)? == u32::MAX as usize {
                    p += 8;
                }
                if local == u32::MAX as usize {
                    local = u64_at(p)?;
                }
            }
            extra += 4 + len;
        }
        if method != 0 {
            return Err(format!("{name}: compressed entries are not supported"));
        }
        let start = local + 30 + u16_at(local + 26)? + u16_at(local + 28)?;
        if start + size > data.len() {
            return Err(format!("{name}: file is truncated"));
        }
        entries.insert(name, (start, size));
        pos = extra_end + comment_len;
    }
    Ok(entries)
}

// ---------------------------------------------------------------- pickle

#[derive(Clone, Copy, PartialEq, Debug)]
enum Kind {
    F32,
    F16,
    I64,
}

#[derive(Clone, Debug, PartialEq)]
struct Tensor {
    kind: Kind,
    key: String,
    shape: Vec<usize>,
}

#[derive(Clone)]
enum Value {
    Mark,
    None,
    Bool,
    Int(i64),
    Str(Rc<str>),
    Tuple(Rc<Vec<Value>>),
    Dict(Rc<RefCell<Vec<(Value, Value)>>>),
    Global(Rc<str>),
    Storage(Kind, Rc<str>),
    Tensor(Rc<Tensor>),
}

fn storage_kind(global: &str) -> Res<Kind> {
    match global {
        "torch FloatStorage" => Ok(Kind::F32),
        "torch HalfStorage" => Ok(Kind::F16),
        "torch LongStorage" => Ok(Kind::I64),
        other => Err(format!("unsupported tensor storage {other}")),
    }
}

fn ints(value: &Value) -> Res<Vec<usize>> {
    let Value::Tuple(items) = value else { return Err("expected a tuple of integers".into()) };
    items
        .iter()
        .map(|v| match v {
            Value::Int(i) if *i >= 0 => Ok(*i as usize),
            _ => Err("expected a tuple of integers".to_string()),
        })
        .collect()
}

/// `torch._utils._rebuild_tensor_v2(storage, offset, size, stride, ...)`.
fn rebuild_tensor(args: &[Value]) -> Res<Value> {
    let [Value::Storage(kind, key), Value::Int(offset), size, stride, ..] = args else {
        return Err("unexpected tensor record".into());
    };
    let (shape, stride) = (ints(size)?, ints(stride)?);
    if shape.len() != stride.len() {
        return Err("tensor record with mismatched size and stride".into());
    }
    let mut expected = 1;
    for (&dim, &step) in shape.iter().zip(&stride).rev() {
        if dim > 1 && step != expected {
            return Err("non-contiguous tensor in the checkpoint".into());
        }
        expected *= dim;
    }
    if *offset != 0 {
        return Err("tensor with a storage offset in the checkpoint".into());
    }
    Ok(Value::Tensor(Rc::new(Tensor { kind: *kind, key: key.to_string(), shape })))
}

/// Runs the small part of the pickle machine that `torch.save` uses and returns the
/// state dict in file order. Anything else is an error, so nothing is executed.
fn read_pickle(p: &[u8]) -> Res<Vec<(String, Tensor)>> {
    let mut stack: Vec<Value> = Vec::new();
    let mut memo: HashMap<u32, Value> = HashMap::new();
    let mut i = 0;

    fn take<'a>(p: &'a [u8], i: &mut usize, n: usize) -> Res<&'a [u8]> {
        let bytes = p.get(*i..*i + n).ok_or("truncated pickle")?;
        *i += n;
        Ok(bytes)
    }
    fn line(p: &[u8], i: &mut usize) -> Res<String> {
        let end = p[*i..].iter().position(|&b| b == b'\n').ok_or("truncated pickle")?;
        let text = String::from_utf8_lossy(&p[*i..*i + end]).into_owned();
        *i += end + 1;
        Ok(text)
    }
    fn pop(stack: &mut Vec<Value>) -> Res<Value> {
        stack.pop().ok_or_else(|| "pickle stack underflow".to_string())
    }
    fn to_mark(stack: &mut Vec<Value>) -> Res<Vec<Value>> {
        let mark = stack.iter().rposition(|v| matches!(v, Value::Mark)).ok_or("pickle has no mark")?;
        let items = stack.split_off(mark + 1);
        stack.pop();
        Ok(items)
    }
    fn tuple(stack: &mut Vec<Value>, n: usize) -> Res<()> {
        let at = stack.len().checked_sub(n).ok_or("pickle stack underflow")?;
        let items = stack.split_off(at);
        stack.push(Value::Tuple(Rc::new(items)));
        Ok(())
    }

    loop {
        let op = *p.get(i).ok_or("truncated pickle")?;
        i += 1;
        match op {
            0x80 => i += 1, // PROTO
            b'(' => stack.push(Value::Mark),
            b'N' => stack.push(Value::None),
            0x88 | 0x89 => stack.push(Value::Bool),
            b'J' => stack.push(Value::Int(i32::from_le_bytes(take(p, &mut i, 4)?.try_into().unwrap()) as i64)),
            b'K' => stack.push(Value::Int(take(p, &mut i, 1)?[0] as i64)),
            b'M' => stack.push(Value::Int(u16::from_le_bytes(take(p, &mut i, 2)?.try_into().unwrap()) as i64)),
            b'X' => {
                let n = u32::from_le_bytes(take(p, &mut i, 4)?.try_into().unwrap()) as usize;
                let text = std::str::from_utf8(take(p, &mut i, n)?).map_err(|_| "bad text in pickle")?;
                stack.push(Value::Str(text.into()));
            }
            b'c' => {
                let module = line(p, &mut i)?;
                let name = line(p, &mut i)?;
                stack.push(Value::Global(format!("{module} {name}").into()));
            }
            b')' => stack.push(Value::Tuple(Rc::new(Vec::new()))),
            b'}' => stack.push(Value::Dict(Rc::new(RefCell::new(Vec::new())))),
            0x85 => tuple(&mut stack, 1)?,
            0x86 => tuple(&mut stack, 2)?,
            0x87 => tuple(&mut stack, 3)?,
            b't' => {
                let items = to_mark(&mut stack)?;
                stack.push(Value::Tuple(Rc::new(items)));
            }
            b'q' => {
                let key = take(p, &mut i, 1)?[0] as u32;
                memo.insert(key, stack.last().ok_or("pickle stack underflow")?.clone());
            }
            b'r' => {
                let key = u32::from_le_bytes(take(p, &mut i, 4)?.try_into().unwrap());
                memo.insert(key, stack.last().ok_or("pickle stack underflow")?.clone());
            }
            b'h' => {
                let key = take(p, &mut i, 1)?[0] as u32;
                stack.push(memo.get(&key).ok_or("bad pickle memo")?.clone());
            }
            b'j' => {
                let key = u32::from_le_bytes(take(p, &mut i, 4)?.try_into().unwrap());
                stack.push(memo.get(&key).ok_or("bad pickle memo")?.clone());
            }
            b'Q' => {
                // BINPERSID: ('storage', <class>, key, 'cpu', numel)
                let Value::Tuple(id) = pop(&mut stack)? else { return Err("unexpected storage id".into()) };
                let [_, Value::Global(class), Value::Str(key), ..] = id.as_slice() else {
                    return Err("unexpected storage id".into());
                };
                stack.push(Value::Storage(storage_kind(class)?, key.clone()));
            }
            b'R' => {
                let Value::Tuple(args) = pop(&mut stack)? else { return Err("unexpected call in pickle".into()) };
                let Value::Global(callable) = pop(&mut stack)? else { return Err("unexpected call in pickle".into()) };
                match &*callable {
                    "collections OrderedDict" => stack.push(Value::Dict(Rc::new(RefCell::new(Vec::new())))),
                    "torch._utils _rebuild_tensor_v2" => stack.push(rebuild_tensor(&args)?),
                    other => return Err(format!("refusing to call {other} from a checkpoint")),
                }
            }
            b's' => {
                let value = pop(&mut stack)?;
                let key = pop(&mut stack)?;
                let Some(Value::Dict(dict)) = stack.last() else { return Err("pickle SETITEM without a dict".into()) };
                dict.borrow_mut().push((key, value));
            }
            b'u' => {
                let items = to_mark(&mut stack)?;
                let Some(Value::Dict(dict)) = stack.last() else { return Err("pickle SETITEMS without a dict".into()) };
                let mut items = items.into_iter();
                while let (Some(key), Some(value)) = (items.next(), items.next()) {
                    dict.borrow_mut().push((key, value));
                }
            }
            b'b' => {
                pop(&mut stack)?; // BUILD: the state dict's `_metadata`, not needed
            }
            b'.' => break,
            other => return Err(format!("unsupported pickle opcode 0x{other:02x}")),
        }
    }

    let Some(Value::Dict(dict)) = stack.pop() else { return Err("the checkpoint is not a state dict".into()) };
    let dict = dict.borrow();
    dict.iter()
        .map(|(key, value)| match (key, value) {
            (Value::Str(name), Value::Tensor(tensor)) => Ok((name.to_string(), (**tensor).clone())),
            _ => Err("the state dict holds something other than tensors".to_string()),
        })
        .collect()
}

// ---------------------------------------------------------------- conversion

struct Checkpoint<'a> {
    names: Vec<String>,
    tensors: HashMap<String, Tensor>,
    storages: HashMap<String, &'a [u8]>,
}

impl<'a> Checkpoint<'a> {
    fn parse(data: &'a [u8]) -> Res<Self> {
        let entries = read_zip(data)?;
        let (pickle, &(start, len)) =
            entries.iter().find(|(n, _)| n.ends_with("/data.pkl")).ok_or("not a PyTorch checkpoint: no data.pkl")?;
        let prefix = pickle.trim_end_matches("data.pkl");
        let order = read_pickle(&data[start..start + len])?;
        let mut storages = HashMap::with_capacity(order.len());
        for (name, tensor) in &order {
            let &(start, len) = entries
                .get(&format!("{prefix}data/{}", tensor.key))
                .ok_or_else(|| format!("{name}: its data is missing"))?;
            storages.insert(tensor.key.clone(), &data[start..start + len]);
        }
        let names = order.iter().map(|(n, _)| n.clone()).collect();
        Ok(Self { names, tensors: order.into_iter().collect(), storages })
    }

    fn floats(&self, name: &str) -> Res<Vec<f32>> {
        let tensor = self.tensors.get(name).ok_or_else(|| format!("{name} is missing"))?;
        let bytes = self.storages[&tensor.key];
        let count: usize = tensor.shape.iter().product();
        let out: Vec<f32> = match tensor.kind {
            Kind::F32 => bytes.chunks_exact(4).take(count).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect(),
            Kind::F16 => bytes
                .chunks_exact(2)
                .take(count)
                .map(|b| f16::from_le_bytes(b.try_into().unwrap()).to_f32())
                .collect(),
            Kind::I64 => return Err(format!("{name}: unexpected integer tensor")),
        };
        if out.len() != count {
            return Err(format!("{name}: its data is too short"));
        }
        Ok(out)
    }

    fn doubles(&self, name: &str) -> Res<Vec<f64>> {
        Ok(self.floats(name)?.into_iter().map(f64::from).collect())
    }
}

fn split(name: &str) -> (&str, &str) {
    name.rsplit_once('.').unwrap_or(("", name))
}

fn join(module: &str, leaf: &str) -> String {
    if module.is_empty() { leaf.to_string() } else { format!("{module}.{leaf}") }
}

/// Module of each BatchNorm, keyed by the layer it folds into.
fn find_batchnorms(names: &[String]) -> Res<HashMap<String, String>> {
    let mut out = HashMap::new();
    for name in names {
        let (module, leaf) = split(name);
        if leaf != "running_var" {
            continue;
        }
        let (parent, bn) = split(module);
        let (_, target) = BN_TARGETS
            .iter()
            .find(|(leaf, _)| *leaf == bn)
            .ok_or_else(|| format!("unexpected BatchNorm {module}: do not know which layer it folds into"))?;
        out.insert(join(parent, target), module.to_string());
    }
    Ok(out)
}

/// Eval-mode BatchNorm as per-channel (scale, shift), in f64.
fn fold_params(c: &Checkpoint, bn: &str) -> Res<(Vec<f64>, Vec<f64>)> {
    let gamma = c.doubles(&format!("{bn}.weight"))?;
    let beta = c.doubles(&format!("{bn}.bias"))?;
    let mean = c.doubles(&format!("{bn}.running_mean"))?;
    let var = c.doubles(&format!("{bn}.running_var"))?;
    let scale: Vec<f64> = gamma.iter().zip(&var).map(|(g, v)| g / (v + BN_EPS).sqrt()).collect();
    let shift = beta.iter().zip(&mean).zip(&scale).map(|((b, m), s)| b - m * s).collect();
    Ok((scale, shift))
}

/// The engine's layout. Returns the data, its shape, and whether it is a GEMM weight.
fn relayout(name: &str, data: Vec<f32>, shape: &[usize]) -> Res<(Vec<f32>, Vec<usize>, bool)> {
    let leaf = split(name).1;
    if leaf == "rel_pos_emb" {
        if shape.len() != 4 || shape[3] != 1 {
            return Err(format!("{name}: unexpected shape {shape:?}"));
        }
        return Ok((data, shape[..3].to_vec(), false));
    }
    if leaf != "weight" {
        return Ok((data, shape.to_vec(), false));
    }
    match *shape {
        [k, 1, 1, h, w] => Ok((data, vec![k, h * w], false)),
        [o, i, 1, 1] => Ok((data, vec![o, i], true)),
        [c, 1, 3, 3] => {
            let mut out = vec![0.0; data.len()];
            for ch in 0..c {
                for tap in 0..9 {
                    out[tap * c + ch] = data[ch * 9 + tap];
                }
            }
            Ok((out, vec![3, 3, c], false))
        }
        [o, i, 3, 3] => {
            let mut out = vec![0.0; data.len()];
            for row in 0..o {
                for ch in 0..i {
                    for tap in 0..9 {
                        out[(row * 9 + tap) * i + ch] = data[(row * i + ch) * 9 + tap];
                    }
                }
            }
            Ok((out, vec![o, 3, 3, i], true))
        }
        [_, _, _, _] | [_, _, _, _, _] => Err(format!("{name}: unsupported kernel {shape:?}")),
        [_, _] => Ok((data, shape.to_vec(), !name.ends_with("embedding_layer.weight"))),
        _ => Ok((data, shape.to_vec(), false)),
    }
}

struct Output {
    dtype: Dtype,
    shape: Vec<usize>,
    bytes: Vec<u8>,
}

fn f32_output(data: &[f32], shape: Vec<usize>) -> Output {
    Output { dtype: Dtype::F32, shape, bytes: bytemuck::cast_slice(data).to_vec() }
}

fn convert(c: &Checkpoint) -> Res<BTreeMap<String, Output>> {
    let batchnorms = find_batchnorms(&c.names)?;
    let skip: HashSet<String> =
        batchnorms.values().flat_map(|bn| BN_KEYS.iter().map(move |key| format!("{bn}.{key}"))).collect();

    let mut out = BTreeMap::new();
    for name in &c.names {
        if skip.contains(name) {
            continue;
        }
        let (module, leaf) = split(name);
        let shape = &c.tensors[name].shape;
        let mut data = c.floats(name)?;
        if let Some(bn) = batchnorms.get(module) {
            if leaf == "bias" {
                continue; // merged with the weight below
            }
            if leaf != "weight" {
                return Err(format!("{name}: unexpected tensor on a folded layer"));
            }
            let (scale, shift) = fold_params(c, bn)?;
            let per_row = data.len() / scale.len();
            for (row, values) in data.chunks_exact_mut(per_row).enumerate() {
                for v in values {
                    *v = (f64::from(*v) * scale[row]) as f32;
                }
            }
            let bias_name = join(module, "bias");
            let bias: Vec<f32> = if c.tensors.contains_key(&bias_name) {
                let old = c.doubles(&bias_name)?;
                old.iter().zip(&scale).zip(&shift).map(|((b, s), t)| (b * s + t) as f32).collect()
            } else {
                shift.iter().map(|&t| t as f32).collect()
            };
            let len = bias.len();
            out.insert(bias_name, f32_output(&bias, vec![len]));
        }
        let (data, shape, gemm) = relayout(name, data, shape)?;
        let output = if gemm && data.iter().all(|v| v.abs() <= F16_MAX) {
            let halves: Vec<f16> = data.iter().map(|&v| f16::from_f32(v)).collect();
            Output { dtype: Dtype::F16, shape, bytes: bytemuck::cast_slice(&halves).to_vec() }
        } else {
            f32_output(&data, shape)
        };
        out.insert(name.clone(), output);
    }
    Ok(out)
}

/// Every name and shape must match `docs/weights-manifest.txt`, section `label`.
fn validate(c: &Checkpoint, label: &str) -> Res<()> {
    let header = format!("[{label}]");
    let expected: HashMap<&str, Vec<usize>> = MANIFEST
        .lines()
        .map(str::trim_end)
        .skip_while(|line| *line != header)
        .skip(1)
        .take_while(|line| !line.starts_with('['))
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(|line| {
            let (name, shape) = line.split_once('\t').expect("manifest line");
            let dims = shape.trim_matches(['[', ']']).split(", ").filter(|d| !d.is_empty());
            (name, dims.map(|d| d.parse().expect("manifest shape")).collect())
        })
        .collect();
    if expected.is_empty() {
        return Err(format!("the manifest has no [{label}] section"));
    }
    for (name, shape) in &expected {
        match c.tensors.get(*name) {
            None => return Err(format!("{label}: {name} is missing, so this is the wrong checkpoint")),
            Some(t) if t.shape != *shape => return Err(format!("{label}: {name} has shape {:?}", t.shape)),
            Some(_) => {}
        }
    }
    match c.names.iter().find(|n| !expected.contains_key(n.as_str())) {
        Some(name) => Err(format!("{label}: unexpected tensor {name}")),
        None => Ok(()),
    }
}

fn sha256(data: &[u8]) -> String {
    Sha256::digest(data).iter().map(|b| format!("{b:02x}")).collect()
}

fn map(path: &Path) -> Res<Mmap> {
    let file = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    // Safety: the installer owns this file while it runs.
    unsafe { Mmap::map(&file) }.map_err(|e| format!("{}: {e}", path.display()))
}

fn write(tensors: &BTreeMap<String, Output>, metadata: HashMap<String, String>, path: &Path) -> Res<()> {
    let views = tensors
        .iter()
        .map(|(name, t)| {
            TensorView::new(t.dtype, t.shape.clone(), &t.bytes).map(|v| (name.as_str(), v)).map_err(|e| format!("{name}: {e}"))
        })
        .collect::<Res<Vec<_>>>()?;
    // A partial file must not look like a finished model to the next install.
    let partial = path.with_extension("partial");
    if let Err(e) = safetensors::serialize_to_file(views, Some(metadata), &partial) {
        let _ = fs::remove_file(&partial);
        return Err(format!("{}: {e}", partial.display()));
    }
    fs::rename(&partial, path).map_err(|e| format!("{}: {e}", path.display()))?;
    println!("wrote {} ({} tensors)", path.display(), tensors.len());
    Ok(())
}

fn convert_file(
    source: &Path,
    expected_sha256: Option<&str>,
    label: &str,
    extra: Option<(&str, String)>,
    out: &Path,
) -> Res<()> {
    let data = map(source)?;
    let hash = sha256(&data);
    if expected_sha256.is_some_and(|expected| !expected.eq_ignore_ascii_case(&hash)) {
        return Err(format!("{}: wrong SHA-256 {hash}. The download is damaged", source.display()));
    }
    let checkpoint = Checkpoint::parse(&data)?;
    validate(&checkpoint, label)?;
    let tensors = convert(&checkpoint)?;
    let mut metadata = HashMap::from([
        ("format".to_string(), FORMAT_VERSION.to_string()),
        ("dtype".to_string(), "f16".to_string()),
        ("source_sha256".to_string(), hash),
    ]);
    if let Some((key, value)) = extra {
        metadata.insert(key.to_string(), value);
    }
    write(&tensors, metadata, &out.join(format!("{label}.safetensors")))
}

fn run(args: &Args) -> Res<()> {
    let config = args.vae.join("config.json");
    let config = fs::read_to_string(&config).map_err(|e| format!("{}: {e}", config.display()))?;
    let config: serde_json::Value = serde_json::from_str(&config).map_err(|e| format!("VAE config.json: {e}"))?;
    let scaling = config["scaling_factor"].as_f64().ok_or("VAE config.json has no scaling_factor")?;
    fs::create_dir_all(&args.out).map_err(|e| format!("{}: {e}", args.out.display()))?;
    convert_file(&args.unet, args.unet_sha256.as_deref(), "unet", None, &args.out)?;
    let vae = args.vae.join("diffusion_pytorch_model.bin");
    convert_file(&vae, args.vae_sha256.as_deref(), "vae", Some(("scaling_factor", format!("{scaling}"))), &args.out)
}

fn main() -> ExitCode {
    match run(&Args::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("error: {message}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_3x3_moves_input_channels_last() {
        // [o=1, i=2, 3, 3]: channel 0 holds 0..9, channel 1 holds 9..18.
        let data: Vec<f32> = (0..18).map(|v| v as f32).collect();
        let (out, shape, gemm) = relayout("x.weight", data, &[1, 2, 3, 3]).unwrap();
        assert_eq!(shape, [1, 3, 3, 2]);
        assert!(gemm);
        assert_eq!(out[..4], [0.0, 9.0, 1.0, 10.0]);
    }

    #[test]
    fn depthwise_moves_channels_last_and_is_not_a_gemm() {
        let data: Vec<f32> = (0..18).map(|v| v as f32).collect();
        let (out, shape, gemm) = relayout("x.weight", data, &[2, 1, 3, 3]).unwrap();
        assert_eq!(shape, [3, 3, 2]);
        assert!(!gemm);
        assert_eq!(out[..4], [0.0, 9.0, 1.0, 10.0]);
    }

    #[test]
    fn the_token_embedding_stays_out_of_the_gemm_weights() {
        assert!(!relayout("embedding_layer.weight", vec![0.0; 4], &[2, 2]).unwrap().2);
        assert!(relayout("x.to_k.weight", vec![0.0; 4], &[2, 2]).unwrap().2);
    }

    #[test]
    fn pickle_reads_a_state_dict_with_memoised_storage_classes() {
        // {"a": _rebuild_tensor_v2(storage, 0, (2,), (1,), False, ())}
        let mut p = vec![0x80, 2, b'}', b'q', 0];
        p.extend(b"X\x01\0\0\0a");
        p.extend(b"ctorch._utils\n_rebuild_tensor_v2\nq\x01");
        p.extend(b"((X\x07\0\0\0storagectorch\nFloatStorage\nq\x02X\x01\0\0\x000X\x03\0\0\0cpuK\x02tQ");
        p.extend(b"K\0K\x02\x85K\x01\x85\x89)t");
        p.extend(b"Rs.");
        let tensors = read_pickle(&p).unwrap();
        assert_eq!(tensors.len(), 1);
        assert_eq!(tensors[0].0, "a");
        assert_eq!(tensors[0].1, Tensor { kind: Kind::F32, key: "0".into(), shape: vec![2] });
    }

    #[test]
    fn tensor_records_need_one_stride_per_dimension() {
        let int_tuple = |values: &[i64]| Value::Tuple(Rc::new(values.iter().map(|&v| Value::Int(v)).collect()));
        let record = |stride: &[i64]| {
            [Value::Storage(Kind::F32, "0".into()), Value::Int(0), int_tuple(&[2, 3]), int_tuple(stride)]
        };
        assert!(rebuild_tensor(&record(&[3, 1])).is_ok());
        assert!(rebuild_tensor(&record(&[1])).unwrap_err().contains("mismatched"));
    }

    #[test]
    fn pickle_refuses_to_call_anything_else() {
        let p = b"\x80\x02cos\nsystem\nX\x04\0\0\0calc\x85R.";
        assert!(read_pickle(p).unwrap_err().contains("refusing"));
    }
}
