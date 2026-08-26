use std::{collections::HashMap, sync::OnceLock};

use cubecl_core::{
    device::DeviceId,
    ir::ElemType,
    server::{CommunicationId, ReduceOperation},
    stub::Mutex,
};

/// Global state map from [`CommunicationId`] to boxed [`cudarc::nccl::sys::ncclUniqueId`].
static UNIQUE_IDS_MAP: OnceLock<Mutex<HashMap<CommunicationId, cudarc::nccl::sys::ncclUniqueId>>> =
    OnceLock::new();

pub(crate) fn get_nccl_comm_id(device_ids: Vec<DeviceId>) -> cudarc::nccl::sys::ncclUniqueId {
    let mut unique_ids_map = UNIQUE_IDS_MAP.get_or_init(Default::default).lock().unwrap();
    let comm_id = CommunicationId::from(device_ids);
    match unique_ids_map.get_mut(&comm_id) {
        Some(id) => *id,
        None => {
            let id = cudarc::nccl::result::get_uniqueid().unwrap();
            unique_ids_map.insert(comm_id, id);
            id
        }
    }
}

pub(crate) fn to_nccl_op(op: ReduceOperation) -> cudarc::nccl::sys::ncclRedOp_t {
    match op {
        ReduceOperation::Sum => cudarc::nccl::sys::ncclRedOp_t::ncclSum,
        ReduceOperation::Mean => cudarc::nccl::sys::ncclRedOp_t::ncclAvg,
    }
}

pub(crate) fn get_nccl_dtype_count(
    dtype: ElemType,
    size: u64,
) -> (cudarc::nccl::sys::ncclDataType_t, usize) {
    match dtype {
        ElemType::Float(
            cubecl_core::ir::FloatKind::E2M1
            | cubecl_core::ir::FloatKind::E2M3
            | cubecl_core::ir::FloatKind::E3M2
            | cubecl_core::ir::FloatKind::UE8M0,
        ) => panic!("Minifloat not supported in NCCL"),
        ElemType::Float(cubecl_core::ir::FloatKind::E4M3) => (
            cudarc::nccl::sys::ncclDataType_t::ncclFloat8e4m3,
            size as usize,
        ),
        ElemType::Float(cubecl_core::ir::FloatKind::E5M2) => (
            cudarc::nccl::sys::ncclDataType_t::ncclFloat8e5m2,
            size as usize,
        ),
        ElemType::Float(cubecl_core::ir::FloatKind::F16) => (
            cudarc::nccl::sys::ncclDataType_t::ncclFloat16,
            (size / 2) as usize,
        ),
        ElemType::Float(cubecl_core::ir::FloatKind::BF16) => (
            cudarc::nccl::sys::ncclDataType_t::ncclBfloat16,
            (size / 2) as usize,
        ),
        ElemType::Float(cubecl_core::ir::FloatKind::Flex32) => {
            panic!("NCCL doesn't support Flex32 format.")
        }

        ElemType::Float(cubecl_core::ir::FloatKind::F32) => (
            cudarc::nccl::sys::ncclDataType_t::ncclFloat32,
            (size / 4) as usize,
        ),
        ElemType::Float(cubecl_core::ir::FloatKind::TF32) => {
            panic!("NCCL doesn't support TF32 format.")
        }
        ElemType::Float(cubecl_core::ir::FloatKind::F64) => (
            cudarc::nccl::sys::ncclDataType_t::ncclFloat64,
            (size / 8) as usize,
        ),
        ElemType::Int(int_kind) => match int_kind {
            cubecl_core::ir::IntKind::I8 => {
                (cudarc::nccl::sys::ncclDataType_t::ncclInt8, size as usize)
            }
            cubecl_core::ir::IntKind::I16 => panic!("NCCL doesn't support Int16 format."),
            cubecl_core::ir::IntKind::I32 => (
                cudarc::nccl::sys::ncclDataType_t::ncclInt32,
                (size / 4) as usize,
            ),
            cubecl_core::ir::IntKind::I64 => (
                cudarc::nccl::sys::ncclDataType_t::ncclInt64,
                (size / 8) as usize,
            ),
        },
        ElemType::UInt(uint_kind) => match uint_kind {
            cubecl_core::ir::UIntKind::U8 => {
                (cudarc::nccl::sys::ncclDataType_t::ncclUint8, size as usize)
            }
            cubecl_core::ir::UIntKind::U16 => panic!("NCCL doesn't support UInt16 format."),
            cubecl_core::ir::UIntKind::U32 => (
                cudarc::nccl::sys::ncclDataType_t::ncclUint32,
                (size / 4) as usize,
            ),
            cubecl_core::ir::UIntKind::U64 => (
                cudarc::nccl::sys::ncclDataType_t::ncclUint64,
                (size / 8) as usize,
            ),
        },
        ElemType::Bool => panic!("NCCL doesn't support Bool format."),
    }
}

// ---------------------------------------------------------------------------
// An externally bootstrapped communicator, for the multi-PROCESS case.
// ---------------------------------------------------------------------------
//
// [`get_nccl_comm_id`] mints a unique id and remembers it in a process-global
// map. That is exactly right for one process driving several GPUs — every
// participant reads the same map — and it is unusable across two processes on
// two nodes, in two independent ways:
//
//  1. Each process mints its OWN id, so `ncclCommInitRank` never pairs them.
//     The failure is a HANG in the rendezvous rather than an error, which is
//     the worst shape for it to take.
//  2. The rank is derived as this device's position in the sorted device list.
//     Two boxes each holding their local CUDA device 0 both derive rank 0, so
//     even a shared id would produce two rank-0s and no rank 1.
//
// Both are properties of how the group was FORMED, and a process that spans
// nodes already has to form it itself: one rank mints the id and ships the 128
// bytes to the others over whatever channel it already has. This is the seam
// where it hands the result back. Set it before the first collective and the
// derivation above is bypassed whole; leave it unset and nothing here changes.

/// A communicator bootstrapped outside this process: the shared id, and who
/// this process is inside the group.
#[derive(Clone, Copy, Debug)]
pub struct ExternalComm {
    /// The unique id one rank minted and every rank received.
    pub id: cudarc::nccl::sys::ncclUniqueId,
    /// This process's rank, `0..world`.
    pub rank: i32,
    /// How many ranks are in the group.
    pub world: i32,
}

static EXTERNAL_COMM: OnceLock<Mutex<Option<ExternalComm>>> = OnceLock::new();

/// Mint a unique id, as plain bytes to send over a socket.
///
/// Bytes rather than `ncclUniqueId` because the caller's job is to TRANSPORT
/// this, and because `c_char` is signed on x86_64 and unsigned on aarch64 —
/// these two boxes are aarch64, but a type whose signedness depends on the
/// target has no business in a wire format. The round trip through
/// [`set_external_comm`] is bit-exact either way.
pub fn mint_unique_id() -> Result<[u8; 128], String> {
    let id = cudarc::nccl::result::get_uniqueid()
        .map_err(|e| format!("ncclGetUniqueId failed: {e:?}"))?;
    Ok(id.internal.map(|c| c as u8))
}

/// Install the group this process belongs to, overriding the single-process
/// derivation for every collective from here on.
pub fn set_external_comm(id: [u8; 128], rank: i32, world: i32) {
    let id = cudarc::nccl::sys::ncclUniqueId {
        internal: id.map(|b| b as core::ffi::c_char),
    };
    *EXTERNAL_COMM.get_or_init(Default::default).lock().unwrap() =
        Some(ExternalComm { id, rank, world });
}

pub(crate) fn external_comm() -> Option<ExternalComm> {
    *EXTERNAL_COMM.get_or_init(Default::default).lock().unwrap()
}
