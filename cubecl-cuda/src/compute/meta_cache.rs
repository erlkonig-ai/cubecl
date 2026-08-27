//! The per-launch dynamic-metadata cache.
//!
//! # What it is for
//!
//! A launch's packed argument blob (`MetadataBindingInfo::data`) is two things
//! end to end, split at `dynamic_metadata_offset`. Below the offset sit the
//! scalars and the STATIC metadata -- buffer lengths, ranks, and the offsets
//! into the second half -- which ride to the kernel by value as a
//! `__grid_constant__` struct. At or above it sit every bound tensor's SHAPE
//! and STRIDE list, which are variable-length and therefore cannot be a
//! fixed-size kernel parameter. They go to the device in a buffer of their own,
//! and `create_with_data` builds that buffer from scratch on every launch:
//! reserve a pinned host buffer, memcpy the shapes and strides into it,
//! allocate a device buffer, `cuMemcpyHtoDAsync`.
//!
//! That happens on every launch binding a ranked tensor, every time, whether or
//! not a single shape moved.
//!
//! MEASURED, per decode step per node, `mary`'s inkling decode path with
//! `INK_LAYERS=0:21`, `INK_KV=1`, ctx3732, on one GB10: 483 host-to-device
//! memcpy nodes out of 1783 kernel launches (27% of launches carry one),
//! 19,280 bytes in total, with a size histogram of 16 B x306, 32 B x103,
//! 64 B x57, 144 B x2 and 208 B x36 -- identical between two captures of
//! consecutive steps. Only 29 DISTINCT DESTINATION ADDRESSES across all 483.
//! The payload is constant for 128-step epochs at a fixed decode config,
//! because the KV pages are pre-allocated at `PAGE = 128` rows and handed over
//! whole, so on 127 of every 128 steps every one of those 483 uploads copies
//! the same bytes it copied last step, from a different host address, into the
//! same device address.
//!
//! So: key a small per-stream map on those bytes, and when a launch is about to
//! describe a shape list some device buffer already holds, bind that buffer and
//! skip the reserve, the copy and the upload entirely.
//!
//! # What is keyed, and why the key is the bytes themselves
//!
//! The key is the dynamic half's exact bytes -- which IS the binding set's
//! (rank, shape, stride) tuples, in the packed form the kernel will read. The
//! map is indexed by a cheap 64-bit hash of them, and every entry keeps the
//! full bytes alongside. A lookup that finds a hash match still compares all
//! the bytes before returning the handle, so a collision is CAUGHT (counted,
//! and served by the ordinary upload path) rather than trusted. A silently
//! wrong shape list is a silently wrong kernel, and a wrong kernel here would
//! not fail -- it would compute a plausible answer on the wrong strides.
//!
//! # Why it never inserts while a capture is open
//!
//! This is a correctness requirement, not caution. Inside a capture the H2D
//! copy that fills a freshly created buffer is not executed; it is RECORDED as
//! a graph node and does not run until a replay. Caching that buffer would let
//! a later eager launch bind memory whose contents are still whatever the
//! allocator last left there. The same reasoning rules out inserting while the
//! capture arena is open: an arena slice is capture-scoped scratch, matched by
//! size in a fixed order on every reopening, and an entry that outlives the
//! region must not be one.
//!
//! A lookup inside a capture is a different matter and is what
//! [`MetaCacheMode::Captures`] enables -- see its documentation for what it
//! buys and what it perturbs.

use cubecl_core::server::Handle;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

/// How much of the launch path the cache is allowed to touch.
///
/// Default is [`MetaCacheMode::Off`], which is today's behaviour exactly: the
/// cache is never consulted, never allocates, and costs not even a hash. The
/// other two arms are selected with `CUBECL_META_CACHE`, so one binary can be
/// A/B'd against itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetaCacheMode {
    /// `CUBECL_META_CACHE` unset, `0`, or `off`. No cache at all.
    Off,
    /// `CUBECL_META_CACHE=1`. The cache serves the EAGER path and is bypassed
    /// entirely -- neither read nor written -- while a capture is open on the
    /// stream.
    ///
    /// This is the arm the measured decode-step figures above are about, and it
    /// cannot perturb a captured region in any way, because a captured region
    /// never sees it.
    Eager,
    /// `CUBECL_META_CACHE=2`. As `Eager`, and additionally a launch made while
    /// a capture is open may HIT the cache (it still never inserts). A hit
    /// inside a capture is what removes the 483 memcpy nodes from the recorded
    /// graph, which is the point of the exercise for the cross-step graph work.
    ///
    /// TWO THINGS THIS PERTURBS, named rather than left to be discovered.
    ///
    /// 1. A hit inside a capture is one fewer request the capture makes of the
    ///    arena, so `arena_signature` -- the hash of the sizes served, in order
    ///    -- depends on what the cache held when the arena opened. The cache
    ///    never grows while a capture or an arena window is open, so a single
    ///    window is self-consistent. A region re-captured in a LATER window,
    ///    after eager steps have warmed the cache further, will sign
    ///    differently from the first capture, and `graph_replay` refuses the
    ///    pair. That refusal is correct -- the two really did allocate
    ///    different arena slices -- but its message names the wrong cause, so
    ///    it names this one too.
    /// 2. A launch that hit the cache has no pinned staging buffer, because it
    ///    performed no upload. `graph_patch_launch` cannot then rewrite its
    ///    dynamic half, and says so. That is the safe outcome: the device
    ///    buffer a hit bound is SHARED with every other launch describing the
    ///    same shapes, so writing a patch into it would silently change theirs.
    Captures,
}

impl MetaCacheMode {
    /// Read `CUBECL_META_CACHE` once.
    pub fn current() -> Self {
        static MODE: std::sync::OnceLock<MetaCacheMode> = std::sync::OnceLock::new();
        *MODE.get_or_init(|| match std::env::var("CUBECL_META_CACHE").as_deref() {
            Ok("1") | Ok("on") | Ok("eager") => MetaCacheMode::Eager,
            Ok("2") | Ok("capture") | Ok("captures") => MetaCacheMode::Captures,
            _ => MetaCacheMode::Off,
        })
    }

    /// Whether the cache does anything at all.
    pub fn enabled(&self) -> bool {
        !matches!(self, MetaCacheMode::Off)
    }
}

/// How many entries the cache may hold before it evicts, from
/// `CUBECL_META_CACHE_CAP`.
///
/// The default of 4096 is chosen against two measurements rather than picked.
/// A decode step's whole distinct set is of the order of the 29 distinct
/// destination addresses the census found, so in the steady state the cache
/// never reaches the bound and the eviction path never runs. A PREFILL is the
/// case that can grow: its shapes move with the sequence, so the distinct set
/// scales with the number of distinct sequence lengths a run passes through
/// times the kernels that bind a ranked tensor. 4096 entries at the observed
/// 16--208 bytes a payload is under a megabyte of device memory, which is
/// nothing against a 119 GiB box, while still refusing to grow without limit on
/// a pathological run.
fn capacity() -> usize {
    static CAP: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CAP.get_or_init(|| {
        std::env::var("CUBECL_META_CACHE_CAP")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(4096)
    })
}

/// Print a one-line summary every this many metadata launches, from
/// `CUBECL_META_CACHE_STATS`. Zero (the default) prints nothing.
fn stats_every() -> u64 {
    static N: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("CUBECL_META_CACHE_STATS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0)
    })
}

/// Whether the metadata half of a launch times itself, from
/// `CUBECL_TIME_META_CACHE`. Read once: a `getenv` per launch would be a cost
/// on the path this exists to measure.
///
/// WHAT THE NUMBER IS, and it is the whole reason this knob exists rather than
/// a hit-rate alone. With the cache OFF every metadata launch is charged as a
/// MISS, and the miss time is then the cost of the thing the cache replaces:
/// a pinned reserve, a host memcpy, a device reservation and a
/// `cuMemcpyHtoDAsync` enqueue. With the cache ON the hit time is what it
/// costs instead: a hash of at most a few hundred bytes, a map probe and a
/// full-payload comparison. The two arms of one binary therefore answer "what
/// does a hit cost against the upload it replaces" directly, in the same units,
/// rather than by inference from a count.
///
/// FRAMING: it is the METADATA half of `launch_checked` only -- the lookup, or
/// the lookup plus the buffer it had to build -- per LAUNCH THAT BINDS A RANKED
/// TENSOR. It is not the launch, it does not include the driver's
/// `cuLaunchKernel`, and it does not include the device time of the copy it
/// enqueues. Off by default: it costs two clock reads a launch, so the timed
/// arm is slower than the arm it measures.
fn time_meta() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("CUBECL_TIME_META_CACHE").as_deref() == Ok("1"))
}

/// Whether the launch path accounts for the metadata half at all. False by
/// default, and then a launch pays nothing for this module beyond the map
/// probe the cache itself is.
pub fn observing() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| stats_every() > 0 || time_meta())
}

/// Whether to start a clock for [`observe`]. `None` when the timing is off, so
/// the caller's `Instant::now` is not paid for.
pub fn clock() -> Option<std::time::Instant> {
    time_meta().then(std::time::Instant::now)
}

/// Process-wide counters. The cache itself is per stream; the counters are not,
/// because what a reader wants to know is what the RUN did, and a run with one
/// compute stream is the case every figure above was measured on.
static LOOKUPS: AtomicU64 = AtomicU64::new(0);
static HITS: AtomicU64 = AtomicU64::new(0);
static MISSES: AtomicU64 = AtomicU64::new(0);
static COLLISIONS: AtomicU64 = AtomicU64::new(0);
static BYPASSED: AtomicU64 = AtomicU64::new(0);
static INSERTS: AtomicU64 = AtomicU64::new(0);
static EVICTIONS: AtomicU64 = AtomicU64::new(0);
static BYTES_SAVED: AtomicU64 = AtomicU64::new(0);
static HIT_NANOS: AtomicU64 = AtomicU64::new(0);
static MISS_NANOS: AtomicU64 = AtomicU64::new(0);
/// The cumulative counters as of the last report, so a report can print a
/// WINDOW as well as a total. A window is what separates the prefill from the
/// decode steps: the two do different amounts of this work and a run-long mean
/// is the wrong number for either.
static LAST: [AtomicU64; 5] = [
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
];

/// Account for one launch's metadata half.
///
/// Called from the launch path in EVERY mode, including `Off`, and only when
/// [`observing`] -- which is what lets the off arm report the cost of the
/// upload the on arm elides. `hit` is whether the cache supplied the buffer;
/// with the cache off it is always false and the launch is charged as a miss,
/// which is exactly what it is.
pub fn observe(
    hit: bool,
    mode: MetaCacheMode,
    capture_open: bool,
    bytes: usize,
    since: Option<std::time::Instant>,
) {
    if capture_open && mode == MetaCacheMode::Eager {
        BYPASSED.fetch_add(1, Ordering::Relaxed);
    }
    let n = LOOKUPS.fetch_add(1, Ordering::Relaxed) + 1;
    let nanos = since.map(|t| t.elapsed().as_nanos() as u64).unwrap_or(0);
    match hit {
        true => {
            HITS.fetch_add(1, Ordering::Relaxed);
            HIT_NANOS.fetch_add(nanos, Ordering::Relaxed);
            BYTES_SAVED.fetch_add(bytes as u64, Ordering::Relaxed);
        }
        false => {
            MISSES.fetch_add(1, Ordering::Relaxed);
            MISS_NANOS.fetch_add(nanos, Ordering::Relaxed);
        }
    }
    let every = stats_every();
    if every > 0 && n % every == 0 {
        report();
    }
}

/// One cached shape-and-stride list, and the device buffer that holds it.
#[derive(Debug)]
struct Entry {
    /// The full dynamic half, kept so a hash match is VERIFIED and not
    /// believed. At 16--208 bytes the comparison is far cheaper than the
    /// upload it replaces.
    bytes: Vec<u8>,
    /// The device buffer. Holding the handle is what keeps the slice out of the
    /// allocator's hands, which is the whole of the invalidation obligation:
    /// nothing may be given this address while a launch -- or a graph node --
    /// still names it.
    handle: Handle,
    /// Lookup counter at the last hit, for eviction order.
    last_used: u64,
    /// Whether this entry's buffer was ever bound by a launch made while a
    /// capture was open. Such a buffer's ADDRESS is baked into a graph node
    /// that may be replayed at any time, so the entry is never evicted.
    pinned: bool,
}

/// A per-stream map from a launch's dynamic metadata to the device buffer that
/// already holds it.
#[derive(Debug, Default)]
pub struct MetaCache {
    entries: HashMap<u64, Entry>,
    /// Monotonic, used only to order eviction.
    clock: u64,
}

impl MetaCache {
    /// Look for a device buffer already holding exactly `bytes`.
    ///
    /// `capture_open` is passed rather than read from the global flag so the
    /// caller's one read of it is the one that decides, and so the two arms of
    /// [`MetaCacheMode`] differ in exactly one place.
    pub fn get(&mut self, bytes: &[u8], mode: MetaCacheMode, capture_open: bool) -> Option<Handle> {
        if !mode.enabled() {
            return None;
        }
        if capture_open && mode != MetaCacheMode::Captures {
            return None;
        }
        self.clock += 1;
        match self.entries.get_mut(&hash(bytes)) {
            // A hash match whose bytes DIFFER is the collision this design
            // exists to catch. Counted, and then served by the ordinary upload
            // path: the entry is left alone rather than replaced, so the
            // incumbent payload keeps its buffer and the newcomer keeps paying
            // for its upload. Both are correct; neither is silent.
            Some(e) if e.bytes != bytes => {
                COLLISIONS.fetch_add(1, Ordering::Relaxed);
                None
            }
            Some(e) => {
                e.last_used = self.clock;
                e.pinned |= capture_open;
                Some(e.handle.clone())
            }
            None => None,
        }
    }

    /// Remember that `handle` holds `bytes`.
    ///
    /// The caller must have created `handle` on the EAGER path: see the module
    /// documentation for why a buffer created inside a capture, or served from
    /// the capture arena, may not be cached. Both conditions are asserted by
    /// the caller passing `false` here; this function does not read them
    /// itself, so that the one place that decides is the launch path.
    pub fn insert(&mut self, bytes: &[u8], handle: Handle) {
        self.clock += 1;
        INSERTS.fetch_add(1, Ordering::Relaxed);
        self.entries.insert(
            hash(bytes),
            Entry {
                bytes: bytes.to_vec(),
                handle,
                last_used: self.clock,
                pinned: false,
            },
        );
        self.evict();
    }

    /// Bring the cache back under its bound, oldest unpinned entry first.
    ///
    /// Evicting in a BATCH once over the bound, rather than one entry per
    /// insert, is what keeps the O(n) scan off the steady-state path: with the
    /// measured 29-entry working set the bound is never reached at all, and on
    /// a prefill that does reach it the scan runs once per quarter-capacity of
    /// new payloads instead of once per launch.
    ///
    /// Dropping a handle is safe for a launch already enqueued against it: the
    /// slice returns to the pool under the same stream-cursor discipline that
    /// governs today's metadata buffer, which is dropped at the end of the very
    /// launch that created it. What is NOT safe is dropping one a graph node
    /// names, which is what `pinned` refuses.
    fn evict(&mut self) {
        let cap = capacity();
        if self.entries.len() <= cap {
            return;
        }
        let target = cap - cap / 4;
        let mut ages: Vec<(u64, u64)> = self
            .entries
            .iter()
            .filter(|(_, e)| !e.pinned)
            .map(|(k, e)| (e.last_used, *k))
            .collect();
        ages.sort_unstable();
        // Never more than there are unpinned entries to give: a cache whose
        // pinned set alone exceeds the bound stays over it, which is the only
        // honest answer when every entry is named by a live graph node.
        let wanted = self.entries.len().saturating_sub(target).min(ages.len());
        for (_, key) in ages.into_iter().take(wanted) {
            self.entries.remove(&key);
            EVICTIONS.fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// FNV-1a over the payload, seeded with its length so two payloads of different
/// lengths cannot collide on their common prefix.
///
/// This is a bucket index, not a proof of equality -- the equality is the byte
/// comparison [`MetaCache::get`] does on every hit -- so a fast weak hash is
/// the right instrument. Over a working set of tens of keys and payloads of at
/// most a few hundred bytes it is a handful of cycles against a reserve, a
/// memcpy and a driver call.
fn hash(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325 ^ (bytes.len() as u64).wrapping_mul(0x1000_0000_01b3);
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(0x1000_0000_01b3);
    }
    h
}

/// Two lines, each carrying its own framing rule.
///
/// The TOTAL line is counts since process start, over every stream. The WINDOW
/// line is the same counts since the previous report, which is the one that
/// separates a prefill from a decode step: set `CUBECL_META_CACHE_STATS` to
/// roughly one step's metadata launches and each window is about one pass.
///
/// The two mean times are per LAUNCH THAT BINDS A RANKED TENSOR, and they are
/// the whole point of the timing: `hit` is what the cache costs, `miss` is what
/// the upload it replaces costs, in the same units, in the same binary. They
/// print as `--` unless `CUBECL_TIME_META_CACHE=1`.
fn report() {
    let now = [
        LOOKUPS.load(Ordering::Relaxed),
        HITS.load(Ordering::Relaxed),
        MISSES.load(Ordering::Relaxed),
        HIT_NANOS.load(Ordering::Relaxed),
        MISS_NANOS.load(Ordering::Relaxed),
    ];
    let prev: Vec<u64> = LAST.iter().map(|a| a.load(Ordering::Relaxed)).collect();
    for (a, v) in LAST.iter().zip(now.iter()) {
        a.store(*v, Ordering::Relaxed);
    }
    let mean = |nanos: u64, n: u64| match (time_meta(), n) {
        (false, _) => "--".to_string(),
        (true, 0) => "n/a".to_string(),
        (true, n) => format!("{:.0} ns", nanos as f64 / n as f64),
    };
    let rate = |h: u64, l: u64| match l {
        0 => 0.0,
        l => 100.0 * h as f64 / l as f64,
    };
    eprintln!(
        "[meta-cache] TOTAL since process start: {} metadata launches, {} hits ({:.1}%), \
         {} misses, {} bytes not uploaded, {} entries inserted, {} evicted, \
         {} hash collisions caught, {} bypassed inside a capture; \
         mean per launch: hit {}, miss {}",
        now[0],
        now[1],
        rate(now[1], now[0]),
        now[2],
        BYTES_SAVED.load(Ordering::Relaxed),
        INSERTS.load(Ordering::Relaxed),
        EVICTIONS.load(Ordering::Relaxed),
        COLLISIONS.load(Ordering::Relaxed),
        BYPASSED.load(Ordering::Relaxed),
        mean(now[3], now[1]),
        mean(now[4], now[2]),
    );
    let (dl, dh, dm) = (now[0] - prev[0], now[1] - prev[1], now[2] - prev[2]);
    eprintln!(
        "[meta-cache] WINDOW since last report: {dl} metadata launches, {dh} hits \
         ({:.1}%), {dm} misses; mean per launch: hit {}, miss {}",
        rate(dh, dl),
        mean(now[3] - prev[3], dh),
        mean(now[4] - prev[4], dm),
    );
}

#[cfg(test)]
mod tests {
    use super::{Entry, MetaCache, MetaCacheMode, hash};
    use cubecl_common::stream_id::StreamId;
    use cubecl_core::server::Handle;

    /// A handle carrying nothing but an identity. `MetaCache` never reads a
    /// handle's memory -- it stores one and hands clones back -- so a bare
    /// `Handle::new` is the whole of what these tests need, and they run with
    /// no device.
    fn handle(size: u64) -> Handle {
        Handle::new(StreamId::current(), size)
    }

    /// The length seed is load-bearing: without it a payload and its own
    /// prefix-extension by zero bytes are not distinguished by FNV-1a's own
    /// mixing as cheaply as one would like.
    #[test]
    fn hash_separates_lengths() {
        assert_ne!(hash(&[1, 2, 3]), hash(&[1, 2, 3, 0]));
        assert_ne!(hash(&[]), hash(&[0]));
    }

    #[test]
    fn hash_is_deterministic() {
        let bytes: Vec<u8> = (0..208u16).map(|i| i as u8).collect();
        assert_eq!(hash(&bytes), hash(&bytes.clone()));
    }
    #[test]
    fn a_hit_hands_back_the_buffer_that_holds_those_bytes() {
        let mut c = MetaCache::default();
        c.insert(&[1, 2, 3, 4], handle(11));
        c.insert(&[9, 9, 9, 9], handle(22));
        let hit = c
            .get(&[1, 2, 3, 4], MetaCacheMode::Eager, false)
            .expect("the payload was inserted");
        assert_eq!(
            hit.size(),
            11,
            "a hit must name the buffer holding ITS bytes"
        );
        assert!(c.get(&[7, 7], MetaCacheMode::Eager, false).is_none());
    }

    /// The point of keeping the full bytes. A hash match whose payload differs
    /// must be a MISS -- the caller then uploads, which is correct -- and not a
    /// handle to somebody else's shapes, which would be a wrong kernel that
    /// does not fail.
    #[test]
    fn a_hash_match_with_different_bytes_is_a_miss() {
        let mut c = MetaCache::default();
        let probe: &[u8] = &[1, 2, 3, 4];
        // Planted directly: two payloads that really collide under FNV-1a are
        // not constructible by hand, and what is under test is the VERIFY, not
        // the hash.
        c.entries.insert(
            hash(probe),
            Entry {
                bytes: vec![5, 6, 7, 8],
                handle: handle(33),
                last_used: 0,
                pinned: false,
            },
        );
        assert!(c.get(probe, MetaCacheMode::Eager, false).is_none());
        // And the incumbent is left alone, so its own lookups still hit.
        assert_eq!(
            c.get(&[5, 6, 7, 8], MetaCacheMode::Eager, false)
                .map(|h| h.size()),
            Some(33)
        );
    }

    #[test]
    fn off_never_looks_and_captures_is_what_lets_a_capture_look() {
        let mut c = MetaCache::default();
        c.insert(&[1, 2, 3, 4], handle(11));
        assert!(c.get(&[1, 2, 3, 4], MetaCacheMode::Off, false).is_none());
        assert!(c.get(&[1, 2, 3, 4], MetaCacheMode::Eager, true).is_none());
        assert!(
            c.get(&[1, 2, 3, 4], MetaCacheMode::Captures, true)
                .is_some()
        );
    }

    /// An entry handed out inside a capture has its ADDRESS in a graph node,
    /// so eviction may not take it however old it is.
    #[test]
    fn eviction_bounds_the_map_and_spares_what_a_graph_points_at() {
        let cap = super::capacity();
        let mut c = MetaCache::default();
        // The oldest entry, pinned by a capture, and one more that is not.
        c.insert(&[0, 0, 0, 1], handle(1));
        let _ = c.get(&[0, 0, 0, 1], MetaCacheMode::Captures, true);
        c.insert(&[0, 0, 0, 2], handle(2));
        for i in 0..(cap as u32 + 64) {
            c.insert(&i.to_le_bytes(), handle(100 + i as u64));
        }
        assert!(
            c.entries.len() <= cap,
            "the map grew past its bound: {} > {cap}",
            c.entries.len()
        );
        assert!(
            c.get(&[0, 0, 0, 1], MetaCacheMode::Captures, true)
                .is_some(),
            "a pinned entry was evicted, and a graph node still names its buffer"
        );
    }
}
