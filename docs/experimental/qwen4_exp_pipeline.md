# Qwen4-Exp pipeline across N Macs

Qwen3.8-Flash-Next (`model_type: qwen4_exp`) is served across a configurable
number of Macs by splitting its decoder layers into contiguous pipeline stages.
This page records what exists, how each part was checked, and what is still
refused.

## Shape of the chain

```
DistributedBatchedEngine -> DistributedJobSupervisor -> inference_worker (per rank)
  -> ModelProvider (mlx_lm.server) -> progressive_sharded_load
  -> mlx_lm.models.qwen4_exp  (bridge, omlx/patches/qwen4_exp_mlx_lm)
  -> vendored mlx-vlm Qwen4-Exp (omlx/patches/mlx_vlm_qwen4_exp_compat/vendor)
```

* **Planner** (`cluster/planner.py`): a Qwen4-Exp checkpoint is offered
  pipeline stages (`_supports_pipeline`) from its configuration alone. Only
  `language_model.*.layers.N` tensors count as decoder layers; the vision tower
  and the MTP head are fixed, replicated weights. The link cost model charges the
  whole stage boundary (see below). The planner already handled 2 to 48 stages
  contiguously, in reverse rank order, and refuses more stages than layers.
* **Admission** (`engine_pool.py`): a cluster deployment may serve a VLM entry
  when its registered `PipelineModelAdapter` declares a serving path. The
  Qwen4-Exp adapter declares text and image input.
* **Bridge** (`omlx/patches/qwen4_exp_mlx_lm`): registers
  `mlx_lm.models.qwen4_exp` so each rank, which is an `mlx_lm.server`, can load
  the model. `Model` *subclasses* the vendored mlx-vlm model, so the parameter
  tree (and with it the checkpoint's per-tensor quantization overrides) is the
  checkpoint's own, and the root-level MTP head keeps its owner. `__call__`
  returns logits for `generate` and `BatchGenerator`. There is no second copy of
  the model.
* **Loader**: the model builds only the layers its rank owns (the others stay
  `None`: no weights, PLE table or storage handle exists for them) and
  `sanitize` drops other stages' tensors before any stacking or dequantization
  graph is built. The RMSNorm centering vote of the vendored sanitizer samples
  every layer's hyper-connection norm, so those few tensors are kept through the
  vote and dropped after it; otherwise a small stage would decide differently
  from its peers. The vision tower is built only by the stage that owns layer 0.
* **Hook recognition** (`cluster/pipeline_compat.py`): the bridge names the exact
  class carrying `pipeline`; `pipeline_assignment_is_honored` accepts the model
  only if that method carries the assignment marker. The declaration is a
  pointer to evidence, not evidence.
* **Worker validation** (`cluster/inference_worker.py`): `_validate_loaded_stage`
  also compares the stage the model recorded, the world position and the decoder
  layers its *parameters* belong to with the approved assignment, and
  `verify_pipeline_contract` runs one collective after the load: every rank must
  agree on the wire layout and the layer ranges must chain from layer 0 on the
  highest rank to the last layer on rank 0.

## Runtime contract

Architecture-specific answers live in `cluster/model_adapters.py` and the
Qwen4-Exp adapter: decoder tensor ownership, boundary size, loader setup and
served modalities. The planner, worker and engine consume the same interface.

The inspected layout carries explicit runtime options. They survive remote
layout serialization, contribute to the plan hash and placement approval, and
are copied into both probe and production deployments. Workers never choose a
PLE storage mode from their own environment. The coordinator resolves the saved PLE SSD setting before planning and sends
`ple_mode=resident` or `ple_mode=mmap` to every rank. Performance replanning
preserves the approved options. A changed setting refuses loading until a new
plan is approved. Distributed loading does not apply a storage decision based
on the coordinator's local RAM alone.

Admission remains conservative: all PLE bytes remain in the approved weight
budget, including file-backed pages. mmap storage is supported; increased
cluster capacity or reduced physical RAM consumption is not claimed. Unknown Qwen4-Exp options and non-finite JSON values fail.

## The stage boundary

Ranks are numbered as MLX-LM's `PipelineMixin` numbers them: the highest rank
owns the first layers, execution flows from rank `size - 1` to rank `0`, and
rank `0` produces the output and serves HTTP.

The tensor leaving a stage is not one hidden state. With deferred residual
writes (the default) each layer returns a pending `(branch, gate)` write that
the next hyper-connection norm applies, so a stage sends one packed tensor
`[residual | branch | gate]` on the last axis (`hc_count * hidden + hidden +
hc_count` wide). One message means one collective per boundary and slicing it
moves bits only. With `OMLX_QWEN4_HC_FUSED_WRITE=0` the boundary is the residual
alone. The closing `hyper_connection_mixer` runs once, on the last stage, and
its output is gathered so every rank can produce logits for MLX-LM's
synchronized sampler. Token ids reach the PLE layer through the ordinary
request sharing, whichever stage owns the embeddings.

Each stage keeps a cache list with one entry per *local* layer, and `fa_idx` /
`ssm_idx` are positions in that list; a stage may hold only one attention
family.

## What was demonstrated, and how

All numeric checks compare a pipeline stage with the same reduced Qwen4-Exp run
whole in the same process, from the same weights, with tolerances fixed in the
test files (`float32` 1e-5, `float16`/`bfloat16` 1e-2, identical tokens). The
reduced architecture keeps the GatedDeltaNet and QSA layers (QSA budget lowered
to 8 tokens so its selection mask is active), hyper-connections, a MoE with a
shared expert, PLE on the second layer, and a vision tower. No user checkpoint
is loaded.

| Level | Evidence |
| --- | --- |
| Real ring collectives, 2, 3 and 4 ranks | `tests/test_qwen4_exp_pipeline_ring.py` |
| Uneven splits; a stage with no full attention; a stage with GatedDeltaNet absent; a one-layer first stage | same |
| Fragmented prefill (5/7/9 tokens), six decodes, logits and every cache tensor (GDN conv/recurrent state, PLE history, QSA keys/values/indexer keys/positions) | same |
| Deferred write crossing the boundary (width asserted) and the eager variant | same |
| float32, float16, bfloat16; batch of 2 | same |
| Production loader, 3 ranks: out-of-stage tensors never sanitized, resident layers equal the stage, weights bit-equal to a whole load | same |
| Legacy RMSNorm checkpoint on stages smaller than the vote quorum | same, plus a negative control in `test_qwen4_exp_pipeline.py` |
| 4-bit affine checkpoint with 8-bit gate overrides and a quantized embedding | same |
| Image + text, 2 and 3 ranks: vision tower on the first stage only, multimodal rope positions on every attention stage, fragmented prefill and decodes | same |
| Real `inference_worker` processes + `mlx_lm.server`, 2 and 3 ranks: chat, streaming, rank-local prompt-cache hit, concurrent batched requests, equal to the whole model | `tests/test_qwen4_exp_worker_e2e.py` |
| `DistributedBatchedEngine` -> real workers: text and image HTTP, streaming | same |
| Image HTTP on 2 and 3 ranks, fragmented prefill, repeated-image cache hit, distinct pixels isolated, text/image transitions and invalid-image recovery | same |
| Loss of a rank: survivors exit within a bound; SIGTERM closes every rank | same |

A loopback ring proves the computation and the stage hand-off. It is not a
measurement of inter-Mac networking, JACCL or RDMA, and no throughput, TTFT or
memory figure from it applies to a real cluster.

## Optimization status and remaining validation

* **PLE SSD:** two-rank HTTP generation, prompt-cache reuse and streaming have
  also been checked with mmap storage against the resident whole-model output.
  These use a synthetic checkpoint on one Mac.
* **Image serving:** implemented through the existing MLX-LM request broadcast
  and sequential generation path. Rank 0 applies the mlx-vlm processor once;
  the first pipeline stage runs the vision tower. Each attention stage receives
  matching multimodal positions. Images run sequentially; existing text batching
  remains enabled. The cache key includes processed pixels, grid and token IDs.
  A prefill snapshot preserves recurrent state for image-cache reuse. No image
  batching or vision-feature cache speedup is claimed.
* **Audio/video:** remain refused by the distributed engine.
* **Native MTP/Lightning:** fixed or adaptive text generation now uses the
  existing MTP BatchGenerator loop. Runtime options carry activation and depth
  in the approved plan; missing embedded head weights fail before serving.
  Rank zero owns draft samples and final acceptance/correction choices. Cache
  commits still check agreement before modifying any stage. Ordinary sampling
  is coordinated too, so transitions out of MTP do not depend on rank-local RNG
  state. Tests cover greedy parity with ordinary generation, stochastic rank
  agreement despite deliberately different seeds, HTTP, streaming and concurrent
  requests on two and three local ranks. No speedup is claimed.
  Multiple text requests use existing shared verification and per-request cache
  commits, including cohorts shrinking back to one request. Batch cost samples
  and timing exclusions are owned by rank zero. PLE caches initialize token
  history with EOS and keep only valid prompt tokens when batching padded rows.
  Tests compare unequal-length requests with independent ordinary generation.
  Image requests use the sequential
  vision path. Adaptive depth reuses the existing
  controller with rank zero timing samples, constructor priors and re-entry
  readiness, keeping depth and performance parking decisions identical.
  Unspecified depth currently selects fixed depth 1.
* **SpecPrefill positional core:** the common sparse-prefill loop now supports
  models that accept explicit positions and own their decode offset. Qwen4's
  bridge supplies that offset from each stage's local attention cache; stages
  without attention need no RoPE wrapper. Tests compare selected-token prefill
  and subsequent decode with independent whole-model forwards, with and without
  a cached prefix, on two and three local ranks. Cleanup clears the offset after
  generation or a failed prefill. This does not establish quality parity with
  full-context generation: token dropping remains approximate.
* **SpecPrefill shared selection:** an isolated coordinator loads and scores a
  draft only on rank zero, then broadcasts selected indices or preparation
  errors through the worker's existing object broadcast. A reservation estimate
  covers draft weights, KV, scores and explicit workspace; admission precedes
  loading. Synthetic two/three-rank tests cover loader failure, recovery,
  insufficient budget and identical selections. Query capture supports both
  scalar-offset RoPE and explicit-position rotary interfaces. This component
  exposes a production-loader factory shared with the local engine. Loading
  temporarily hides target pipeline assignments and disables target MTP,
  materializes lazy buffers, and restores both settings even on failure.
  Model loading must remain serialized because these settings are global.
  Planning now inspects the configured draft and reserves its weights, KV,
  lookahead scores and at least 1 GiB of workspace for the requested context.
  This estimate is charged only to rank zero, reducing its target-layer budget.
  Assignment serialization preserves the charge for the worker memory guard
  without lowering the operator's process ceiling. Both pipeline allocation
  policies honor it; TP and contexts exceeding the reserved bound are refused.
  HTTP text serving now uses the existing sequential generation loop after
  shared sparse prefill. Sparse caches never enter the ordinary prefix LRU;
  original token history is preserved for logits processors and API usage.
  Preparation errors reach the client before response headers, and subsequent
  requests can recover. Seeded sampling remains rank-zero-owned. The collective
  broadcast explicitly restores distributed mode while sequential telemetry
  temporarily masks it for upstream cancellation handling.
  Synthetic two/three-rank HTTP tests cover missing-draft recovery, streaming,
  repeated requests, opt-out, a truly sparse 640-token prompt against a complete
  reference model, and reproducible sampled output. Text below the threshold
  and images keep the ordinary path. Per-request `specprefill=False` opts out;
  `True` requires an approved draft reservation. SpecPrefill and MTP cannot be
  enabled together. No speed or real-checkpoint quality claim follows.
* **DFlash:** one rank-zero drafter feeds the existing MTP verifier, including
  mixed-length text batches and singleton image streaming. Global target-layer
  captures are gathered in order without consuming the pending residual write.
  Draft proposals, stochastic proposal distributions and initialization failures
  are shared. Draft residency is reserved on rank zero. Defaults come from the
  draft checkpoint; optional weight quantization uses the existing loader.
  Optional sinks and RAM/SSD capture sidecars are supported (see the configuration
  section below). Alternative verifier algorithms remain explicitly refused. Draft/target compatibility still requires a suitable
  trained checkpoint; synthetic random drafts prove execution and correctness,
  not acceptance rate or speed. Speculative pre-drafting stays disabled.
* **External VLM MTP:** a compatible Qwen4 checkpoint can supply only its MTP
  head. The loader never constructs a second target trunk; existing sanitization,
  legacy RMSNorm centering and quantization are reused. Geometry and state-contract
  checks reject Qwen3.5/Gemma heads. Head weights and caches are replicated and
  reserved on every rank, like native MTP. Tests cover exact head weights including
  quantized/legacy exports, HTTP text and image streaming, and request reuse.
  The external path must be readable on every rank. No trained external pairing
  or speedup is claimed by the synthetic tests.
* **TurboQuant KV:** stage-local QSA caches retain unquantized indexer keys and
  multimodal positions alongside compressed K/V. The final attention-layer skip
  is global, not repeated per stage. Existing attention kernels are reused;
  text serving supports continuous batching. Packed cache merge, filter, extraction
  and padded rollback retain QSA keys and positions; rollback moves packed state
  without dequantizing and requantizing it. Image requests remain sequential.
  Copy, extraction, trim and durable snapshots preserve
  the complete state, including fractional MSE codecs. Planning uses conservative per-layer compressed cache budgets, including QSA
  index storage, fixed codec buffers and allocation rounding.
* **Combination policy:** native MTP, external MTP, DFlash, SpecPrefill and
  TurboQuant are selectable. TurboQuant can accompany exactly one of native MTP,
  external Qwen4 MTP or DFlash. Multiple draft strategies, and SpecPrefill combined
  with these strategies, remain refused. The separate `speculative_verify` API remains
  refused; supported speculation uses the existing transactional verifier.
* **Rank-only sampling / rank-zero logits:** enabled for ordinary batched text
  decode, including TurboQuant, through an explicit scoped output contract.
  Peers skip the vocabulary projection; only sampled tokens cross ranks during
  these decode steps. Ordinary singleton image requests also use coordinator-only logits.
  MTP, external MTP, DFlash and SpecPrefill retain their current distributed
  verification contract, so this optimization is disabled for those deployments.
* **Pipeline prefill overlap:** enabled for batched text, including TurboQuant,
  native MTP, external Qwen4 MTP and DFlash, when async overlap is enabled. Each chunk
  computes its stage output before submitting the transport asynchronously.
  Outstanding sends are bounded and drained before decoding. Singleton speculative
  image requests use the same scheduler; their prefix snapshots run after transport
  drains. Ordinary images use this singleton scheduler too. SpecPrefill retains
  its selected-position loop and uses the same bounded send queue.
  Speculative decode retains its verifier.
* **Tensor parallelism, pipeline + TP, MoE expert sharding:** not provided; a
  stage holds whole MoE layers.
* **Replicated weights:** embeddings and the output head are on every rank (each
  rank turns the gathered final state into logits); the planner counts them as
  fixed weights.
* **RDMA stage links, JACCL, multi-Mac memory, TTFT and throughput:** untested.
  The wire contract resolves `mx.distributed.send` / `recv_like` at call time so
  the worker's RDMA wrappers apply, but no RDMA edge was exercised with this
  model.
* **RAM/SSD/vision caches:** the existing rank-local LRU and durable prefix-chain
  store restore GDN, PLE, QSA and image positions. QSA uses its registered serializer;
  recurrent snapshots retain the MLX-VLM class, padding and lengths rather than
  resolving the same class name to MLX-LM. Cache misses or restoration failures
  still join the shared prefix vote, avoiding a stranded peer. Snapshot manifest
  version 3 invalidates older incompatible recurrent snapshots. Two/three-rank
  HTTP tests cover text/image hits after worker restart; a three-rank test covers
  fractional TurboQuant persistence. This reuses the cluster snapshot store,
  rather than introducing a second block allocator or DFlash-private cache tier.
* **KV reservation:** the planner reserves KV bytes per layer from the attention
  geometry for every layer, which over-reserves for a hybrid model whose
  GatedDeltaNet layers hold a fixed-size state instead.
* **Real checkpoint:** nothing here loads or measures it.

## Provenance

New code is Apache-2.0 (oMLX). It follows MLX-LM's `PipelineMixin` (MIT) for the
reverse rank convention and Exo PR #2283 (Apache-2.0) for the idea of repairing
`fa_idx` / `ssm_idx` after a pipeline split; no code was copied from either, and
no fork is imported. The vendored mlx-vlm Qwen4-Exp tree keeps its upstream
license notices.


### Compressed speculative batches (2 October 2026)

TurboQuant uses the existing QSA batch position contract and mlx-vlm packed batch
storage. The serving path no longer disables text batching. Each row retains its
own offset, padding, QSA keys and multimodal positions. Partial speculative
acceptance rolls packed state before the next step; no lossy requantization is
introduced by rollback. Stage-send dependencies preserve packed state types.

Synthetic two- and three-rank ring tests compare ragged MTP/DFlash batches against
independent compressed-target generation and assert actual verification cycles.
HTTP tests cover native MTP, external Qwen4 MTP and DFlash with fractional-bit
TurboQuant, text, images, repeated prompts, streaming and concurrent text requests.
The memory planner remains conservative; no measured RAM savings or performance
on physical multi-Mac/RDMA deployments are claimed.


### Per-layer compressed memory planning (2 October 2026)

Model layouts optionally carry per-layer KV byte rates, fixed cache allowances and
an allocation step. These fields survive serialization and participate in plan
hashes. The unequal partitioner uses each layer's cache budget, and each stage's
fit and maximum context use the same rates and fixed allowances. Older layouts
without these fields preserve their previous behavior. Unvalidated TP division of
these per-layer cache contracts is refused.

The Qwen4 adapter resolves geometry without importing MLX. It accepts both source
and normalized QSA layer names and the model's default geometry. Packed estimates
include fractional-codec padding, float32 raw and pooled index keys, multimodal
positions, codec buffers, stepped allocation and geometric-growth headroom.
Recurrent layers retain a conservative per-token allowance. These remain estimates
for cache residency; transient computation still uses the existing runtime reserve.
Changing or disabling TurboQuant recalculates or clears this profile, including on
replanning. Serving now recognizes normalized QSA names for the final-layer skip.

Validation: 189 planner/route/deployment tests, 26 targeted cache/HTTP tests, and one
additional route enable/disable test passed (the suites overlap). Five synthetic
allocation checks cover 3, 3.5, 4, 4.5 and 8 bits. Physical multi-Mac measurements,
rank-zero logits, prefill overlap and tensor/expert partitioning remain pending.


### Rank-zero ordinary decode (2 October 2026)

The generic runtime accepts an explicit model-owned output scope in addition to
its existing legacy pipeline contract. Qwen4 returns rank-local outputs only inside
that scope; ordinary prefill, sequential generation and image requests continue to
use the original final-state collectives. The bridge exposes `skip_logits` and its
vocabulary width. Non-coordinator ranks advance their caches and finish the stage
send before participating in the token collective, without projecting vocabulary
logits. The output scope restores its prior state on success or exception.

The worker passes approved runtime options to capability selection. Speculative
and sparse-prefill deployments cannot accidentally activate ordinary decode's
shorter collective sequence. Native prefill capability participates in the existing cross-rank vote.

Synthetic two/three-rank tests instrument the projection and final gather: only
rank zero projects during coordinated decode and no hidden-state gather occurs.
Greedy output matches a complete model; stochastic tokens agree across ranks even
with different RNG seeds. Both compressed and ordinary caches are covered. The
HTTP suite covers text, image, streaming and speculative paths. Weight residency is
unchanged: this reduces executed projection work, not replicated model weights.
No physical multi-Mac speedup or RDMA result is claimed.


### Bounded asynchronous prefill transport (2 October 2026)

The Qwen4 output scope now also surrounds each ordinary batched prefill chunk.
The existing generic chunk scheduler and queued-send implementation are reused;
there is no architecture-specific scheduler. Prefill does not calculate vocabulary
logits and does not gather final hidden states. Cache preparation/finalization,
right-padded rows and the admitted chunk size follow the existing prompt contract.

After each chunk, transport is submitted with `mx.async_eval`. At most the current
and previous send buffers are retained: after submitting a second send, the older
one is awaited. The last outstanding send is awaited before leaving prefill.
This allows transport to remain in flight while the next chunk is computed without
retaining activations for an entire long prompt. Pending queues and the model's
output scope are cleared/restored on exit.

Ring tests instrument send submission, assert that transport is deferred outside
the model output scope, and count asynchronous sends on every non-final stage.
Two and three ranks, ragged prompts, plain/compressed caches, deterministic output
parity and stochastic token agreement are exercised. Unit tests verify the bound
and final drain over two and four chunks. HTTP tests cover cache reuse and restart.
Physical overlap duration, TTFT gains and RDMA throughput remain unmeasured.

Final regression: 92 tests passed in `work/codex-native-prefill-final.log`
(shared performance contracts, Qwen4 HTTP serving and local ring execution).


### Independent prefill capability for speculative text (2 October 2026)

The runtime now selects scoped prefill separately from coordinator-only decoding.
An explicit prefill-output capability is voted across ranks before any prompt
handler is replaced. Native MTP, external Qwen4 MTP and DFlash can therefore use
the bounded asynchronous text prefill scheduler while leaving
`GenerationBatch._step` and the existing speculative verifier untouched. This also
works when `sampling_rank_only` is disabled. A peer veto disables scoped prefill.

Only cache-producing prefill forwards enter the output scope; verification cycles
still exchange the hidden states and decisions required by their existing
contract. SpecPrefill remains excluded. The initial implementation retained the
original image prompt handler; the image-prefix follow-up below removes that
fallback for singleton speculative image requests.

Instrumented three-rank tests exercise actual MTP/DFlash verification cycles,
ragged batches, short prefill chunks, plain/TurboQuant caches and asynchronous send
submission. They compare generated tokens against the complete target and verify
that the speculative generation step is not replaced. HTTP tests additionally
cover external MTP, image fallback, streaming and cache reuse.

Validation: 48 capability/HTTP tests and 8 ring tests passed (56 distinct);
see `work/codex-speculative-prefill-regression.log` and
`work/codex-speculative-prefill-final.log` (four unit tests overlap).


### Image prefixes after asynchronous prefill (2 October 2026)

Singleton image requests using the speculative BatchGenerator now share the
bounded asynchronous prefill scheduler. The model suppresses its inline prefix
snapshot inside the rank-local output scope. A completion hook saves the prefix
after all scheduled chunks and outgoing sends have completed, before decoding.
This prevents snapshot coordination from blocking a downstream activation receive.
Unscoped sequential generation retains its existing snapshot timing.

Instrumented tests use fragmented image prompts on two and three local ranks,
with plain and TurboQuant caches. They require multiple scoped prefill chunks,
one prefix snapshot with a collective across every rank, actual MTP verification
cycles, and exact greedy token parity against the whole synthetic model.
Ordinary sequential images and SpecPrefill still use their existing prefill.
Physical multi-Mac/RDMA behavior and throughput remain unmeasured.

Validation: 100 tests passed (52 ring, 29 HTTP, 19 capability tests);
four strengthened image cases passed again and overlap that count. Logs:
`work/codex-image-prefill-regression.log` and
`work/codex-image-prefill-scoped.log`. Ruff and diff whitespace checks passed.


### Ordinary images share the singleton scheduler (2 October 2026)

Image serving now routes ordinary requests through the same BatchGenerator stream
adapter as speculative requests. This enables bounded asynchronous prefill and
coordinator-only ordinary decoding without another image-specific prompt loop.
Request admission remains singleton; this does not enable concurrent image batching.
Prefix snapshots use the post-transport hook, and the stream adapter returns the
committed final cache for the existing request cache.

Validation: 29 HTTP tests and 8 instrumented image ring tests passed. The latter
cover ordinary and speculative images, two/three ranks, plain/TurboQuant caches,
fragmented scoped prefill, a coordinated prefix snapshot, and whole-model token
parity. HTTP coverage includes streaming and cache reuse/restart. Logs:
`work/codex-ordinary-image-prefill.log` and `work/codex-ordinary-image-ring.log`.
SpecPrefill still keeps its existing prompt path. Physical multi-Mac/RDMA and
throughput measurements remain deferred.

### Cache-only sparse prefill forwards (2 October 2026)

SpecPrefill now uses the declared native output scope and skip-logits capability
for intermediate cache-producing chunks. Explicit selected positions are preserved.
The final selected token retains the shared output contract and supplies logits;
decode offsets and subsequent generation are unchanged. Models without these
capabilities retain their original forwards. This removes intermediate vocabulary
projections and hidden-state gathers, but transport remains synchronous.

Validation: 67 distinct tests passed (60 SpecPrefill unit tests, four instrumented
ring cases, three HTTP cases). Ring spies reject projections and gathers inside
the output scope and compare final logits plus three decode steps with the whole
model, including sparse selection and existing prefixes. HTTP checks cover
streaming and failure recovery. Logs: `work/codex-sparse-cache-only.log` and
`work/codex-sparse-cache-only-http.log` (four ring cases overlap).
The worker passes Ruff; specprefill.py retains 36 lint findings on lines unchanged
by this increment. Physical multi-Mac/RDMA performance remains unverified.


### Bounded transport for sparse prefill (2 October 2026)

The runtime exposes its bounded send queue as a scoped transport hook for prompt
loops that own their positions. SpecPrefill uses that hook only for cache-producing
chunks with the native rank-local output contract. Each stage evaluates its output,
submits transport asynchronously, and retains at most two sends. All outstanding
sends drain before the final selected token computes shared logits. The hook is
restored on runtime exit; per-request queue state is cleared on success or failure.

The prefill capability vote now also permits SpecPrefill deployments. Coordinator-
only ordinary decoding remains disabled for those deployments. Draft selection,
selected positions, progress callbacks and decode offsets retain their existing
contracts. Models without the hook retain synchronous transport. No physical
multi-Mac/RDMA overlap or throughput improvement has been measured.

Validation: 164 tests passed (60 SpecPrefill, 19 capability, 56 ring, 29 HTTP).
Two ring cases additionally passed with async-send instrumentation and overlap
that count. Logs: `work/codex-sparse-async.log` and
`work/codex-sparse-async-instrumented.log`. Runtime/worker Ruff and diff whitespace
checks passed.


### Coordinator-owned speculative sampling draws (2 October 2026)

The coordinated sampler now draws ordinary tokens only on rank zero. Peers evaluate
their lazy log-probability graphs before joining the token collective, preserving
pipeline progress. For samplers exposing the deterministic sampling-logits contract,
peers compute the same float32 filtered acceptance density without drawing a token.
Custom sample-with-density implementations without that contract keep their existing
path. Rank zero retains its sampler and random draw sequence.

This reduces redundant sampler draws; it does not eliminate peer vocabulary
projections, verification calculations, or shared hidden states. No throughput
improvement or physical multi-Mac/RDMA behavior is claimed.

Validation: 97 tests passed (12 sampler/density unit tests, 56 ring, 29 HTTP).
Coverage includes MTP/DFlash, TurboQuant, images, streaming and cache reuse.
Log: `work/codex-mtp-root-draw-final.log`. Changed Python files pass Ruff;
diff whitespace checks pass.


### Distributed DFlash window override (2 October 2026)

Distributed DFlash now forwards an optional `dflash_draft_window_size` to the
existing drafter loader. The loader sets the draft configuration before binding
and cache construction, so the existing sliding context, ring capacity and pending
capture limits use the requested window. Omitting it preserves the checkpoint
default. Explicit values must be integers of at least two; booleans, zero, one,
negative values, strings and fractional values are rejected.

This reuses the existing batched drafter window implementation. Sink retention,
context cutoff and alternate distributed verifier modes remain unsupported.

Validation: 75 tests passed (29 distributed options, 15 drafter/cache, 31 HTTP).
Loader tests cover default/2/4/32-token windows; HTTP tests cover a four-token
window on two and three local ranks with reference token parity. Log:
`work/codex-dflash-window-final.log`. Ruff and diff whitespace checks passed.
Physical multi-Mac/RDMA validation remains deferred.

### Singleton fallback cache restoration (2 October 2026)

When an active singleton becomes ineligible for speculation, the common generation
patch now reconciles its committed history before dropping its MTP state. Previously
that branch discarded the state and resumed with stale ordinary-generation tokens.
The restoration applies only when the state still belongs to the current singleton;
stale ownership continues through the existing drop path.

A forced DFlash eligibility transition on three local ranks checks actual verification
cycles followed by ordinary decoding, including singleton and batched requests.
The public DFlash context cutoff remains rejected: trial coverage exposed a separate
image cache-reuse divergence. Enabling that option is deferred until image state
restoration and reuse are proven; the drafter eviction policy is also unresolved.

Validation: 118 regression tests passed, plus three targeted checks with two
overlapping cases (119 distinct). Logs: `work/codex-dflash-fallback-final.log`
and `work/codex-dflash-fallback-targeted.log`. The cutoff remains explicitly
rejected; failed experimental image-cache coverage is retained in
`work/codex-dflash-cutoff.log`.

### Cached image prefixes remain in generation history (2 October 2026)

The singleton image stream now supplies the cached prefix through BatchGenerator's
existing `all_tokens` argument. Previously only the uncached suffix entered the
generation history. A repeated image with one remaining prompt token could therefore
appear to be a one-token context, reactivate speculation under a context threshold,
then rebuild an incomplete history when it became ineligible.

The saved prefix itself was correct: direct prefix restoration matched the
reference. The HTTP tests exercise the real coordinated LRU path. Four HTTP cases reproduced the divergence when a
test-only one-token eligibility threshold was applied, with default/four-token
DFlash windows on two/three ranks. The production cutoff remains rejected pending
validation of image state reconstruction during an active-to-ordinary transition.
This change does not enable that setting.

Validation: all 35 HTTP cases pass, including the four formerly failing cutoff
simulations; all ten strengthened image-reuse ring cases pass after correcting
the test harness. The other 50 ring cases passed in the broader regression.
Logs: `work/codex-image-prefix-history-final.log` (includes six initial harness
failures) and `work/codex-image-repeat-prefix-final.log` (ten passes). Changed files
pass Ruff and diff whitespace checks.


### Image-aware replay and distributed DFlash cutoff (2 October 2026)

The common singleton reconciliation now asks the model for replay segments when
that hook exists. Qwen4 resets image positions and the consumed-token cursor, then
splits replay at the image-prompt boundary as well as the normal chunk boundaries.
Generated tokens therefore never receive prompt image embeddings. Prefix-save
callbacks are suppressed during reconstruction. Successful replay leaves the
committed cursor in place; a failed replay restores the original image state.

Distributed `dflash_max_ctx` is now supported. Omitted or zero disables the cutoff;
a positive integer stops speculation once a row's committed history exceeds it.
Checks occur between generation steps, not inside an in-flight verification block.
The whole active batch returns to ordinary decoding through cache reconciliation.
The cutoff metadata is shared from rank zero. This supersedes earlier notes that
the public setting was rejected.

The distributed drafter remains resident, with its memory reservation unchanged.
This cutoff is a speculation policy, not a memory eviction mechanism. Sink retention
and alternate verifier modes remain unsupported; physical multi-Mac/RDMA behavior
and throughput remain unmeasured.

Validation: 145 regression tests and two replay-state tests passed (147 distinct).
Four active image-transition HTTP cases failed before the fix and pass afterward.
Logs: `work/codex-dflash-cutoff-enabled.log` and
`work/codex-image-replay-state.log`. Public cutoff tests cover values 1/14;
ring tests exercise real shared cutoff transitions in singleton/batched generation.

### Coordinator-owned stochastic verification (2 October 2026)

The common stochastic verifier now runs acceptance draws, residual distributions
and correction/bonus sampling only on rank zero when its sampler is coordinated.
Peers evaluate their lazy target outputs and draft densities, then receive the
compact verification result. The ordinary final decision collective remains,
because cache rollback limits and stop/budget clipping still apply afterward.

Dense and top-k verification retain rank zero's existing algorithm and random
draw sequence. Greedy verification is unchanged. Target vocabulary projections
remain replicated. This trades peer verification work for an additional compact
collective; no net throughput improvement is claimed without hardware measurements.

Sink retention remains open: the current prefill capture paths do not always
preserve the beginning of the prompt, so accepting a sink setting alone would
silently produce the wrong draft context.

Validation: 125 regression tests passed, followed by 21 sampler tests with
18 overlapping cases (128 distinct). Tests forbid peer verification draws and
compare rank-zero results with seeded local dense/top-k/legacy sampling. Logs:
`work/codex-root-stochastic-verification.log` and
`work/codex-root-stochastic-unit-final.log`.


### Coordinator-owned greedy verification (2026-10-02)

Greedy speculative verification now uses one shared helper for singleton and
fused batches. Rank zero computes targets and accepted draft counts; peers
complete the lazy forward before receiving the compact result. Local verification
and pre-drafting preserve their existing token semantics. Final rollback/stop
clipping and its decision collective remain unchanged.

Speculative vocabulary projections remain replicated. This adds a compact
collective; throughput benefits require hardware measurement.

Validation: 134 sampler/ring/HTTP tests and 372 local MTP/DFlash tests pass
(506 distinct). Unit tests cover fully accepted, partially accepted and rejected
batches, forbid peer target selection, and check evaluation before the collective.
Logs: `work/codex-root-greedy.log`, `work/codex-root-greedy-local.log`.


### Peer MTP draft projections (2026-10-02)

Qwen4 MTP draft chains now omit vocabulary projection on non-coordinator ranks
for greedy generation without logits processors, in singleton and fused batches.
The head still runs and evaluates its recurrent output/cache before providing
shape-compatible placeholder logits. Rank zero supplies the sampled draft tokens
and retains the real draft probabilities. Other adapters must explicitly advertise
support; their behavior is unchanged.

This does not skip target verification projections, initial MTP priming projections,
stochastic draft projections, processor-bearing draft projections or DFlash heads.
No throughput gain is claimed: evaluating the recurrent head introduces a sync.

Validation: 227 unit/ring/HTTP tests plus 372 local MTP/DFlash tests pass (599
 distinct). Eight existing ring cases rerun with explicit spies also pass, proving
peer projection skips actually occur in greedy mode and never in stochastic mode.
Logs: `work/codex-peer-draft-projection.log`, `work/codex-peer-draft-local.log`,
`work/codex-peer-draft-exercised.log`.


### Stochastic MTP peer projections (2026-10-02)

The peer draft projection skip now also covers stochastic Qwen4 MTP chains,
including fused batches without logits processors. The coordinator already owns
both draft sampling and stochastic verification, including the genuine normalized
q distribution. Peer placeholder probabilities are not used for acceptance or
residual sampling. Target verification, initial priming, DFlash projections and
processor-bearing draft paths remain unchanged. No throughput gain is claimed.

Validation: 158 distinct sampler/ring/HTTP cases pass across the regression and
focused rerun. The first regression had 157 passes and one failure in a new test
comparing two timing-adaptive runs at the same RNG seed. Adaptive depth changes
the draw sequence, so exact seeded identity is now checked at fixed depth only;
adaptive tests still require rank agreement and actual projection skips. All eight
focused singleton/batch cases pass after that test correction. Ruff on the touched
helper/test files and git diff --check pass.
Logs: `work/codex-peer-stochastic-projection.log` and
`work/codex-peer-stochastic-comparison-final.log`.


### Pending DFlash request captures (2026-10-02)

Captures queued before a request receives its generation UID now use the same
bounded pending context as an active row. Discarded positions remain counted and
transfer to the bound row, preserving absolute positions and subsequent proposals.
This uses the existing draft window; it introduces no new cutoff or context limit.
Sink support remains unimplemented. Retained logical rows are bounded; no physical
memory or throughput improvement has been measured.

Validation: 53 DFlash unit/lifecycle/prefill tests and 21 HTTP DFlash cases pass.
The new regression checks the bound after each chunk, compares positions and
proposals with direct row seeding, and verifies request release. Logs:
`work/codex-dflash-request-seeds.log`, `work/codex-dflash-request-seeds-http.log`.

The deferred real multi-Mac A/B projection performance protocol is recorded in
[TESTING.md](../TESTING.md#deferred-multi-mac-mtp-projection-performance-comparison).
It explicitly requires checking whether synchronization outweighs saved computation.


### DFlash prefill capture positions (2026-10-02)

Both scheduler prefill paths now pass the absolute start of retained captures to
DFlash. Request seeds preserve this origin while trimming the pending window and
binding a generation UID. Explicitly positioned chunks must be contiguous; invalid
positions are rejected before mutating the pending context. Callers that omit the
position retain the existing relative-position behavior. SharedDFlash forwards the
optional position to its resident drafter.

This is groundwork for sink retention, not sink support. Capturing and retaining
the initial prompt positions, including cache-hit handling, remains unfinished.
Validation: 178 focused tests plus 21 HTTP cases pass (199 distinct). Seventeen
DFlash unit cases also pass again with the final scheduler-offset assertion.
Logs: `work/codex-dflash-capture-positions.log`,
`work/codex-dflash-capture-positions-http.log`.


### Internal DFlash sink attention (2026-10-02)

The drafter's internal constructor accepts a nonnegative `sink_size` (default 0).
It retains the first committed hidden captures separately from the sliding ring,
including before request UID binding. Attention projects these prefix captures at
their original positions and masks duplicate ring entries and padded prefix rows.
Missing initial captures are rejected instead of interpreting a cached suffix as
the prompt beginning. Prefix rows are released with their request or generation row.

This is not a public serving feature yet: scheduler capture selection and cache-hit
recovery still need to guarantee the prompt prefix before the distributed option
can be accepted. Existing public nonzero sink settings remain rejected. Predrafting
is disabled for the internal sink path because its captures include uncommitted
verification positions. Prefix KV is recomputed for each draft; performance and
physical memory usage have not been measured.

Validation: 82 tests pass, covering retained prefix values, short/mixed row masks,
no duplicate attention entries, missing-prefix rejection, release, and identical
batched/singleton proposals across ring rotations and changing cohorts. HTTP tests
exercise the unchanged default path only. Log: `work/codex-dflash-sink-core-final.log`.
Ruff on the changed drafter/test files and git diff --check pass.


### Sink-aware scheduler prefill (2026-10-02)

When an internally configured drafter has sinks, scheduler prefill now requests
captures from every prompt chunk, including the beginning. The drafter keeps only
the sink prefix and its bounded pending tail. Capturing intermediate chunks keeps
absolute positions contiguous; it adds capture work but does not enlarge the
retained logical context. With zero sinks the existing tail-only capture remains.
The internal loader accepts and validates `draft_sink_size` before loading weights.

Validation: 193 tests pass. New cases compare chunk sizes 1, 5 and 29 against
full-prompt seeding, including identical subsequent proposals, bounded pending
rows, preserved prefix and UID transfer. Loader cases cover sink/window combinations
and reject invalid sinks before checkpoint loading. Log:
`work/codex-dflash-sink-prefill.log`. Ruff on the drafter/test files and
git diff --check pass.

Cache-hit reconstruction and distributed capture integration are still unfinished.
Public nonzero sink settings remain rejected; the internal loader argument is not
wired into production settings. No hardware performance claim is made.


### Safe sink prefill when snapshots lack captures (2026-10-02)

The scheduler's internal sink mode now bypasses KV-only prefix lookup and
re-prefills the complete prompt. It clears stale request seeds and resets cached
 token accounting before marking admission prepared. Repeated preparation does
not discard captures already produced by that admission. This reconstructs the
required prefix rather than treating a cached suffix as the prompt beginning.

This is a correctness fallback, not capture restoration from a sidecar: every
internal sink request currently forgoes prefix-cache reuse, with potentially
substantial first-token latency cost. Default sink-free requests keep their cache
path. Public sinks remain disabled; the distributed serving path still needs its
own capture integration. A reusable capture sidecar remains future work.

Validation: 194 DFlash/prefill tests plus 456 scheduler/prefix/GDN/position tests
pass (650 distinct). The new test checks reset, idempotent admission, preserved
captures and proposal identity against a complete prefill. Logs:
`work/codex-dflash-sink-cache-replay.log`,
`work/codex-dflash-sink-cache-regression.log`.


### Capture sidecars and explicit switches (2026-10-02)

DFlash sinks and capture reuse are now available in the batched scheduler and the
Qwen4 distributed serving path. Sidecars store the retained initial captures and
recent window at an exact KV boundary. On a hit, only the uncached suffix is
processed. If captures are absent, evicted, corrupt or incompatible, the prompt is
recomputed; there is no unconditional prefix-cache bypass anymore.

The admin model-settings panel and API expose these controls:

| Setting | Default | Behavior |
| --- | --- | --- |
| `dflash_draft_sink_size` | `0` | Retain this many initial tokens in addition to the draft window. |
| `dflash_capture_cache` | `false` | Store and restore DFlash captures alongside matching target cache boundaries. |
| `dflash_in_memory_cache` and its entry/byte limits | existing defaults | Control sidecar RAM retention when capture reuse is enabled. |
| `dflash_ssd_cache` and its byte limit | `false` | Persist sidecars; requires the configured SSD cache directory and, on workers, distributed prompt-cache SSD. |
| `mtp_peer_projection_skip` | `false` | Opt into skipping peer draft vocabulary projections on supported Qwen4 MTP paths without logits processors. |
| `mtp_peer_verify_projection_skip` | `false` | Independent opt-in: non-coordinator ranks skip the target-verification vocabulary projection (MTP and DFlash, singleton and fused) without logits processors. |
| `dflash_async_prefill` | `false` | Experimental: capture-bearing distributed prefill overlaps like ordinary staggered prefill; layer captures ride the stage boundary sends toward rank zero instead of per-chunk collectives. |
| `dflash_predraft` | `false` | Experimental: rank zero drafts the next DFlash block before the cycle decision is final (single request, no sinks, fixed depth); adopted only when the final accepted count is unchanged. |
| `dflash_evict_on_fallback` | `false` | Experimental, needs `dflash_max_ctx`: rank zero drops the draft weights while the cutoff keeps the whole batch in ordinary decoding and reloads them when speculation is eligible again. |
| `dflash_verify_mode=ddtree` | off | Distributed: branched verification of drafter top-k candidates; greedy by exact match, sampled requests by a target-distribution tree walk (no draft q). Needs `dflash_ddtree_memory_bytes` (incremental per-stage budget, reserved by the planner on every rank); `dflash_ddtree_max_branches` (2-16, default 4) and `dflash_ddtree_max_nodes` (2-64, default 8) bound the tree. Local batched Qwen4 (`qwen4_exp`) engines accept it too (same options and semantics; other local targets refuse it). |

The two new booleans are opt-in. Sinks remain disabled at size zero. Existing
window and optional context-cutoff controls are preserved. Settings changes flow
through model reload signatures and distributed runtime options. No existing user
model configuration was enabled automatically.

Sidecars use schema-versioned safetensors with atomic replacement. Keys include
checkpoint file identities, runtime settings, exact token prefix/boundary and
multimodal identity. RAM has entry/byte bounds; SSD eviction uses access recency
and the configured byte budget across identities in its dedicated capture root.
Incompatible identities never restore one another. On distributed deployments only
rank zero owns the store, and every restore decision is shared before processing.
Admission tracks individual UIDs, including changing concurrent cohorts. Shutdown
releases RAM while preserving SSD snapshots for the next compatible deployment.

Capture-bearing distributed prefill uses the synchronous transport path: gathering
hidden states before queued sends flush caused a collective ordering failure.
Capture-only forwards do not open speculative rollback transactions. This may cost
cold-prefill throughput; warm reuse avoids repeated capture work. Predrafting with
sinks remains disabled. Prefix KV projections are reused while the active batch and retained sink captures remain unchanged. Growing prefixes and batch membership/order changes rebuild this bounded cache. The cache is released with the batch; capture snapshots on SSD still contain hidden states, not projected KV.
None of these changes establish a hardware throughput gain.

Validation includes RAM and cold SSD restoration, worker restart with proven disk
hits, opt-out fallback, distinct images, streaming, concurrent requests, partial
boundaries, incompatible identities, corruption, eviction and settings round trips.
Logs: `work/codex-capture-unit-final.log` (1031 passing cases before two additional
validation guards), `work/codex-capture-final.log` (339 passing regression cases),
`work/codex-capture-http-final.log` (7 instrumented HTTP cases, overlapping the
regression). Follow-up guard checks are in `work/codex-capture-guards.log`.
Real inter-Mac/JACCL/RDMA throughput and quality on trained drafts remain unmeasured.

Final validation across the suites and focused follow-ups covers 1,218 distinct
cases. `work/codex-capture-guards.log` has 146 passing cases (two new guards), and
`work/codex-capture-restore-final.log` has 411 passing cases (two new zero-sink
restoration cases). Targeted Ruff, JavaScript syntax, locale JSON parsing and
`git diff --check` pass. Historical lint findings in large existing modules are
outside this change.

### DFlash capture cache maintenance

Rank-local hot, SSD and combined cache-clear commands clear the corresponding DFlash capture storage. SSD purges cover all configuration namespaces in the dedicated rank-local capture directory, including stale files when no drafter is resident or the current drafter uses RAM only. Active requests reject the operation before either cache is touched. A disk-only purge preserves RAM captures. Symlinked namespace directories are skipped. Rank responses keep target prompt counters separate from `capture_hot_cleared` and `capture_ssd_deleted`. The distributed engine sums each field across ranks; admin totals include both target entries and capture entries. Missing capture fields from older workers count as zero. These are entry/file counts, not measured physical memory reclamation.

### Optional sink KV retention

The admin checkbox and API/profile field `dflash_sink_kv_cache` control retention of projected sink KV in local batched and distributed DFlash. It defaults to true to preserve existing sink behavior; zero sinks make it inert. False recomputes prefix projections each draft and retains no projected sink KV between calls. Capture reuse and sink selection remain independent. Changing this setting reloads the engine.

### Verification mode contract

The shared block verifier supports `dflash_verify_mode=None`, `"dflash"` or `"adaptive"`. `ddtree` is accepted by distributed loading and by local batched Qwen4 loading only; every other loader, `off` and unknown modes are rejected explicitly. Direct drafter loading validates the mode before checkpoint loading.

The opt-in adaptive mode uses the existing native singleton depth controller and whole-batch cost controller. Distributed controllers make decisions on the coordinator and synchronize them across ranks. For positive depth, DFlash generates its trained full block; only the chosen prefix and its corresponding draft probabilities enter target verification. At depth zero, draft computation and sampling are skipped. Confirmed captures remain bounded for exact re-entry, including sinks and mixed batches. The controllers may calibrate ordinary decoding and temporarily park speculation when measured cost warrants it. This is independent of the optional context cutoff. Predrafting is disabled in adaptive mode.

This is the batched engine's adaptive policy, not a reproduction of the standalone DFlash engine's full/reduced/probe policy. The default remains fixed-block verification. At positive depth, draft computation is not shortened; reduced target verification and controller overhead need physical throughput measurement. The standalone engine is unchanged, and tree verification remains unimplemented here.

### Peer target-verification projections and DFlash prefix depth (2026-10-02)

DFlash prefix depth: the drafter mask keeps every block key visible to every block
query (bidirectional within the block), so a verified prefix computed from a shorter
block is not the prefix of the trained block. Positive adaptive depths therefore
still compute the full block and truncate afterwards; no equivalence is claimed for a
reduced computation. `test_block_attention_is_bidirectional_so_prefix_depth_is_not_a_shorter_block`
checks the all-visible block mask and that first-position logits change with block length.

Peer target projections: with `mtp_peer_verify_projection_skip` enabled (separate from the draft option `mtp_peer_projection_skip`, whose behavior is unchanged), non-coordinator ranks
run the speculative verify forward (singleton and fused batches, MTP and DFlash) with
`skip_logits` when the sampler is coordinated and no logits processor is active. The
forward's hidden state and layer captures are evaluated before the collective, then
zero logits of the full shape stand in; rank zero keeps the real logits and still
decides tokens and accepted counts through the existing greedy/stochastic collectives
and final decision. Peer log-probabilities are placeholders and never drive control
flow. Default (`false`) is unchanged. DFlash drafter head projections on peers and
processor-bearing paths remain replicated. No throughput gain is claimed: the extra
evaluation is a sync, to be measured with the deferred A/B protocol.

Contract checks for the verify skip: rank zero never skips (its logprobs stay real);
only HTTP requests reach rank zero, and peers drain responses into a throwaway queue, so
placeholder zeros cannot appear in API output. Sites that sample real tokens from target
logits (post-init, ordinary/standard fallback, boundary materialization, legacy verify)
never pass `skip_logits`. DFlash drafting is rank-zero only (`SharedDFlash` keeps
metadata on peers), so there is no peer DFlash draft projection to remove.

### Remaining DFlash/pipeline limits: concrete contracts (2026-10-02)

These were investigated against the code; none is implemented, and none is claimed.

* **Optional drafter eviction on context fallback.** Not applicable as a safe batched or
  distributed feature. The cutoff (`dflash_max_ctx`) is evaluated per generation step over
  the whole active batch (`_mtp_common_eligible`), so a short request joining later would need
  a reload on the decode path; eviction would also drop per-row rings, sinks and pending
  captures that exact re-entry relies on. Distributed deployments admit the drafter's
  reservation at planning time (`DraftReservation.admit`), so releasing weights would not
  return planned capacity. The standalone DFlash engine already evicts and delegates to the
  batched engine. A future design needs an engine-idle trigger and a reload outside the step.
* **Async prefill with DFlash captures.** A capture-bearing chunk ends with collectives
  inside the forward (`gather_mtp_output`, `gather_layer_captures`), while the staggered
  prefill queues stage sends until the forward returns; the collective would wait for a send
  that is not yet issued. Per-chunk collectives also force lockstep, which removes the
  stagger overlap. Only rank zero consumes captures (`SharedDFlash.seed_request`). A correct
  design carries each stage's captured layers on the existing boundary send toward rank zero
  (a wire-contract change that every rank must fingerprint) or defers captures to bounded
  group exchanges issued at the same logical point on every rank. Memory bounds and the
  contract fingerprint need validation on the ring before code. The synchronous path stays.
* **Distributed predraft.** `SharedDFlash.predraft` returns False by design. A predraft
  would overlap rank zero's draft with the decision collective only; proposals still need
  the pickled broadcast (two collectives) at adoption, and rank zero's eligibility
  (`ring_slots` overflow, pending rows) is rank-local, so a shared flag would add a
  collective. The gain is the round trip of one decision collective at best; it needs
  physical measurement before the extra state is worth carrying.
* **DDTree.** Requires per-branch target caches with rollback (KV and recurrent state) and a
  tree verifier with a branch-aware attention mask. The refusal stays until those exist.
* **Pipeline+TP, expert sharding, new architectures.** Each needs its own wire contract
  (what crosses stages, which collectives are uniform across ranks), a planner extension and
  a model adapter that advertises explicit capabilities (`skip_logits`, captures, rollback).
  No code before that contract is written for the specific architecture.

### Async capture-bearing prefill (`dflash_async_prefill`, 2026-10-02)

Default off; the synchronous capture prefill is unchanged. With the option on and the
staggered prefill active, each chunk's forward holds no collective. A stage appends the
layers it captured to its boundary message, after the `[residual | branch | gate]` layout
and in ascending layer order; the next stage is told how many upstream captures to expect
(layer ids below its own start), forwards them with its own, and rank zero, the only
consumer (`SharedDFlash.seed_request`), returns all of them. Peers keep no captures. The
existing bounded in-flight send queue is reused, so ordering stays deterministic and one
message per boundary means no extra transfer to order. `unpack_boundary` now bounds the
gate slice explicitly. Rank zero evaluates each chunk's captures in the capture callback so
nothing hangs on a receive past its chunk. Positions, sinks, padding of ragged rows,
capture-store restore and image replay use the unchanged callback. The option is checked
across ranks (`capture_prefill` vote); any rank without support turns it off for all.
Capability `dflash_async_prefill` reports the effective state.

Also fixed (pre-existing): a ragged batch split or a full split at prompt completion lost
`_omlx_dflash_prepared`, so DFlash capture prefill replayed the whole prompt a second time
when no capture sidecar could be restored. `PromptProcessingBatch._copy`, `extend` and the
scheduler's full-split path now carry it. On three ranks the HTTP capture-reuse tests only
saw a capture hit through that replay; they now accept the logged full-prefill rebuild.

Validation (local rings, no hardware claim): 2 and 3 ranks, ragged padded batch, 5 chunks of
2 tokens, sinks 0 and 3: zero collectives inside the capture window, boundary messages
carry the expected capture count, rank zero's seeded captures equal the whole-model
reference per row and position, greedy tokens match, stochastic coordination agrees.
HTTP (2 and 3 ranks, RAM and SSD captures, text, image, streaming, repeated prompts) passes.
Throughput, memory and RDMA behavior remain unmeasured.

### Distributed DFlash predraft (`dflash_predraft`, 2026-10-02)

Default off. Flow checked first: in distributed verification the acceptance counts are
already the result of the coordinator collective when predraft would start, so the only
overlap is rank zero's draft with the remaining host work (clamps, decision collective,
cache commit). That window is small and no gain is claimed; the option exists so it can
be measured. Added synchronization: none. Every rank answers `predraft` from shared metadata
only (option on, no sinks, fixed depth), and every rank performs exactly one proposal
share per committed cycle, inside `adopt_predraft` or `draft`. Rank zero drafts ahead on
its own drafter; if it could not (ring overflow) or the final count differs, it drafts from
the job given at adoption, so eligibility is never agreed by a collective. A mismatch
discards the queued block. Draft sampling stays local to rank zero (coordinator wrapper
removed before drafting). `adopt_predraft` now receives the draft job (local drafter
ignores it).

Validation (local rings, no hardware claim): 2 and 3 ranks, greedy and stochastic
generation match the whole model; all ranks take identical predraft/adopt/discard
counts; rank zero's real adoption count is positive.

Review notes on the async prefill lot: `_omlx_dflash_prepared` carry is covered by
`test_prepared_rows_survive_batch_split_extend_and_scheduler_full_split`. Removing the carry
shows the old 3-rank `DFLASH_CAPTURE_HIT` came from a redundant second restore after the
split, not from a prompt-cache hit; HTTP tests now require a real hit on two ranks and
excuse only a logged cross-rank plan divergence on three ranks. Whether that divergence
is a pre-existing defect is not investigated here.

### Prompt-cache plan divergence on three ranks (2026-10-02)

Cause: speculative decoding (MTP/DFlash) finishes one token short of the emitted tokens
(the final token is `next_main`, not yet in the backbone cache), yet MLX-LM stores the
finished request under all emitted tokens. A trimmable attention-only stage (rank 1 of the
3-rank test plan) then returned a hit one position behind the claimed prefix (offset 5 for a
reused length of 6); the stages with recurrent state could not trim and served their
boundary snapshots instead. The existing plan vote flagged the incoherent offset and every
rank rebuilt with a full prefill, so output stayed correct but cache reuse was lost.

Fix: `covered_cache_key` keys the stored entry by the length its offset-bearing caches
actually hold (recurrent-only, pooled and disagreeing caches are left untouched, and the
plan vote still guards them). The conservative vote, stage-local caches, snapshots and
rollback are unchanged. The HTTP tests now require a real DFlash capture hit on every
topology and no divergence warning on any rank (they failed on three ranks before the fix).
Remaining risk: a recurrent-only stage still stores the longer key; its hit is protected
only by the vote of the attention-bearing ranks.

### Optional drafter weight eviction (`dflash_evict_on_fallback`, 2026-10-02)

Default off. The cutoff is batch-wide, so a batch above it has no row using the draft;
`_mtp_common_eligible` then calls `fallback()` (never inside a forward or draft; a queued
predraft is dropped). Rank zero releases only the weights: rings, pending captures, sinks
and the capture store are activations and stay, so re-entry resumes from the confirmed
context. Identity is checked on reload (layer ids, block size, source path, window, sinks).
Both calls sit at the same step boundary on every rank, so eviction needs no collective;
reload shares rank zero's outcome. A failed reload is seen by all ranks, which then stay in
ordinary decoding for the deployment's lifetime (no retry loop). A mixed batch never evicts,
because any row above the cutoff already sends the whole batch to fallback and a batch
without such a row keeps the draft. The plan's memory reservation is unchanged and no
memory is redistributed; the release is of references, with no physical figure claimed.

Validation (local rings): 2 and 3 ranks, unbatched and ragged batches, cutoff crossed then
later short requests: weights freed once per eviction (weakref), one reload per eviction,
tokens match the whole model, never-crossed cutoff evicts nothing, injected reload failure
is shared and keeps ordinary decoding. Unit: identical tokens and sampled q after re-entry
with captures observed while evicted, identity mismatch refused, option round trips.

### DDTree foundation: branched target caches (2026-10-02)

DDTree is NOT implemented; `dflash_verify_mode=ddtree` is still refused. Only the cache
primitive is in (`omlx/speculative/branch_cache.py`). Contract chosen after reading the
installed `dflash_mlx` tree helpers and the real rollback calls: a flat tree attention mask
cannot cover recurrent (GDN) layers, and the pipeline stage refuses `speculative_verify`,
so independent branches are verified as the rows of one batched forward, the same shape
as batched speculation. `fork_cache` merges the committed cache into one row per branch
(existing `merge`; the source is never written), the forward's speculative transaction
and `rollback_speculative_cache` give each row its own accepted prefix (every rank passes
the same list, as `agree_accepted` requires), and `commit_branch` extracts the chosen row
(existing `extract`) as an ordinary cache. GDN, QSA/KV, PLE and TurboQuant state go through
these existing paths. `branch_cache_bytes` is a conservative pre-activation estimate (each
row copies the source state plus per-row metadata, the source stays resident); `fork_cache`
raises `MemoryError` under an undersized budget. Verification activations of the wider
batch are not included.

Greedy only: stochastic acceptance over several children needs the multi-draft rule over
the true draft distribution q, not implemented, so no sampled support is claimed.

Validation (local rings, whole-model reference): 2 and 3 ranks, plus TurboQuant on 3
ranks. Diverging branches (shared first slot, different later slots) give logits equal to
sequential forwards; a ragged accept vector commits one row whose continuation equals the
sequential accepted path; committing one draft too many is detected; the source cache
keeps identical state and offsets and remains usable. The primitive also runs on the
unsharded whole model in each rank.

Next integration: a tree/branch proposer from the drafter (top-k per slot), a branch-aware
greedy acceptance replacing the linear block, rollback of the unchosen rows' PLE/pooled
state, branch counts bounded by the estimate, and an explicit option before ddtree is
accepted.

### DDTree proposer and branch verifier (2026-10-02)

Still not ddtree: `dflash_verify_mode=ddtree` stays refused and nothing in generation
calls this yet. `omlx/speculative/ddtree_branches.py` adds, on top of `branch_cache`:
`propose_branches` (the installed `dflash_mlx` flat-tree builder over per-slot top-k,
then an explicit bound of `max_nodes` tree nodes and `max_branches` root-to-leaf paths,
best cumulative score first, flat-tree order on ties), and `verify_branches` (paths as
independent cache rows through the existing `_call_backbone_captured`, greedy acceptance
with the generator's own argmax tie rule, longest confirmed prefix wins, lowest row on
ties, only that row's cache, hidden state and layer captures are kept; padding is never
counted as accepted). Sampling raises `NotImplementedError`; no q or stochastic support is
claimed. Admission: `branch_forward_bytes` = forked caches plus rows x tokens x a
caller-supplied per-token activation estimate (`token_bytes_estimate`: logits plus three
residual-stream copies per layer); a budget without that estimate is a `ValueError`, an
insufficient one a `MemoryError`, both before any forward. The estimate is not a measured
bound beyond the tiny test model.

Validation (rings, ordinary whole-model greedy generation as oracle): 2 and 3 ranks and
TurboQuant on 3 ranks. Seven speculative steps with a different selected row per step
(rows 1,0,2,2,1,0,2), accepted counts 2,0,1,2,2,0,1 (including total rejection), two
ordinary decoding steps between cycles with re-entry, 3 branches and 5 nodes per step;
emitted tokens equal the ordinary target's, captures of the confirmed positions equal an
ordinary forward, the committed cache gives the ordinary next token. Peak memory stayed
within the estimate on the test model. Unit: bounds, determinism, guards.

Integration still to do: a drafter top-k per slot (the DFlash drafter returns only the
argmax block), hooking `verify_branches` into the cycle through `verify_result` and
`commit_cache` so stop/limit/clamp logic stays shared, the multi-draft q rule, and an
admission that also covers drafter activations before the option is public.

### DDTree in distributed generation (`dflash_verify_mode=ddtree`, 2026-10-02)

First delivery: greedy only, distributed deployments only, one request per cycle.
Contract. The drafter keeps the top-k logits per slot (k = max branches) with its own
token first, so the linear block is always a branch; rank zero drafts and the proposals,
including candidates, are shared through the existing proposal share. Inside
`_run_verify_cycle_chain`, `plan_tree_cycle` builds at most `max_branches` paths over at
most `max_nodes` nodes, verifies them as cache rows of one forward, and hands the
chosen row to the shared acceptance code through `verify_result`/`commit_cache`, so
stop/limit/clamp/matcher logic, hidden/capture slices (confirmed positions only) and the
next draft are the linear ones. Rank zero picks the longest confirmed branch (lowest row on
ties) and broadcasts the row; the row count is the minimum across ranks (one extra
all-gather per cycle); unchosen rows roll back with zero accepted drafts on every rank, so
the pipeline's accepted-count agreement holds. Image requests work: the position tracker
reads the committed row (`_omlx_branch_row`). Multi-row groups keep the existing fused
linear verification (the tree runs for single-request cycles, including a single row of a
mixed batch); predraft is off with ddtree.

Sampling: a deployment with ddtree refuses `temperature > 0` before any request work
(`ValueError` naming the limit and `temperature=0`); the generator also raises if a
sampled or processor-bearing row reaches the tree. No q or stochastic support is claimed:
the multi-draft acceptance rule is the next step.

Admission (`omlx/speculative/branch_memory.py`), before the fork and per cycle.
`dflash_ddtree_memory_bytes` is an INCREMENTAL per-stage budget: bytes a stage may add on
top of what the plan already admits (weights, the planned KV cache, the draft reservation
on rank zero and one linear verification). The planner charges it as `runtime_reserve_bytes`
on every rank, so it must fit each stage's headroom at planning time (the draft reservation
stays rank zero's; nothing is counted twice). Per cycle, for `rows` branches of `width`
tokens, each stage sums, from live shapes: forked cache copies (live tokens plus per-row
offsets); growth of the next `width` tokens (BatchKVCache allocates whole 256-token steps
and concatenates, so old copy, new block and result coexist; the QSA indexer concatenates
raw keys and positions); the recurrent state recorded after each of the `width - 1` steps
per row; the extracted committed row; and payloads (logits plus two fp32 log-prob
temporaries, gathered hidden and residual streams over the world size, the capture stack and
its all_sum result, stage send/receive buffers). Forward-internal scratch (attention and
indexer workspace, MLP/MoE intermediates) has no closed form from cache shapes, so it is
measured: the logical MLX peak of the last linear cycle, per row-token, scaled by rows x
width and by context growth. Until one linear cycle is measured nothing forks. Rows are
cut until the estimate fits; one row (or no measurement) runs the linear block and forks
nothing. Families without a bound (rotating, quantized, TurboQuant, pooled, unknown) are
refused: at startup, agreed by every rank, with the reason, and again before any fork.
No fork happens unless the estimate fits.

What is and is not shown: every branched cycle's measured logical MLX peak stayed within
its admitted estimate on the tiny test model (ring and generation tests), and the formula
covers real merged caches, growth across the step boundary and recurrent steps. Not shown:
physical memory, the Metal allocator's cache, RDMA/JACCL registered buffers, real
checkpoint shapes, or drafter activations on rank zero (outside the budget).

Validation (local rings, whole-model greedy generation as oracle): 2 and 3 ranks, three
modes: crafted candidates (selected row varies, total and partial rejection), the
drafter's real top-k, and a budget too small for two rows (zero forks); all ranks make
the same branched decisions and tokens match. HTTP (2 and 3 ranks, RAM and SSD captures,
sinks, text, image, streaming, capture hits) matches the reference. Unit: option bounds,
required memory bound, local refusal, API/settings/profile round trips, sampling refusal.
No physical throughput or memory figure is claimed.

### Sampled ddtree: target-distribution tree walk (2026-10-02)

Contract, stated before the code. The tree is built from the drafter's deterministic
top-k, a function of the context only. Every visited node has the target's own sampling
distribution for its prefix (the request's temperature, top-k, top-p and min-p, applied by
its real sampler to the verified logits). The walk takes exactly ONE draw per visited
prefix from that distribution. If the drawn token is a child of the node, the walk moves to
the child; otherwise (or at a leaf, which has no children) the draw is the bonus token and the
walk ends. Each emitted token is thus a single draw from the target conditional given the
emitted prefix, and the output is sequential target sampling cut at a stopping time that
depends only on the draws themselves. The tree shape therefore changes how many tokens one
verification yields, never their law. Not claimed: multi-draft speculative sampling, rejection
sampling against a draft q, or any gain from a better acceptance rate; the drafter's
distribution is not used. No branch is chosen by comparing independent concurrent draws
(that would bias), and unvisited nodes are never drawn.

Mechanics. Rank zero draws with the unwrapped sampler and shares one bounded decision (row,
accepted count, bonus token) through the existing coordinator collective, so every rank
commits the same row with the same count (rows are verified on every rank, peers skip nothing
conditional). A node's logits come from the lowest row containing its prefix (stable ties);
the committed row is the lowest row containing the accepted prefix, and the bonus token is
not yet processed, so cache, hidden state and captures stay aligned with the seed token plus
the accepted prefix. The shared acceptance code is reused unchanged: stop tokens, max tokens
and boundary clamps cut the accepted prefix, and an emitted draft token is itself a draw, so
clamping is exact. Single-row (linear) cycles and multi-row fused groups of a ddtree
deployment use the same walk (`draft_accept_lps is None`), so no q is ever read.
Logits processors: superseded on 2026-10-03 (see "Processors in ddtree" below); penalties
are replayed per branch, only grammar stays refused.
Admission adds 13 vocabulary-sized fp32 buffers for the sampler chain's temporaries to the
incremental budget of a sampled cycle (`SAMPLER_VOCAB_BUFFERS`, counted from
`omlx/utils/sampling.py`); draws are sequential, so it is counted once. The per-draw host
sync (one per visited node, rank zero) is a cost not measured here.

Validation. Exact law by enumeration over a toy vocabulary and a tree with a shared prefix,
an absent branch and leaves: every (accepted prefix, bonus) event has the sequential target
probability, the first token is the root distribution, and changing any unvisited draw changes
nothing. Real samplers (temperature, top-k, top-p, min-p) match the sampler's own distribution
on the first two emitted tokens (seeded, 4-sigma bounds). On 2 and 3 local rings, with all four
sampler configurations and crafted/real candidates: the emitted tokens equal ordinary
sampling of the whole model with the same seed on rank zero (a stronger check than a
distribution, because the walk consumes one draw per token in order), exactly one draw per
visited prefix, identical decisions on every rank, branched cycles with total and partial
acceptance, and a stop token drawn mid-run ends generation like ordinary sampling.
HTTP: temperature 0.8 with top_k 1 equals the greedy reference token for token (text, stream,
image); a random temperature/top-p request completes. Greedy ddtree is unchanged.

### DDTree on continuous-batching cohorts (2026-10-02)

`dflash_verify_mode=ddtree` now covers multi-request groups, with no new option. In
`fused_batch._tree_group` every request of a verification group proposes its branches
(drafter top-k); the cohort's branch rows (a fork of the request's own cache per branch,
requests of different prompt length, branch count and width, padded to the longest) are
merged into ONE cache and verified in ONE target forward, with each branch row carrying its
request's rope delta (`_branch_rope_deltas`). Row mapping is request order then branch order.
Rank zero decides every request in stable order: greedy picks the longest confirmed branch
(lowest row on ties); a sampled request walks the target distribution with its own sampler
(one draw per visited prefix, never choosing among competing draws). One decision array
(row, accepted count, bonus per request) is shared by the existing collective on every rank
whatever the outcome. A single vector rollback over all branch rows commits each request's
chosen row (unchosen rows roll back to zero accepted drafts); each request then goes through
the shared acceptance code (stop, max tokens, clamps, captures, logprobs of the chosen row,
sinks). Mixed greedy and sampled requests share a cohort; requests arriving or finishing change
the cohort between cycles as before.

Admission is for the whole cohort before any fork: `cohort_total` sums per-request fork,
growth, recurrent-step and extract terms, adds payload and measured scratch for the total row
count, and counts the sampler buffers when any request samples. `reduce_cohort` removes one
branch at a time from the request holding the most (highest index on ties) until it fits;
the branch counts are reduced to the elementwise minimum across ranks (an all-gather every
cohort cycle), and the estimate only shrinks with fewer rows, so each stage still fits. Branch
counts all equal to one run the existing linear verification with the drafter's own blocks.
No linear measurement yet also runs linear. Batch caches (single-row BatchKVCache and
BatchQSAKVCache of a lone request) are now bounded by the memory model. Image requests keep
their singleton path (the cohort runner returns to the linear code). Logits processors:
see "Processors in ddtree" below.

Validation: 2 and 3 ring ranks x 2 and 4 simultaneous requests of different lengths and
generation limits (requests finish at different cycles, so the cohort shrinks), crafted
candidates giving varied accepted lengths, greedy tokens equal to per-request ordinary
generation, mixed greedy/controlled-sampler cohorts, and (not mixed) sampled requests with a
controlled per-request RNG equal to ordinary per-request sampling, same decisions on every
rank, grouped branch forward observed (`requests >= 2`, rows > requests) and the measured
logical MLX peak of each cohort inside its admitted estimate. HTTP (2 and 3 ranks, three
concurrent requests, two rounds so the second hits the prompt cache after branch commits,
greedy and temperature 0.8 / top_k 1): answers equal the reference and the worker logs a
branched cohort of at least two requests, i.e. no linear fallback. Not covered: DFlash sinks
inside a cohort (the capture/sink paths run per request through the shared code, tested in
HTTP only with sinks on single requests), TurboQuant cohorts (refused at startup as unbounded),
physical memory and throughput. A pre-existing defect surfaced here and is fixed (next section): batches whose longest prompt reached the 8-token QSA budget diverged from per-request generation.

### Right-padded batch prefill and recurrent state (2026-10-03)

Symptom: a batch whose longest prompt was 8 or more tokens (the test checkpoint's QSA budget
is 8) generated different tokens than the same requests alone, with plain batching, MTP and
DFlash alike, and with no ddtree. Cause, established by replaying the first recurrent layer:
batch prefill right-pads prompts and prepares per-row `lengths`, but the caches come from
merging empty caches, which carry ZERO left padding. mlx-vlm's `ArraysCache.make_mask`
answers from left padding alone (and its mask builder returns nothing when no row has left
padding), so the recurrent (GDN) layers got no mask: padded positions updated every short
row's recurrent state (a state error of about 8e-3 in layer 0 for a 3-token row beside a
9-token one; the key/value caches and the full-length row were exact). The logit error was
~5e-2 at the first decode step and only flipped argmax tokens once discrete QSA selection
cut in at the budget, which is why shorter mixes passed. The oracle (per-request ordinary
generation) is right: it matches a full recompute of the sequence token for token, the
padded batch does not.
Fix: `_recurrent_row_mask` in the vendored Qwen4 language model adds the in-length condition
from `cache.lengths` (combined with left padding when both exist), only while some row is
shorter than the step; decode steps and full-length rows keep mlx-vlm's answer, so batching
cost and collectives are unchanged. The dependency's `ArraysCache.make_mask` itself is
outside this repository and was not edited; other models that use it keep the defect.
Validation: 2 and 3 ranks x 2 and 4 requests x longest prompt 8 and 9 x MTP and DFlash,
all equal to independent generation (they failed before the fix); mask unit test (zero left
padding, full-length rows, combined left and right padding); ddtree cohort, greedy, sampled,
and HTTP concurrency controls pass.


### Local ddtree and sink HTTP coverage (2026-10-03)

`dflash_verify_mode=ddtree` now loads on a local batched `qwen4_exp` engine (this also
enables the local batched DFlash drafter for `qwen4_exp`). It reuses the distributed
primitives unchanged: `plan_tree_cycle` for a lone request and `fused_batch._tree_group`
for a cohort (ONE grouped target verification), greedy by longest confirmed prefix and
sampled by the target-distribution walk; no coordinator, so no collective runs.
`enable_local_tree` arms the spec (same options and incremental `dflash_ddtree_memory_bytes`
admission; TurboQuant and other unbounded cache families are refused). Refused before the
engine is built: other local targets, a missing memory budget, TurboQuant KV. Refused per
request before any work: repetition/presence/frequency penalties, logit bias, grammar. A
ddtree drafter load failure aborts the load instead of falling back to the linear block.
Fixes found: the local adapter dropped per-branch rope deltas (the language model then
used its stale prefill-batch deltas), so branch rows get the request's delta for the
forward (`_repeat_rope_delta`, `_branch_rope_deltas` through `_language_model`).
Validation: `tests/test_ddtree_local.py` (tiny Qwen4, CPU/MLX: 1/2/4 requests, late join,
greedy and sampled point-mass walk equal ordinary generation, branched cycles observed,
refusals); distributed ring/HTTP ddtree controls 53 passed. HTTP with sinks and captures:
`test_dflash_ddtree_concurrent_sinks_staggered_requests` (2 and 3 ranks, 2 requests RAM /
4 requests SSD with 2-token prefill chunks, staggered arrival, different lengths and
budgets, greedy and top_k=1 sampled, second round restores captures, grouped branch marker
>= 2 requests, tokens equal independent generation). Test harness fix: the capture and
cohort traces shared one name each and overwrote each other's `sitecustomize`.
Limits: the local path is validated on the tiny model only, not through a full local
engine/scheduler load nor on physical hardware; sampled equality uses a point-mass sampler
(not a random-draw equality); images stay singleton; no throughput claim.
(Superseded in part by the next section: an engine-level test now exists.)

### Processors in ddtree and engine-level local test (2026-10-03)

Processors that exist in the API: repetition, presence and frequency penalties (mlx-lm
closures, pure in (prefix, logits)), token suppression (pure), the thinking-budget processor
(stateful, `snapshot_state`/`restore_state`) and the grammar automaton (stateful, no
snapshot). No logit-bias processor exists (the old `logit_bias` refusal stays; nothing builds
one). `ddtree_branches.processed_logprobs` evaluates each visited node through the standard
`_apply_processors` on the EXACT prefix (request context + branch path), on raw logits before
the log-softmax, as the linear cycle does; stateful processors replay the path from their
pre-cycle snapshot and are rewound afterwards, so the shared acceptance code replays the chosen
branch from the pristine state. Greedy branch choice (`greedy_hits`) and the sampled walk both
use these log-probs; only the root evaluates them, peers never gate a collective on their own
verdict. `check_processors` refuses before any fork a processor that is neither snapshottable
nor a known pure closure (grammar, arbitrary callables with side effects: not cloneable per
branch, hence unsupported). Admission counts the sampler's vocabulary buffers whenever
processors are present.
Engine-level local test (`tests/test_zz_ddtree_engine.py`, named to run last because loading a
Qwen4 engine applies process-global mlx-vlm patches): `EnginePool.get_engine` on a tiny Qwen4
checkpoint + the DFlash test drafter with the public options; greedy, penalized and cached
(SSD, 4-token blocks) requests equal the ordinary engine, a grouped branch forward (>=2
requests, more rows than requests) is observed, real temperature/top-p/top-k/min-p sampling
walks the tree and its token 3-4 law stays within total variation 0.3 of ordinary sampling
(draw order differs: no draw-for-draw equality), incompatible settings (no budget, TurboQuant)
are refused before the engine is built. Walk exactness stays covered by the enumeration tests
in `test_ddtree_walk.py`. Bugs found by the engine test: `enable_local_tree` needed the
language model's `make_cache`; the scheduler seeded the drafter with Qwen4's extra
final-residual capture (now cut to the drafter's layers).
Limits: no physical hardware; the TV check is statistical (seeded); processor replay costs one
call per compared node on the root; the HTTP penalty test proves equality with a non-ddtree
deployment, not that penalties changed tokens on this tiny model.
