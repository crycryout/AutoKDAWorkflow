# Optional BF16 GEMM adapter

This adapter is a narrow interface for a task-authored shared library. Other
operators, layouts and numerical contracts need their own validation harness.
The manifest template names source and output paths relative to `KDA_THOR_ROOT`.
Set `CUTLASS_ROOT` or the manifest's `cutlass_root` before building.

## C ABI

Export these functions with C linkage:

```cpp
const char* ct_error();
void* ct_create(int m, int n, int k, void* x, void* weight,
                void* bias, void* output, void* cuda_stream);
int ct_run(void* plan, void* x, void* weight, void* bias,
           void* output, void* cuda_stream);
void ct_destroy(void* plan);
```

`ct_create` returns null on failure; `ct_run` returns zero on success and a
nonzero status on failure. `ct_error` supplies a valid error string. Device
buffers remain owned by the caller. The implementation must use the supplied
stream and must respect CUDA Graph capture/replay and buffer lifetimes.

For the checker's `row_row` layout, `x` has contiguous shape `[M,K]`, `weight`
has contiguous shape `[N,K]`, and `output` has shape `[M,N]`.
Compute `output = x @ weight.T + bias` with BF16 inputs/output and FP32
accumulation. The optional BF16 bias has shape `[N]`; its pointer is null when
`bias` is false. The build adapter also accepts a `row_col` manifest, but the
included checker deliberately requires `row_row`.

## Validation scope

Fill `tolerance_relative_l2` from the frozen task contract; the template leaves
it unset. The checker uses deterministic random inputs and a float32 reference,
checks finite output and relative L2 error, tests a nondefault stream, and
poisons output before changed-input Graph replay. It tests the primary shape
and any additional `validation_shapes`. `--capture-only` collects a kernel
capture without establishing numerical correctness.

Add real inputs, high-precision references and numerical boundary cases for
the task. A relative L2 check does not establish bitwise equality, model
quality, or safe behavior for every distribution. The task's model harness
must supply those checks and the paired unprofiled E2E measurements separately.
