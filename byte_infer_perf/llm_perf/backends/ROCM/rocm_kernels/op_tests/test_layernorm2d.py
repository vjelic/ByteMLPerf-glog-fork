import torch
import torch.nn.functional as F
import numpy as np
import rocmKernels

num_iters = 100


def run_torch(input, normalized_shape, weight, bias, eps, residual=None):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    latencies = []
    for i in range(num_iters):
        start_event.record()
        if residual is None:
            output = F.layer_norm(
                input=input,
                normalized_shape=(input.shape[-1],),
                weight=weight,
                bias=bias,
                eps=eps
            )
        else:
            residual = input + residual
            output = F.layer_norm(
                input=residual,
                normalized_shape=(input.shape[-1],),
                weight=weight,
                bias=bias,
                eps=eps
            )
        end_event.record()
        end_event.synchronize()
        latencies.append(start_event.elapsed_time(end_event))
    avg = np.mean(latencies[10:]) * 1000  # us
    # print(f"run_torch avg time: {avg} us")
    return output, avg


def run_ck(input, normalized_shape, weight, bias, eps, residual=None):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    latencies = []
    for i in range(num_iters):
        start_event.record()
        if residual is None:
            output = torch.empty_like(input)
            rocmKernels.layernorm2d_fwd(
                output,
                input,
                weight,
                bias,
                eps
            )
        else:
            residual_out = torch.empty_like(input)
            output = torch.empty_like(input)
            rocmKernels.layernorm2d_fwd_with_add(
                output,
                input,
                residual,
                residual_out,
                weight,
                bias,
                eps
            )
        end_event.record()
        end_event.synchronize()
        latencies.append(start_event.elapsed_time(end_event))
    avg = np.mean(latencies[10:]) * 1000  # us
    # print(f"run_ck    avg time: {avg} us")
    return output, avg


def checkAllclose(a, b, rtol, atol):
    assert torch.allclose(
        a, b, rtol, atol), f'''torch and ck results are not close\n{a.shape}\n{a}\n{b.shape}\n{b}\nmax delta:{(a-b).max()}\n 
        details{(a-b)}'''


for dtype in [torch.float16, torch.bfloat16]:
    # for dtype in [torch.float16]:
    for dim in [4096, 8192, 16384, 32768, 65536]:
        input = torch.randn(dim, dtype=dtype, device="cuda")
        weight = torch.randn(dim, dtype=dtype, device="cuda")
        bias = torch.randn(dim, dtype=dtype, device="cuda")
        residual = torch.randn(dim, dtype=dtype, device="cuda")
        a, avg_a = run_torch(input, (dim,), weight, bias, 1e-5)
        b, avg_b = run_ck(input, (dim,), weight, bias, 1e-5)
        print(
            f"dim: {dim}, dtype: {dtype}, torch avg: {avg_a:.2f} us, ck avg: {avg_b:.2f} us, uplift: {avg_a/avg_b-1:.1%}")
        checkAllclose(a, b, 1e-2, 1e-2)

print('start fuse add test')
for dtype in [torch.float16, torch.bfloat16]:
    # for dtype in [torch.bfloat16]:
    for dim in [4096, 8192, 16384, 32768, 65536]:
        input = torch.randn(dim, dtype=dtype, device="cuda")
        weight = torch.randn(dim, dtype=dtype, device="cuda")
        bias = torch.randn(dim, dtype=dtype, device="cuda")
        residual = torch.randn(dim, dtype=dtype, device="cuda")
        a, avg_a = run_torch(input, (dim,), weight, bias, 1e-5)
        b, avg_b = run_ck(input, (dim,), weight, bias, 1e-5)
        # fuse add
        a, avg_a = run_torch(input, (dim,), weight, bias,
                             1e-5, residual=residual)
        b, avg_b = run_ck(input, (dim,), weight, bias, 1e-5, residual=residual)
        print(
            f"dim: {dim}, dtype: {dtype}, torch avg: {avg_a:.2f} us, ck avg: {avg_b:.2f} us, uplift: {avg_a/avg_b-1:.1%}")

        checkAllclose(a, b, 1e-2, 1e-2)
