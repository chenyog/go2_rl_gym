# Go1 C++ Deploy

This deploy path uses the local Free-Dog low-level UDP headers in `include/ucl`
and ONNX Runtime C++ for policy inference.

## Prerequisites

System packages required for both real deploy and latency benchmark:

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake libyaml-cpp-dev
```

Keep non-system runtime dependencies inside this deploy directory:

```text
deploy/deploy_real_go1_cpp/third_party/
```

Install CPU ONNX Runtime 1.10.0 from the repo root:

```bash
mkdir -p deploy/deploy_real_go1_cpp/third_party
cd deploy/deploy_real_go1_cpp/third_party
wget https://github.com/microsoft/onnxruntime/releases/download/v1.10.0/onnxruntime-linux-x64-1.10.0.tgz
tar -xzf onnxruntime-linux-x64-1.10.0.tgz
```

Expected layout:

```text
deploy/deploy_real_go1_cpp/
  config/
  include/
  scripts/
  src/
  third_party/
    onnxruntime-linux-x64-1.10.0/
      include/
      lib/
```

The C++ executable links against the ONNX Runtime archive selected by
`-DONNXRUNTIME_ROOT=...`. For CPU inference, use the Linux x64 CPU release. For
CUDA inference, use the Linux x64 GPU release that matches the local CUDA/cuDNN
runtime.

## Build

From the repo root:

```bash
cmake -S deploy/deploy_real_go1_cpp \
  -B deploy/deploy_real_go1_cpp/build \
  -DONNXRUNTIME_ROOT=deploy/deploy_real_go1_cpp/third_party/onnxruntime-linux-x64-1.10.0
cmake --build deploy/deploy_real_go1_cpp/build -j
```

This builds:

```text
deploy/deploy_real_go1_cpp/build/go1_deploy
deploy/deploy_real_go1_cpp/build/go1_benchmark_onnx_latency
```

Use the GPU ONNX Runtime package by changing `ONNXRUNTIME_ROOT` to the local GPU
directory:

```bash
cmake -S deploy/deploy_real_go1_cpp \
  -B deploy/deploy_real_go1_cpp/build_gpu \
  -DONNXRUNTIME_ROOT=deploy/deploy_real_go1_cpp/third_party/onnxruntime-linux-x64-gpu-1.10.0
cmake --build deploy/deploy_real_go1_cpp/build_gpu -j
```

If the ONNX Runtime shared libraries are not in the system loader path, add the
selected `lib` directory before running:

```bash
export LD_LIBRARY_PATH=deploy/deploy_real_go1_cpp/third_party/onnxruntime-linux-x64-1.10.0/lib:$LD_LIBRARY_PATH
```

For the GPU build, use the GPU package directory in `LD_LIBRARY_PATH` instead.

## C++ ONNX Latency Benchmark

`go1_benchmark_onnx_latency` uses ONNX Runtime C++ and matches the deploy input
layout in `src/onnx_policy.cpp`: it keeps the same history buffer, concatenates
terms in `ang_vel, gravity, command, q, dq, last_action` order, runs ONNX
Runtime, and checks that the first output is a 12-D action.

Benchmark an exported ONNX policy on CPU:

```bash
deploy/deploy_real_go1_cpp/build/go1_benchmark_onnx_latency \
  --policy logs/go1_moe_cts/exported/policies/policy.onnx \
  --provider cpu \
  --warmup 200 \
  --repeat 2000
```

Benchmark CUDA with a GPU ONNX Runtime build:

```bash
deploy/deploy_real_go1_cpp/build_gpu/go1_benchmark_onnx_latency \
  --policy logs/go1_moe_cts/exported/policies/policy.onnx \
  --provider cuda \
  --warmup 200 \
  --repeat 2000
```

Benchmark both providers when the executable is built against a GPU ONNX Runtime
package:

```bash
deploy/deploy_real_go1_cpp/build_gpu/go1_benchmark_onnx_latency \
  --policy logs/go1_moe_cts/exported/policies/policy.onnx \
  --provider both
```

Benchmark the configured real-deploy policy slots:

```bash
deploy/deploy_real_go1_cpp/build/go1_benchmark_onnx_latency --slot initial --provider cpu
deploy/deploy_real_go1_cpp/build/go1_benchmark_onnx_latency --slot all --provider cpu
```

If CUDA provider support is missing from the linked ONNX Runtime package, the
CUDA benchmark fails explicitly. It does not fall back to CPU.

## Real Robot Runtime: CPU Or GPU

`go1_deploy` selects the ONNX Runtime execution provider from
`config/go1.yaml`:

```yaml
policy_provider: cpu
cuda_device_id: 0
```

Use CPU for the real robot unless measured latency shows that CPU inference
misses the 50 Hz policy deadline. The current Go1 policy is small, and CPU
inference is simpler for real robot control. GPU inference adds CUDA provider
installation, device scheduling, and host-device transfer failure modes.

To use GPU on the real robot, build against a GPU ONNX Runtime package and set:

```yaml
policy_provider: cuda
cuda_device_id: 0
```

If CUDA provider support is missing from the linked ONNX Runtime package,
`go1_deploy` fails during startup. It does not fall back to CPU. Installing a
Python ONNX Runtime package does not change the C++ deploy path.

## Real Deploy

WiFi connection settings live in `config/go1.yaml`:

```yaml
connection: low_wifi
robot_ip: 192.168.123.10
local_ip: 192.168.12.14
listen_port: 8090
send_port: 8007
```

`robot_ip` is the Go1 target address. `local_ip` must match the PC WiFi address.
The UDP socket binds to `local_ip:listen_port`, so a wrong local IP fails at
startup instead of silently waiting for packets.

Run dry-run first:

```bash
deploy/deploy_real_go1_cpp/build/go1_deploy \
  --config deploy/deploy_real_go1_cpp/config/go1.yaml \
  --dry-run
```

Run real deploy:

```bash
deploy/deploy_real_go1_cpp/build/go1_deploy \
  --config deploy/deploy_real_go1_cpp/config/go1.yaml \
  --connection low_wifi
```

Runtime keys:

```text
S: move to default stand
r: run policy
1/2/3/4: switch up/down/left/right policy slot
p: passive damping
Q: damping and quit
h: help
```

When `command_source: keyboard`:

```text
w/s or up/down: vx +1/-1
a/d or left/right: vy +1/-1
q/e: yaw +1/-1
space: zero command
```
