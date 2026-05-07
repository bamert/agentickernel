# Agentic Kernel: Automatic Iterative Kernel Optimization

Agentic Kernel is a compact benchmark harness for iterative LLM-driven compute-kernel optimization. Starting from a baseline reference implementation, it asks a model to propose new kernels, compiles and tests them, benchmarks them against the baseline, and feeds the results back into the next iteration.

Under a spending cap, local models such as `gemma4:31b` found speedups of over 40x in this benchmark, exceeding frontier models on performance/cost.

> For a full write-up, see the [**blog post**](https://nikbamert.com/article/agentic-kernel-optimization)

# Why

Optimizing compute kernels often requires time-consuming hand-tuning. This project explores how far a small agent loop can get by combining code generation, correctness checks, benchmarking, and cost tracking. It also compares local and frontier models under budget constraints.

The implementation is intentionally small and contained so the optimization loop, benchmark feedback, and cost/performance trade-offs are easy to inspect and reproduce.

# Requirements and how to use

- A C compiler, `cmake`, and Google Benchmark.
- An OpenAI-compatible API for the agent loop to call. If an API key is required, it can be passed through `OPENROUTER_API_KEY`.

# Setup

Install requirements:

```bash
# On Debian/Ubuntu
apt install build-essential cmake libbenchmark-dev

# -- OR --

xcode-select --install
brew install cmake google-benchmark
```


Set up the build directory

```
mkdir build
cd build
cmake ../sandbox_bmm
```

Run the optimization loop with the following command:
```
kernel_optimization.py  --url <OPENAI_COMPATIBLE_API_URL> --model <MODEL_NAME> 
```

Optionally, the following arguments can be passed:
- `--cost_in_per_million`: Cost per million tokens for the input to the model.
- `--cost_out_per_million`: Cost per million tokens for the model being used. 
- `--budget_limit`: Spending cap for one run in USD.

