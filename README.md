# Agentic Kernel: Automatic Iterative Kernel Optimization
Automatically iterates compute kernels on a baseline reference implementation to optimize towards performance improvements.
Under a spending cap, local models (Such as `gemma4:31b`) found speedups of over 40x, exceeding frontier models on performance/cost.

> 📖 For a full write-up, see the [**blog post**](https://nikbamert.com/article/agentic-kernel-optimization)

# Why
The performance of compute kernels is critical for a wide range of applications. Optimizing these kernels often requires time-consuming hand-tuning. In this project we consider the trade-offs between cost (for API access to the models) of various frontier and local models and demonstrate that local models can be very competitive under budget constraints.

*Note* This repository is an experimental benchmark harness, not a production framework. The implementation is intentionally kept small and contained to make the optimization loop easy to inspect and reproduce.

# Requirements and how to use
- A C compiler, `cmake` and google benchmark.
- An OpenAI compatible API for the agent loop to call. If an API key is required for the host, it can be passed throuhgh `OPENROUTER_API_KEY`.

# Setup

Install requirements
```
# On Debian/Ubuntu
apt install build-essential cmake libbenchmark-dev 
# -- OR --
xcode-select --install
brew install cmake google-benchmark
```

Setup the build directory
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

