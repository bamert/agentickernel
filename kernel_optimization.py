import traceback
import argparse
import os
from typing import Literal, get_args
import time
from datetime import datetime
import re
import json
import inspect
import subprocess
import requests
from pathlib import Path
from jinja2 import Environment, FileSystemLoader

BANNED_KEYWORDS = [r'\bstatic\b', r'\bthread_local\b', r'\bextern\b', r'\basm\b', r'\b__asm__\b']
SYSTEM_PROMPT = (
    "You are a C++ Optimization Agent. Do NOT use #includes in your generated files. Do NOT use the `static` keyword in your code or anything else to persist data across invocations."
    "Every file you evaluate is tested 1v1 against the baseline. If an approach fails to compile or is too slow, "
    "you can either try to fix it in the same file, or abandon it and start a completely new file."
)
SPEC_PROMPT = "Optimize baseline.hpp by writing more efficient versions. Fore each attempt, create a new .hpp with the same structure as baseline.hpp and same function signature for matmul. Use the same function signature for matmul. Start without intrinsics. If you later want to use intrinsics, use NEON, but stay on one core (no openmp or similar). neon intrinsics are already included in the harness. you don't need to add it. Do not write code in the chat, always use the write_and_evaluate_kernel tool."
CompactionMode = Literal["none", "flush"]
def run_cmd(cmd: list[str], cwd: Path) -> tuple[bool, str]:
    try:
        proc = subprocess.run(
            cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=15
        )
        return proc.returncode == 0, proc.stdout
    except subprocess.TimeoutExpired as e:
        out = e.stdout.decode('utf-8', errors='replace') if e.stdout else "No output."
        return False, f"TIMED OUT after 15s.\n\n{out}"
    except Exception as e:
        return False, f"SYSTEM ERROR: {str(e)}"

class KernelTools:
    def __init__(self, sandbox_dir: Path, implementation_name:str, build_dir: Path):
        self.script_dir = Path(__file__).resolve().parent
        self.sandbox_dir = sandbox_dir
        self.implementation_name = implementation_name
        self.implementation_dir = sandbox_dir / implementation_name
        self.build_dir = build_dir
        self.implementation_dir.mkdir(parents=True, exist_ok=True)
        self.jinja_env = Environment(loader=FileSystemLoader(str(self.sandbox_dir)))
        self.perf: dict[str,str] = {} 
        self.log = [] 
        self.fastest_time_ms = float('inf')
        self.fastest_method = "baseline"
        self.current_loop_iter = 0
        self.tokens_in = 0
        self.tokens_out = 0
        self.cumulative_cost = 0.0
        self.start_time = time.time()
        print(f"[*] Sandbox loaded at: {self.sandbox_dir}")
        print(f"[*] Agent is allowed to read and write files inside : {self.implementation_dir}")


    def log_perf(self, target_name:str, message: str, baseline_ms:float=-1., target_ms:float=-1.):
        wall_time = time.time() - self.start_time # Calculate elapsed time
        
        if target_ms > 0 and target_ms < self.fastest_time_ms:
            self.fastest_time_ms = target_ms
            self.fastest_method = target_name
            
        if len(self.log) == 0:
            self.log.append(f"{self.current_loop_iter}, baseline,{baseline_ms}, 1.00, {self.tokens_in}, {self.tokens_out}, 0, {wall_time:.1f}")
        
        self.log.append(f"{self.current_loop_iter}, {target_name},{target_ms}, {(baseline_ms/target_ms):.2f}, {self.tokens_in}, {self.tokens_out}, {self.cumulative_cost}, {wall_time:.1f}")
        self.perf["baseline"] = f"{baseline_ms}ms"
        self.perf[target_name] = f"{target_ms}ms"
    def get_perf_log_for_llm(self) -> str:
        status = ""
        for k,v in self.perf.items():
            status += f"{k}: {v}. \n"
        return status
    def export_csv(self, compaction_mode:str):
        """Exports the chronological log to a CSV file using pathlib."""
        timestamp = datetime.now().strftime("%y%m%d%H%M")
        output_path = self.script_dir / f"benchmark_{self.implementation_name}_results_{timestamp}.csv"
        
        content = "model,compaction,iteration,target,target_ms,speedup,tokens_in,tokens_out,cumulative_cost_usd,wall_time_sec\n"
        content += "".join(f'{self.implementation_name},{compaction_mode},{entry}\n' for entry in self.log)
        
        output_path.write_text(content, encoding='utf-8')
        print(f"\n[*] Run results saved to {output_path}")
    def get_tool_schemas(self) -> list[dict]:
        tools = []
        type_map = {str: "string", int: "integer", float: "number", bool: "boolean", list: "array", dict: "object"}
        for name, fn in inspect.getmembers(self, predicate=inspect.ismethod):
            if name != "write_and_evaluate_kernel":
                continue
            props = {arg: {"type": type_map.get(t, "string")} for arg, t in fn.__annotations__.items() if arg != "return"}
            tools.append({
                "type": "function",
                "function": {"name": name, "description": fn.__doc__ or "", "parameters": {"type": "object", "properties": props, "required": list(props.keys())}}
            })
        return tools

    def _sync_registry(self, target_name: str):
        """Builds a 1v1 registry containing ONLY the baseline and the current attempt."""
        method_name = Path(target_name).stem
        template = self.jinja_env.get_template("registry.hpp.template")
        (self.sandbox_dir / "registry.hpp").write_text(template.render(method_name=method_name, implementation_name=self.implementation_name), encoding="utf-8")

    def write_and_evaluate_kernel(self, name: str, content: str) -> dict:
        if not name.endswith(".hpp"): name += ".hpp"
        target_path = self.implementation_dir / name
        target_name = Path(name).stem
        for keyword in BANNED_KEYWORDS:
            if re.search(keyword, content):
                return make_ban_response(keyword)
      
        print(f"\n[PROPOSING] {name}")
        target_path.write_text(content, encoding="utf-8")
        self._sync_registry(name)
        
        print("  -> Compiling 1v1 Benchmark...")
        compile_ok, compile_out = run_cmd(["make"], self.build_dir)
        if not compile_ok:
            print("  !! Compilation Failed.")
            self.log_perf(target_name, "compilation failed")
            lines = compile_out.splitlines()
            important_lines = [line for line in lines if "error:" in line or "note:" in line]
            filtered_out = "\n".join(important_lines)
            if not filtered_out.strip():
                filtered_out = compile_out[:1000] # Fallback for linker errors
            return {"success": False, "output": f"COMPILATION FAILED:\n\n{filtered_out}"}

        bench_ok, bench_out = run_cmd(["./test_and_bench", "--benchmark_format=csv"], self.build_dir)
        print(f"  -> Benchmark {"complete" if bench_ok else "failed"}")
        if not bench_ok:
            self.log_perf(target_name, "tests failed")
        if compile_ok and bench_ok:
            base_match = re.search(r'^"baseline",\d+,([^,]+),', bench_out, re.MULTILINE)
            target_match = re.search(rf'^"{target_name}",\d+,([^,]+),', bench_out, re.MULTILINE)

            if base_match and target_match:
                base_ns = float(base_match.group(1))
                target_ns = float(target_match.group(1))
                base_ms = base_ns / 1e6
                target_ms = target_ns / 1e6
                
                is_faster = target_ns < base_ns
                ratio = (base_ns / target_ns) if is_faster else (target_ns / base_ns)
                is_fastest = target_ms < self.fastest_time_ms
                if is_fastest and is_faster:
                    status = "SUCCESS (This is the new fastest solution)"
                elif is_faster:
                    status = f"GOOD (Faster than baseline, but slower than current best: {self.fastest_time_ms:.2f}ms)"
                else:
                    status = "FAILURE (slower than baseline)"
                comp = "FASTER" if is_faster else "SLOWER"
                
                # Append a bright, unmissable summary for the LLM
                self.log_perf(target_name, status, base_ms, target_ms)
                summary = self.get_perf_log_for_llm()
                perfsum = f"\n\n[SYSTEM NOTE: {status}! Your code ran in {target_ms:.2f}ms. It is {ratio:.2f}x {comp} than the baseline ({base_ms:.2f}ms).]"
                print(summary)
                print(perfsum)
                bench_out += summary + perfsum

        return {"success": bench_ok, "output": bench_out}

    def read_file(self, name: str) -> dict:
        p = self.implementation_dir / name
        print(f"\n[READ FILE] {name}")
        return {"success": True, "content": p.read_text()} if p.is_file() else {"error": "Not found"}

    def list_files(self) -> dict:
        return {"success": True, "files": sorted([f.name for f in self.implementation_dir.iterdir() if f.is_file()])}

def call_llm(conv, tools, model, url, key):
    headers = {"Content-Type": "application/json", **({"Authorization": f"Bearer {key}"} if key else {})}
    payload = {"model": model, "messages": conv, "tools": tools}
    for _ in range(3):
        r = requests.post(url, json=payload, headers=headers)
        r.raise_for_status()
        resp = r.json()
        if "choices" not in resp or len(resp["choices"]) == 0:
            time.sleep(5)
            continue
        else:
            return resp["choices"][0]["message"], resp.get("usage", {})
    raise RuntimeError("LLM did not return a valid response after 3 attempts.")
def make_ban_response(keyword:str) -> dict:
    return {
        "success": False, 
        "output": f"COMPILATION FAILED:\n\nRULE VIOLATION: The keyword '{keyword.replace(r'\\b', '')}' is strictly banned. The function must be 100% stateless and use pure C++ intrinsics without inline assembly."
                }
def make_base_conversation(baseline_code: str | None) -> list[dict]:
    system = SYSTEM_PROMPT
    if baseline_code:
        system += f"\n\nBaseline Reference:\n{baseline_code}\n"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": SPEC_PROMPT},
    ]

def make_continue_message() -> dict:
       return {
            "role": "user",
            "content": (
                "Please continue working by calling a tool. "
                "Pick the optimization path most likely to succeed based on feedback so far."
            ),
        }
def make_flush_catchup(target_name: str, perf_summary: str) -> dict:
    return {
        "role": "user",
        "content": (
            "System Note: The context window was flushed to save memory.\n\n"
            f"Your last kernel (`{target_name}`) compiled and benched successfully. "
            f"Leaderboard of all attempts so far:\n{perf_summary}\n\n"
            "If you need to see the code for any previous attempt, use the `read_file` tool. "
            "Please continue optimizing."
        ),
    }
def run_autonomous_loop(toolkit:KernelTools, model:str, url:str, key:str, max_iterations:int, cost_in_per_million:float, cost_out_per_million:float, budget_limit:float, context_compaction: CompactionMode):
    total_in_tokens = 0
    total_out_tokens = 0
    cumulative_cost = 0.0

   
    baseline_code = (toolkit.sandbox_dir / "baseline.hpp").read_text()
    conversation = make_base_conversation(baseline_code)

    for iteration in range(max_iterations):
        print(f"\n--- Iteration {iteration}/{max_iterations} ---")
        msg, usage = call_llm(conversation, toolkit.get_tool_schemas(), model, url, key)
        if usage:
            total_in_tokens += usage.get("prompt_tokens", 0)
            total_out_tokens += usage.get("completion_tokens", 0)
            in_cost = (total_in_tokens / 1_000_000) * cost_in_per_million 
            out_cost = (total_out_tokens / 1_000_000) * cost_out_per_million
            cumulative_cost = in_cost + out_cost if (cost_in_per_million > 0 or cost_out_per_million > 0) else 0.0
            print(f"  [Tokens] In: {total_in_tokens} | Out: {total_out_tokens} | Cumulative Cost: ${cumulative_cost:.4f}")
        toolkit.current_loop_iter = iteration + 1
        toolkit.tokens_in = total_in_tokens
        toolkit.tokens_out = total_out_tokens
        if budget_limit > 0 and cumulative_cost >= budget_limit:
            print(f"\n[!] BUDGET LIMIT REACHED (${cumulative_cost:.2f} >= ${budget_limit}). Terminating")
            break 
        toolkit.cumulative_cost = cumulative_cost
        conversation.append(msg)
        
        if msg.get("content"):
            print(f"Agent:\n{msg['content']}\n")
            
        if not msg.get("tool_calls"):
            conversation.append(make_continue_message())
            continue
            
        for call in msg["tool_calls"]:
            t_name = call["function"]["name"]
            t_args = json.loads(call["function"]["arguments"])
            
            try:
                res = getattr(toolkit, t_name)(**t_args)
            except Exception as e:
                res = {"error": str(e)}
            if context_compaction == "none":
                conversation.append({
                    "role": "tool", 
                    "tool_call_id": call["id"], 
                    "name": t_name, 
                    "content": json.dumps(res)
                })
                conversation.append({"role": "user", "content": "Please continue working by calling a tool. Pick the optimization path that you think is most likely to succeed based on the feedback you have received so far."})
            elif context_compaction == "flush":
                if t_name == "write_and_evaluate_kernel" and res.get("success"):
                    print("Flushing context to save tokens...")
                    conversation = make_base_conversation(baseline_code)
                    perf_state = toolkit.get_perf_log_for_llm()
                    target = t_args.get('name', 'your last attempt')
                    conversation.append(make_flush_catchup(target, perf_state))
                    continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agentic Kernel Optimization")
    parser.add_argument("--url", type=str, required=True, help="OpenAI compatible /completions API endpoint URL (e.g., http://localhost/v1/chat/completions)")
    parser.add_argument("--model", type=str, required=True, help="Model identifier (e.g., 'gpt-oss:20b')")
    parser.add_argument("--cost_out_per_million", type=float, required=False, default=0, help="Cost per million output tokens in $")
    parser.add_argument("--cost_in_per_million", type=float, required=False, default=0, help="Cost per million input tokens in $")
    parser.add_argument("--budget_limit", type=float, required=False, default=0, help="Budget Limit for the$")
    parser.add_argument(
        "--compaction", 
        type=str, 
        choices=get_args(CompactionMode),
        default="none", 
        help="Context compaction mode: 'none' (default) or 'flush'."
    )
    args = parser.parse_args()
    key = os.getenv("OPENROUTER_API_KEY")
    script_dir = Path(__file__).resolve().parent
    sandbox_dir = script_dir / "sandbox_bmm"
    build_dir = script_dir / "build"
    implementation_name =  args.model.replace("/","_").replace(":","_")
    kernel_tools = KernelTools(sandbox_dir, implementation_name, build_dir)
    try:
        run_autonomous_loop(kernel_tools, args.model, args.url, key, 50, args.cost_in_per_million, args.cost_out_per_million, args.budget_limit, args.compaction)
    except KeyboardInterrupt:
        print("\n[!] Loop aborted by user (Ctrl+C). Saving progress...")
        sys.exit(130) 
    except Exception as e:
        print(f"\n[!] Error encountered: {e}")
        traceback.print_exc()  
        sys.exit(1) 
    finally:
        kernel_tools.export_csv(args.compaction)
