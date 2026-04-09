import os
from typing import Literal
import time
import re
import json
import inspect
import subprocess
import requests
from pathlib import Path
from jinja2 import Environment, FileSystemLoader


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
    def __init__(self, sandbox_dir: Path, build_dir: Path):
        self.script_dir = Path(__file__).resolve().parent
        self.sandbox_dir = sandbox_dir
        self.build_dir = build_dir
        self.sandbox_dir.mkdir(parents=True, exist_ok=True)
        self.jinja_env = Environment(loader=FileSystemLoader(str(self.sandbox_dir)))
        self.perf: dict[str,str] = {} # method to outcome mapping (if multiple attempts were made, only latest one)
        self.log = [] # chronological log and perf  (for all attempts)
        print(f"[*] Sandbox loaded at: {self.sandbox_dir}")


    def log_perf(self, target_name:str, message: str, baseline_ms:float=-1., target_ms:float=-1.):
        iteration = len(self.log)
        if len(self.log) == 0:
            self.log.append(f"{iteration}, baseline,{message},{baseline_ms}, 1.00")
        self.log.append(f"{iteration+1}, {target_name},{message},{target_ms}, {(baseline_ms/target_ms)}")
        self.perf["baseline"] = f"{baseline_ms}ms"
        self.perf[target_name] = f"{target_ms}ms"
    def get_perf_log_for_llm(self) -> str:
        status = ""
        for k,v in self.perf.items():
            status += f"{k}: {v}. \n"
        return status
    def export_csv(self, model: str):
        """Exports the chronological log to a CSV file using pathlib."""
        model_descriptor = model.replace("/","_")
        output_path = self.script_dir / f"benchmark_{model_descriptor}_results.csv"
        
        # Build the entire CSV content as a single string
        content = "model,iteration,target,message,target_ms,speedup\n"
        content += "".join(f'"{model_descriptor}",{entry}\n' for entry in self.log)
        
        # Write it to disk in one shot
        output_path.write_text(content, encoding='utf-8')
        print(f"\n[*] Run results saved to {output_path}")
    def get_tool_schemas(self) -> list[dict]:
        tools = []
        type_map = {str: "string", int: "integer", float: "number", bool: "boolean", list: "array", dict: "object"}
        for name, fn in inspect.getmembers(self, predicate=inspect.ismethod):
            if name.startswith("_") or name == "get_tool_schemas": continue
            props = {arg: {"type": type_map.get(t, "string")} for arg, t in fn.__annotations__.items() if arg != "return"}
            tools.append({
                "type": "function",
                "function": {"name": name, "description": fn.__doc__ or "", "parameters": {"type": "object", "properties": props, "required": list(props.keys())}}
            })
        return tools

    def _sync_registry(self, target_name: str):
        """Builds a 1v1 registry containing ONLY the baseline and the current attempt."""
        target_stem = Path(target_name).stem
        methods = ["baseline", target_stem]
        try:
            template = self.jinja_env.get_template("registry.hpp.template")
            (self.sandbox_dir / "registry.hpp").write_text(template.render(methods=methods), encoding="utf-8")
        except Exception as e:
            print(f"  !! Registry Sync Failed: {e}")

    def write_and_evaluate_kernel(self, name: str, content: str) -> dict:
        if not name.endswith(".hpp"): name += ".hpp"
        target_path = self.sandbox_dir / name
        target_name = Path(name).stem
        
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

        # Add the CSV flag right here!
        bench_ok, bench_out = run_cmd(["./test_and_bench", "--benchmark_format=csv"], self.build_dir)
        print(f"  -> Benchmark {"complete" if bench_ok else "failed"}")
        if not bench_ok:
            self.log_perf(target_name, "tests failed")
        # --- EXTRACT RUNTIMES FROM CSV ---
        if compile_ok and bench_ok:
            # Look for the quoted names in the CSV output
            base_match = re.search(r'^"baseline",\d+,([^,]+),', bench_out, re.MULTILINE)
            target_match = re.search(rf'^"{target_name}",\d+,([^,]+),', bench_out, re.MULTILINE)

            if base_match and target_match:
                base_ns = float(base_match.group(1))
                target_ns = float(target_match.group(1))
                base_ms = base_ns / 1e6
                target_ms = target_ns / 1e6
                
                is_faster = target_ns < base_ns
                ratio = (base_ns / target_ns) if is_faster else (target_ns / base_ns)
                status = "SUCCESS" if is_faster else "FAILURE"
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
        p = self.sandbox_dir / name
        print(f"\n[READ FILE] {name}")
        return {"success": True, "content": p.read_text()} if p.is_file() else {"error": "Not found"}

    def list_files(self) -> dict:
        return {"success": True, "files": sorted([f.name for f in self.sandbox_dir.iterdir() if f.is_file()])}

# --- PURE STATELESS REPL ---

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

def run_autonomous_loop(toolkit:KernelTools, model:str, url:str, key:str, max_iterations:int, context_compaction: Literal["no_compaction", "flush"]):
    total_in_tokens = 0
    total_out_tokens = 0

    spec_prompt = "Optimize baseline.hpp by writing more efficient versions. Fore each attempt, create a new .hpp with the same structure as baseline.hpp and same function signature for matmul. Use the same function signature for matmul. Start without intrinsics. If you later want to use intrinsics, use NEON, but stay on one core (no openmp or similar). neon intrinsics are already included in the harness. you don't need to add it."
    sys_prompt = (
        "You are a C++ Optimization Agent. Do NOT use #includes in your generated files. "
        "Every file you evaluate is tested 1v1 against the baseline. If an approach fails to compile or is too slow, "
        "you can either try to fix it in the same file, or abandon it and start a completely new file."
    )
    
    base_file = toolkit.sandbox_dir / "baseline.hpp"
    if base_file.exists(): 
        sys_prompt += f"\n\nBaseline Reference:\n{base_file.read_text()}\n"
    
    conversation = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": spec_prompt}]

    for iteration in range(max_iterations):
        print(f"\n--- Iteration {iteration}/{max_iterations} ---")
        msg, usage = call_llm(conversation, toolkit.get_tool_schemas(), model, url, key)
        if usage:
            total_in_tokens += usage.get("prompt_tokens", 0)
            total_out_tokens += usage.get("completion_tokens", 0)
            print(f"  [Tokens] In: {total_in_tokens} | Out: {total_out_tokens}")

        conversation.append(msg)
        
        if msg.get("content"):
            print(f"Agent:\n{msg['content']}\n")
            
        if not msg.get("tool_calls"):
            conversation.append({"role": "user", "content": "Please continue working by calling a tool. Pick the optimization path that you think is most likely to succeed based on the feedback you have received so far."})
            continue
            
        for call in msg["tool_calls"]:
            t_name = call["function"]["name"]
            t_args = json.loads(call["function"]["arguments"])
            
            try:
                res = getattr(toolkit, t_name)(**t_args)
            except Exception as e:
                res = {"error": str(e)}
            if context_compaction == "no_compaction":
                conversation.append({
                    "role": "tool", 
                    "tool_call_id": call["id"], 
                    "name": t_name, 
                    "content": json.dumps(res)
                })
            elif context_compaction == "flush":
                if t_name == "write_and_evaluate_kernel" and res.get("success"):
                    print("Flushing context to save tokens...")
                    
                    # 1. Reset conversation to the original system and user prompts
                    conversation = [
                        {"role": "system", "content": sys_prompt}, 
                        {"role": "user", "content": spec_prompt}
                    ]
                    
                    # 2. Build the "catch-up" message using your new logging system
                    perf_state = toolkit.get_perf_log_for_llm()
                    target = t_args.get('name', 'your last attempt')
                    
                    catch_up_msg = (
                        f"System Note: The context window was flushed to save memory.\n\n"
                        f"Your last kernel (`{target}`) compiled and benched successfully. "
                        f"Here is the leaderboard of all attempts so far:\n{perf_state}\n\n"
                        f"If you need to see the code for any previous attempt, use the `read_file` tool. "
                        f"Please continue optimizing."
                    )
                    
                    # 3. Append the catch-up message and skip the raw tool append
                    conversation.append({"role": "user", "content": catch_up_msg})
                    continue

if __name__ == "__main__":
    url = "https://openrouter.ai/api/v1/chat/completions"
    model = "google/gemini-3-flash-preview"
    #model = "google/gemini-3.1-pro-preview"
    #model = "google/gemma-4-31b-it"
    #model = "google/gemini-2.5-pro"
    #model = "z-ai/glm-5.1"
    #model = "openai/gpt-5-nano"
    #model ="google/gemini-2.5-flash"
    #model = "openai/gpt-5.4-mini"
    #url = "http://gx10:11434/v1/chat/completions" 
    #model = "gpt-oss:20b"
    #model = "nemotron-3-super:120b"
    #model = "qwen3-coder-next"
    #model = "gemma4:31b"
    #model = "gemma4:e2b"
    #model = "gemma4:e4b"
    key = os.getenv("OPENROUTER")
    script_dir = Path(__file__).resolve().parent
    sandbox_dir = script_dir / "sandbox_bmm"
    build_dir = script_dir / "build"
    kernel_tools = KernelTools(sandbox_dir, build_dir)
    run_autonomous_loop( kernel_tools , model, url, key, 5, "no_compaction")
    kernel_tools.export_csv(model)
