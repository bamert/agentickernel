import os
import re
import json
import inspect
import subprocess
import requests
from pathlib import Path
from jinja2 import Environment, FileSystemLoader

MAX_ITERATIONS = 50

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
    def __init__(self):
        self.script_dir = Path(__file__).resolve().parent
        self.sandbox_dir = self.script_dir / "sandbox"
        self.build_dir = self.script_dir / "build"
        self.sandbox_dir.mkdir(parents=True, exist_ok=True)
        self.jinja_env = Environment(loader=FileSystemLoader(str(self.script_dir)))
        print(f"[*] Sandbox loaded at: {self.sandbox_dir}")

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
            return {"success": False, "output": f"COMPILATION FAILED:\n\n{compile_out[:2000]}"}

        print("  -> Benchmarking...")
        # Add the CSV flag right here!
        bench_ok, bench_out = run_cmd(["./test_and_bench", "--benchmark_format=csv"], self.build_dir)
        print(f"  -> Benchmark complete. (Exit code 0: {bench_ok})")

        # --- EXTRACT RUNTIMES FROM CSV ---
        if compile_ok and bench_ok:
            # Look for the quoted names in the CSV output
            base_match = re.search(r'^"baseline",\d+,([^,]+),', bench_out, re.MULTILINE)
            target_match = re.search(rf'^"{target_name}",\d+,([^,]+),', bench_out, re.MULTILINE)

            if base_match and target_match:
                base_ns = float(base_match.group(1))
                target_ns = float(target_match.group(1))
                
                is_faster = target_ns < base_ns
                ratio = (base_ns / target_ns) if is_faster else (target_ns / base_ns)
                status = "SUCCESS" if is_faster else "FAILURE"
                comp = "FASTER" if is_faster else "SLOWER"
                
                # Append a bright, unmissable summary for the LLM
                summary = f"\n\n[SYSTEM NOTE: {status}! Your code ran in {target_ns/1e6:.2f}ms. It is {ratio:.2f}x {comp} than the baseline ({base_ns/1e6:.2f}ms).]"
                print(summary)
                bench_out += summary

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
    r = requests.post(url, json=payload, headers=headers)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]

def run_autonomous_loop(spec_prompt, toolkit, model, url, key):
    sys_prompt = (
        "You are a C++ Optimization Agent. Do NOT use #includes in your generated files. "
        "Every file you evaluate is tested 1v1 against the baseline. If an approach fails to compile or is too slow, "
        "you can either try to fix it in the same file, or abandon it and start a completely new file."
    )
    
    base_file = toolkit.sandbox_dir / "baseline.hpp"
    if base_file.exists(): 
        sys_prompt += f"\n\nBaseline Reference:\n{base_file.read_text()}\n"
    
    conversation = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": spec_prompt}]

    for iteration in range(MAX_ITERATIONS):
        print(f"\n--- Iteration {iteration + 1}/{MAX_ITERATIONS} ---")
        msg = call_llm(conversation, toolkit.get_tool_schemas(), model, url, key)
        conversation.append(msg)
        
        if msg.get("content"):
            print(f"Agent:\n{msg['content']}\n")
            
        if not msg.get("tool_calls"):
            conversation.append({"role": "user", "content": "Please continue working by calling a tool."})
            continue
            
        for call in msg["tool_calls"]:
            t_name = call["function"]["name"]
            t_args = json.loads(call["function"]["arguments"])
            
            try:
                res = getattr(toolkit, t_name)(**t_args)
            except Exception as e:
                res = {"error": str(e)}

            conversation.append({
                "role": "tool", 
                "tool_call_id": call["id"], 
                "name": t_name, 
                "content": json.dumps(res)
            })

if __name__ == "__main__":
    url = "https://openrouter.ai/api/v1/chat/completions"
    model = "google/gemini-3-flash-preview"
    #url = "http://gx10:11434/v1/chat/completions" 
    #model = "gemma4:e4b"
    key = os.getenv("OPENROUTER")
    
    run_autonomous_loop(
        "Optimize baseline.hpp by writing more efficient versions. Create a new .hpp file for each attempt with the same structure as baseline.hpp for each attempt. Use the same function signature for dot product. ", 
        KernelTools(), model, url, key
    )
