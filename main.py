import requests
from pathlib import Path 
import os
import re
import json
import inspect
from jinja2 import Environment, FileSystemLoader
import subprocess

MAX_ITERATIONS = 50
MAX_RETRIES = 5
PERF_GOAL_MS = 1.5

def run_make_in_sandbox() -> str:
    # Safely resolve cwd/sandbox relative to this Python file
    sandbox_path = Path(__file__).parent / "sandbox"
    
    try:
        # stderr=subprocess.STDOUT merges errors directly into standard output.
        # text=True automatically decodes the byte stream to a string.
        process = subprocess.run(
            ["make"],
            cwd=sandbox_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=10 # Crucial: Prevents a weird build state from hanging the agent forever
        )
        
        # We return everything. The LLM gets the raw terminal experience.
        if process.returncode == 0:
            return f"Make completed successfully:\n\n{process.stdout}"
        else:
            return f"Make failed (Exit code {process.returncode}):\n\n{process.stdout}"
            
    except subprocess.TimeoutExpired as e:
        # If make hangs, we catch it and feed the partial output back to the LLM
        partial_out = e.stdout.decode('utf-8', errors='replace') if e.stdout else "No output."
        return f"Build TIMED OUT after 10 seconds. Partial output:\n\n{partial_out}"
    except FileNotFoundError:
        return "System error: 'make' command not found. Is build-essential installed?"
    except Exception as e:
        return f"System error executing make: {str(e)}"
def build_registry(method_names: list[str]):
    env = Environment(loader=FileSystemLoader("."))
    template = env.get_template("registry.hpp.template")
    rendered_cpp = template.render(methods=method_names)
    output_path = Path(__file__).resolve().parent / "sandbox" / "registry.hpp"
    Path(output_path).write_text(rendered_cpp, encoding="utf-8")
class KernelTools:
    """Base class that auto-discovers methods and generates OpenAI tool schemas."""
    
    def __init__(self):
        self.sandbox_dir = Path(__file__).resolve().parent / "sandbox"
    def get_tool_schemas(self) -> list[dict]:
        """Generates OpenAI-compatible function schemas from instance methods."""
        tools = []
        type_map = {str: "string", int: "integer", float: "number", bool: "boolean", list: "array", dict: "object"}
        
        for name, fn in inspect.getmembers(self, predicate=inspect.ismethod):
            # Skip private methods and the schema generator itself
            if name.startswith("_") or name == "get_tool_schemas": 
                continue
            
            props = {
                arg: {"type": type_map.get(t, "string")} 
                for arg, t in fn.__annotations__.items() if arg != "return"
            }
            
            tools.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": fn.__doc__ or "",
                    "parameters": {
                        "type": "object",
                        "properties": props,
                        "required": list(props.keys())
                    }
                }
            })
        return tools

    def write_and_evaluate_kernel(self, name: str, content: str) -> dict:
        """Writes kernel code to a file and triggers the evaluation pipeline."""
        
        if name.lower().endswith(".cpp"):
            return {"error": "Direct .cpp file writing is blocked. Please write implementation into the hpp file like in baseline.hpp"}
        # Enforces a flat filename. Allows an optional extension but completely blocks slashes and directory traversal.
        if not re.match(r"^[a-zA-Z0-9_-]+(\.[a-zA-Z0-9]+)?$", name):
            return {"error": "Invalid name: must be a flat filename without subfolders (e.g., 'kernel' or 'kernel.cpp')."}
            
        file_path = self.sandbox_dir / name
        
        try:
            file_path.write_text(content, encoding="utf-8")
        except Exception as e:
            return {"error": f"Failed to write file: {str(e)}"}
            
        # Compile -> Test -> Benchmark 
        # (To be implemented)
        # Write method names

        print("Rebuilding registry with current sandbox files...")
        method_names = [f.stem for f in self.sandbox_dir.iterdir() if f.is_file() and str(f).endswith(".hpp") and f.stem != "registry"]
        print("Current kernel files in sandbox:", method_names)
        build_registry(method_names)
        output = run_make_in_sandbox()
        
        return {
            "success": True, 
            "filepath": name,
            "output": output
        }
    def read_file(self, name: str) -> dict:
        """Reads a file, strictly enforcing a flat filename (extensions allowed, subfolders blocked)."""
        # Whitelist: Allow alphanumeric, dashes, underscores, and an optional extension.
        # This completely blocks slashes (/) and directory traversal (../).
        if not re.match(r"^[a-zA-Z0-9_-]+(\.[a-zA-Z0-9]+)?$", name):
            return {"error": "Invalid name: must be a flat filename without subfolders (e.g., 'kernel' or 'kernel.cpp')."}
        
        file_path = self.sandbox_dir / name
        
        if not file_path.exists():
            return {"error": f"File '{name}' does not exist."}
        if not file_path.is_file():
            return {"error": f"'{name}' is a directory, not a file."}
            
        try:
            return {"success": True, "content": file_path.read_text(encoding="utf-8")}
        except Exception as e:
            return {"error": f"Failed to read file: {str(e)}"}
    def list_files(self) -> dict:
        try:
            # Iterate through the immediate directory only and filter out folders
            files = [f.name for f in self.sandbox_dir.iterdir() if f.is_file()]
            return {"success": True, "files": sorted(files)}
        except Exception as e:
            return {"error": f"Failed to list files: {str(e)}"}

def call_llm(conversation: list[dict], new_prompt: str, tools: list[dict], model: str, base_url: str, api_key: str = "") -> dict:
    """
    Appends a new prompt to the conversation and calls the LLM.
    Works with OpenAI, OpenRouter, or Ollama depending on base_url.
    """
    if new_prompt:
        conversation.append({"role": "user", "content": new_prompt})
        
    payload = {
        "model": model,
        "messages": conversation,
        "tools": tools,
        "stream": False
    }
    
    # Remove tools if empty to prevent API validation errors
    if not tools:
        del payload["tools"]

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    print(headers)

    # Ensure correct endpoint routing
    endpoint = base_url
    
    response = requests.post(endpoint, json=payload, headers=headers)
    response.raise_for_status()
    
    # Return the raw message object (contains 'content' and potentially 'tool_calls')
    return response.json()["choices"][0]["message"]

def process_tool_call(tool_call: dict, toolkit: KernelTools) -> dict:
    """
    Executes a specific tool call and returns the properly formatted 
    dictionary ready to be appended to the conversation history.
    """
    name = tool_call["function"]["name"]
    args = json.loads(tool_call["function"]["arguments"])
    
    try:
        # Dynamically call the method on the toolkit
        fn = getattr(toolkit, name)
        result = fn(**args)
        content = json.dumps(result)
    except Exception as e:
        content = json.dumps({"error": str(e)})
        
    # Return the exact dict format OpenAI expects for tool results.
    # You can intercept and modify this return value outside this function.
    return {
        "role": "tool",
        "tool_call_id": tool_call["id"],
        "name": name,
        "content": content
    }


def build_system_context(locked_files: set, summaries: list, graveyard: list) -> str:
    """Dynamically generates the system prompt based on current episodic memory."""
    prompt = "You are an autonomous kernel optimization agent. Your goal is to maximize performance.\n"
    prompt += f"Target Performance: {PERF_GOAL_MS}ms.\n\n"
    
    if locked_files:
        prompt += f"LOCKED FILES (Passed tests, do not modify): {', '.join(locked_files)}\n"
    if summaries:
        prompt += "SUCCESSFUL OPTIMIZATIONS:\n" + "\n".join(f"- {s}" for s in summaries) + "\n"
    if graveyard:
        prompt += "GRAVEYARD (Failed approaches, do not repeat):\n" + "\n".join(f"- {g}" for g in graveyard) + "\n"
        
    prompt += "\nRules: Read files to understand state. Use `write_and_evaluate_kernel` to propose, compile, and test optimizations in one step. Attempts to overwrite locked files will be blocked."
    prompt += "\n Read `baseline.hpp` for the desired function signature"
    return prompt

def run_autonomous_loop(spec_prompt: str, toolkit: KernelTools, model: str, base_url: str, api_key: str = ""):
    locked_files = set()
    summaries = []
    graveyard = []
    
    # Initialize the episodic context
    conversation = [{"role": "system", "content": build_system_context(locked_files, summaries, graveyard)}]
    conversation.append({"role": "user", "content": spec_prompt})
    
    local_retries = 0
    tools_schema = toolkit.get_tool_schemas()
    
    for iteration in range(MAX_ITERATIONS):
        print(f"\n--- Iteration {iteration + 1} | Retries: {local_retries}/{MAX_RETRIES} ---")
        
        # 1. Call LLM
        llm_msg = call_llm(conversation, None, tools_schema, model, base_url, api_key)
        conversation.append(llm_msg)
        
        if llm_msg.get("content"):
            print(f"Agent: {llm_msg['content']}")
            
        # 2. Process Tool Calls
        if "tool_calls" not in llm_msg or not llm_msg["tool_calls"]:
            conversation.append({"role": "user", "content": "Please continue optimizing or call the write_and_evaluate_kernel tool."})
            continue
            
        for t_call in llm_msg["tool_calls"]:
            name = t_call["function"]["name"]
            args = json.loads(t_call["function"]["arguments"])
            
            # A. WRITE-LOCK SAFEGUARD (Now checks the unified tool)
            if name == "write_and_evaluate_kernel" and args.get("filepath") in locked_files:
                print(f"[BLOCKED] Agent attempted to write to locked file: {args.get('filepath')}")
                conversation.append({
                    "role": "tool",
                    "tool_call_id": t_call["id"],
                    "name": name,
                    "content": json.dumps({"error": f"File {args.get('filepath')} is LOCKED. It already passed tests. Read-only."})
                })
                continue
                
            # B. EXECUTE TOOL
            tool_result_msg = process_tool_call(t_call, toolkit)
            conversation.append(tool_result_msg)
            
            # C. EVALUATION & EPISODIC MEMORY TRIGGER
            if name == "write_and_evaluate_kernel":
                result_data = json.loads(tool_result_msg["content"])
                
                # Success Path
                if result_data.get("success") == True:
                    perf = result_data.get("perf_ms", 999)
                    
                    target_file = result_data.get("filepath") or args.get("filepath") 
                    if target_file:
                        locked_files.add(target_file)
                        print(f"[LOCKED] {target_file} passed tests at {perf}ms.")
                        
                    # Append the prompt to the EXISTING conversation so the LLM can see its own work
                    conversation.append({
                        "role": "user", 
                        "content": f"The code passed tests! Perf: {perf}ms. Look at the code you just wrote above and write a 1-sentence summary of the exact optimization technique you successfully applied."
                    })
                    
                    # Call the LLM with the full context (no tools needed for the summary)
                    summary_msg = call_llm(conversation, None, [], model, base_url, api_key)
                    summaries.append(summary_msg.get("content", f"Successful optimization applied at {perf}ms."))
                    
                    if perf <= PERF_GOAL_MS:
                        print(f"\n[GOAL REACHED] Performance goal of {PERF_GOAL_MS}ms met!")
                        return
                        
                    # Context Reset: NOW we flush tokens, keeping only the episodic memory
                    print("[CONTEXT RESET] Optimization successful. Flushing tokens.")
                    conversation = [{"role": "system", "content": build_system_context(locked_files, summaries, graveyard)}]
                    conversation.append({"role": "user", "content": f"Last optimization succeeded. Current perf: {perf}ms. Propose the NEXT distinct optimization."})
                    local_retries = 0
                    
                # Failure Path
                else:
                    local_retries += 1
                    print(f"[FAILED] Test/Compile failed. Retry {local_retries}/{MAX_RETRIES}")
                    
                    if local_retries >= MAX_RETRIES:
                        # (Keep the Graveyard logic here exactly as it was)
                        conversation.append({
                            "role": "user", 
                            "content": f"We failed {MAX_RETRIES} times. Look at the code and error above. Write a 1-sentence post-mortem explaining why this specific approach failed so we don't repeat it."
                        })
                        pm_msg = call_llm(conversation, None, [], model, base_url, api_key)
                        graveyard.append(pm_msg.get("content", "Approach failed after max retries."))
                        
                        print("[GRAVEYARD] Approach abandoned. Flushing tokens.")
                        conversation = [{"role": "system", "content": build_system_context(locked_files, summaries, graveyard)}]
                        conversation.append({"role": "user", "content": "Previous approach failed and is in the graveyard. Start a completely NEW angle. Do not repeat failed methods."})
                        local_retries = 0
                        
                    else:
                        # ADD THIS: Explicitly nudge the model to fix the error it just received
                        conversation.append({
                            "role": "user",
                            "content": f"The evaluation failed (see tool output above). You have {MAX_RETRIES - local_retries} retries left for this approach. Please fix the exact error and call the evaluation tool again."
                        })
if __name__ == "__main__":
    # Example usage assuming toolkit, MODEL, and BASE_URL are defined
    #BASE_URL = "http://gx10:11434/v1/chat/completions" 
    BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
    MODEL = "gemini-2.5-flash-lite-preview-09-2025"
    API_KEY = os.getenv("OPENROUTER")
    
    toolkit = KernelTools()
    run_autonomous_loop("Optimize the kernel in baseline.hpp by writing more efficient versions of it with the same function signature. Create new files for each version" , toolkit, MODEL, BASE_URL, API_KEY)
