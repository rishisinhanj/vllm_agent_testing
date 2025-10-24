#!/usr/bin/env python3
# Single-process vLLM version - no API server needed

import json
from typing import TypedDict
from datasets import load_dataset
from langgraph.graph import StateGraph, START, END

# Try to import vLLM - if it fails, we'll know immediately
try:
    from vllm import LLM, SamplingParams
    print("vLLM imported successfully")
except ImportError as e:
    print(f"❌ vLLM import failed: {e}")
    print("💡 Make sure you're in the correct environment with vLLM installed")
    exit(1)

# ===============================
# Configuration
# ===============================
MODEL_NAME = "distilgpt2"  # Much smaller model (82M vs 124M parameters)
RESULTS_PATH = "delta_workflow_results.json"
MAX_TOKENS = 64  # Very short outputs
TEMPERATURE = 0.7

# ===============================
# Initialize vLLM
# ===============================
print(f"Loading model: {MODEL_NAME}")
try:
    # Initialize vLLM with minimal settings
    llm = LLM(
        model=MODEL_NAME,
        dtype="half",
        max_model_len=256,  # Very small context window
        gpu_memory_utilization=0.5  # Use less GPU memory
    )
    
    # Create sampling parameters
    sampling_params = SamplingParams(
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        stop=None
    )
    
    print("vLLM model loaded successfully!")
    
except Exception as e:
    print(f"Failed to load vLLM model: {e}")
    exit(1)

# ===============================
# Helper function to generate text
# ===============================
def generate_text(prompt: str) -> str:
    """Generate text using vLLM"""
    try:
        outputs = llm.generate([prompt], sampling_params)
        return outputs[0].outputs[0].text.strip()
    except Exception as e:
        print(f"Generation failed: {e}")
        return f"Error: {str(e)}"

# ===============================
# Load dataset
# ===============================
print("Loading HumanEval dataset...")
dataset = load_dataset("openai/openai_humaneval")
test_split = dataset['test']['prompt']

tasks_for_chatdev = []
for example in test_split:
    task_description = f"Implement the Python function as described below: {example} Make sure the implementation passes all test cases."
    tasks_for_chatdev.append(task_description)

print(f"📋 Loaded {len(tasks_for_chatdev)} tasks")

# ===============================
# LangGraph Workflow Definition
# ===============================
class WorkflowState(TypedDict):
    task: str
    ceo_plan: str
    code: str
    review: str
    final_output: str

def ceo_agent(state: WorkflowState) -> dict:
    print("  👔 CEO planning...")
    # Very short prompt
    task_tiny = state['task'][:50] + "..." if len(state['task']) > 50 else state['task']
    prompt = f"Plan: {task_tiny}"
    plan = generate_text(prompt)
    return {"ceo_plan": plan}

def programmer_agent(state: WorkflowState) -> dict:
    print("  💻 Programmer coding...")
    # Very minimal
    plan_tiny = state['ceo_plan'][:30] if state['ceo_plan'] else "code"
    prompt = f"Code for: {plan_tiny}"
    code = generate_text(prompt)
    return {"code": code}

def reviewer_agent(state: WorkflowState) -> dict:
    print("  🔍 Reviewer checking...")
    # Very minimal
    code_tiny = state['code'][:40] if state['code'] else "review"
    prompt = f"Review: {code_tiny}"
    review = generate_text(prompt)
    return {"review": review}

def tester_agent(state: WorkflowState) -> dict:
    print("  🧪 Tester validating...")
    # Very minimal
    prompt = "Test result:"
    final_output = generate_text(prompt)
    return {"final_output": final_output}

# ===============================
# Build Graph
# ===============================
print(" Building workflow graph...")
builder = StateGraph(WorkflowState)
builder.add_node("ceo", ceo_agent)
builder.add_node("programmer", programmer_agent)
builder.add_node("reviewer", reviewer_agent)
builder.add_node("tester", tester_agent)

builder.add_edge(START, "ceo")
builder.add_edge("ceo", "programmer")
builder.add_edge("programmer", "reviewer")
builder.add_edge("reviewer", "tester")
builder.add_edge("tester", END)

workflow = builder.compile()

# ===============================
# Run workflows
# ===============================
print("🚀 Starting workflow execution...")
results = []

# process all tasks
tasks_to_process = tasks_for_chatdev


for idx, task in enumerate(tasks_to_process):
    task_input = {"task": task}
    print(f"\n Running task {idx+1}/{len(tasks_to_process)}: {task[:80]}...")
    try:
        output = workflow.invoke(task_input)
        results.append({
            "task_index": idx,
            "task": task,
            "ceo_plan": output.get("ceo_plan", ""),
            "code": output.get("code", ""),
            "review": output.get("review", ""),
            "final_output": output.get("final_output", "")
        })
        print(f" Task {idx+1} completed successfully")
    except Exception as e:
        print(f"⚠️ Task {idx+1} failed: {str(e)}")
        results.append({"task_index": idx, "task": task, "error": str(e)})

# ===============================
# Save results
# ===============================
with open(RESULTS_PATH, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n✅ All tasks processed. Results saved to {RESULTS_PATH}")
print("🎉 Workflow complete!")

# Show a sample result
if results:
    print(f"\n📄 Sample result:")
    print(f"Task: {results[0]['task'][:100]}...")
    print(f"CEO Plan: {results[0]['ceo_plan'][:100]}...")
    print(f"Code: {results[0]['code'][:100]}...")
