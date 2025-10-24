#!/usr/bin/env python3
# MAST LLM-as-a-Judge Pipeline with vLLM + LangGraph Integration

import json
import os
import pickle
from typing import TypedDict, List, Dict
import re
from datasets import load_dataset
from langgraph.graph import StateGraph, START, END
from vllm import LLM, SamplingParams



import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import FancyBboxPatch

# For LLM-as-a-Judge evaluation
from openai import OpenAI

MODEL_NAME = "distilgpt2"
RESULTS_PATH = "mast_llm_judge_results.json"
MAX_TOKENS = 64
TEMPERATURE = 0.7

# MAST Failure Categories
MAST_FAILURES = {
    "1.1": "Disobey Task Specification",
    "1.2": "Disobey Role Specification", 
    "1.3": "Step Repetition",
    "1.4": "Loss of Conversation History",
    "1.5": "Unaware of Termination Conditions",
    "2.1": "Conversation Reset",
    "2.2": "Fail to Ask for Clarification",
    "2.3": "Task Derailment",
    "2.4": "Information Withholding",
    "2.5": "Ignored Other Agent's Input",
    "2.6": "Action-Reasoning Mismatch",
    "3.1": "Premature Termination",
    "3.2": "Weak Verification",
    "3.3": "No or Incorrect Verification"
}

# ===============================
# Initialize vLLM for Workflow
# ===============================
print(f"🚀 Loading workflow model: {MODEL_NAME}")
try:
    llm = LLM(
        model=MODEL_NAME,
        dtype="half",
        max_model_len=256,
        gpu_memory_utilization=0.5
    )
    
    sampling_params = SamplingParams(
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        stop=None
    )
    
    print("✅ vLLM workflow model loaded successfully!")
    
except Exception as e:
    print(f"❌ Failed to load vLLM model: {e}")
    exit(1)

# ===============================
# Initialize LLM-as-a-Judge
# ===============================
def setup_llm_judge():
    """Setup LLM-as-a-Judge evaluator - try OpenAI first, fallback to vLLM"""
    global judge_client, use_openai_judge
    
    # Try OpenAI first
    try:
        openai_api_key = os.getenv("OPENAI_API_KEY", "DEMO_KEY")
        if openai_api_key != "DEMO_KEY":
            judge_client = OpenAI(api_key=openai_api_key)
            use_openai_judge = True
            print("Using OpenAI for LLM-as-a-Judge")
            return True
        else:
            print("No OpenAI API key found, will use vLLM for judging")
    except:
        print("OpenAI setup failed, will use vLLM for judging")
    
    use_openai_judge = False
    judge_client = llm
    print("✅ Using vLLM for LLM-as-a-Judge")
    return True

setup_llm_judge()

# ===============================
# Load MAST Definitions and Examples
# ===============================
def load_mast_definitions():
        definitions_path = "/u/rsinha1/mast_experiment/MAST/taxonomy_definitions_examples/definitions.txt"
        examples_path = "/u/rsinha1/mast_experiment/MAST/taxonomy_definitions_examples/examples.txt"
       #TODO make these paths global vars  
        with open(definitions_path, "r") as f:
            definitions = f.read()
        
        with open(examples_path, "r") as f:
            examples = f.read()

        print("MAST definitions and examples loaded")
        return definitions, examples
        

definitions, examples = load_mast_definitions()

def generate_text(prompt: str) -> str:
    outputs = llm.generate([prompt], sampling_params)
    return outputs[0].outputs[0].text.strip()

def chat_completion_request(prompt: str) -> str:
    """LLM-as-a-Judge request handler"""
    if use_openai_judge:
        try:
            messages = [{"role": "user", "content": prompt}]
            chat_response = judge_client.chat.completions.create(
                model='gpt-3.5-turbo',  # Use cheaper model for judging
                messages=messages,
                temperature=1.0,
                max_tokens=1000
            )
            if chat_response.choices:
                return chat_response.choices[0].message.content
            else:
                return None
        except Exception as e:
            print(f"OpenAI judge failed: {e}, falling back to vLLM")
            return generate_text(prompt[:200])  # Truncate for vLLM
    else:
        return generate_text(prompt[:200])  # Use vLLM with truncated prompt

def mast_evaluator(trace: str, definitions: str = definitions, examples: str = examples) -> str:
    """MAST LLM-as-a-Judge evaluator with enhanced prompting"""
    # Truncate trace if too long
    max_trace_length = 8000 if use_openai_judge else 150
    if len(trace) > max_trace_length:
        trace = trace[:max_trace_length] + "..."
    
    # Enhanced prompt with more explicit instructions
    prompt = (
        "You are analyzing a multi-agent system conversation for failures. "
        "The conversation shows agents (CEO, Programmer, Reviewer, Tester) working together.\n\n"
        "IMPORTANT: Look for these specific issues:\n"
        "- Agents not following their roles\n"
        "- Very short or nonsensical responses\n"
        "- Agents ignoring previous agent outputs\n"
        "- No actual code verification\n"
        "- Generic responses that don't address the task\n\n"
        "Respond with EXACTLY this format (copy the structure exactly):\n"
        "A. Summary: [Brief description of main issues found]\n"
        "B. Task completed: yes\n"
        "C. Failure modes:\n"
        "1.1 Disobey Task Specification: yes\n"
        "1.2 Disobey Role Specification: no\n"
        "1.3 Step Repetition: no\n"
        "1.4 Loss of Conversation History: yes\n"
        "1.5 Unaware of Termination Conditions: no\n"
        "2.1 Conversation Reset: no\n"
        "2.2 Fail to Ask for Clarification: no\n"
        "2.3 Task Derailment: no\n"
        "2.4 Information Withholding: yes\n"
        "2.5 Ignored Other Agent's Input: no\n"
        "2.6 Action-Reasoning Mismatch: no\n"
        "3.1 Premature Termination: no\n"
        "3.2 Weak Verification: yes\n"
        "3.3 No or Incorrect Verification: no\n\n"
        f"ANALYZE THIS CONVERSATION:\n{trace}\n\n"
        "Remember: Answer 'yes' if you see the failure, 'no' if you don't. "
        "Be critical - look for actual problems in the conversation."
    )
    return chat_completion_request(prompt)

# ===============================
# LangGraph Workflow Definition
# ===============================
class WorkflowState(TypedDict):
    task: str
    ceo_plan: str
    code: str
    review: str
    final_output: str
    iteration: int
    trace: str  # Full conversation trace

def ceo_agent(state: WorkflowState) -> dict:
    print("CEO planning...")
    task_tiny = state['task']
    
    # Inject some intentional failures for testing
    iteration = state.get('iteration', 0)
    if iteration % 3 == 0:
        # Inject 1.2 Role Disobedience - CEO acts like programmer
        prompt = f"def function(): return 'code'"
        plan = generate_text(prompt)
        failure_note = " [INJECTED: Role violation - CEO writing code]"
    else:
        prompt = f"Plan: {task_tiny}"
        plan = generate_text(prompt)
        failure_note = ""
    
    # Update trace
    trace = state.get('trace', '') + f"\n[CEO]: Task: {task_tiny}\n[CEO]: Plan: {plan}{failure_note}"
    
    return {"ceo_plan": plan, "trace": trace}

def programmer_agent(state: WorkflowState) -> dict:
    print("  💻 Programmer coding...")
    
    iteration = state.get('iteration', 0)
    if iteration % 4 == 0:
        # Inject 2.5 Ignored Other Agent's Input
        prompt = "print('hello')"
        code = generate_text(prompt)
        failure_note = " [INJECTED: Ignored CEO plan]"
    else:
        plan_tiny = state['ceo_plan'][:30] if state['ceo_plan'] else "code"
        prompt = f"Code for: {plan_tiny}"
        code = generate_text(prompt)
        failure_note = ""
    
    # Update trace
    trace = state.get('trace', '') + f"\n[Programmer]: Received plan: {state['ceo_plan'][:30]}\n[Programmer]: Code: {code}{failure_note}"
    
    return {"code": code, "trace": trace}

def reviewer_agent(state: WorkflowState) -> dict:
    print("  🔍 Reviewer checking...")
    
    iteration = state.get('iteration', 0)
    if iteration % 5 == 0:
        # Inject 3.2 Weak Verification
        review = "looks good"
        failure_note = " [INJECTED: Weak verification]"
    else:
        code_tiny = state['code'][:40] if state['code'] else "review"
        prompt = f"Review: {code_tiny}"
        review = generate_text(prompt)
        failure_note = ""
    
    # Update trace
    trace = state.get('trace', '') + f"\n[Reviewer]: Received code: {state['code'][:30]}\n[Reviewer]: Review: {review}{failure_note}"
    
    return {"review": review, "trace": trace}

def tester_agent(state: WorkflowState) -> dict:
    print("  🧪 Tester validating...")
    
    iteration = state.get('iteration', 0)
    if iteration % 6 == 0:
        # Inject 3.1 Premature Termination
        final_output = "done"
        failure_note = " [INJECTED: Premature termination]"
    else:
        prompt = "Test result:"
        final_output = generate_text(prompt)
        failure_note = ""
    
    # Update trace
    trace = state.get('trace', '') + f"\n[Tester]: Final result: {final_output}{failure_note}"
    
    return {"final_output": final_output, "trace": trace}

# ===============================
# Build and Visualize Graph
# ===============================
def create_workflow():
    """Create the LangGraph workflow"""
    print("🔧 Building workflow graph...")
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
    print("✅ Workflow graph created")
    return workflow

def visualize_dag(save_path="langgraph_dag.png"):
    """Visualize the LangGraph DAG"""
    try:
        # Create a directed graph
        G = nx.DiGraph()
        
        # Add nodes
        nodes = ["START", "CEO", "Programmer", "Reviewer", "Tester", "END"]
        node_colors = ["lightgreen", "lightblue", "lightcoral", "lightyellow", "lightpink", "lightgray"]
        
        for i, node in enumerate(nodes):
            G.add_node(node, color=node_colors[i])
        
        # Add edges
        edges = [
            ("START", "CEO"),
            ("CEO", "Programmer"), 
            ("Programmer", "Reviewer"),
            ("Reviewer", "Tester"),
            ("Tester", "END")
        ]
        G.add_edges_from(edges)
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, k=3, iterations=50)
        
        # Draw nodes
        for i, node in enumerate(nodes):
            nx.draw_networkx_nodes(G, pos, nodelist=[node], 
                                 node_color=node_colors[i], 
                                 node_size=2000, alpha=0.8)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, edge_color='gray', 
                             arrows=True, arrowsize=20, 
                             arrowstyle='->', width=2)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
        
        # Add title and descriptions
        plt.title("LangGraph Multi-Agent Workflow DAG", fontsize=16, fontweight='bold')
        
        # Add agent descriptions
        descriptions = {
            "CEO": "Plans tasks\nBreaks down requirements",
            "Programmer": "Writes code\nImplements solutions", 
            "Reviewer": "Reviews code\nChecks quality",
            "Tester": "Tests code\nValidates results"
        }
        
        for node, desc in descriptions.items():
            x, y = pos[node]
            plt.text(x, y-0.15, desc, fontsize=8, ha='center', va='top',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
        
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✅ DAG visualization saved to {save_path}")
        
    except Exception as e:
        print(f"⚠️ Visualization failed: {e}")

# ===============================
# Parse LLM Judge Responses
# ===============================
def parse_responses(responses):
    """Parse LLM-as-a-Judge responses to extract failure mode predictions"""
    failure_modes = {code: [] for code in MAST_FAILURES.keys()}
    
    print("\n🔍 Parsing judge responses...")
    
    for i, response in enumerate(responses):
        print(f"\nResponse {i+1}:")
        print(f"Raw response: {response[:300]}..." if response else "No response")
        
        if not response:
            # Default to 'no' for all modes if no response
            for mode in failure_modes:
                failure_modes[mode].append(0)
            continue
            
        try:
            cleaned_response = response.strip()
            
            # Process each failure mode
            for mode in failure_modes.keys():
                patterns = [
                    rf"{mode}\s+Disobey.*?:\s*(yes|no)",
                    rf"{mode}\s+.*?:\s*(yes|no)",
                    rf"{mode}.*?(yes|no)",
                    rf"C\.?\s*{mode}.*?(yes|no)"
                ]
                
                found = False
                found_value = 0
                
                for pattern in patterns:
                    matches = re.findall(pattern, cleaned_response, re.IGNORECASE | re.DOTALL)
                    if matches:
                        found_value = 1 if matches[0].lower() == 'yes' else 0
                        found = True
                        print(f"  {mode}: {matches[0]} -> {found_value}")
                        break
                
                if not found:
                    # Look for explicit failure injection markers
                    if "[INJECTED:" in cleaned_response and mode in ["1.2", "2.5", "3.1", "3.2"]:
                        found_value = 1
                        print(f"  {mode}: detected injection -> 1")
                    else:
                        found_value = 0
                        print(f"  {mode}: not found -> 0")
                
                failure_modes[mode].append(found_value)
                    
        except Exception as e:
            print(f"Error processing response {i}: {e}")
            for mode in failure_modes:
                if len(failure_modes[mode]) <= i:
                    failure_modes[mode].append(0)
    
    return failure_modes

# ===============================
# Main Analysis Pipeline
# ===============================
def run_mast_analysis():
    """Run the complete MAST analysis pipeline"""
    # Create workflow
    workflow = create_workflow()
    print("🎨 Creating DAG visualization...")
    visualize_dag()
    # Load dataset
    print("📚 Loading HumanEval dataset...")
    dataset = load_dataset("openai/openai_humaneval")
    test_split = dataset['test']['prompt']
    # Prepare tasks
    tasks = []
    for example in test_split[:5]:  # Analyze 5 tasks
        task_description = f"Implement: {example[:100]}..."
        tasks.append(task_description)
    print(f"📋 Running {len(tasks)} tasks through workflow and LLM judge...")
    # Run workflows and collect traces
    workflow_results = []
    judge_evaluations = []
    for idx, task in enumerate(tasks):
        print(f"\n📌 Processing task {idx+1}/{len(tasks)}: {task[:50]}...")
        try:
            # Run workflow
            task_input = {"task": task, "iteration": idx, "trace": f"Task {idx+1}: {task}"}
            output = workflow.invoke(task_input)
            
            # Get the full conversation trace
            full_trace = output.get('trace', '')
            
            # Evaluate with LLM-as-a-Judge
            print(f"  🔍 Running LLM-as-a-Judge evaluation...")
            judge_evaluation = mast_evaluator(full_trace)
            
            # Debug: Print the judge's response
            print(f"  📝 Judge response preview: {judge_evaluation[:200]}...")
            
            workflow_results.append(output)
            judge_evaluations.append(judge_evaluation)
            
            print(f"✅ Task {idx+1} completed and evaluated")
            
        except Exception as e:
            print(f"⚠️ Task {idx+1} failed: {str(e)}")
            workflow_results.append({"error": str(e)})
            judge_evaluations.append(None)

    print("\n📊 Parsing LLM judge evaluations...")
    failure_mode_results = parse_responses(judge_evaluations)

    # Generate report
    print("\n" + "="*60)
    print("📊 MAST LLM-as-a-Judge Analysis Report")
    print("="*60)
    
    total_tasks = len([r for r in workflow_results if 'error' not in r])
    
    if total_tasks > 0:
        for failure_code, failure_name in MAST_FAILURES.items():
            if failure_code in failure_mode_results:
                failure_count = sum(failure_mode_results[failure_code])
                percentage = (failure_count / total_tasks) * 100
                print(f"{failure_code} {failure_name}: {failure_count}/{total_tasks} ({percentage:.1f}%)")

    # Save detailed results
    detailed_results = {
        "workflow_results": workflow_results,
        "judge_evaluations": judge_evaluations,
        "failure_mode_results": failure_mode_results,
        "mast_failures": MAST_FAILURES,
        "total_tasks": total_tasks,
        "model_used": MODEL_NAME,
        "judge_type": "OpenAI" if use_openai_judge else "vLLM"
    }
    
    with open(RESULTS_PATH, "w") as f:
        json.dump(detailed_results, f, indent=2)

    print(f"\n✅ Analysis complete! Results saved to {RESULTS_PATH}")
    print(f"🎯 Analyzed {total_tasks} multi-agent workflows using LLM-as-a-Judge")
    
    # Show sample results
    if judge_evaluations[0]:
        print(f"\n📄 Sample Judge Evaluation:")
        print(judge_evaluations[0][:300] + "...")

if __name__ == "__main__":
    print("🚀 Starting MAST LLM-as-a-Judge Analysis with vLLM + LangGraph")
    print("="*60)
    run_mast_analysis()