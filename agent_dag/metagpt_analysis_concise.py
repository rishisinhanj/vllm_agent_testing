import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import json
import re
from collections import Counter, defaultdict
import ast
import numpy as np
from datetime import datetime

class MetaGPTAnalyzer:
    """Analyzer for MetaGPT communication workflows"""
    
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.actions = []
    
    def parse_trace(self, trace_value):
        """Extract trajectory from MetaGPT log format"""
        if isinstance(trace_value, str):
            return trace_value
        else:
            return str(trace_value) if trace_value else ''
    
    def extract_actions(self, trajectory):
        """Extract agent actions and communications from MetaGPT log"""
        actions = []
        
        # Parse MetaGPT specific format
        # 1. Look for Human requirements first
        if 'FROM: Human TO:' in trajectory:
            content_match = re.search(r'FROM: Human.*?CONTENT:\s*(.*?)(?=------|$)', trajectory, re.DOTALL)
            if content_match:
                content = content_match.group(1).strip()
                actions.append({
                    'agent': 'Human',
                    'action_type': 'requirement_specification',
                    'tools': ['requirement_input'],
                    'content': content,
                    'content_length': len(content),
                    'has_code': False,
                    'has_test': False
                })
        
        # 2. Look for SimpleCoder responses - fixed regex (no newline requirement)
        coder_matches = re.findall(r'SimpleCoder:\s*(.*?)(?=--------------------------------------------------------------------------------|\nSimpleTester:|\nSimpleReviewer:|$)', trajectory, re.DOTALL)
        for content in coder_matches:
            content = content.strip()
            if content and len(content) > 10:
                tools = self._extract_metagpt_tools('SimpleCoder', content)
                actions.append({
                    'agent': 'SimpleCoder',
                    'action_type': 'code_implementation',
                    'tools': tools,
                    'content': content,
                    'content_length': len(content),
                    'has_code': 'def ' in content or 'import ' in content,
                    'has_test': False
                })
        
        # 3. Look for SimpleTester responses - fixed regex (no newline requirement)
        tester_matches = re.findall(r'SimpleTester:\s*(.*?)(?=--------------------------------------------------------------------------------|\nSimpleCoder:|\nSimpleReviewer:|$)', trajectory, re.DOTALL)
        for content in tester_matches:
            content = content.strip()
            if content and len(content) > 10:
                tools = self._extract_metagpt_tools('SimpleTester', content)
                actions.append({
                    'agent': 'SimpleTester',
                    'action_type': 'test_creation',
                    'tools': tools,
                    'content': content,
                    'content_length': len(content),
                    'has_code': 'def ' in content or 'import ' in content,
                    'has_test': 'test_' in content or 'assert ' in content
                })
        
        # 4. Look for SimpleReviewer responses - working regex
        reviewer_matches = re.findall(r'SimpleReviewer:\s*(.*?)(?=--------------------------------------------------------------------------------|\nSimpleCoder:|\nSimpleTester:|$)', trajectory, re.DOTALL)
        for content in reviewer_matches:
            content = content.strip()
            if content and len(content) > 10:
                tools = self._extract_metagpt_tools('SimpleReviewer', content)
                actions.append({
                    'agent': 'SimpleReviewer',
                    'action_type': 'code_review',
                    'tools': tools,
                    'content': content,
                    'content_length': len(content),
                    'has_code': False,
                    'has_test': False
                })
        
        return actions
    
    def _extract_metagpt_tools(self, agent, content):
        """Extract tools/capabilities used by MetaGPT agents"""
        tools = []
        
        if 'SimpleCoder' in agent:
            if 'import ' in content:
                tools.append('python_imports')
            if 'def ' in content:
                tools.append('function_definition')
            if 'argparse' in content:
                tools.append('cli_argument_parsing')
            if 'base64' in content:
                tools.append('base64_encoding')
            if 'class ' in content:
                tools.append('class_definition')
            tools.append('code_generation')
        
        elif 'SimpleTester' in agent:
            if 'pytest' in content:
                tools.append('pytest_framework')
            if 'assert ' in content:
                tools.append('assertion_testing')
            if 'test_' in content:
                tools.append('unit_testing')
            if 'unicode' in content.lower():
                tools.append('unicode_testing')
            tools.append('test_generation')
        
        elif 'SimpleReviewer' in agent:
            if 'comment' in content.lower():
                tools.append('code_review')
            if 'critical' in content.lower():
                tools.append('critical_analysis')
            if 'unicode' in content.lower() or 'utf-8' in content.lower():
                tools.append('encoding_analysis')
            if 'security' in content.lower():
                tools.append('security_review')
            tools.append('quality_assurance')
        
        elif agent == 'Human':
            tools.append('requirement_input')
        
        return tools if tools else ['communication']
    
    def _infer_action_type(self, agent, content):
        """Infer the type of action based on agent and content"""
        if 'SimpleCoder' in agent:
            if 'import ' in content and 'def ' in content:
                return 'code_implementation'
            else:
                return 'code_snippet'
        elif 'SimpleTester' in agent:
            return 'test_creation'
        elif 'SimpleReviewer' in agent:
            return 'code_review'
        elif agent == 'Human':
            return 'requirement_specification'
        else:
            return 'communication'
    
    def analyze_all(self):
        """Analyze all traces"""
        print(f"Analyzing {len(self.df)} traces...")
        
        for idx, row in self.df.iterrows():
            trajectory = self.parse_trace(row['trace'])
            actions = self.extract_actions(trajectory)
            
            for action in actions:
                self.actions.append({
                    'trace_id': idx,
                    'agent': action['agent'],
                    'tools': action['tools'],
                    'action_type': action['action_type'],
                    'content_length': action['content_length'],
                    'has_code': action['has_code'],
                    'has_test': action['has_test'],
                    'benchmark': row.get('benchmark_name', 'unknown')
                })
        
        print(f"Extracted {len(self.actions)} actions")
    
    def build_graph(self):
        """Build workflow graph for MetaGPT communication flow"""
        G = nx.DiGraph()
        agent_transitions = defaultdict(int)
        agent_tools = defaultdict(int)
        
        # Group by trace for transitions
        for trace_id in set(a['trace_id'] for a in self.actions):
            trace_actions = [a for a in self.actions if a['trace_id'] == trace_id]
            
            
            # Agent transitions
            for i in range(len(trace_actions) - 1):
                curr_agent = trace_actions[i]['agent']
                next_agent = trace_actions[i + 1]['agent']
                if curr_agent != next_agent:
                    agent_transitions[(curr_agent, next_agent)] += 1
            
            # Agent-tool usage
            for action in trace_actions:
                for tool in action['tools']:
                    agent_tools[(action['agent'], tool)] += 1
        
        # Add nodes
        agents = set(a['agent'] for a in self.actions)
        tools = set(tool for a in self.actions for tool in a['tools'])
        
        for agent in agents:
            G.add_node(agent, type='agent')
        for tool in tools:
            G.add_node(tool, type='tool')
        
        # Add edges
        for (agent1, agent2), weight in agent_transitions.items():
            G.add_edge(agent1, agent2, weight=weight, type='transition')
        
        for (agent, tool), weight in agent_tools.items():
            G.add_edge(agent, tool, weight=weight, type='usage')
        
        return G
    
    def visualize(self, G):
        """Create visualization for MetaGPT workflow"""
        plt.figure(figsize=(16, 12))
        
        # Separate nodes
        agents = [n for n in G.nodes() if G.nodes[n].get('type') == 'agent']
        tools = [n for n in G.nodes() if G.nodes[n].get('type') == 'tool']
        
        # Layout with Human at center
        pos = {}
        
        # Position Human at center if present
        if 'Human' in agents:
            pos['Human'] = (0, 0)
            other_agents = [a for a in agents if a != 'Human']
        else:
            other_agents = agents
        
        # Position other agents in a circle
        if other_agents:
            angles = np.linspace(0, 2*np.pi, len(other_agents), endpoint=False)
            for i, agent in enumerate(other_agents):
                pos[agent] = (2.5 * np.cos(angles[i]), 2.5 * np.sin(angles[i]))
        
        # Position tools in outer ring
        if tools:
            angles = np.linspace(0, 2*np.pi, len(tools), endpoint=False)
            for i, tool in enumerate(tools):
                pos[tool] = (4.5 * np.cos(angles[i]), 4.5 * np.sin(angles[i]))
        
        # Draw nodes with different colors for different agent types
        human_nodes = [n for n in agents if n == 'Human']
        coder_nodes = [n for n in agents if 'Coder' in n]
        tester_nodes = [n for n in agents if 'Tester' in n]
        reviewer_nodes = [n for n in agents if 'Reviewer' in n]
        
        if human_nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=human_nodes,
                                  node_color='gold', node_size=2000, alpha=0.9)
        if coder_nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=coder_nodes,
                                  node_color='lightblue', node_size=1800, alpha=0.8)
        if tester_nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=tester_nodes,
                                  node_color='lightcoral', node_size=1800, alpha=0.8)
        if reviewer_nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=reviewer_nodes,
                                  node_color='lightgreen', node_size=1800, alpha=0.8)
        
        # Draw tool nodes
        nx.draw_networkx_nodes(G, pos, nodelist=tools,
                              node_color='lightyellow', node_size=1200, alpha=0.7, node_shape='s')
        
        # Draw edges
        transition_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('type') == 'transition']
        usage_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('type') == 'usage']
        
        if transition_edges:
            weights = [G[u][v]['weight'] for u, v in transition_edges]
            max_w = max(weights) if weights else 1
            widths = [(w / max_w) * 5 + 1 for w in weights]
            nx.draw_networkx_edges(G, pos, edgelist=transition_edges,
                                  width=widths, edge_color='blue', alpha=0.7, arrows=True, arrowsize=25)
        
        if usage_edges:
            nx.draw_networkx_edges(G, pos, edgelist=usage_edges,
                                  width=2, edge_color='green', alpha=0.5, 
                                  style='dashed', arrows=True, arrowsize=20)
        
        # Labels
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
        
        # Legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gold', 
                   markersize=15, label='Human'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', 
                   markersize=12, label='SimpleCoder'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='lightcoral', 
                   markersize=12, label='SimpleTester'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen', 
                   markersize=12, label='SimpleReviewer'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='lightyellow', 
                   markersize=10, label='Tools'),
            Line2D([0], [0], color='blue', linewidth=3, label='Communication Flow'),
            Line2D([0], [0], color='green', linewidth=2, linestyle='--', label='Tool Usage')
        ]
        plt.legend(handles=legend_elements, loc='upper left', fontsize=10)
        
        plt.title('MetaGPT Agent Communication Workflow', fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        return plt
    
    def print_stats(self):
        """Print analysis statistics for MetaGPT"""
        print("\n=== METAGPT WORKFLOW ANALYSIS ===")
        
        # Agent counts
        agent_counts = Counter(a['agent'] for a in self.actions)
        print(f"Agent Activity:")
        for agent, count in agent_counts.most_common():
            avg_content = sum(a['content_length'] for a in self.actions if a['agent'] == agent) / count
            has_code_count = sum(1 for a in self.actions if a['agent'] == agent and a['has_code'])
            has_test_count = sum(1 for a in self.actions if a['agent'] == agent and a['has_test'])
            print(f"  {agent}: {count} actions (avg {avg_content:.0f} chars, {has_code_count} with code, {has_test_count} with tests)")
        
        # Tool counts
        all_tools = [tool for a in self.actions for tool in a['tools']]
        tool_counts = Counter(all_tools)
        print(f"\nTool Usage:")
        for tool, count in tool_counts.most_common():
            print(f"  {tool}: {count} uses")
        
        # Action types
        action_type_counts = Counter(a['action_type'] for a in self.actions)
        print(f"\nAction Types:")
        for action_type, count in action_type_counts.most_common():
            print(f"  {action_type}: {count} times")
        
        # Communication patterns
        workflows = []
        for trace_id in set(a['trace_id'] for a in self.actions):
            trace_agents = [a['agent'] for a in self.actions if a['trace_id'] == trace_id]
            if len(set(trace_agents)) > 1:
                # Remove consecutive duplicates but preserve order
                pattern_agents = []
                prev_agent = None
                for agent in trace_agents:
                    if agent != prev_agent:
                        pattern_agents.append(agent)
                        prev_agent = agent
                pattern = ' → '.join(pattern_agents)
                workflows.append(pattern)
        
        workflow_counts = Counter(workflows)
        print(f"\nCommunication Patterns:")
        for pattern, count in workflow_counts.most_common(5):
            print(f"  {pattern}: {count} times")
        
        # Development lifecycle analysis
        code_traces = sum(1 for a in self.actions if a['has_code'])
        test_traces = sum(1 for a in self.actions if a['has_test'])
        review_traces = sum(1 for a in self.actions if 'review' in a['action_type'])
        
        print(f"\nDevelopment Lifecycle:")
        print(f"  Code generation actions: {code_traces}")
        print(f"  Test creation actions: {test_traces}")
        print(f"  Code review actions: {review_traces}")

# Main execution
if __name__ == "__main__":
    # Run analysis - update the CSV path as needed
    analyzer = MetaGPTAnalyzer('MetaGPT.csv')
    analyzer.analyze_all()
    analyzer.print_stats()
    
    # Build and visualize graph
    G = analyzer.build_graph()
    print(f"\nGraph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    if G.number_of_nodes() > 0:
        plt = analyzer.visualize(G)
        plt.savefig('metagpt_workflow_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Save graph
        import networkx as nx
        nx.write_gexf(G, 'metagpt_workflow.gexf')
        print("✅ Saved MetaGPT workflow analysis")
    else:
        print("❌ No workflow data found")