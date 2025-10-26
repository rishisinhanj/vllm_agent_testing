import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import json
import re
from collections import Counter, defaultdict
import ast

# USE THIS ONE

class MagenticAnalyzer:
    """Concise analyzer for MagenticOne workflows"""
    
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.actions = []
    
    def parse_trace(self, trace_value):
        # confirming that trace value is a string
        if isinstance(trace_value, str):
            data = ast.literal_eval(trace_value)
        else:
            data = trace_value
        return data.get('trajectory', '')
    
    def extract_actions(self, trajectory):
        """Extract agent actions and tools from trajectory using structured patterns"""
        actions = []
        
        # Split by agent headers eg. ---------- AgentName ---------
        sections = re.split(r'---------- ([^-\n]+) ----------', trajectory) 
        for i in range(1, len(sections), 2): # going in pairs to get each instance of agent + content
            if i + 1 < len(sections):
                agent = sections[i].strip()
                content = sections[i + 1].strip()

                # tools and actions is a list 
                tools = self._extract_structured_tools(agent, content) # this is specific to MagenticOne!!
                actions_found = self._extract_agent_actions(agent, content)
                
                actions.append({
                    'agent': agent,
                    'tools': tools,
                    'actions': actions_found,
                    'content_length': len(content),
                    'has_content': len(content) > 20
                })
        
        return actions
    
    def _extract_structured_tools(self, agent, content):
        """Extract tools based on agent type and structured patterns"""
        tools = []
        
        # WebSurfer specific tools
        if 'WebSurfer' in agent:
            if re.search(r"typed '[^']+' into the browser search bar", content):
                tools.append('web_search')
            if re.search(r"typed '[^']+' into the browser address bar", content):
                tools.append('web_navigate')
            if re.search(r"clicked '[^']+'", content):
                tools.append('web_click')
            if "The web browser is open to the page" in content:
                tools.append('web_page_access')
            if "scrolled" in content.lower():
                tools.append('web_scroll')
        
        # ComputerTerminal specific tools
        elif 'ComputerTerminal' in agent:
            if '```python' in content:
                tools.append('python_execution')
            if '```sh' in content:
                tools.append('shell_execution')
            if 'pip install' in content:
                tools.append('package_installation')
        
        # FileSurfer specific tools
        elif 'FileSurfer' in agent:
            if any(word in content.lower() for word in ['read', 'open', 'load']):
                tools.append('file_read')
            if any(word in content.lower() for word in ['write', 'save', 'create']):
                tools.append('file_write')
            if 'filepath:' in content:
                tools.append('file_operation')
        
        # Assistant reasoning
        elif 'Assistant' in agent and len(content) > 50:
            tools.append('reasoning')
        
        # MagenticOneOrchestrator coordination
        elif 'MagenticOneOrchestrator' in agent:
            if 'FINAL ANSWER:' in content:
                tools.append('final_answer')
            elif 'Here is the plan to follow' in content:
                tools.append('task_planning')
            elif len(content) > 30:
                tools.append('agent_coordination')
        
        return tools
    
    def _extract_agent_actions(self, agent, content):
        """Extract specific actions performed by agents"""
        actions = []
        
        if 'WebSurfer' in agent:
            # Search actions
            search_matches = re.findall(r"typed '([^']+)' into the browser search bar", content)
            for query in search_matches:
                actions.append(f"search: {query[:50]}")
            
            # Navigation actions
            nav_matches = re.findall(r"typed '([^']+)' into the browser address bar", content)
            for url in nav_matches:
                actions.append(f"navigate: {url[:50]}")
            
            # Click actions
            click_matches = re.findall(r"clicked '([^']+)'", content)
            for element in click_matches:
                actions.append(f"click: {element[:30]}")
        
        elif 'ComputerTerminal' in agent:
            # Extract code snippets
            python_matches = re.findall(r'```python\n(.*?)\n```', content, re.DOTALL)
            for code in python_matches:
                first_line = code.split('\n')[0][:50]
                actions.append(f"python: {first_line}")
            
            shell_matches = re.findall(r'```sh\n(.*?)\n```', content, re.DOTALL)
            for cmd in shell_matches:
                first_line = cmd.split('\n')[0][:50]
                actions.append(f"shell: {first_line}")
        
        elif 'MagenticOneOrchestrator' in agent:
            if 'FINAL ANSWER:' in content:
                answer = content.split('FINAL ANSWER:')[-1].strip()[:50]
                actions.append(f"final_answer: {answer}")
            elif 'Here is the plan' in content:
                actions.append("task_planning")
        
        return actions
    
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
                    'actions': action['actions'],
                    'content_length': action['content_length'],
                    'benchmark': row.get('benchmark_name', 'unknown')
                })
        
        print(f"Extracted {len(self.actions)} actions")
    
    def build_graph(self):
        # the edge weights are based on frequency of agent to agent interactions 
        """Build workflow graph"""
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
        """Create visualization"""
        plt.figure(figsize=(14, 10))
        
        # Separate nodes
        agents = [n for n in G.nodes() if G.nodes[n].get('type') == 'agent']
        tools = [n for n in G.nodes() if G.nodes[n].get('type') == 'tool']
        
        # Layout
        pos = {}
        if agents:
            agent_pos = nx.circular_layout({a: i for i, a in enumerate(agents)})
            for a, (x, y) in agent_pos.items():
                pos[a] = (x - 1.5, y * 2)
        
        if tools:
            tool_pos = nx.circular_layout({t: i for i, t in enumerate(tools)})
            for t, (x, y) in tool_pos.items():
                pos[t] = (x + 1.5, y * 1.5)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, nodelist=agents,
                              node_color='lightblue', node_size=1500, alpha=0.8)
        nx.draw_networkx_nodes(G, pos, nodelist=tools,
                              node_color='lightgreen', node_size=1000, alpha=0.8)
        
        # Draw edges
        transition_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('type') == 'transition']
        usage_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('type') == 'usage']
        
        if transition_edges:
            weights = [G[u][v]['weight'] for u, v in transition_edges]
            max_w = max(weights) if weights else 1
            widths = [(w / max_w) * 4 + 1 for w in weights]
            nx.draw_networkx_edges(G, pos, edgelist=transition_edges,
                                  width=widths, edge_color='blue', alpha=0.7, arrows=True)
        
        if usage_edges:
            weights = [G[u][v]['weight'] for u, v in usage_edges]
            max_w = max(weights) if weights else 1
            widths = [(w / max_w) * 3 + 1 for w in weights]
            nx.draw_networkx_edges(G, pos, edgelist=usage_edges,
                                  width=widths, edge_color='green', alpha=0.6, 
                                  style='dashed', arrows=True)
        
        # Labels
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
        
        plt.title('MagenticOne Workflow Graph', fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        return plt
    
    def print_stats(self):
        """Print analysis statistics"""
        print("\n=== WORKFLOW ANALYSIS ===")
        
        # Agent counts
        agent_counts = Counter(a['agent'] for a in self.actions)
        print(f"Agent Activity:")
        for agent, count in agent_counts.most_common():
            avg_content = sum(a['content_length'] for a in self.actions if a['agent'] == agent) / count
            print(f"  {agent}: {count} actions (avg {avg_content:.0f} chars)")
        
        # Tool counts
        all_tools = [tool for a in self.actions for tool in a['tools']]
        tool_counts = Counter(all_tools)
        print(f"\nTool Usage:")
        for tool, count in tool_counts.most_common():
            print(f"  {tool}: {count} uses")
        
        # Specific actions
        all_actions = [action for a in self.actions for action in a['actions']]
        if all_actions:
            action_counts = Counter(all_actions)
            print(f"\nSpecific Actions (top 10):")
            for action, count in action_counts.most_common(10):
                print(f"  {action}: {count} times")
        
        # Agent-tool combinations
        print(f"\nAgent-Tool Combinations:")
        agent_tool_pairs = []
        for a in self.actions:
            for tool in a['tools']:
                agent_tool_pairs.append(f"{a['agent']} → {tool}")
        
        pair_counts = Counter(agent_tool_pairs)
        for pair, count in pair_counts.most_common(10):
            print(f"  {pair}: {count} times")
        
        # Workflow patterns
        workflows = []
        for trace_id in set(a['trace_id'] for a in self.actions):
            trace_agents = [a['agent'] for a in self.actions if a['trace_id'] == trace_id]
            if len(set(trace_agents)) > 1:  # Multiple unique agents
                pattern = ' → '.join(dict.fromkeys(trace_agents))  # Remove consecutive duplicates
                workflows.append(pattern)
        
        workflow_counts = Counter(workflows)
        print(f"\nCommon Workflows:")
        for pattern, count in workflow_counts.most_common(5):
            print(f"  {pattern}: {count} times")

if __name__ == "__main__":
    # Run analysis
    analyzer = MagenticAnalyzer('magentic_dataframe.csv')
    analyzer.analyze_all()
    analyzer.print_stats()
    
    # Build and visualize graph
    G = analyzer.build_graph()
    print(f"\nGraph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    plt = analyzer.visualize(G)
    plt.savefig('magentic_concise_workflow.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Saved concise workflow analysis")