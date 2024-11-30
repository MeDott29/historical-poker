import ollama
import networkx as nx
from typing import List, Tuple, Dict
import re
from colorama import init, Fore, Style
import threading
import time
import random
import numpy as np
from scipy.stats import entropy
from scipy.io import wavfile
from dataclasses import dataclass
import logging  # Added for performance monitoring
from fairlearn.metrics import MetricFrame, selection_rate  # Added for bias mitigation
import warnings
import os  # Added for security enhancements

init(autoreset=True)  # Initialize colorama

@dataclass
class ConversationMetrics:
    entropy: float
    coherence: float
    opposition_strength: float

    def validate(self):
        """Ensure metrics are within expected ranges"""
        if not 0 <= self.entropy <= 10:  # Typical entropy range
            raise ValueError(f"Entropy {self.entropy} outside expected range [0,10]")
        if not -1000 <= self.coherence <= 1000:  # Typical coherence range
            raise ValueError(f"Coherence {self.coherence} outside expected range [-1000,1000]")
        if not 0 <= self.opposition_strength <= 5:  # Typical opposition range
            raise ValueError(f"Opposition strength {self.opposition_strength} outside expected range [0,5]")

class NoiseChannel:
    def __init__(self, dimension: int = 512):
        self.dimension = dimension
        self.opposite_pairs = []
        self.noise_thread = None
        self.current_noise = np.zeros(dimension)
        self.truth_metrics = {"entropy": [], "coherence": [], "opposition_strength": []}
        self.stop_thread = False  # For graceful shutdown

    def generate_opposites(self, vector: np.ndarray) -> np.ndarray:
        return -vector + np.random.normal(0, 0.1, self.dimension)

    def measure_truth_factors(self) -> Dict[str, float]:
        entropy_score = entropy(np.abs(self.current_noise))
        coherence = np.mean([np.dot(a, b) for a, b in self.opposite_pairs[-10:]]) if self.opposite_pairs else 0
        opposition_strength = np.mean([np.linalg.norm(a + b) for a, b in self.opposite_pairs[-10:]]) if self.opposite_pairs else 0

        return {
            "entropy": entropy_score,
            "coherence": coherence,
            "opposition_strength": opposition_strength
        }

    def run_noise(self):
        while not self.stop_thread:
            # Generate base noise vector
            self.current_noise = np.random.normal(0, 1, self.dimension)

            # Generate its opposite
            opposite_noise = self.generate_opposites(self.current_noise)

            # Store the pair
            self.opposite_pairs.append((self.current_noise, opposite_noise))

            # Measure truth factors
            metrics = self.measure_truth_factors()
            for key, value in metrics.items():
                self.truth_metrics[key].append(value)

            time.sleep(0.1)
        print(f"{Fore.CYAN}Noise channel stopped.{Style.RESET_ALL}")

class DialogueEncoder:
    def __init__(self, noise_channel: NoiseChannel):
        self.noise_channel = noise_channel
        self.dialogue_buffer = []

    def encode_dialogue(self, message: Dict[str, str]) -> np.ndarray:
        # Convert dialogue to binary with truth metrics as weights
        metrics = self.noise_channel.measure_truth_factors()

        # Encode message content
        content_bytes = message['content'].encode('utf-8')
        role_bytes = message['role'].encode('utf-8')

        # Weight the encoding based on truth metrics
        weighted_data = np.frombuffer(content_bytes + role_bytes, dtype=np.uint8)
        weighted_data = weighted_data * (1 + metrics['entropy'])

        return weighted_data.astype(np.int16)

class Agent:
    def __init__(self, name: str, model: str, color: str, noise_channel: NoiseChannel):
        self.name = name
        self.model = model
        self.client = ollama.Client()
        self.conversation_history: List[Dict[str, str]] = []
        self.background_thoughts: List[str] = []
        self.color = color
        self.noise_channel = noise_channel
        self.role_description = ""  # Added for enhanced learning mechanisms

    def generate_response(self, user_input: str, context: str) -> str:
        # Incorporate noise channel metrics into response generation
        truth_metrics = self.noise_channel.measure_truth_factors()

        # Bias mitigation: Check for biased language
        if self.detect_bias(user_input):
            warnings.warn(f"{self.name} detected potential bias in user input.")
            user_input = self.mitigate_bias(user_input)

        prompt = f"""Generate a focused response (max 150 words) that:
- Addresses the specific question
- Avoids repeating information
- Builds on previous responses
- Stays within your role's expertise

As {self.name}, consider the following:

User Input: {user_input}
Context: {context}
Conversation History: {self.conversation_history[-5:]}
Current Truth Metrics:
- Entropy: {truth_metrics['entropy']:.2f}
- Coherence: {truth_metrics['coherence']:.2f}
- Opposition Strength: {truth_metrics['opposition_strength']:.2f}

Provide your response below:
"""
        response = self.client.generate(self.model, prompt=prompt)
        return response['response']

    def detect_bias(self, text: str) -> bool:
        # Simple bias detection logic (can be expanded)
        biased_terms = ["biased_term1", "biased_term2"]
        return any(term in text.lower() for term in biased_terms)

    def mitigate_bias(self, text: str) -> str:
        # Simple bias mitigation (can be expanded)
        for term in ["biased_term1", "biased_term2"]:
            text = text.replace(term, "[REDACTED]")
        return text

    def background_process(self, other_thoughts: List[str]):
        prompt = f"""As {self.name}, reflect on the recent conversation and other agents' thoughts:

Conversation history: {self.conversation_history[-5:]}
Other agents' thoughts: {other_thoughts}

Provide insights or ideas for improving future responses:
"""
        response = self.client.generate(self.model, prompt=prompt)
        self.background_thoughts.append(response['response'])

class NeuroSymbolicEngine(Agent):
    def __init__(self, model: str, noise_channel: NoiseChannel):
        super().__init__("NeuroSymbolicEngine", model, Fore.MAGENTA, noise_channel)
        self.role_description = "An engine that combines neural networks with symbolic reasoning."  # For enhanced learning
        self.knowledge_graph = nx.Graph()
        self.ontology = {}  # For improved knowledge graphs

    def extract_symbolic_knowledge(self, text: str) -> List[Tuple[str, str, str]]:
        prompt = f"""Extract symbolic knowledge triplets (subject, predicate, object) from the text.
Format each triplet as 'subject,predicate,object' on a new line:

{text}

Triplets:
"""
        response = self.client.generate(self.model, prompt=prompt)
        triplets = []
        for line in response['response'].split('\n'):
            parts = line.split(',')
            if len(parts) == 3:
                triplets.append(tuple(part.strip() for part in parts))
        return triplets

    def update_knowledge_graph(self, triplets: List[Tuple[str, str, str]]):
        for triplet in triplets:
            if len(triplet) == 3:
                subject, predicate, obj = triplet
                self.knowledge_graph.add_edge(subject, obj, relation=predicate)
                # Update ontology
                self.ontology.setdefault(subject, []).append((predicate, obj))
            else:
                print(f"{Fore.YELLOW}Skipping invalid triplet: {triplet}{Style.RESET_ALL}")

    def symbolic_reasoning(self, query: str) -> List[str]:
        if "path" in query.lower():
            start, end = re.findall(r'\b\w+\b', query)[-2:]
            try:
                path = nx.shortest_path(self.knowledge_graph, start, end)
                return [f"Path found: {' -> '.join(path)}"]
            except nx.NetworkXNoPath:
                return ["No path found"]
            except nx.NodeNotFound:
                return ["One or both nodes not found in the knowledge graph"]

        reasoning_prompt = f"""Perform symbolic reasoning on the following query:

{query}

Knowledge Graph Edges:
{list(self.knowledge_graph.edges(data=True))}

Ontology:
{self.ontology}

Reasoning:
"""
        response = self.client.generate(self.model, prompt=reasoning_prompt)
        return [response['response']]

class ConversationalAgent(Agent):
    def __init__(self, model: str, noise_channel: NoiseChannel):
        super().__init__("ConversationalAgent", model, Fore.YELLOW, noise_channel)
        self.role_description = "A friendly conversational agent who communicates in a casual, engaging manner."

    def generate_response(self, user_input: str, context: str) -> str:
        # Override parent method with 2000s texting style prompt
        truth_metrics = self.noise_channel.measure_truth_factors()

        # Bias mitigation: Check for biased language
        if self.detect_bias(user_input):
            warnings.warn(f"{self.name} detected potential bias in user input.")
            user_input = self.mitigate_bias(user_input)

        prompt = f"""You are a teenager texting in the 2000s. ALWAYS respond using:
- Lots of abbreviations (ur, r, u, thx, plz, etc)
- Multiple punctuation marks (!!! ???)
- Text emoticons like :) ;) :P xD
- No capital letters unless for emphasis
- Common phrases like "omg", "lol", "rofl", "brb"
- Shortened words (wat, wut, dis, dat)

User Text: {user_input}
Context: {context}

Respond in 2000s texting style:"""

        response = self.client.generate(self.model, prompt=prompt)
        return response['response']

class OverseerAgent(Agent):
    def __init__(self, model: str, noise_channel: NoiseChannel):
        super().__init__("OverseerAgent", model, Fore.GREEN, noise_channel)
        self.role_description = "An agent that oversees the conversation, ensuring clarity and providing decisions."

    def make_decision(self, responses: List[str]) -> str:
        prompt = f"""As the OverseerAgent, synthesize the following responses into a clear, concise decision:
1. Identify key insights from each response.
2. Remove redundant information.
3. Provide clear next steps or recommendations.
4. Keep response focused and brief.

Responses:
{responses}

Decision:
"""
        response = self.client.generate(self.model, prompt=prompt)
        return response['response']

class PerformanceMonitor:
    def __init__(self):
        logging.basicConfig(filename='performance.log', level=logging.INFO)
        self.start_time = time.time()

    def log_performance(self, event: str):
        elapsed = time.time() - self.start_time
        logging.info(f"{event} at {elapsed:.2f} seconds")

class SecurityManager:
    def __init__(self):
        self.allowed_users = ["user1", "user2"]  # Example allowed users
        self.logged_in_user = None

    def authenticate_user(self):
        username = input("Enter username: ")
        if username in self.allowed_users:
            # Simple two-factor authentication
            code = random.randint(100000, 999999)
            print(f"Authentication code sent: {code}")
            user_code = int(input("Enter authentication code: "))
            if user_code == code:
                self.logged_in_user = username
                print("Authentication successful.")
            else:
                print("Incorrect authentication code.")
                exit()
        else:
            print("User not recognized.")
            exit()

    def check_permissions(self, action: str):
        if self.logged_in_user is None:
            self.authenticate_user()
        # Add permission checks as needed

class NeuroSymbolicConversationalSystem:
    def __init__(self, model: str):
        self.noise_channel = NoiseChannel()
        self.neuro_symbolic_engine = NeuroSymbolicEngine(model, self.noise_channel)
        self.conversational_agent = ConversationalAgent(model, self.noise_channel)
        self.overseer_agent = OverseerAgent(model, self.noise_channel)
        self.agents = [self.neuro_symbolic_engine, self.conversational_agent, self.overseer_agent]
        self.background_thread = None
        self.dialogue_encoder = DialogueEncoder(self.noise_channel)
        self.audio_buffer = np.array([], dtype=np.int16)
        self.performance_monitor = PerformanceMonitor()
        self.security_manager = SecurityManager()
        self.stop_threads = False  # For graceful shutdown

    def initialize_knowledge(self, text: str):
        triplets = self.neuro_symbolic_engine.extract_symbolic_knowledge(text)
        self.neuro_symbolic_engine.update_knowledge_graph(triplets)
        print(f"{Fore.CYAN}Initialized knowledge graph with {len(triplets)} triplets.{Style.RESET_ALL}")
        self.performance_monitor.log_performance("Knowledge graph initialized")

    def background_discussion(self):
        while not self.stop_threads:
            for agent in self.agents:
                other_thoughts = [a.background_thoughts[-1] for a in self.agents if a != agent and a.background_thoughts]
                agent.background_process(other_thoughts)
            time.sleep(5)  # Wait for 5 seconds before the next background discussion
        print(f"{Fore.CYAN}Background discussion stopped.{Style.RESET_ALL}")

    def save_dialogue_audio(self, filename: str):
        # Convert accumulated dialogue to audio
        if len(self.audio_buffer) > 0:
            wavfile.write(filename, 44100, self.audio_buffer)
            self.performance_monitor.log_performance("Dialogue audio saved")

    def interactive_session(self):
        # Authenticate user
        self.security_manager.authenticate_user()

        print(f"{Fore.CYAN}Welcome to the enhanced NeuroSymbolic AI session with three agents. Type 'exit' to end the conversation.{Style.RESET_ALL}")

        # Start the background discussion thread
        self.background_thread = threading.Thread(target=self.background_discussion)
        self.background_thread.start()

        while True:
            user_input = input(f"{Fore.BLUE}Human: {Style.RESET_ALL}")
            if user_input.lower() == 'exit':
                break

            # Prepare context
            context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in self.neuro_symbolic_engine.conversation_history[-5:]])

            # Agents generate responses
            responses = []
            agents_responses = []
            for agent in self.agents[:2]:  # NeuroSymbolicEngine and ConversationalAgent
                response = agent.generate_response(user_input, context)
                responses.append(response)
                agents_responses.append({'agent': agent, 'response': response})
                print(f"{agent.color}{agent.name}: {response}{Style.RESET_ALL}")
                self.performance_monitor.log_performance(f"{agent.name} generated response")

            # Overseer makes decision
            overseer_decision = self.overseer_agent.make_decision(responses)
            print(f"{self.overseer_agent.color}{self.overseer_agent.name} (Decision): {overseer_decision}{Style.RESET_ALL}")
            self.performance_monitor.log_performance("OverseerAgent made a decision")

            # Update conversation histories
            for agent in self.agents:
                agent.conversation_history.append({"role": "human", "content": user_input})
                for res in agents_responses:
                    agent.conversation_history.append({"role": res['agent'].name, "content": res['response']})
                if agent == self.overseer_agent:
                    agent.conversation_history.append({"role": agent.name, "content": overseer_decision})

            # Display background thoughts periodically
            if random.random() < 0.3:  # 30% chance to show background thoughts
                print(f"\n{Fore.CYAN}Background Thoughts:{Style.RESET_ALL}")
                for agent in self.agents:
                    if agent.background_thoughts:
                        print(f"{agent.color}{agent.name}: {agent.background_thoughts[-1]}{Style.RESET_ALL}")
                        agent.background_thoughts.clear()

            # After each response, encode the dialogue
            for res in agents_responses:
                encoded_dialogue = self.dialogue_encoder.encode_dialogue({
                    "role": res['agent'].name,
                    "content": res['response']
                })
                self.audio_buffer = np.concatenate([self.audio_buffer, encoded_dialogue])

            # Encode Overseer's decision
            encoded_dialogue = self.dialogue_encoder.encode_dialogue({
                "role": self.overseer_agent.name,
                "content": overseer_decision
            })
            self.audio_buffer = np.concatenate([self.audio_buffer, encoded_dialogue])

            # Periodically save audio (e.g., every 10 seconds of audio)
            if len(self.audio_buffer) > 44100 * 10:  # Assuming 44100 Hz sample rate
                self.save_dialogue_audio("dialogue_encoding.wav")
                self.audio_buffer = np.array([], dtype=np.int16)

            # Performance monitoring
            self.performance_monitor.log_performance("End of interactive session loop")

        # Graceful shutdown
        self.stop_threads = True
        self.noise_channel.stop_thread = True
        self.background_thread.join()
        self.performance_monitor.log_performance("Interactive session ended")

        # Save any remaining audio
        if len(self.audio_buffer) > 0:
            self.save_dialogue_audio("dialogue_encoding_final.wav")

    def run(self):
        # Start noise thread
        self.noise_thread = threading.Thread(target=self.noise_channel.run_noise)
        self.noise_thread.start()

        initial_knowledge = """
The sky is blue. Water is composed of hydrogen and oxygen. 
Trees produce oxygen through photosynthesis. The Earth orbits around the Sun.
Humans need oxygen to breathe. Plants absorb carbon dioxide and release oxygen.
The moon orbits around the Earth. Gravity keeps planets in orbit.
"""
        self.initialize_knowledge(initial_knowledge)
        self.interactive_session()

if __name__ == "__main__":
    system = NeuroSymbolicConversationalSystem("qwen2.5:1.5b")
    system.run()