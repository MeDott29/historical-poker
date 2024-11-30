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

init(autoreset=True)

@dataclass
class ConversationMetrics:
    entropy: float
    coherence: float
    opposition_strength: float

    def validate(self):
        """Validate that metrics are within expected ranges."""
        if not 0 <= self.entropy <= 10:
            raise ValueError(f"Entropy {self.entropy} is not in range [0, 10]")
        if not -1000 <= self.coherence <= 1000:
            raise ValueError(f"Coherence {self.coherence} is not in range [-1000, 1000]")
        if not 0 <= self.opposition_strength <= 5:
            raise ValueError(f"Opposition Strength {self.opposition_strength} is not in range [0, 5]")

class NoiseChannel:
    """Simulates environmental noise and tracks interaction metrics."""

    def __init__(self, dimension: int = 512):
        self.dimension = dimension
        self.opposite_pairs: List[Tuple[np.ndarray, np.ndarray]] = []
        self.current_noise = np.zeros(dimension)
        self.truth_metrics = {"entropy": [], "coherence": [], "opposition_strength": []}
        self.thread = None

    def generate_opposites(self, vector: np.ndarray) -> np.ndarray:
        """Generate a noisy opposite vector."""
        return -vector + np.random.normal(0, 0.1, self.dimension)

    def measure_truth_factors(self) -> ConversationMetrics:
        """Calculate metrics like entropy, coherence, and opposition strength."""
        entropy_score = entropy(np.abs(self.current_noise))
        coherence = (
            np.mean([np.dot(a, b) for a, b in self.opposite_pairs[-10:]])
            if self.opposite_pairs
            else 0
        )
        opposition_strength = (
            np.mean([np.linalg.norm(a + b) for a, b in self.opposite_pairs[-10:]])
            if self.opposite_pairs
            else 0
        )

        metrics = ConversationMetrics(
            entropy=entropy_score, coherence=coherence, opposition_strength=opposition_strength
        )
        metrics.validate()
        return metrics

    def run_noise(self):
        """Continuously generates noise and updates metrics."""
        while True:
            self.current_noise = np.random.normal(0, 1, self.dimension)
            opposite_noise = self.generate_opposites(self.current_noise)
            self.opposite_pairs.append((self.current_noise, opposite_noise))

            metrics = self.measure_truth_factors()
            for key, value in vars(metrics).items():
                self.truth_metrics[key].append(value)
            time.sleep(0.1)  # Cycle every 100ms

class CommonContext:
    """Shared context for agents to collaborate and align."""

    def __init__(self):
        self.conversation_history: List[Dict[str, str]] = []
        self.global_metrics: ConversationMetrics = None

    def update_metrics(self, metrics: ConversationMetrics):
        """Update shared truth metrics."""
        self.global_metrics = metrics

class DialogueEncoder:
    """Encodes dialogue into a format that can be shared or stored."""

    def __init__(self, noise_channel: NoiseChannel):
        self.noise_channel = noise_channel
        self.audio_buffer = np.array([], dtype=np.int16)

    def encode_dialogue(self, message: Dict[str, str]) -> np.ndarray:
        """Encode dialogue message into an audio-like binary format."""
        metrics = self.noise_channel.measure_truth_factors()
        content_bytes = message['content'].encode('utf-8')
        role_bytes = message['role'].encode('utf-8')
        weighted_data = np.frombuffer(content_bytes + role_bytes, dtype=np.uint8)
        weighted_data = weighted_data * (1 + metrics.entropy)
        return weighted_data.astype(np.int16)

    def append_to_buffer(self, encoded_data: np.ndarray):
        """Append encoded data to the audio buffer."""
        self.audio_buffer = np.concatenate([self.audio_buffer, encoded_data])

    def save_audio(self, filename: str):
        """Save the encoded dialogue buffer as an audio file."""
        if self.audio_buffer.size > 0:
            wavfile.write(filename, 44100, self.audio_buffer)
            self.audio_buffer = np.array([], dtype=np.int16)

class Agent:
    """Base class for an agent that generates responses collaboratively."""

    def __init__(self, name: str, model: str, color: str, noise_channel: NoiseChannel, shared_context: CommonContext):
        self.name = name
        self.model = model
        self.color = color
        self.noise_channel = noise_channel
        self.shared_context = shared_context
        self.client = ollama.Client()
        self.conversation_history = []
        self.background_thoughts = []

    def generate_response(self, user_input: str) -> str:
        """Generate a response using shared context and metrics."""
        truth_metrics = self.noise_channel.measure_truth_factors()
        self.shared_context.update_metrics(truth_metrics)
        context = "\n".join(
            [f"{entry['role']}: {entry['content']}" for entry in self.shared_context.conversation_history[-5:]]
        )
        prompt = f"""
Role: {self.name}

Collaborate with other agents to produce a meaningful and accurate response.

- Shared Context:
{context}

- Metrics:
  - Entropy: {truth_metrics.entropy:.2f}
  - Coherence: {truth_metrics.coherence:.2f}
  - Opposition Strength: {truth_metrics.opposition_strength:.2f}

User Input: {user_input}

Constraints:
- Address the user's input directly.
- Build upon the shared context and other agents' contributions.
- Stay within your expertise and role.
- Keep the response succinct (max 150 words).

Provide your response below:
"""
        response = self.client.generate(self.model, prompt=prompt)
        return response["response"].strip()

    def background_process(self, other_thoughts: List[str]):
        """Generate background thoughts based on other agents' thoughts."""
        context = "\n".join(
            [f"{entry['role']}: {entry['content']}" for entry in self.shared_context.conversation_history[-5:]]
        )
        prompt = f"""
Role: {self.name}

Reflect on the recent conversation and other agents' thoughts.

- Shared Context:
{context}

- Other Agents' Thoughts:
{other_thoughts}

Task:
- Provide insights or ideas for improving future responses.
- Consider how you can better collaborate with the other agents.

Provide your thoughts below:
"""
        response = self.client.generate(self.model, prompt=prompt)
        self.background_thoughts.append(response["response"].strip())

class NeuroSymbolicEngine(Agent):
    """Agent specializing in symbolic reasoning and knowledge extraction."""

    def __init__(self, name: str, model: str, noise_channel: NoiseChannel, shared_context: CommonContext):
        super().__init__(name, model, Fore.MAGENTA, noise_channel, shared_context)
        self.knowledge_graph = nx.DiGraph()

    def extract_symbolic_knowledge(self, text: str) -> List[Tuple[str, str, str]]:
        """Extract symbolic knowledge triplets from a given text."""
        prompt = f"""
Extract symbolic knowledge from the following text.

Text:
{text}

Task:
- Identify and list triplets in the form (subject, predicate, object).
- Each triplet should be on a new line, formatted as 'subject,predicate,object'.

Provide the triplets below:
"""
        response = self.client.generate(self.model, prompt=prompt)
        triplets = [
            tuple(line.strip() for line in triplet.split(","))
            for triplet in response["response"].strip().split("\n")
            if len(triplet.split(",")) == 3
        ]
        return triplets

    def update_knowledge_graph(self, triplets: List[Tuple[str, str, str]]):
        """Update the symbolic knowledge graph."""
        for s, p, o in triplets:
            self.knowledge_graph.add_edge(s, o, relation=p)

    def symbolic_reasoning(self, query: str) -> str:
        """Perform symbolic reasoning based on the knowledge graph."""
        prompt = f"""
Using the knowledge graph, perform symbolic reasoning to answer the following query:

Query:
{query}

Provide a concise answer below:
"""
        response = self.client.generate(self.model, prompt=prompt)
        return response["response"].strip()

class ConversationalAgent(Agent):
    """Agent focusing on natural language dialogue and empathy."""

    def __init__(self, name: str, model: str, noise_channel: NoiseChannel, shared_context: CommonContext):
        super().__init__(name, model, Fore.YELLOW, noise_channel, shared_context)

    # Additional methods specific to conversational interaction can be added here.

class OverseerAgent(Agent):
    """Agent responsible for synthesizing inputs and guiding collaboration."""

    def __init__(self, name: str, model: str, noise_channel: NoiseChannel, shared_context: CommonContext):
        super().__init__(name, model, Fore.GREEN, noise_channel, shared_context)

    def make_decision(self, responses: List[Dict[str, str]]) -> str:
        """Combine agent outputs into a coherent decision or next step."""
        context = "\n".join(
            [f"{entry['role']}: {entry['content']}" for entry in self.shared_context.conversation_history[-5:]]
        )
        responses_text = "\n".join(
            [f"{response['role']}: {response['content']}" for response in responses]
        )
        prompt = f"""
Role: {self.name}

You are responsible for synthesizing the following responses and providing a clear, concise decision or recommendation.

Shared Context:
{context}

Agents' Responses:
{responses_text}

Task:
- Identify key insights from each response.
- Eliminate redundant or conflicting information.
- Provide clear next steps or recommendations for the user.
- Keep the response focused and brief (max 150 words).

Provide your decision below:
"""
        response = self.client.generate(self.model, prompt=prompt)
        return response["response"].strip()

class NeuroSymbolicConversationalSystem:
    """Main system orchestrating collaboration between agents."""

    def __init__(self, model: str):
        self.shared_context = CommonContext()
        self.noise_channel = NoiseChannel()
        self.dialogue_encoder = DialogueEncoder(self.noise_channel)

        self.neuro_symbolic_engine = NeuroSymbolicEngine("NeuroSymbolicEngine", model, self.noise_channel, self.shared_context)
        self.conversational_agent = ConversationalAgent("ConversationalAgent", model, self.noise_channel, self.shared_context)
        self.overseer_agent = OverseerAgent("OverseerAgent", model, self.noise_channel, self.shared_context)

        self.agents = [
            self.neuro_symbolic_engine,
            self.conversational_agent,
            self.overseer_agent
        ]

        self.background_thread = threading.Thread(target=self.background_discussion, daemon=True)
        self.noise_thread = threading.Thread(target=self.noise_channel.run_noise, daemon=True)

    def initialize_knowledge(self, text: str):
        """Initialize the knowledge graph with given text."""
        triplets = self.neuro_symbolic_engine.extract_symbolic_knowledge(text)
        self.neuro_symbolic_engine.update_knowledge_graph(triplets)
        print(f"{Fore.CYAN}Knowledge graph initialized with {len(triplets)} triplets.{Style.RESET_ALL}")

    def background_discussion(self):
        """Agents share background thoughts periodically to enhance collaboration."""
        while True:
            other_thoughts = {}
            for agent in self.agents:
                other_agents_thoughts = [
                    thought for a in self.agents if a != agent for thought in a.background_thoughts[-1:]
                ]
                agent.background_process(other_agents_thoughts)
                other_thoughts[agent.name] = agent.background_thoughts[-1] if agent.background_thoughts else ""
            time.sleep(5)  # Adjust the frequency as needed

    def interactive_session(self):
        """Start interactive collaboration between the system and the user."""
        print(f"{Fore.CYAN}Welcome to the NeuroSymbolic Conversational System. Type 'exit' to quit.{Style.RESET_ALL}")

        self.background_thread.start()
        self.noise_thread.start()

        while True:
            user_input = input(f"{Fore.BLUE}User: {Style.RESET_ALL}")
            if user_input.lower() == "exit":
                break

            # Each agent generates a response
            responses = []
            for agent in self.agents[:-1]:  # Exclude OverseerAgent for initial responses
                response_text = agent.generate_response(user_input)
                response = {"role": agent.name, "content": response_text}
                responses.append(response)
                print(f"{agent.color}{agent.name}: {response_text}{Style.RESET_ALL}")

            # Overseer agent synthesizes the responses
            overseer_decision = self.overseer_agent.make_decision(responses)
            print(f"{self.overseer_agent.color}{self.overseer_agent.name}: {overseer_decision}{Style.RESET_ALL}")

            # Update shared context with the new conversation entries
            self.shared_context.conversation_history.append({"role": "User", "content": user_input})
            for response in responses:
                self.shared_context.conversation_history.append(response)
            self.shared_context.conversation_history.append({"role": self.overseer_agent.name, "content": overseer_decision})

            # Each agent updates their conversation history
            for agent in self.agents:
                agent.conversation_history = self.shared_context.conversation_history

            # Encode the dialogue and store it
            for agent in self.agents:
                encoded_dialogue = self.dialogue_encoder.encode_dialogue({
                    "role": agent.name,
                    "content": agent.conversation_history[-1]['content']
                })
                self.dialogue_encoder.append_to_buffer(encoded_dialogue)

            # Periodically save the encoded dialogue as audio
            if self.dialogue_encoder.audio_buffer.size > 44100 * 10:  # Save every ~10 seconds
                self.dialogue_encoder.save_audio("dialogue_encoding.wav")
                print(f"{Fore.CYAN}Dialogue encoding saved to 'dialogue_encoding.wav'.{Style.RESET_ALL}")

            # Display background thoughts occasionally
            if random.random() < 0.3:
                print(f"\n{Fore.CYAN}Background Thoughts:{Style.RESET_ALL}")
                for agent in self.agents:
                    if agent.background_thoughts:
                        print(f"{agent.color}{agent.name}: {agent.background_thoughts[-1]}{Style.RESET_ALL}")
                        agent.background_thoughts.clear()

    def run(self):
        """Start the system."""
        initial_knowledge = """
        The sky is blue. Water consists of hydrogen and oxygen.
        Trees produce oxygen through photosynthesis. The Earth orbits the Sun.
        Humans need oxygen to breathe. Plants absorb carbon dioxide.
        The moon orbits the Earth. Gravity keeps planets in orbit.
        """
        self.initialize_knowledge(initial_knowledge)
        self.interactive_session()

if __name__ == "__main__":
    system = NeuroSymbolicConversationalSystem("llama3.2")
    system.run()