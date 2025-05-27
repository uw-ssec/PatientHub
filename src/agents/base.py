from abc import ABC, abstractmethod


class BaseAgent(ABC):
    def __init__(self, role: str, model):
        self.role = role
        self.model = model
        self.messages = []
        self.memory = []

    def reset_messages(self):
        self.messages = []

    def add_message(self, msg):
        self.messages.append(msg)

    def remove_message(self, idx):
        if idx < len(self.messages):
            self.messages.pop(idx)

    def edit_message(self, idx, msg):
        if idx < len(self.messages):
            self.messages[idx] = msg

    @abstractmethod
    def set_prompt(self, prompt):
        """Set the system prompt for the agent."""
        pass

    @abstractmethod
    def receive_message(self, msg: str, role: str = "user"):
        """Receive a message from the user."""
        pass

    @abstractmethod
    def generate_response(self):
        """Generate a response based on the current state."""
        pass
