import re
import random
from src.base import ChatAgent
from omegaconf import DictConfig
from dataclasses import dataclass
from pydantic import BaseModel, Field


@dataclass
class ElizaTherapistConfig:
    """Configuration for Eliza Therapist agent."""

    agent_type: str = "eliza"
    lang: str = "en"


PATTERNS = [
    (
        r"hello|hi|hey",
        [
            "Hello... I'm glad you could drop by today.",
            "Hi there... how are you today?",
            "Hello... what seems to be troubling you?",
        ],
    ),
    (
        r"how are you",
        [
            "I'm doing well, but let's focus on you.",
            "Fine, thank you. How about you?",
        ],
    ),
    (
        r"I need (.*)",
        [
            "Why do you need {0}?",
            "Would it really help if you had {0}?",
            "Are you sure you need {0}?",
        ],
    ),
    (
        r"why don'?t you (.*)",
        [
            "Do you really think I don't {0}?",
            "Perhaps eventually I will {0}.",
            "Do you want me to {0}?",
        ],
    ),
    (
        r"why can'?t I (.*)",
        [
            "Do you think you should be able to {0}?",
            "If you could {0}, what would you do?",
        ],
    ),
    (
        r"I can'?t (.*)",
        [
            "How do you know you can't {0}?",
            "Perhaps you could {0} if you tried.",
            "What would it take for you to {0}?",
        ],
    ),
    (
        r"I'?m (.*)",
        [
            "How does being {0} make you feel?",
            "Do you enjoy being {0}?",
            "Did you come to me because you are {0}?",
            "How long have you been {0}?",
        ],
    ),
    (
        r"you'?re (.*)",
        [
            "What makes you think I am {0}?",
            "Does it please you to believe I am {0}?",
            "Why do you say I'm {0}?",
        ],
    ),
    (
        r"(.*)\b(mother|mom)\b(.*)",
        [
            "Tell me more about your mother.",
            "What is your relationship with your mother like?",
        ],
    ),
    (
        r"(.*)\b(father|dad)\b(.*)",
        [
            "Tell me more about your father.",
            "How does your father make you feel?",
        ],
    ),
    (
        r"(.*)\bchild(.*)",
        [
            "Did you have close friends as a child?",
            "What is your favorite childhood memory?",
        ],
    ),
    (
        r"(.*)\?$",
        [
            "Why do you ask that?",
            "Please consider whether you can answer your own question.",
            "Perhaps the answer lies within yourself?",
        ],
    ),
    (r"^yes\b", ["You seem quite sure.", "OK, but can you elaborate a bit?"]),
    (
        r"^no\b",
        [
            "Why not?",
            "You are being a bit negative.",
            "Are you saying no just to be negative?",
        ],
    ),
    (
        r"(.*)\bsorry\b(.*)",
        [
            "There are many times when no apology is needed.",
            "What feelings do you have when you apologize?",
        ],
    ),
]

FALLBACK_RESPONSES = [
    "Please tell me more.",
    "Let's change focus a bit... Tell me about your family.",
    "Can you elaborate on that?",
    "I see. And what does that tell you?",
    "How does that make you feel?",
    "Very interesting.",
]

REFLECTIONS = {
    "i": "you",
    "me": "you",
    "my": "your",
    "am": "are",
    "you": "I",
    "your": "my",
    "mine": "yours",
    "myself": "yourself",
    "yourself": "myself",
    "are": "am",
    "was": "were",
    "were": "was",
    "i'm": "you are",
    "you're": "I am",
    "i've": "you have",
    "you've": "I have",
    "i'll": "you will",
    "you'll": "I will",
}


class Response(BaseModel):
    content: str = Field(
        description="The content of eliza's generated response based on the client's input"
    )


class ElizaTherapist(ChatAgent):
    def __init__(self, configs: DictConfig):
        self.name = "Eliza"
        self.is_first_message = True

    def set_client(self, client):
        self.client = client.get("name", "Client")

    def reflect(self, text: str) -> str:
        """Swap first/second person pronouns."""
        tokens = text.lower().split()
        return " ".join(self.REFLECTIONS.get(t, t) for t in tokens)

    def preprocess(self, text: str) -> str:
        """Normalize input text."""
        if self.client_name:
            text = text.replace(self.client_name, "")
        return re.sub(r"\s+", " ", text).strip()

    def match_pattern(self, text: str) -> str:
        """Find matching pattern and generate response."""
        for pattern, responses in self.PATTERNS:
            if match := re.search(pattern, text, re.IGNORECASE):
                response = random.choice(responses)
                if "{0}" in response:
                    phrase = match.group(1) if match.lastindex else ""
                    response = response.format(self.reflect(phrase))
                return response
        return random.choice(self.FALLBACK_RESPONSES)

    def generate(self, msg):
        text = self.preprocess(msg)
        content = self.match_pattern(text)

        return Response(content=content)

    def generate_response(self, msg: str):
        if self.is_first_message:
            self.is_first_message = False
            return Response(content="Hello. How can I help you today?")
        else:
            res = self.generate(msg)
            return res

    def reset(self):
        self.is_first_message = True
        self.client = None
