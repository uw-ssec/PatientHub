import random
from typing import Any, Dict, List, Optional

from omegaconf import DictConfig
from pydantic import BaseModel, Field

from src.agents import InferenceAgent
from src.utils import get_model_client, get_reranker, load_json, load_prompts

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


class Response(BaseModel):
    content: str = Field(description="The content of your generated response in this turn")


class ActionDistribution(BaseModel):
    Deny: int
    Downplay: int
    Blame: int
    Inform: int
    Engage: int


class BinaryAnswer(BaseModel):
    answer: bool = Field(description="True if the answer is yes, false otherwise")
    rationale: Optional[str] = Field(default=None, description="Short explanation for the decision")


class consistentMIClient(InferenceAgent):
    def __init__(self, configs: DictConfig):
        print("[consistentMIClient.__init__] init with configs:", configs)
        self.configs = configs
        self.prompts = load_prompts(role="client", agent_type="consistentMI", lang=configs.lang)
        self.model_client = get_model_client(configs.model_client)
        self.reranker = get_reranker(configs.model_retriever)

        self.data = {}
        self.name = "ConsistentMI"
        self.therapist: Optional[str] = None

        self.topic_graph: Dict[str, Dict[str, int]] = {}
        self.all_topics: List[str] = []
        self.topic_passages: List[str] = []

        self.messages: List[Any] = []

        print("[consistentMIClient.__init__] calling _load_profile")
        self._load_profile()
        print("[consistentMIClient.__init__] calling _init_topic_graph")
        self._init_topic_graph()
        print("[consistentMIClient.__init__] calling _init_topics")
        self._init_topics()
        print("[consistentMIClient.__init__] calling _init_conversation")
        self._init_conversation()

    def generate(self, messages: List[Any], response_format: BaseModel):
        print(
            f"[consistentMIClient.generate] format={response_format.__name__}, "
            f"messages_count={len(messages)}"
        )
        model_client = self.model_client.with_structured_output(response_format)
        res = model_client.invoke(messages)
        print("[consistentMIClient.generate] raw result:", res)
        return res

    def set_therapist(self, therapist, prev_sessions: Optional[List[Dict[str, str]]] = None):
        print("[consistentMIClient.set_therapist] therapist:", therapist)
        self.therapist = therapist.get("name", "Therapist")

    def generate_response(self, msg: str):
        print("[consistentMIClient.generate_response] incoming msg:", msg)
        # Treat incoming message as therapist utterance
        self.messages.append(HumanMessage(content=msg))

        client_text = self.reply()

        # Persist clean conversation history for subsequent turns
        self.messages.append(AIMessage(content=client_text))

        content = client_text[len("Client: ") :].strip() if client_text.startswith("Client: ") else client_text
        print("[consistentMIClient.generate_response] client_text:", client_text)
        print("[consistentMIClient.generate_response] content:", content)
        return Response(content=content)

    def reset(self):
        print("[consistentMIClient.reset] resetting state")
        # Keep model_client and prompts; reload profile-dependent state
        self._load_profile()
        self._init_topics()
        self._init_conversation()

    # ===== Internal helpers =====
    def _load_profile(self):
        print("[consistentMIClient._load_profile] loading profiles from:", self.configs.data_path)
        profiles = load_json(self.configs.data_path)
        index = getattr(self.configs, "data_index", 0)
        print("[consistentMIClient._load_profile] chosen index:", index)
        self.data = profiles[index] if isinstance(profiles, list) else profiles
        self.name = self.data.get("name", "ConsistentMI")

        self.goal = self.data.get("topic", "")
        self.behavior = self.data.get("Behavior", "")
        self.personas: List[str] = self.data.get("Personas", [])
        self.beliefs: List[str] = self.data.get("Beliefs", [])
        self.acceptable_plans: List[str] = self.data.get("Acceptable Plans", []).copy()
        motivation_list = self.data.get("Motivation", [])
        self.motivation_text = motivation_list[-1] if motivation_list else ""
        self.engagemented_topics = motivation_list[:-1] if len(motivation_list) > 1 else []
        self.initial_stage = self.data.get("initial_stage", "Precontemplation")
        suggestibilities = self.data.get("suggestibilities", [])
        self.receptivity = (
            sum(suggestibilities) / len(suggestibilities) if suggestibilities else 3
        )
        print(
            "[consistentMIClient._load_profile] profile:",
            f"name={self.name}, goal={self.goal}, behavior={self.behavior},",
            f"initial_stage={self.initial_stage}, receptivity={self.receptivity}",
        )
        print(
            "[consistentMIClient._load_profile] personas={}/beliefs={}/plans={}".format(
                len(self.personas), len(self.beliefs), len(self.acceptable_plans)
            )
        )
        self.engagement = self.receptivity
        self.state = self.initial_stage
        self.error_topic_count = 0

    def _init_topic_graph(self):
        # Copied from the original ConsistentMI simulator
        self.topic_graph = {
            "Health": {
                "Mental Disorders": 2,
                "Diseases": 2,
                "Sexual Health": 2,
                "Fitness": 2,
                "Health Care": 2,
                "Workplace Wellness": 1,
                "Interpersonal Relationships": 3,
                "Law": 3,
                "Economy": 3,
                "Education": 3,
            },
            "Interpersonal Relationships": {
                "Parenting": 2,
                "Family": 2,
                "Health": 3,
                "Law": 3,
                "Economy": 3,
            },
            "Parenting": {
                "Interpersonal Relationships": 2,
                "Family": 2,
                "Health": 3,
                "Education": 2,
                "Role Model": 1,
                "Child Development": 1,
                "Paternal Bond": 1,
                "Child Care": 1,
                "Habituation": 1,
            },
            "Family": {
                "Interpersonal Relationships": 2,
                "Parenting": 2,
                "Law": 2,
                "Health": 3,
                "Family Estrangement": 1,
                "Family Disruption": 1,
                "Divorce": 1,
            },
            "Mental Disorders": {
                "Health": 2,
                "Diseases": 3,
            },
            "Diseases": {
                "Health": 2,
                "Mental Disorders": 3,
                "Infection": 1,
                "Hypertension": 1,
                "Flu": 1,
                "Inflammation": 1,
                "Liver Disease": 1,
                "Lung Cancer": 1,
                "COPD": 1,
                "Asthma": 1,
                "Stroke": 1,
                "Diabetes": 1,
            },
            "Sexual Health": {
                "Health": 2,
                "Maternal Health": 1,
                "Safe Sex": 1,
                "Preterm Birth": 1,
                "Miscarriage": 1,
                "Birth Defects": 1,
            },
            "Fitness": {
                "Health": 2,
                "Physical Activity": 1,
                "Sport": 1,
                "Physical Fitness": 1,
                "Strength": 1,
                "Flexibility": 1,
                "Endurance": 1,
            },
            "Health Care": {
                "Health": 2,
                "Dentistry": 1,
                "Caregiver Burden": 1,
                "Independent Living": 1,
                "Human Appearance": 1,
            },
            "Economy": {
                "Health": 3,
                "Interpersonal Relationships": 3,
                "Law": 2,
                "Education": 2,
                "Employment": 1,
                "Personal Finance": 1,
                "Cost of Living": 1,
            },
            "Law": {
                "Health": 3,
                "Interpersonal Relationships": 3,
                "Economy": 2,
                "Education": 2,
                "Criminal Law": 1,
                "Family Law": 1,
                "Traffic Law": 1,
            },
            "Education": {
                "Health": 3,
                "Interpersonal Relationships": 3,
                "Law": 2,
                "Economy": 2,
                "Student Affairs": 1,
                "Academic Achievement": 1,
            },
            "Employment": {
                "Economy": 1,
                "Productivity": 1,
                "Absenteeism": 1,
                "Workplace Relationships": 1,
                "Career Break": 1,
                "Career Assessment": 1,
                "Absence Rate": 1,
                "Salary": 1,
                "Workplace Wellness": 2,
                "Workplace Incivility": 2,
            },
            "Personal Finance": {
                "Economy": 1,
                "Cost of Living": 1,
                "Personal Budget": 1,
                "Debt": 1,
                "Income Deficit": 1,
            },
            "Student Affairs": {
                "Education": 1,
                "Attendance": 1,
                "Suspension": 1,
                "Scholarship": 1,
            },
            "Academic Achievement": {
                "Education": 1,
                "Exam": 1,
            },
            "Criminal Law": {
                "Law": 1,
                "Arrest": 1,
                "Imprisonment": 1,
                "Complaint": 1,
            },
            "Family Law": {
                "Law": 1,
                "Family": 1,
                "Child Custody": 1,
            },
            "Traffic Law": {
                "Law": 1,
                "Traffic Ticket": 1,
            },
            "Workplace Wellness": {
                "Health": 1,
                "Employment": 2,
            },
            "Workplace Incivility": {
                "Employment": 2,
            },
            "Productivity": {
                "Employment": 1,
            },
            "Absenteeism": {
                "Employment": 1,
            },
            "Workplace Relationships": {
                "Employment": 1,
            },
            "Career Break": {
                "Employment": 1,
            },
            "Career Assessment": {
                "Employment": 1,
            },
            "Absence Rate": {
                "Employment": 1,
            },
            "Salary": {
                "Employment": 1,
            },
            "Cost of Living": {
                "Economy": 1,
                "Personal Finance": 1,
            },
            "Personal Budget": {
                "Personal Finance": 1,
            },
            "Debt": {
                "Personal Finance": 1,
            },
            "Income Deficit": {
                "Personal Finance": 1,
            },
            "Role Model": {
                "Parenting": 1,
            },
            "Child Development": {
                "Parenting": 1,
            },
            "Paternal Bond": {
                "Parenting": 1,
            },
            "Child Care": {
                "Parenting": 1,
            },
            "Habituation": {
                "Parenting": 1,
            },
            "Family Estrangement": {
                "Family": 1,
            },
            "Family Disruption": {
                "Family": 1,
            },
            "Divorce": {
                "Family": 1,
            },
            "Arrest": {
                "Criminal Law": 1,
            },
            "Imprisonment": {
                "Criminal Law": 1,
            },
            "Complaint": {
                "Criminal Law": 1,
            },
            "Child Custody": {
                "Family Law": 1,
            },
            "Traffic Ticket": {
                "Traffic Law": 1,
            },
            "Attendance": {
                "Student Affairs": 1,
            },
            "Suspension": {
                "Student Affairs": 1,
            },
            "Scholarship": {
                "Student Affairs": 1,
            },
            "Exam": {
                "Academic Achievement": 1,
            },
            "Infection": {
                "Diseases": 1,
            },
            "Hypertension": {
                "Diseases": 1,
            },
            "Flu": {
                "Diseases": 1,
            },
            "Inflammation": {
                "Diseases": 1,
            },
            "Liver Disease": {
                "Diseases": 1,
            },
            "Lung Cancer": {
                "Diseases": 1,
            },
            "COPD": {
                "Diseases": 1,
            },
            "Asthma": {
                "Diseases": 1,
            },
            "Stroke": {
                "Diseases": 1,
            },
            "Diabetes": {
                "Diseases": 1,
            },
        }

        self.all_topics = []
        for node in self.topic_graph:
            if node not in self.all_topics:
                self.all_topics.append(node)
            for neighbor in self.topic_graph[node]:
                if neighbor not in self.all_topics:
                    self.all_topics.append(neighbor)

    def _init_topics(self):
        print("[consistentMIClient._init_topics] initializing topic passages")
        topics_path = getattr(self.configs, "topics_path", None)
        print("[consistentMIClient._init_topics] topics_path:", topics_path)
        topics_data = load_json(topics_path) if topics_path else []
        topic_to_content = {
            item.get("topic"): item.get("content", "") for item in topics_data if "topic" in item
        }

        self.topic_passages = []
        for topic in self.all_topics:
            description = self.prompts["topic_description_prompt"].render(
                behavior=self.behavior, goal=self.goal, topic=topic
            )
            passage = description.strip() + " " + topic_to_content.get(topic, "")
            self.topic_passages.append(passage)
        print("[consistentMIClient._init_topics] total topics:", len(self.all_topics))

    def _init_conversation(self):
        print("[consistentMIClient._init_conversation] building initial messages")
        personas_and_beliefs = "\n".join(
            f"- {text}" for text in (self.personas + self.beliefs)
        )
        system_prompt = self.prompts["consistentmi_client_system_prompt"].render(
            behavior=self.behavior,
            goal=self.goal,
            personas_and_beliefs=personas_and_beliefs,
        )

        self.messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content="Hello. How are you?"),
            AIMessage(content="Client: I am good. What about you?"),
        ]
        print("[consistentMIClient._init_conversation] initial messages length:", len(self.messages))

    # ===== Core reasoning blocks =====
    def verify_motivation(self):
        context_block = "\n- ".join(self._get_recent_context(5))
        prompt = self.prompts["verify_motivation_prompt"].render(
            goal=self.goal,
            context_block=context_block,
            motivation=self.motivation_text,
        )
        res = self.generate([SystemMessage(content=prompt)], response_format=BinaryAnswer)
        if res.answer:
            self.state = "Motivation"
        return res.rationale or ""

    def top_related_topics(self) -> List[str]:
        last_therapist = self._get_last_therapist_utterance()
        if not last_therapist:
            return []

        # Use LangChain reranker when available
        if getattr(self, "reranker", None) is not None:
            from langchain_core.documents import Document

            documents = [
                Document(page_content=passage, metadata={"topic": topic})
                for topic, passage in zip(self.all_topics, self.topic_passages)
            ]
            try:
                ranked_docs = self.reranker.rerank(query=last_therapist, documents=documents)
                top_docs = ranked_docs[: min(5, len(ranked_docs))]
                return [doc.metadata.get("topic", "") for doc in top_docs if doc.metadata.get("topic")]
            except Exception:
                pass

        # Lightweight lexical fallback
        lower_query = last_therapist.lower()
        scores: List[int] = []
        for passage in self.topic_passages:
            score = sum(1 for token in lower_query.split() if token in passage.lower())
            scores.append(score)

        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[
            : min(5, len(scores))
        ]
        return [self.all_topics[index] for index in top_indices]

    def dijkstra(self, graph: Dict[str, Dict[str, int]], start_node: str, target_node: str):
        distances = {node: float("inf") for node in graph}
        distances[start_node] = 0
        pq = [(0, start_node)]
        visited = set()

        while pq:
            current_distance, current_node = min(pq, key=lambda x: x[0])
            pq.remove((current_distance, current_node))

            if current_node == target_node:
                return current_distance
            if current_node in visited:
                continue

            visited.add(current_node)
            for neighbor, weight in graph.get(current_node, {}).items():
                if neighbor in visited:
                    continue
                distance = current_distance + weight
                if distance < distances.get(neighbor, float("inf")):
                    distances[neighbor] = distance
                    pq.append((distance, neighbor))

        return float("inf")

    def update_state(self):
        if self.state == "Contemplation":
            if not self.beliefs:
                self.state = "Preparation"
            return None
        elif self.state == "Preparation":
            return None

        if not self.engagemented_topics:
            return None

        top_topics = self.top_related_topics()
        predicted_topic = top_topics[0] if top_topics else self.engagemented_topics[0]

        if predicted_topic == self.engagemented_topics[0]:
            self.engagement = 4
            self.error_topic_count = 0
            return self.verify_motivation()

        distance = self.dijkstra(self.topic_graph, self.engagemented_topics[0], predicted_topic)
        if distance <= 3:
            self.engagement = 3
            self.error_topic_count = 0
            return f"The client's perceived topic is {predicted_topic}."
        if distance <= 5:
            self.engagement = 2
            return f"The client's perceived topic is {predicted_topic}."

        self.engagement = 1
        # Approximate long conversations by number of therapist utterances
        if len([m for m in self.messages if isinstance(m, HumanMessage)]) > 10:
            self.error_topic_count += 1
        return f"The client's perceived topic is {predicted_topic}."

    def select_action(self) -> str:
        print("[consistentMIClient.select_action] state:", self.state, "receptivity:", self.receptivity)
        recent_context = "\n".join(self._get_recent_context(3)).replace(
            "Client:", "**Client**:"
        ).replace("Counselor:", "**Counselor**:")
        print("[consistentMIClient.select_action] recent_context:\n", recent_context)
        prompt = self.prompts["select_action_prompt"].render(recent_context=recent_context)

        try:
            res = self.generate([SystemMessage(content=prompt)], response_format=ActionDistribution)
            context_dist = {
                "Deny": res.Deny,
                "Downplay": res.Downplay,
                "Blame": res.Blame,
                "Inform": res.Inform,
                "Engage": res.Engage,
            }
            print("[consistentMIClient.select_action] model context_dist:", context_dist)
        except Exception as e:
            print("[consistentMIClient.select_action] exception, using uniform dist:", e)
            context_dist = {"Deny": 20, "Downplay": 20, "Blame": 20, "Inform": 20, "Engage": 20}

        if self.receptivity < 2:
            receptivity_dist = {"Deny": 23, "Downplay": 28, "Blame": 15, "Engage": 11, "Inform": 22}
        elif self.receptivity < 3:
            receptivity_dist = {"Deny": 20, "Downplay": 25, "Blame": 10, "Engage": 15, "Inform": 30}
        elif self.receptivity < 4:
            receptivity_dist = {"Deny": 19, "Downplay": 21, "Blame": 11, "Engage": 13, "Inform": 36}
        elif self.receptivity < 5:
            receptivity_dist = {"Deny": 9, "Downplay": 20, "Blame": 13, "Engage": 14, "Inform": 44}
        else:
            receptivity_dist = {"Deny": 7, "Downplay": 13, "Blame": 4, "Engage": 16, "Inform": 60}
        print("[consistentMIClient.select_action] receptivity_dist:", receptivity_dist)

        action_distribution = {
            action: context_dist.get(action, 0) + receptivity_dist[action]
            for action in receptivity_dist
        }

        if not self.personas:
            action_distribution["Inform"] = 0
        if not self.beliefs:
            action_distribution["Blame"] = 0

        total = sum(action_distribution.values())
        if total == 0:
            print("[consistentMIClient.select_action] total=0, fallback Engage")
            return "Engage"
        actions = list(action_distribution.keys())
        probs = [v / total for v in action_distribution.values()]
        action = random.choices(actions, weights=probs, k=1)[0]
        print("[consistentMIClient.select_action] combined:", action_distribution, "sampled:", action)
        return action

    def select_information(self, action: str) -> Optional[str]:
        print("[consistentMIClient.select_information] action:", action)
        last_therapist = self._get_last_therapist_utterance()
        print("[consistentMIClient.select_information] last_therapist:", last_therapist)
        if not last_therapist or "?" not in last_therapist:
            print("[consistentMIClient.select_information] no question, return None")
            return None

        if action == "Inform":
            personas = self.personas
            prompt_template = self.prompts["select_information_inform_prompt"]
        elif action == "Downplay":
            personas = self.beliefs
            prompt_template = self.prompts["select_information_downplay_prompt"]
        elif action == "Blame":
            personas = self.beliefs
            prompt_template = self.prompts["select_information_blame_prompt"]
        elif action == "Hesitate":
            personas = self.beliefs
            prompt_template = self.prompts["select_information_hesitate_prompt"]
        else:
            print("[consistentMIClient.select_information] unsupported action, return None")
            return None

        print("[consistentMIClient.select_information] personas_count:", len(personas))
        for persona in personas:
            prompt = prompt_template.render(persona=persona)
            res = self.generate([SystemMessage(content=prompt)], response_format=BinaryAnswer)
            print("[consistentMIClient.select_information] persona:", persona, "answer:", res.answer)
            if res.answer:
                print("[consistentMIClient.select_information] selected persona:", persona)
                return persona

        chosen = random.choice(personas) if personas else None
        print("[consistentMIClient.select_information] random fallback persona:", chosen)
        return chosen

    def get_engage_instruction(self):
        print("[consistentMIClient.get_engage_instruction] engagement:", self.engagement, "topics:", self.engagemented_topics)
        if not self.engagemented_topics or len(self.engagemented_topics) < 3:
            return ""
        text = self.prompts["engage_instruction_prompt"].render(
            engagement_level=int(self.engagement),
            engagemented_topics=self.engagemented_topics,
            motivation=self.motivation_text,
        ).strip()
        print("[consistentMIClient.get_engage_instruction] text:", text)
        return text

    def _render_state_instruction(self):
        text = self.prompts["state_instruction_prompt"].render(
            stage=self.state, behavior=self.behavior, goal=self.goal
        ).strip()
        print("[consistentMIClient._render_state_instruction] state:", self.state, "text:", text)
        return text

    def _render_action_instruction(self, action: str):
        text = self.prompts["action_instruction_prompt"].render(action=action).strip()
        print("[consistentMIClient._render_action_instruction] action:", action, "text:", text)
        return text

    def _get_recent_context(self, n: int) -> List[str]:
        """Return recent conversation as prefixed text lines: 'Counselor: ...' / 'Client: ...'."""
        print("[consistentMIClient._get_recent_context] n:", n)
        lines: List[str] = []
        # Skip initial system message
        for msg in self.messages[1:]:
            if isinstance(msg, HumanMessage):
                lines.append(f"Counselor: {msg.content}")
            elif isinstance(msg, AIMessage):
                content = msg.content
                if not content.startswith("Client:"):
                    content = f"Client: {content}"
                lines.append(content)
        recent = lines[-n:] if n > 0 else lines
        print("[consistentMIClient._get_recent_context] recent:", recent)
        return recent

    def _get_last_therapist_utterance(self) -> Optional[str]:
        """Return raw text of the last therapist (Human) utterance."""
        print("[consistentMIClient._get_last_therapist_utterance] scanning messages")
        for msg in reversed(self.messages):
            if isinstance(msg, HumanMessage):
                print("[consistentMIClient._get_last_therapist_utterance] found:", msg.content)
                return msg.content
        print("[consistentMIClient._get_last_therapist_utterance] none found")
        return None

    def reply(self) -> str:
        print("========== [consistentMIClient.reply] ==========")
        engagement_analysis = self.update_state()
        print("[consistentMIClient.reply] engagement_analysis:", engagement_analysis)
        engage_instruction = self.get_engage_instruction()
        print("[consistentMIClient.reply] engage_instruction:", engage_instruction)
        information = None

        if self.state == "Motivation":
            print("[consistentMIClient.reply] branch=Motivation")
            action = "Acknowledge"
            state_instruction = ""
            action_instruction = self._render_action_instruction(action)
            instruction = self.prompts["reply_instruction_prompt"].render(
                stage="Motivation",
                action=action,
                engage_instruction=engage_instruction,
                state_instruction=state_instruction,
                action_instruction=action_instruction,
                information="",
                motivation=self.motivation_text,
            )
            self.state = "Contemplation"
        elif self.state == "Precontemplation":
            print("[consistentMIClient.reply] branch=Precontemplation, error_topic_count:", self.error_topic_count)
            action = "Terminate" if self.error_topic_count >= 5 else self.select_action()
            state_instruction = self._render_state_instruction()
            action_instruction = self._render_action_instruction(action)
            if action in ["Inform", "Downplay", "Blame"]:
                information = self.select_information(action)
            instruction = self.prompts["reply_instruction_prompt"].render(
                stage="Precontemplation",
                action=action,
                engage_instruction=engage_instruction,
                state_instruction=state_instruction,
                action_instruction=action_instruction,
                information=information or "",
                motivation=self.motivation_text,
            )
        elif self.state == "Contemplation":
            print("[consistentMIClient.reply] branch=Contemplation")
            action = self.select_action()
            state_instruction = self._render_state_instruction()
            action_instruction = self._render_action_instruction(action)
            if action in ["Hesitate", "Inform"]:
                information = self.select_information(action)
            instruction = self.prompts["reply_instruction_prompt"].render(
                stage="Contemplation",
                action=action,
                engage_instruction=engage_instruction,
                state_instruction=state_instruction,
                action_instruction=action_instruction,
                information=information or "",
                motivation=self.motivation_text,
            )
        else:  # Preparation
            print("[consistentMIClient.reply] branch=Preparation")
            action = self.select_action()
            state_instruction = self._render_state_instruction()
            action_instruction = self._render_action_instruction(action)
            if action == "Inform":
                information = self.select_information(action)
            elif action == "Plan" and self.acceptable_plans:
                information = self.acceptable_plans.pop(0)
            instruction = self.prompts["reply_instruction_prompt"].render(
                stage="Preparation",
                action=action,
                engage_instruction=engage_instruction,
                state_instruction=state_instruction,
                action_instruction=action_instruction,
                information=information or "",
                motivation=self.motivation_text,
            )

        # Build model call with conversation history and inline instruction
        last_therapist = self._get_last_therapist_utterance() or ""
        print("[consistentMIClient.reply] last_therapist:", last_therapist)
        print("[consistentMIClient.reply] instruction:", instruction)
        user_message = f"{last_therapist} {instruction.replace(chr(10), ' ')}"
        messages_for_model = list(self.messages) + [HumanMessage(content=user_message)]
        print("[consistentMIClient.reply] messages_for_model_count:", len(messages_for_model))
        res = self.generate(messages_for_model, response_format=Response)
        print("[consistentMIClient.reply] model Response:", res)

        patient_text = res.content.strip()
        if not patient_text.startswith("Client:"):
            patient_text = f"Client: {patient_text}"
        if "Counselor:" in patient_text:
            patient_text = patient_text.split("Counselor:")[0].strip()

        print("[consistentMIClient.reply] final patient_text:", patient_text)
        return patient_text
