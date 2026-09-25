"""
Jev (TypeSafe System One) decision step for OASIS agents.

Replaces `env.step({agent: LLMAction()})` with a cascade:
  1. Jev picks the agent's action (and target post/comment/user) from the
     persona + feed, returning calibrated probabilities.
  2. Actions that need no text (like, repost, follow, do_nothing...) are
     executed directly - no LLM call.
  3. Actions that need text (create_post, create_comment, quote_post,
     search...) go to the LLM, told which action Jev chose.
Any Jev failure falls back to the plain LLM step for that agent.

Enable with JEV_ENABLED=true. Reads TYPESAFE_API_KEY / TYPESAFE_BASE_URL
(OpenRouter: https://openrouter.ai/api).
"""

import asyncio
import json
import logging
import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import httpx
from camel.messages import BaseMessage
from oasis.social_platform.typing import DefaultPlatformType

logger = logging.getLogger("mirofish.jev")

DEFAULT_MODEL = "jev-1.13"
MAX_CHOICES = 255           # Jev limit per choice question
MAX_STATE_CHARS = 80_000    # keeps state well under Jev's 32k-token context
SNIPPET_CHARS = 200

# action -> (argument name, target question); None = no argument
DIRECT_ACTIONS: Dict[str, Tuple[Optional[str], Optional[str]]] = {
    "do_nothing": (None, None),
    "like_post": ("post_id", "post"),
    "dislike_post": ("post_id", "post"),
    "repost": ("post_id", "post"),
    "like_comment": ("comment_id", "comment"),
    "dislike_comment": ("comment_id", "comment"),
    "follow": ("followee_id", "user"),
    "mute": ("mutee_id", "user"),
}

ACTION_DESCRIPTIONS = {
    "create_post": "Write a new original post.",
    "create_comment": "Write a comment replying to a post in the feed.",
    "quote_post": "Share a post from the feed adding their own commentary.",
    "repost": "Share a post from the feed without commentary.",
    "like_post": "Like a post in the feed.",
    "dislike_post": "Dislike a post in the feed.",
    "like_comment": "Like a comment in the feed.",
    "dislike_comment": "Dislike a comment in the feed.",
    "follow": "Follow the author of a post or comment in the feed.",
    "mute": "Mute the author of a post or comment in the feed.",
    "search_posts": "Search for posts about some topic.",
    "search_user": "Search for some user.",
    "trend": "Look at the trending posts.",
    "refresh": "Refresh the feed to see other posts.",
    "do_nothing": "Stay passive and do nothing right now.",
}

# Same wording as oasis SocialAgent.perform_action_by_llm
LLM_PROMPT = (
    "Please perform social media actions after observing the platform "
    "environments. Notice that don't limit your actions for example to just "
    "like the posts. Here is your social media environment: {env_prompt}"
)


@dataclass
class Observation:
    env_prompt: str
    posts: List[Dict[str, Any]]


@dataclass
class Decision:
    action: str
    targets: Dict[str, str] = field(default_factory=dict)


def _snippet(text: Any) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= SNIPPET_CHARS else text[:SNIPPET_CHARS] + "..."


def _pick(answer: Dict[str, Any], sample: bool, rng: random.Random) -> str:
    """Sample from Jev's distribution (keeps population diversity) or argmax."""
    probs = answer.get("probabilities") or {}
    if sample and probs and sum(probs.values()) > 0:
        labels = list(probs)
        return rng.choices(labels, weights=[probs[k] for k in labels])[0]
    return answer["choice"]


def build_targets(posts: List[Dict[str, Any]], self_id: int) -> Dict[str, Dict[str, str]]:
    """Candidate targets per question, as Jev `criteria` dicts."""
    post_c, comment_c, user_c = {}, {}, {}
    for post in posts:
        pid = post.get("post_id")
        if pid is not None and len(post_c) < MAX_CHOICES:
            post_c[str(pid)] = _snippet(post.get("content"))
        authors = [(post.get("user_id"), post.get("content"))]
        for comment in post.get("comments") or []:
            cid = comment.get("comment_id")
            if cid is not None and len(comment_c) < MAX_CHOICES:
                comment_c[str(cid)] = _snippet(comment.get("content"))
            authors.append((comment.get("user_id"), comment.get("content")))
        for uid, content in authors:
            if uid is None or uid == self_id or str(uid) in user_c:
                continue
            if len(user_c) < MAX_CHOICES:
                user_c[str(uid)] = f"Author of: {_snippet(content)}"
    return {"post": post_c, "comment": comment_c, "user": user_c}


def build_questions(actions: List[str], targets: Dict[str, Dict[str, str]]) -> Dict[str, Any]:
    questions: Dict[str, Any] = {
        "action": {
            "type": "choice",
            "instructions": (
                "Given this user's profile and what they currently see on the "
                "platform, which action do they take now?"
            ),
            "criteria": {a: ACTION_DESCRIPTIONS.get(a, a) for a in actions},
        }
    }
    needed = {DIRECT_ACTIONS[a][1] for a in actions if a in DIRECT_ACTIONS}
    prompts = {
        "post": "If this user interacts with a post in the feed, which post is it?",
        "comment": "If this user interacts with a comment in the feed, which comment is it?",
        "user": "If this user follows or mutes someone from the feed, who is it?",
    }
    for name, criteria in targets.items():
        # A single-option choice carries no information; skip it
        if name in needed and len(criteria) >= 2:
            questions[name] = {"type": "choice", "instructions": prompts[name], "criteria": criteria}
    return questions


def build_state(agent, env_prompt: str) -> str:
    persona = agent.system_message.content.split("# RESPONSE FORMAT")[0].strip()
    state = f"# USER PROFILE\n{persona}\n\n# WHAT THE USER SEES\n{env_prompt}"
    return state[:MAX_STATE_CHARS]


async def observe(agent) -> Observation:
    """Like SocialEnvironment.to_text_prompt, but keeps the structured posts."""
    env = agent.env
    result = await env.action.refresh()
    posts = result.get("posts", []) if result.get("success") else []
    if posts:
        posts_env = env.posts_env_template.substitute(posts=json.dumps(posts, indent=4))
    else:
        posts_env = "After refreshing, there are no existing posts."
    env_prompt = env.env_template.substitute(
        posts_env=posts_env, groups_env=await env.get_group_env()
    )
    return Observation(env_prompt=env_prompt, posts=posts)


class JevDecider:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://openrouter.ai/api",
        model: str = DEFAULT_MODEL,
        sample: bool = True,
        concurrency: int = 50,
        seed: Optional[int] = None,
        http_client: Optional[httpx.AsyncClient] = None,
    ):
        self.url = base_url.rstrip("/") + "/v1/systemone"
        self.model = model
        self.sample = sample
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._semaphore = asyncio.Semaphore(concurrency)
        self._rng = random.Random(seed)
        self._client = http_client or httpx.AsyncClient(timeout=30)
        self.stats = {"direct": 0, "llm_text": 0, "llm_fallback": 0, "jev_errors": 0, "jev_cost": 0.0}

    @classmethod
    def from_env(cls) -> Optional["JevDecider"]:
        if os.environ.get("JEV_ENABLED", "").lower() not in ("1", "true", "yes"):
            return None
        api_key = os.environ.get("TYPESAFE_API_KEY")
        if not api_key:
            logger.warning("JEV_ENABLED is set but TYPESAFE_API_KEY is missing; using LLM only")
            return None
        seed = os.environ.get("JEV_SEED")
        return cls(
            api_key=api_key,
            base_url=os.environ.get("TYPESAFE_BASE_URL", "https://openrouter.ai/api"),
            model=os.environ.get("JEV_MODEL", DEFAULT_MODEL),
            sample=os.environ.get("JEV_SAMPLING", "sample").lower() != "argmax",
            concurrency=int(os.environ.get("JEV_CONCURRENCY", "50")),
            seed=int(seed) if seed else None,
        )

    async def step(self, env, agents: List[Any]) -> None:
        """Drop-in replacement for env.step({agent: LLMAction()})."""
        await env.platform.update_rec_table()
        await asyncio.gather(*(self.act(agent, env.llm_semaphore) for agent in agents))
        if env.platform_type == DefaultPlatformType.TWITTER:
            env.platform.sandbox_clock.time_step += 1

    async def act(self, agent, llm_semaphore: asyncio.Semaphore):
        obs = await observe(agent)
        actions = [tool.func.__name__ for tool in agent.action_tools]
        try:
            decision = await self.decide(agent, obs, actions)
        except Exception as e:
            logger.warning(f"Jev failed for agent {agent.social_agent_id}: {e}; using LLM")
            self.stats["jev_errors"] += 1
            self.stats["llm_fallback"] += 1
            return await self._llm_act(agent, obs.env_prompt, None, llm_semaphore)

        kwargs = self._direct_kwargs(decision)
        if kwargs is None:
            self.stats["llm_text"] += 1
            return await self._llm_act(agent, obs.env_prompt, decision.action, llm_semaphore)
        self.stats["direct"] += 1
        return await agent.perform_action_by_data(decision.action, **kwargs)

    async def decide(self, agent, obs: Observation, actions: List[str]) -> Decision:
        targets = build_targets(obs.posts, agent.social_agent_id)
        questions = build_questions(actions, targets)
        answers = await self._request(build_state(agent, obs.env_prompt), questions)
        decision = Decision(action=_pick(answers["action"], self.sample, self._rng))
        for name, criteria in targets.items():
            if name in answers:
                decision.targets[name] = _pick(answers[name], self.sample, self._rng)
            elif len(criteria) == 1:
                decision.targets[name] = next(iter(criteria))
        return decision

    def _direct_kwargs(self, decision: Decision) -> Optional[Dict[str, int]]:
        """Arguments to run the action without the LLM, or None if it needs the LLM."""
        if decision.action not in DIRECT_ACTIONS:
            return None
        arg, target = DIRECT_ACTIONS[decision.action]
        if arg is None:
            return {}
        value = decision.targets.get(target)
        return None if value is None else {arg: int(value)}

    async def _request(self, state: str, questions: Dict[str, Any]) -> Dict[str, Any]:
        payload = {"model": self.model, "state": state, "questions": questions}
        async with self._semaphore:
            for attempt in range(3):
                try:
                    resp = await self._client.post(self.url, json=payload, headers=self._headers)
                    if resp.status_code == 429 or resp.status_code >= 500:
                        raise httpx.HTTPStatusError(f"HTTP {resp.status_code}", request=resp.request, response=resp)
                    resp.raise_for_status()
                    break
                except (httpx.TransportError, httpx.HTTPStatusError) as e:
                    retryable = not isinstance(e, httpx.HTTPStatusError) or e.response.status_code == 429 or e.response.status_code >= 500
                    if attempt == 2 or not retryable:
                        raise
                    await asyncio.sleep(2 ** attempt)
        data = resp.json()
        self.stats["jev_cost"] += (data.get("usage") or {}).get("cost") or 0.0
        return data["answers"]

    async def _llm_act(self, agent, env_prompt: str, chosen: Optional[str], llm_semaphore):
        content = LLM_PROMPT.format(env_prompt=env_prompt)
        if chosen:
            content += f"\n\nYou have decided to perform the action `{chosen}` now."
        msg = BaseMessage.make_user_message(role_name="User", content=content)
        async with llm_semaphore:
            try:
                return await agent.astep(msg)
            except Exception as e:
                logger.error(f"Agent {agent.social_agent_id} LLM error: {e}")
                return e

    def summary(self) -> str:
        s = self.stats
        total = s["direct"] + s["llm_text"] + s["llm_fallback"]
        pct = (100 * s["direct"] / total) if total else 0.0
        return (
            f"Jev: {total} decisions, {s['direct']} without LLM ({pct:.0f}%), "
            f"{s['llm_text']} text actions via LLM, {s['llm_fallback']} fallbacks "
            f"({s['jev_errors']} Jev errors), Jev cost ${s['jev_cost']:.4f}"
        )

    async def aclose(self) -> None:
        await self._client.aclose()
