import asyncio
import json
import os
import sys
from types import SimpleNamespace

import httpx
import pytest
from oasis.social_agent.agent_environment import SocialEnvironment
from oasis.social_platform.typing import DefaultPlatformType

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import jev_decider  # noqa: E402
from jev_decider import JevDecider, build_questions, build_targets  # noqa: E402

POSTS = [
    {
        "post_id": 12,
        "user_id": 3,
        "content": "Nova taxa sobre apps aprovada",
        "comments": [{"comment_id": 40, "user_id": 7, "content": "Absurdo!"}],
    },
    {"post_id": 15, "user_id": 1, "content": "Receita de bolo de cenoura", "comments": []},
]


class FakeAction:
    async def refresh(self):
        return {"success": True, "posts": POSTS}

    async def listen_from_group(self):
        return {"success": False}


class FakeAgent:
    def __init__(self, actions):
        self.social_agent_id = 1
        self.env = SocialEnvironment(FakeAction())
        self.system_message = SimpleNamespace(content="Estudante crítico.\n# RESPONSE FORMAT\nuse tools")
        self.action_tools = [SimpleNamespace(func=SimpleNamespace(__name__=a)) for a in actions]
        self.data_calls = []
        self.llm_messages = []

    async def perform_action_by_data(self, func_name, **kwargs):
        self.data_calls.append((func_name, kwargs))
        return {"success": True}

    async def astep(self, msg):
        self.llm_messages.append(msg.content)
        return "llm"


def _answers(action, **targets):
    answers = {"action": {"type": "choice", "choice": action, "probabilities": {action: 1.0}}}
    for name, value in targets.items():
        answers[name] = {"type": "choice", "choice": value, "probabilities": {value: 1.0}}
    return answers


def _decider(handler, **kwargs):
    requests = []

    def recorder(request):
        requests.append(json.loads(request.content))
        return handler(request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(recorder))
    return JevDecider(api_key="k", http_client=client, **kwargs), requests


def _ok(answers):
    return lambda request: httpx.Response(200, json={"answers": answers, "usage": {"cost": 0.001}})


def test_from_env_requires_flag_and_key(monkeypatch):
    monkeypatch.delenv("JEV_ENABLED", raising=False)
    monkeypatch.setenv("TYPESAFE_API_KEY", "k")
    assert JevDecider.from_env() is None

    monkeypatch.setenv("JEV_ENABLED", "true")
    monkeypatch.delenv("TYPESAFE_API_KEY")
    assert JevDecider.from_env() is None

    monkeypatch.setenv("TYPESAFE_API_KEY", "k")
    monkeypatch.setenv("TYPESAFE_BASE_URL", "https://openrouter.ai/api/")
    decider = JevDecider.from_env()
    assert decider.url == "https://openrouter.ai/api/v1/systemone"
    assert decider.sample is True


def test_build_targets_excludes_self_and_collects_comment_authors():
    targets = build_targets(POSTS, self_id=1)
    assert set(targets["post"]) == {"12", "15"}
    assert set(targets["comment"]) == {"40"}
    assert set(targets["user"]) == {"3", "7"}


def test_build_questions_only_asks_needed_targets_with_real_choices():
    targets = build_targets(POSTS, self_id=1)
    questions = build_questions(["like_post", "follow", "create_post"], targets)
    assert set(questions) == {"action", "post", "user"}
    assert set(questions["action"]["criteria"]) == {"like_post", "follow", "create_post"}


def test_direct_action_runs_without_llm():
    agent = FakeAgent(["like_post", "create_comment", "do_nothing"])
    decider, requests = _decider(_ok(_answers("like_post", post="12")))

    asyncio.run(decider.act(agent, asyncio.Semaphore(1)))

    assert agent.data_calls == [("like_post", {"post_id": 12})]
    assert agent.llm_messages == []
    assert "# RESPONSE FORMAT" not in requests[0]["state"]
    assert decider.stats["direct"] == 1
    assert decider.stats["jev_cost"] == pytest.approx(0.001)


def test_single_candidate_target_is_used_without_asking():
    agent = FakeAgent(["like_comment", "do_nothing"])
    decider, requests = _decider(_ok(_answers("like_comment")))

    asyncio.run(decider.act(agent, asyncio.Semaphore(1)))

    assert "comment" not in requests[0]["questions"]
    assert agent.data_calls == [("like_comment", {"comment_id": 40})]


def test_text_action_goes_to_llm_with_chosen_action():
    agent = FakeAgent(["like_post", "create_comment"])
    decider, _ = _decider(_ok(_answers("create_comment", post="12")))

    asyncio.run(decider.act(agent, asyncio.Semaphore(1)))

    assert agent.data_calls == []
    assert "`create_comment`" in agent.llm_messages[0]
    assert "Nova taxa sobre apps" in agent.llm_messages[0]
    assert decider.stats["llm_text"] == 1


def test_jev_error_falls_back_to_plain_llm(monkeypatch):
    agent = FakeAgent(["like_post", "create_comment"])
    decider, _ = _decider(lambda request: httpx.Response(400, json={"error": "bad"}))

    asyncio.run(decider.act(agent, asyncio.Semaphore(1)))

    assert agent.data_calls == []
    assert "You have decided" not in agent.llm_messages[0]
    assert decider.stats["jev_errors"] == 1


def test_server_errors_are_retried(monkeypatch):
    async def no_sleep(_):
        return None

    monkeypatch.setattr(jev_decider.asyncio, "sleep", no_sleep)
    responses = [httpx.Response(503), _ok(_answers("do_nothing"))(None)]
    agent = FakeAgent(["do_nothing", "create_post"])
    decider, requests = _decider(lambda request: responses.pop(0))

    asyncio.run(decider.act(agent, asyncio.Semaphore(1)))

    assert len(requests) == 2
    assert agent.data_calls == [("do_nothing", {})]


def test_sampling_follows_probabilities_and_argmax_uses_choice():
    answer = {"choice": "a", "probabilities": {"a": 0.0, "b": 1.0}}
    rng = jev_decider.random.Random(0)
    assert jev_decider._pick(answer, sample=True, rng=rng) == "b"
    assert jev_decider._pick(answer, sample=False, rng=rng) == "a"


def test_step_updates_recsys_and_twitter_clock():
    calls = []

    async def update_rec_table():
        calls.append("rec")

    platform = SimpleNamespace(update_rec_table=update_rec_table, sandbox_clock=SimpleNamespace(time_step=0))
    env = SimpleNamespace(platform=platform, platform_type=DefaultPlatformType.TWITTER, llm_semaphore=asyncio.Semaphore(1))
    agents = [FakeAgent(["do_nothing"]), FakeAgent(["do_nothing"])]
    decider, requests = _decider(_ok(_answers("do_nothing")))

    asyncio.run(decider.step(env, agents))

    assert calls == ["rec"]
    assert platform.sandbox_clock.time_step == 1
    assert len(requests) == 2
    assert all(a.data_calls == [("do_nothing", {})] for a in agents)
