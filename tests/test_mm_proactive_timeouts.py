from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from memory.mental_model_memory import MentalModelMemory


class _FakeAgent:
    def generate_openai_response(self, prompt: str) -> str:
        # Keep LLM output neutral so timeout responders are tested deterministically.
        return json.dumps(
            {
                "imm_update": {},
                "smm_update": {},
                "response_decision": {
                    "should_respond": False,
                    "response_type": "none",
                    "query": "",
                    "reason": "",
                },
            },
            ensure_ascii=False,
        )


def _iso_ago(seconds: int) -> str:
    dt = datetime.now(timezone.utc) - timedelta(seconds=seconds)
    return dt.replace(microsecond=0).isoformat()


class TestMentalModelProactiveTimeouts(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.jl = Path(self.tmp.name)
        (self.jl / "smm_shared_models.json").write_text("{}", encoding="utf-8")
        self.mm = MentalModelMemory(jl_dir=str(self.jl))
        self.agent = _FakeAgent()
        self.uid = "U_TEST"
        self.cid = "C_TEST"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _run(self, message: str = "继续讨论") -> dict:
        return self.mm.analyze_and_update(
            agent=self.agent,
            channel_id=self.cid,
            user_id=self.uid,
            user_name="tester",
            message_text=message,
            convs="A: hi\nB: ok",
        )

    def test_unknown_term_timeout_over_10s_triggers_explain(self) -> None:
        self.mm.upsert_imm(
            user_id=self.uid,
            patch={
                "认知盲区 (未涉及知识)": [
                    {
                        "未知术语": "RAG",
                        "当前状态": "未解决",
                        "触发时间戳": _iso_ago(12),
                        "持续时长_秒": 0,
                    }
                ]
            },
            user_name="tester",
        )
        out = self._run("这个先不处理")
        self.assertTrue(out["decision"]["should_respond"])
        self.assertEqual(out["decision"]["response_type"], "professional_explain")
        self.assertIn("RAG", out["decision"]["query"])

    def test_conflict_timeout_over_10s_triggers_judgment(self) -> None:
        smm = self.mm._default_smm(self.cid)
        smm["任务生命周期"]["阶段进入时间"] = _iso_ago(30)
        smm["团队冲突区 (Conflict Zone)"] = [
            {
                "冲突描述": "A主张自动评测，B主张人工评测",
                "当前状态": "未解决",
                "触发时间戳": _iso_ago(15),
                "持续时长_秒": 0,
            }
        ]
        self.mm._smm_by_channel[self.cid] = smm
        self.mm._flush_smm()

        out = self._run("我还在想")
        self.assertTrue(out["decision"]["should_respond"])
        self.assertEqual(out["decision"]["response_type"], "judgment")

    def test_topic_stage_over_5min_triggers_topic_help(self) -> None:
        smm = self.mm._default_smm(self.cid)
        smm["任务生命周期"]["当前所处阶段"] = "选题"
        smm["任务生命周期"]["阶段进入时间"] = _iso_ago(6 * 60)
        smm["phase_status"] = "未解决"
        self.mm._smm_by_channel[self.cid] = smm
        self.mm._flush_smm()

        out = self._run("先聊聊天")
        self.assertTrue(out["decision"]["should_respond"])
        self.assertEqual(out["decision"]["response_type"], "topic")

    def test_division_stage_over_10min_triggers_division_help(self) -> None:
        smm = self.mm._default_smm(self.cid)
        smm["任务生命周期"]["当前所处阶段"] = "分工"
        smm["任务生命周期"]["阶段进入时间"] = _iso_ago(11 * 60)
        smm["phase_status"] = "未解决"
        self.mm._smm_by_channel[self.cid] = smm
        self.mm._flush_smm()

        out = self._run("继续")
        self.assertTrue(out["decision"]["should_respond"])
        self.assertEqual(out["decision"]["response_type"], "division")


if __name__ == "__main__":
    unittest.main()
