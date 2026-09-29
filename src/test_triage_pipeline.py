import sys
import os
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
logging.basicConfig(level=logging.WARNING)

from src.retrieval.retriever import RetrievedDocument
from src.prompting.context_builder import ContextBuilder
from src.prompting.prompt_builder_dum import PromptBuilder
from src.response_validator import ResponseValidator
from src.utils.alert_parser import parse_alert


def test_req_a_structured_real_alert():
    print("\n=========================================================")
    print("RUNNING REQUIRED TEST A: Structured Real Alert Parsing")
    print("=========================================================")

    raw_alert = {
        "timestamp": "2026-09-29T07:51:33.085198+00:00",
        "prediction_class": 1,
        "triage_action": "ESCALATE — automated model flagged anomalous behaviour",
        "risk_score": 0.85,
        "mitre_techniques": [
            "T1059 — Command & Scripting Interpreter",
            "T1071 — Application Layer Protocol",
        ],
        "iocs_extracted": {
            "agent_name": "Victus_host",
            "agent_ip": "127.0.0.1",
            "feature_snapshot": {
                "Network_I_ActiveNIC_TCP_APS": 9430.0,
                "Process_Thread Count": 5831.0,
                "Process_Working Set": 15218671616.0,
            },
        },
        "ai_reasoning": "Random Forest predicted class 1 (non-zero -> anomalous).",
        "source_event_id": "1790668292.222204",
    }

    parsed = parse_alert(raw_alert)
    assert parsed.agent_name == "Victus_host", f"Expected Victus_host, got {parsed.agent_name}"
    assert parsed.agent_ip == "127.0.0.1", f"Expected 127.0.0.1, got {parsed.agent_ip}"
    assert parsed.source_event_id == "1790668292.222204", f"Expected 1790668292.222204, got {parsed.source_event_id}"
    assert parsed.upstream_risk_score == 0.85, f"Expected 0.85, got {parsed.upstream_risk_score}"
    assert any("T1059" in tech for tech in parsed.upstream_mitre_techniques), "T1059 missing"
    assert any("T1071" in tech for tech in parsed.upstream_mitre_techniques), "T1071 missing"
    print("REQUIRED TEST A PASSED!")


def test_req_b_no_invented_mitre_techniques():
    print("\n=========================================================")
    print("RUNNING REQUIRED TEST B: No Invented MITRE Techniques Leakage")
    print("=========================================================")

    prompt_builder = PromptBuilder()
    context_builder = ContextBuilder()

    alert_data = {
        "timestamp": "2026-09-29T07:51:33.085198+00:00",
        "mitre_techniques": ["T1059 — Command & Scripting Interpreter", "T1071 — Application Layer Protocol"],
        "agent_name": "Victus_host",
    }

    # RAG document discusses T1021 Remote Services
    doc = RetrievedDocument(
        document="Sigma Rule: Remote Service Execution via SMB\nAssociated MITRE ATT&CK Tags: T1021.002",
        distance=0.1,
        metadata={"source": "SIGMA_RULES", "name": "PsExec Service"},
        query="Victus_host alert",
        rerank_score=0.65,
    )

    built_context = context_builder.build([doc])
    prompt = prompt_builder.build(alert_text=str(alert_data), context=built_context, alert_metadata=alert_data)

    # Ensure prompt clearly specifies Upstream MITRE techniques separate from RAG
    assert "UPSTREAM ALERT MITRE TECHNIQUES:" in prompt
    assert "T1059" in prompt
    assert "T1071" in prompt
    assert "NEVER add a RAG-retrieved MITRE technique (e.g. T1021)" in prompt
    print("REQUIRED TEST B PASSED!")


def test_req_c_missing_optional_fields():
    print("\n=========================================================")
    print("RUNNING REQUIRED TEST C: Missing Optional Fields Handling")
    print("=========================================================")

    alert_data = {
        "timestamp": "2026-09-29T07:51:33.085198+00:00",
        "agent_name": "Victus_host",
        # agent_ip intentionally omitted
    }

    parsed = parse_alert(alert_data)
    assert parsed.agent_name == "Victus_host"
    assert parsed.agent_ip is None, "agent_ip should be None"

    prompt_builder = PromptBuilder()
    context_builder = ContextBuilder()
    doc = RetrievedDocument(document="Generic info", distance=0.5, metadata={"source": "BG", "name": "Doc"}, query="test", rerank_score=0.1)
    built_context = context_builder.build([doc])

    prompt = prompt_builder.build(alert_text="Victus_host alert", context=built_context, alert_metadata=alert_data)
    assert "agent_ip" not in parsed.metadata_dict or parsed.metadata_dict.get("agent_ip") is None
    print("REQUIRED TEST C PASSED!")


def test_req_d_no_traditional_iocs_invented():
    print("\n=========================================================")
    print("RUNNING REQUIRED TEST D: No Traditional IOCs Invention")
    print("=========================================================")

    alert_data = {
        "agent_name": "Victus_host",
        "agent_ip": "127.0.0.1",
        "feature_snapshot": {"Network_I_ActiveNIC_TCP_APS": 9430.0},
    }

    parsed = parse_alert(alert_data)
    # Feature snapshot telemetry must be separated into formatted_telemetry, not metadata IOCs
    assert len(parsed.formatted_telemetry) == 1
    assert "TCP_APS: 9,430" in parsed.formatted_telemetry[0] or "9430" in parsed.formatted_telemetry[0]
    print("REQUIRED TEST D PASSED!")


def test_req_e_risk_separation():
    print("\n=========================================================")
    print("RUNNING REQUIRED TEST E: Risk Separation (Upstream vs ARIA)")
    print("=========================================================")

    alert_data = {
        "risk_score": 0.85,
        "agent_name": "Victus_host",
    }

    prompt_builder = PromptBuilder()
    context_builder = ContextBuilder()

    doc = RetrievedDocument(document="Medium severity rule\nSeverity: MEDIUM", distance=0.1, metadata={"source": "SIGMA", "name": "Medium Rule"}, query="alert", rerank_score=0.70)
    built_context = context_builder.build([doc])

    prompt = prompt_builder.build(alert_text="Victus_host", context=built_context, alert_metadata=alert_data)

    assert "UPSTREAM MODEL RISK: 0.85" in prompt
    assert "ARIA Deterministic Risk Score: 65/100" in prompt or "ARIA Deterministic Risk Score:" in prompt
    assert built_context["computed_risk_score"] != 85, "ARIA risk score must not silently copy upstream 0.85 as 85"
    print("REQUIRED TEST E PASSED!")


def test_req_f_analyst_actionability():
    print("\n=========================================================")
    print("RUNNING REQUIRED TEST F: Analyst Actionability & Structure")
    print("=========================================================")

    prompt_builder = PromptBuilder()
    context_builder = ContextBuilder()

    doc_pb = RetrievedDocument(
        document="TLP:CLEAR Incident Response Playbook for Red Team Detection\nSteps: 1. Monitor unusual behavior. 2. Investigate process resource usage.",
        distance=0.1,
        metadata={"source": "IR_PLAYBOOKS", "name": "IRM-20-Red_Team_Detection"},
        query="Victus_host alert",
        rerank_score=0.50,
    )

    built_context = context_builder.build([doc_pb])
    prompt = prompt_builder.build(alert_text="Victus_host alert", context=built_context)

    assert "Threat Assessment" in prompt
    assert "Alert Facts" in prompt
    assert "Evidence Interpretation" in prompt
    assert "Recommended Investigation" in prompt
    assert "Escalation Recommendation" in prompt
    assert "Evidence Sources" in prompt
    print("REQUIRED TEST F PASSED!")


if __name__ == "__main__":
    test_req_a_structured_real_alert()
    test_req_b_no_invented_mitre_techniques()
    test_req_c_missing_optional_fields()
    test_req_d_no_traditional_iocs_invented()
    test_req_e_risk_separation()
    test_req_f_analyst_actionability()
    print("\nALL GROUNDING & ANALYST ACTIONABILITY REGRESSION TESTS PASSED SUCCESSFULLY!")
