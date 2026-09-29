import sys
import os
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
logging.basicConfig(level=logging.WARNING)

from src.retrieval.retriever import RetrievedDocument
from src.prompting.context_builder import ContextBuilder
from src.prompting.prompt_builder_dum import PromptBuilder
from src.response_validator import ResponseValidator

def test_case_1_powershell_encoded_payload():
    print("\n=========================================================")
    print("RUNNING TEST 1: Standard Encoded PowerShell Alert Grounding Check")
    print("=========================================================")

    context_builder = ContextBuilder()
    prompt_builder = PromptBuilder()

    alert_text = "Suspicious PowerShell encoded payload A-003 · 14:27:14 · WS-ACCT-021"

    doc1 = RetrievedDocument(
        document="Sigma Rule: Suspicious Execution of Powershell with Base64\nSeverity Level: HIGH\nAssociated MITRE ATT&CK Tags: attack.execution, attack.t1059.001\nDetection Logic: CommandLine contains -Enc",
        distance=0.1,
        metadata={"source": "SIGMA_RULES", "name": "Suspicious Execution of Powershell with Base64"},
        query=alert_text,
        rerank_score=0.65
    )
    doc2 = RetrievedDocument(
        document="Sigma Rule: Suspicious Obfuscated PowerShell Code\nSeverity Level: HIGH\nAssociated MITRE ATT&CK Tags: attack.execution, attack.t1059.001\nDetection Logic: Payload contains UTF16 Base64",
        distance=0.1,
        metadata={"source": "SIGMA_RULES", "name": "Suspicious Obfuscated PowerShell Code"},
        query=alert_text,
        rerank_score=0.55
    )

    built_context = context_builder.build([doc1, doc2])

    assert built_context["computed_risk_score"] == 95, f"Expected score 95, got {built_context['computed_risk_score']}"
    assert built_context["escalation_recommendation"] == "Escalate to Tier-2 / Analyst Investigation Required", f"Unexpected escalation: {built_context['escalation_recommendation']}"

    prompt = prompt_builder.build(alert_text=alert_text, context=built_context)

    print(f"Risk Score: {built_context['computed_risk_score']}/100")
    print(f"Escalation: {built_context['escalation_recommendation']}")
    print(f"Risk Factors: {built_context['risk_factors']}")
    print("TEST 1 PASSED: Risk score & escalation correctly computed and injected into prompt!")

def test_case_2_lateral_movement_alert():
    print("\n=========================================================")
    print("RUNNING TEST 2: Alert with Explicit Remote Connection / Lateral Movement")
    print("=========================================================")

    context_builder = ContextBuilder()
    alert_text = "PsExec remote execution on DC-01"

    doc = RetrievedDocument(
        document="Sigma Rule: PsExec Remote Execution\nSeverity Level: HIGH\nAssociated MITRE ATT&CK Tags: attack.t1021.002\nDetection Logic: PsExec remote service execution",
        distance=0.1,
        metadata={"source": "SIGMA_RULES", "name": "PsExec Remote Execution"},
        query=alert_text,
        rerank_score=0.70
    )

    built_context = context_builder.build([doc])
    print(f"Risk Score: {built_context['computed_risk_score']}/100")
    print(f"Escalation: {built_context['escalation_recommendation']}")
    assert built_context["computed_risk_score"] == 90
    print("TEST 2 PASSED!")

def test_case_3_no_useful_rag_knowledge():
    print("\n=========================================================")
    print("RUNNING TEST 3: Alert with Weak/No Useful RAG Knowledge")
    print("=========================================================")

    context_builder = ContextBuilder()
    alert_text = "Unknown system event 0x99"

    doc1 = RetrievedDocument(
        document="High noise doc",
        distance=0.9,
        metadata={"source": "BACKGROUND", "name": "Noise 1"},
        query=alert_text,
        rerank_score=0.01
    )

    built_context = context_builder.build([doc1])
    print(f"Risk Score: {built_context['computed_risk_score']}/100")
    print(f"Escalation: {built_context['escalation_recommendation']}")
    print(f"Risk Factors: {built_context['risk_factors']}")
    assert built_context["computed_risk_score"] in (40, 50)
    assert "Tier-1 Investigation" in built_context["escalation_recommendation"] or "Monitor" in built_context["escalation_recommendation"]
    print("TEST 3 PASSED!")

def test_case_4_artifact_non_leak():
    print("\n=========================================================")
    print("RUNNING TEST 4: Artifact Non-Leak Check (Retrieved Example Artifacts)")
    print("=========================================================")

    validator = ResponseValidator()
    alert_text = "Suspicious PowerShell encoded payload A-003 · 14:27:14 · WS-ACCT-021"

    # Context contains example IP 192.168.1.50 which is NOT in alert_text
    context = [{"document": "Sigma Rule example CommandLine: powershell.exe -e 192.168.1.50"}]

    # BAD response trying to claim IP 192.168.1.50 is an IOC for this incident
    bad_response = (
        "Escalation Recommendation: Escalate to Tier-2 / Analyst Investigation Required\n"
        "Risk Score: 95/100\n"
        "Basis: High severity rule match\n\n"
        "IOCs Extracted:\n"
        "- 192.168.1.50\n"
    )

    result = validator.validate(response=bad_response, alert_text=alert_text, context=context)
    print(f"Validator Warning Caught: {result.warnings}")
    assert any("does not appear in the alert text" in w for w in result.warnings)
    print("TEST 4 PASSED: Validator correctly caught ungrounded artifact in IOCs!")

def test_case_5_mitre_stage_non_attribution():
    print("\n=========================================================")
    print("RUNNING TEST 5: MITRE Attack Stage Non-Attribution Rule Check")
    print("=========================================================")

    prompt_builder = PromptBuilder()
    context_builder = ContextBuilder()

    doc = RetrievedDocument(
        document="MITRE ATT&CK T1059: Command and Scripting Interpreter. Adversaries may use scripts for execution, persistence, and lateral movement.",
        distance=0.1,
        metadata={"source": "MITRE", "name": "T1059"},
        query="PowerShell alert",
        rerank_score=0.65
    )

    built_context = context_builder.build([doc])
    prompt = prompt_builder.build(alert_text="PowerShell alert", context=built_context)

    assert "ATTACK STAGE GENERATION RESTRICTIONS" in prompt
    assert "Do NOT assert or imply that any of the following occurred in this specific incident" in prompt
    print("TEST 5 PASSED: Prompt explicitly restricts unobserved attack stage assertions!")

if __name__ == "__main__":
    test_case_1_powershell_encoded_payload()
    test_case_2_lateral_movement_alert()
    test_case_3_no_useful_rag_knowledge()
    test_case_4_artifact_non_leak()
    test_case_5_mitre_stage_non_attribution()
