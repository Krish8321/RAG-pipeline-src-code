from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from src.config.logging_config import setup_logger

logger = setup_logger()


@dataclass
class ParsedAlert:
    raw_input: Any
    timestamp: str | None = None
    prediction_class: int | str | None = None
    triage_action: str | None = None
    upstream_risk_score: float | str | None = None
    upstream_mitre_techniques: list[str] = field(default_factory=list)
    agent_name: str | None = None
    agent_ip: str | None = None
    source_event_id: str | None = None
    feature_snapshot: dict[str, float | int | str] = field(default_factory=dict)
    ai_reasoning: str | None = None
    metadata_dict: dict[str, Any] = field(default_factory=dict)
    formatted_telemetry: list[str] = field(default_factory=list)
    affected_asset: dict[str, str] = field(default_factory=dict)
    traditional_iocs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "prediction_class": self.prediction_class,
            "triage_action": self.triage_action,
            "upstream_risk_score": self.upstream_risk_score,
            "upstream_mitre_techniques": self.upstream_mitre_techniques,
            "agent_name": self.agent_name,
            "agent_ip": self.agent_ip,
            "affected_asset": self.affected_asset,
            "traditional_iocs": self.traditional_iocs,
            "source_event_id": self.source_event_id,
            "feature_snapshot": self.feature_snapshot,
            "ai_reasoning": self.ai_reasoning,
            "formatted_telemetry": self.formatted_telemetry,
        }


def format_bytes(byte_val: float | int) -> str:
    """Format large byte numbers into human-readable MB/GB strings."""
    if byte_val >= 1024**3:
        return f"{byte_val / (1024**3):.2f} GB ({int(byte_val):,} bytes)"
    if byte_val >= 1024**2:
        return f"{byte_val / (1024**2):.2f} MB ({int(byte_val):,} bytes)"
    return f"{int(byte_val):,} bytes"


def format_feature_value(key: str, val: float | int | str) -> str:
    """Format telemetry features cleanly."""
    if not isinstance(val, (int, float)):
        return f"{key}: {val}"

    key_lower = key.lower()
    if "bytes" in key_lower:
        return f"{key}: {format_bytes(val)}"
    if "count" in key_lower or "aps" in key_lower or "connections" in key_lower:
        return f"{key}: {val:,.1f}" if isinstance(val, float) and not val.is_integer() else f"{key}: {int(val):,}"
    return f"{key}: {val}"


def parse_alert(alert_input: Any) -> ParsedAlert:
    """
    Deterministically parse raw alert inputs (dict, JSON string, or multiline key-value string)
    into a standardized ParsedAlert object.
    """
    if alert_input is None:
        return ParsedAlert(raw_input=alert_input)

    data: dict[str, Any] = {}

    if isinstance(alert_input, dict):
        data = alert_input
    elif isinstance(alert_input, str):
        stripped = alert_input.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                data = json.loads(stripped)
            except Exception as exc:
                logger.warning(f"JSON parsing failed for alert text: {exc}")
                data = _parse_kv_string(alert_input)
        else:
            data = _parse_kv_string(alert_input)

    parsed = ParsedAlert(raw_input=alert_input)

    parsed.timestamp = str(data.get("timestamp")) if data.get("timestamp") is not None else None
    parsed.prediction_class = data.get("prediction_class")
    parsed.triage_action = data.get("triage_action")

    # Upstream risk score
    raw_risk = data.get("risk_score")
    if raw_risk is not None:
        try:
            parsed.upstream_risk_score = float(raw_risk)
        except (ValueError, TypeError):
            parsed.upstream_risk_score = raw_risk

    parsed.ai_reasoning = data.get("ai_reasoning")
    parsed.source_event_id = str(data.get("source_event_id")) if data.get("source_event_id") is not None else None

    # IOCs / Entities
    iocs = data.get("iocs_extracted")
    if isinstance(iocs, dict):
        parsed.agent_name = iocs.get("agent_name")
        parsed.agent_ip = iocs.get("agent_ip")
        snapshot = iocs.get("feature_snapshot", {})
        if isinstance(snapshot, dict):
            parsed.feature_snapshot = snapshot
    else:
        parsed.agent_name = data.get("agent_name") or data.get("Agent Name")
        parsed.agent_ip = data.get("agent_ip") or data.get("Agent IP")
        snapshot = data.get("feature_snapshot") or data.get("Feature Snapshot")
        if isinstance(snapshot, dict):
            parsed.feature_snapshot = snapshot

    # MITRE Techniques
    raw_mitre = data.get("mitre_techniques") or data.get("MITRE Techniques")
    if isinstance(raw_mitre, list):
        parsed.upstream_mitre_techniques = [str(tech).strip() for tech in raw_mitre if tech]
    elif isinstance(raw_mitre, str):
        parsed.upstream_mitre_techniques = [tech.strip() for tech in re.split(r"[;,]", raw_mitre) if tech.strip()]

    # Format telemetry snapshot
    formatted_tel = []
    if parsed.feature_snapshot:
        for k, v in parsed.feature_snapshot.items():
            formatted_tel.append(format_feature_value(k, v))
    parsed.formatted_telemetry = formatted_tel

    # Build clean metadata_dict
    meta = {}
    if parsed.timestamp:
        meta["timestamp"] = parsed.timestamp
    if parsed.prediction_class is not None:
        meta["prediction_class"] = parsed.prediction_class
    if parsed.triage_action:
        meta["triage_action"] = parsed.triage_action
    if parsed.upstream_risk_score is not None:
        meta["upstream_risk_score"] = parsed.upstream_risk_score
    if parsed.upstream_mitre_techniques:
        meta["upstream_mitre_techniques"] = parsed.upstream_mitre_techniques
    if parsed.agent_name:
        meta["agent_name"] = parsed.agent_name
    if parsed.agent_ip:
        meta["agent_ip"] = parsed.agent_ip
    if parsed.source_event_id:
        meta["source_event_id"] = parsed.source_event_id
    if parsed.ai_reasoning:
        meta["ai_reasoning"] = parsed.ai_reasoning
    if parsed.feature_snapshot:
        meta["feature_snapshot"] = parsed.feature_snapshot

    parsed.metadata_dict = meta
    return parsed


def _parse_kv_string(text: str) -> dict[str, Any]:
    """Parse multiline key: value alert text."""
    data: dict[str, Any] = {}
    snapshot: dict[str, Any] = {}
    in_snapshot = False

    for line in text.splitlines():
        line_str = line.strip()
        if not line_str:
            continue

        if line_str.startswith("Feature Snapshot:"):
            in_snapshot = True
            continue

        if in_snapshot:
            if ":" in line_str or "=" in line_str:
                parts = re.split(r"[:=]", line_str, maxsplit=1)
                k = parts[0].strip()
                v_str = parts[1].strip()
                if k.lower() in ("ai reasoning", "agent name", "agent ip", "source event id"):
                    in_snapshot = False
                else:
                    try:
                        v = float(v_str)
                    except ValueError:
                        v = v_str
                    snapshot[k] = v
                    continue
            else:
                in_snapshot = False

        if ":" in line_str:
            parts = line_str.split(":", 1)
            k = parts[0].strip()
            v_str = parts[1].strip()

            if k.lower() == "timestamp":
                data["timestamp"] = v_str
            elif k.lower() == "prediction class":
                try:
                    data["prediction_class"] = int(v_str)
                except ValueError:
                    data["prediction_class"] = v_str
            elif k.lower() == "triage action":
                data["triage_action"] = v_str
            elif k.lower() == "risk score":
                try:
                    data["risk_score"] = float(v_str)
                except ValueError:
                    data["risk_score"] = v_str
            elif k.lower() == "mitre techniques":
                data["mitre_techniques"] = v_str
            elif k.lower() == "agent name":
                data["agent_name"] = v_str
            elif k.lower() == "agent ip":
                data["agent_ip"] = v_str
            elif k.lower() == "source event id":
                data["source_event_id"] = v_str
            elif k.lower() == "ai reasoning":
                data["ai_reasoning"] = v_str

    if snapshot:
        data["feature_snapshot"] = snapshot

    return data
