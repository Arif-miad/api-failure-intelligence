def compute_severity(status_code: int, latency_ms: int, root_cause: str) -> str:
    if status_code in (500, 502, 503, 504) and latency_ms >= 1500:
        return "P0"
    if status_code >= 500:
        return "P1"
    if status_code >= 400:
        return "P2"
    return "OK"


def recommended_next_step(retry_prob: float, severity: str) -> str:
    if severity == "P0":
        return "Escalate immediately to on-call SRE"
    if retry_prob < 0.35:
        return "Escalate to SRE (low retry success probability)"
    if retry_prob < 0.60:
        return "Retry with backoff and monitor"
    return "Auto-retry likely to succeed"