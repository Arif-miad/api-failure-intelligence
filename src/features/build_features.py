import pandas as pd

NUM_COLS = [
    "status_code",
    "latency_ms",
    "request_size_bytes",
    "response_size_bytes",
    "retry_count",
    "thread_id",
]

CAT_COLS = [
    "api_name",
    "service_owner",
    "environment",
    "http_method",
    "endpoint",
    "region",
    "log_level",
]

FEATURE_COLS = NUM_COLS + CAT_COLS + [
    "hour",
    "day_of_week",
    "is_weekend",
    "is_error",
    "has_error_type",
]


def parse_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce", utc=True)
    return out


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["hour"] = out["timestamp"].dt.hour.fillna(0).astype(int)
    out["day_of_week"] = out["timestamp"].dt.dayofweek.fillna(0).astype(int)
    out["is_weekend"] = out["day_of_week"].isin([5, 6]).astype(int)
    return out


def add_operational_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["is_error"] = (out["status_code"] >= 400).astype(int)
    out["has_error_type"] = out["error_type"].notna().astype(int)
    out["error_type"] = out["error_type"].fillna("NONE")
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = parse_timestamp(df)
    df = add_time_features(df)
    df = add_operational_features(df)

    for c in FEATURE_COLS:
        if c not in df.columns:
            df[c] = 0

    return df