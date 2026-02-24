"""Alert severity levels for drift monitoring."""

from enum import Enum


class AlertLevel(Enum):
    """Severity tiers for drift alerts."""
    OK = "ok"
    WARNING = "warning"
    CRITICAL = "critical"


ALERT_SEVERITY: dict[AlertLevel, int] = {
    AlertLevel.OK: 0,
    AlertLevel.WARNING: 1,
    AlertLevel.CRITICAL: 2,
}
