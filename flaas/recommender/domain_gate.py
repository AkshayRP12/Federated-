"""
Domain Gate for FLaaS Recommender.
Enforces domain activation policies:
- Active domains: Mobile Apps and IoT Edge.
- Pending domains: Healthcare and Finance marked as 'pending fixes' and excluded.
Labels all empirical data as 1-seed.
"""

from typing import Dict, List, Set, Optional


ACTIVE_DOMAINS: List[str] = ["mobile_apps", "iot_edge"]

PENDING_DOMAINS: Dict[str, str] = {
    "healthcare": "Pending fixes: requires domain-specific capacity tuning and balanced class loss re-weighting.",
    "finance": "Pending fixes: requires severe class imbalance focal loss and cost-sensitive thresholding.",
}

DATASET_LABEL: str = "1-seed (Seed 42)"


class DomainPendingFixesError(ValueError):
    """Raised when querying a domain that is gated under 'pending fixes'."""
    pass


class UnknownDomainError(ValueError):
    """Raised when querying a domain not recognized in the knowledge base."""
    pass


def is_domain_active(domain_id: str) -> bool:
    """Returns True if the domain is currently active and approved for recommendation."""
    return domain_id in ACTIVE_DOMAINS


def validate_domain_active(domain_id: str) -> None:
    """
    Validates that a domain is active.
    Raises DomainPendingFixesError if the domain is pending fixes.
    Raises UnknownDomainError if the domain is unknown.
    """
    if domain_id in PENDING_DOMAINS:
        reason = PENDING_DOMAINS[domain_id]
        raise DomainPendingFixesError(
            f"Domain '{domain_id}' is currently marked as PENDING FIXES and excluded from recommendation. "
            f"Reason: {reason}"
        )
    if domain_id not in ACTIVE_DOMAINS:
        raise UnknownDomainError(
            f"Domain '{domain_id}' is unknown. Active domains: {ACTIVE_DOMAINS}, "
            f"Pending domains: {list(PENDING_DOMAINS.keys())}"
        )


def get_domain_status(domain_id: str) -> Dict[str, str]:
    """Returns domain operational status and diagnostic message."""
    if domain_id in ACTIVE_DOMAINS:
        return {"status": "active", "message": "Domain is active and approved for recommendation."}
    elif domain_id in PENDING_DOMAINS:
        return {"status": "pending_fixes", "message": PENDING_DOMAINS[domain_id]}
    else:
        return {"status": "unknown", "message": f"Domain '{domain_id}' not recognized."}
