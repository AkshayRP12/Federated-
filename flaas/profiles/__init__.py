"""
Domain Profiles module for FLaaS.
"""

from flaas.profiles.kb_v1 import (
    DomainProfile,
    KB_V1_PROFILES,
    get_profile,
    export_kb_v1_json,
)

__all__ = [
    "DomainProfile",
    "KB_V1_PROFILES",
    "get_profile",
    "export_kb_v1_json",
]
