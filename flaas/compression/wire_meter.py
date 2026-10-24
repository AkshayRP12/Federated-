"""
Wire byte accounting and metrics helper for communication measurement.
"""

from typing import List, Dict, Any
import torch


def calculate_raw_bytes(tensors: List[torch.Tensor]) -> int:
    """Calculates standard raw uncompressed 32-bit floating point payload size."""
    total_floats = sum(t.numel() for t in tensors)
    return total_floats * 4


class RoundWireMeter:
    """Tracks cumulative uplink and downlink network communication across rounds."""

    def __init__(self):
        self.uplink_bytes = 0
        self.downlink_bytes = 0

    def record_client_uplink(self, bytes_sent: int) -> None:
        self.uplink_bytes += bytes_sent

    def record_server_downlink(self, bytes_broadcast: int) -> None:
        self.downlink_bytes += bytes_broadcast

    @property
    def total_bytes(self) -> int:
        return self.uplink_bytes + self.downlink_bytes

    def get_summary(self) -> Dict[str, Any]:
        return {
            "uplink_bytes": self.uplink_bytes,
            "downlink_bytes": self.downlink_bytes,
            "total_bytes": self.total_bytes,
            "total_megabytes": self.total_bytes / (1024 * 1024),
        }
