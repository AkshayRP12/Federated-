"""
Data adapters and partitioners module.
"""

from flaas.data.partitioners import (
    partition_iid,
    partition_dirichlet,
    get_client_dataloaders,
)

__all__ = ["partition_iid", "partition_dirichlet", "get_client_dataloaders"]
