"""
Data adapters and partitioners module for FLaaS.
"""

from flaas.data.partitioners import (
    partition_iid,
    partition_dirichlet,
    get_client_dataloaders,
)
from flaas.data.domain_adapters import (
    BaseDomainAdapter,
    HealthcareAdapter,
    FinanceAdapter,
    IoTEdgeAdapter,
    MobileAdapter,
    AutonomousVehiclesAdapter,
    DOMAIN_ADAPTERS,
    get_domain_adapter,
    GPUDataset,
)

__all__ = [
    "partition_iid",
    "partition_dirichlet",
    "get_client_dataloaders",
    "BaseDomainAdapter",
    "HealthcareAdapter",
    "FinanceAdapter",
    "IoTEdgeAdapter",
    "MobileAdapter",
    "AutonomousVehiclesAdapter",
    "DOMAIN_ADAPTERS",
    "get_domain_adapter",
    "GPUDataset",
]
