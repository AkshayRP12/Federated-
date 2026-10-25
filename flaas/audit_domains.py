"""
Domain Audit Script for Stage 4 Pre-flight Checks.
Prints: domain, dataset name, number of classes, number of samples,
dataset_source, and client partitioning details.
"""

from flaas.data.domain_adapters import get_domain_adapter
from flaas.profiles.kb_v1 import get_profile

domains = ['healthcare', 'finance', 'iot_edge', 'mobile_apps', 'autonomous_vehicles']
print("=" * 110)
print(f"{'Domain':22s} | {'Dataset Name':24s} | {'Classes':7s} | {'Train / Test':17s} | {'Source':12s} | {'Partitioning Strategy'}")
print("-" * 110)

for d in domains:
    adapter = get_domain_adapter(d)
    prof = get_profile(d)
    train_ds, test_ds = adapter.load_data()
    ds_name = prof.name.split('(')[-1].rstrip(')')
    partition_info = f"Dirichlet non-IID (alpha={prof.heterogeneity_alpha}), {prof.typical_num_clients} clients, 1k cap"
    print(f"{d:22s} | {ds_name:24s} | {adapter.num_classes:<7d} | {len(train_ds):6d} / {len(test_ds):6d}  | {adapter.dataset_source:12s} | {partition_info}")
print("=" * 110)
