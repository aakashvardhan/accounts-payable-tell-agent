"""Pure formulas for the enterprise workload / break-even model (no I/O).

Break-even threshold formula (as specified for this task):

    required_monthly_net_savings = hardware_cost / target_break_even_months
                                   + monthly_local_operating_cost
    required_workflows_per_month = required_monthly_net_savings
                                   / avoided_cost_per_workflow

`avoided_cost_per_workflow` is the commercial-API cost one workflow would
have incurred. If it is zero or negative there is no break-even (None).
Nothing in this module is imported by corpus selection.
"""

from __future__ import annotations

import math


def api_cost_per_workflow(input_tokens: float, output_tokens: float, usd_per_m_input: float, usd_per_m_output: float) -> float:
    return input_tokens / 1_000_000 * usd_per_m_input + output_tokens / 1_000_000 * usd_per_m_output


def required_monthly_net_savings(hardware_cost: float, target_months: float, monthly_local_operating_cost: float) -> float:
    return hardware_cost / target_months + monthly_local_operating_cost


def required_workflows_per_month(hardware_cost: float, target_months: float, monthly_local_operating_cost: float, avoided_cost_per_workflow: float) -> float | None:
    if avoided_cost_per_workflow <= 0:
        return None
    return required_monthly_net_savings(hardware_cost, target_months, monthly_local_operating_cost) / avoided_cost_per_workflow


def min_whole_workflows(required: float | None) -> int | None:
    return None if required is None else math.ceil(required - 1e-9)


def break_even_months(hardware_cost: float, monthly_avoided_cost: float) -> float | None:
    return None if monthly_avoided_cost <= 0 else hardware_cost / monthly_avoided_cost


def net_savings_over(months: int, hardware_cost: float, monthly_avoided_cost: float) -> float:
    return months * monthly_avoided_cost - hardware_cost


def serial_compute_hours(workflows: int, seconds_per_workflow: float) -> float:
    return workflows * seconds_per_workflow / 3600.0


def largest_remainder(total: int, weights: dict[str, float]) -> dict[str, int]:
    """Deterministic apportionment of `total` integer replays across keys
    in proportion to `weights` (Hamilton / largest-remainder method; ties
    broken by key). Counts always sum exactly to `total`."""
    wsum = sum(weights.values())
    if wsum <= 0:
        raise ValueError("weights must sum to a positive value")
    quotas = {k: total * w / wsum for k, w in weights.items()}
    counts = {k: math.floor(q) for k, q in quotas.items()}
    remaining = total - sum(counts.values())
    order = sorted(weights, key=lambda k: (-(quotas[k] - counts[k]), k))
    for k in order[:remaining]:
        counts[k] += 1
    return counts


__all__ = [
    "api_cost_per_workflow",
    "required_monthly_net_savings",
    "required_workflows_per_month",
    "min_whole_workflows",
    "break_even_months",
    "net_savings_over",
    "serial_compute_hours",
    "largest_remainder",
]
