"""CPU-only, static/structural tests for tool-access boundaries between
Agent 1 and Agent S, and for import-time safety of the GPU-backed modules.
No model, no CUDA, no real payment/email/database anywhere in this file.
"""
from __future__ import annotations

import ast
import importlib
import inspect
import sys

import pytest


def _imported_names(module) -> set[str]:
    tree = ast.parse(inspect.getsource(module))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            names.add(node.module or "")
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
    return names


def test_agent_1_tool_module_never_imports_trusted_lookups():
    import tell.agent.tools as tools_mod

    imported = _imported_names(tools_mod)
    assert "tell.agent.trusted_lookups" not in imported
    assert "tell.agent.agent_s_tools" not in imported
    assert not any("trusted_lookup" in n for n in imported)


def test_agent_s_tools_module_only_wraps_read_only_lookup_functions():
    import tell.agent.agent_s_tools as mod

    public = [name for name in dir(mod) if not name.startswith("_")]
    for name in public:
        obj = getattr(mod, name)
        if inspect.isfunction(obj) or inspect.isclass(obj):
            src = inspect.getsource(obj) if inspect.isfunction(obj) else inspect.getsource(obj)
            for forbidden in ("write", "approve", "clear_alarm", "execute", ".save(", ".insert(", ".update(", ".delete("):
                assert forbidden not in src.lower().replace("tool_name", ""), f"{name} source unexpectedly contains {forbidden!r}"


def test_trusted_lookup_queries_accept_only_stable_ids_not_arbitrary_untrusted_dict():
    """An untrusted document (e.g. attacker-controlled email/invoice text)
    cannot masquerade as a trusted lookup result: the query models are
    `extra="forbid"` and accept only the two stable identifiers, and the
    result models are constructed solely by the provider, never from
    caller-supplied free-form content."""
    from tell.agent.trusted_lookups import DisputeCaseQuery, InvoicePaymentHistoryQuery

    for cls in (InvoicePaymentHistoryQuery, DisputeCaseQuery):
        assert cls.model_config.get("extra") == "forbid"
        fields = set(cls.model_fields.keys())
        assert fields == {"invoice_document_id", "vendor_id"}, f"{cls.__name__} unexpectedly accepts extra fields: {fields}"
        with pytest.raises(Exception):
            cls(invoice_document_id="x", vendor_id="y", beneficiary_account_id="attacker-injected")


def test_no_lookup_result_type_has_a_write_or_approve_field():
    from tell.agent import trusted_lookups as mod

    for name in ("InvoicePaymentHistoryRecord", "InvoicePaymentHistoryResult", "DisputeCaseRecord", "DisputeCaseLookupResult"):
        cls = getattr(mod, name)
        fields = set(cls.model_fields.keys())
        forbidden_field_names = {"approve", "write", "clear_alarm", "execute", "authorize"}
        assert not (fields & forbidden_field_names), f"{name} has a forbidden field: {fields & forbidden_field_names}"


# --- import-time safety of GPU-backed modules -------------------------------------------


# The actual CUDA "inspection" calls this restriction is about -- functions
# that query or initialize device state. `transformers`/`torch._dynamo`'s
# own internals reference type objects like `torch.cuda._CudaDeviceProperties`
# merely by being imported (confirmed empirically: poisoning ALL torch.cuda
# attribute access fails on that harmless reference, which happens purely
# from `import transformers`, an existing, pre-this-task import chain) --
# that is not a device inspection, so only the real query/init functions
# below are poisoned, not the whole namespace.
_INSPECTION_CALLS = ("is_available", "device_count", "get_device_name", "get_device_properties",
                    "get_device_capability", "current_device", "memory_allocated", "memory_reserved",
                    "max_memory_allocated", "max_memory_reserved", "set_device", "synchronize", "init",
                    "reset_peak_memory_stats", "get_rng_state", "set_rng_state")


@pytest.mark.parametrize("module_name", ["tell.safety.adapter_runtime", "tell.safety.agent_model_factory",
                                         "tell.agent.operational_router", "tell.agent.real_agent_s_investigator"])
def test_module_import_performs_no_cuda_inspection(module_name):
    """Importing these GPU-capable modules must never CALL a CUDA query/
    init function (is_available, device_count, get_device_name, ...).
    Poisons only those specific attributes on torch.cuda, leaving harmless
    type/class references (which transformers' own internals touch at
    import time regardless of this task's code) untouched."""
    for name in list(sys.modules):
        if name == module_name or name.startswith(module_name + "."):
            del sys.modules[name]

    import torch

    originals = {}
    for name in _INSPECTION_CALLS:
        if hasattr(torch.cuda, name):
            originals[name] = getattr(torch.cuda, name)

            def _poisoned(*a, _name=name, **kw):
                raise AssertionError(f"import of {module_name} called torch.cuda.{_name}() at import time")

            setattr(torch.cuda, name, _poisoned)
    try:
        importlib.import_module(module_name)
    finally:
        for name, orig in originals.items():
            setattr(torch.cuda, name, orig)


def test_agent_model_factory_construction_does_no_import_time_model_load():
    """Constructing the factory (not calling get_agent_1/get_agent_s) must
    not import transformers/peft's heavy loading machinery beyond what was
    already imported for typing -- checked by asserting the factory object
    holds no live model reference until a get_* method runs."""
    from tell.safety.agent_model_factory import AgentModelFactory
    from tell.safety.agent_s_config import AgentSRuntimeConfig

    c = AgentSRuntimeConfig(agent_s_adapter_path="/nonexistent")
    factory = AgentModelFactory(c, runtime_builder=lambda: (_ for _ in ()).throw(AssertionError("runtime_builder must not be called at construction time")))
    assert factory._runtime is None
