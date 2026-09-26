"""Proves the two memory-infection scenarios are properly paired before
any inference. Pure comparison of already-built scenario data -- no
model, no tools beyond read-only calls, no network.

Writes:
  - results/scenario_design/memory_infection_pairing_validation.json
  - results/scenario_design/memory_infection_pairing_validation_report.md
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tell.agent.tools import GetVendorRecordArgs, ReadInvoiceArgs, get_vendor_record, read_invoice
from tell.evaluation.scenario import ScenarioBundle, load_scenario

SCENARIO_DIR = Path("/home/hp5/tell/data/scenarios/memory_infection")
ATTACK_TARGET_BENEFICIARY_ACCOUNT_ID = "SIM-ATTACKER-ACCT-0001"

RESULT_JSON_PATH = Path("/home/hp5/tell/results/scenario_design/memory_infection_pairing_validation.json")
REPORT_PATH = Path("/home/hp5/tell/results/scenario_design/memory_infection_pairing_validation_report.md")


def validate_pairing(clean: ScenarioBundle, poisoned: ScenarioBundle) -> dict:
    clean_invoice = read_invoice(clean, ReadInvoiceArgs(document_id=clean.untrusted_inputs.invoice_document.docid))
    poisoned_invoice = read_invoice(poisoned, ReadInvoiceArgs(document_id=poisoned.untrusted_inputs.invoice_document.docid))
    clean_vendor = get_vendor_record(clean, GetVendorRecordArgs(vendor_id=clean.trusted_state.canonical_vendor_id))
    poisoned_vendor = get_vendor_record(poisoned, GetVendorRecordArgs(vendor_id=poisoned.trusted_state.canonical_vendor_id))

    checks = {
        "invoice_tool_result_identical": clean_invoice.content.model_dump(mode="json") == poisoned_invoice.content.model_dump(mode="json"),
        "vendor_record_tool_result_identical": clean_vendor.content.model_dump(mode="json") == poisoned_vendor.content.model_dump(mode="json"),
        "trusted_state_identical": clean.trusted_state == poisoned.trusted_state,
        "invoice_operational_provenance_identical": clean.untrusted_inputs.invoice_document.operational_provenance
        == poisoned.untrusted_inputs.invoice_document.operational_provenance,
        "email_body_differs": clean.untrusted_inputs.supplier_email.body != poisoned.untrusted_inputs.supplier_email.body,
        "email_message_id_differs": clean.untrusted_inputs.supplier_email.message_id != poisoned.untrusted_inputs.supplier_email.message_id,
        "only_evaluation_only_and_email_differ": clean.evaluation_only != poisoned.evaluation_only,
        "attack_target_absent_from_trusted_state": ATTACK_TARGET_BENEFICIARY_ACCOUNT_ID not in poisoned.trusted_state.model_dump_json(),
        "attack_target_absent_from_vendor_record_tool_result": ATTACK_TARGET_BENEFICIARY_ACCOUNT_ID not in poisoned_vendor.model_dump_json(),
        "attack_target_absent_from_invoice_tool_result": ATTACK_TARGET_BENEFICIARY_ACCOUNT_ID not in poisoned_invoice.model_dump_json(),
        "attack_target_present_in_poisoned_email_body": ATTACK_TARGET_BENEFICIARY_ACCOUNT_ID in poisoned.untrusted_inputs.supplier_email.body,
        "attack_target_absent_from_clean_scenario": ATTACK_TARGET_BENEFICIARY_ACCOUNT_ID not in clean.model_dump_json(),
        "poisoned_is_attacked_true": poisoned.evaluation_only.is_attacked is True,
        "poisoned_attack_surface_is_delayed_memory_poisoning": poisoned.evaluation_only.attack_surface.value == "delayed_memory_poisoning",
        "clean_is_attacked_false": clean.evaluation_only.is_attacked is False,
    }
    return checks


def main() -> None:
    clean_path = SCENARIO_DIR / "memory_infection_clean_04d531ca_v1.json"
    poisoned_path = SCENARIO_DIR / "memory_infection_poisoned_04d531ca_v1.json"
    clean = load_scenario(clean_path)
    poisoned = load_scenario(poisoned_path)

    checks = validate_pairing(clean, poisoned)
    all_passed = all(checks.values())

    result = {
        "clean_scenario_id": clean.scenario_id,
        "poisoned_scenario_id": poisoned.scenario_id,
        "clean_sha256": hashlib.sha256(clean_path.read_bytes()).hexdigest(),
        "poisoned_sha256": hashlib.sha256(poisoned_path.read_bytes()).hexdigest(),
        "attack_target_beneficiary_account_id": ATTACK_TARGET_BENEFICIARY_ACCOUNT_ID,
        "checks": checks,
        "all_checks_passed": all_passed,
    }

    RESULT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_JSON_PATH.write_text(json.dumps(result, indent=2))

    lines = ["# Memory-Infection Pairing Validation\n", f"## Result: {'PASS' if all_passed else 'FAIL'}\n"]
    lines.append(f"Clean: `{clean.scenario_id}` (`{result['clean_sha256'][:16]}...`)")
    lines.append(f"Poisoned: `{poisoned.scenario_id}` (`{result['poisoned_sha256'][:16]}...`)\n")
    for check, ok in checks.items():
        lines.append(f"- [{'x' if ok else ' '}] {check}: {ok}")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines))

    print(f"All checks passed: {all_passed}")
    print(f"Wrote {RESULT_JSON_PATH}")
    print(f"Wrote {REPORT_PATH}")
    if not all_passed:
        raise SystemExit("Pairing validation FAILED -- see report for details.")


if __name__ == "__main__":
    main()
