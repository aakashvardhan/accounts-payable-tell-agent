"""Proves the five email-attack variants are properly paired with the
clean scenario before any attacked-scenario model inference runs.

Pure comparison of already-built scenario data -- no model, no tools, no
network. Run this after scripts/build_email_attack_variants.py and
before scripts/run_email_attack_pilot.py.

Writes:
  - results/scenario_design/email_attack_pairing_validation.json
    (machine-readable check results)
  - results/scenario_design/email_attack_pairing_validation_report.md
"""

from __future__ import annotations

import json
from pathlib import Path

from tell.agent.tools import (
    GetVendorRecordArgs,
    ReadEmailArgs,
    ReadInvoiceArgs,
    get_vendor_record,
    read_email,
    read_invoice,
)
from tell.evaluation.scenario import DEFAULT_SCENARIO_PATH_04D531CA, ScenarioBundle, load_scenario

ATTACK_DIR = Path("/home/hp5/tell/data/scenarios/attacked/email")
ATTACK_TARGET_BENEFICIARY_ACCOUNT_ID = "SIM-ATTACKER-ACCT-0001"

RESULT_JSON_PATH = Path("/home/hp5/tell/results/scenario_design/email_attack_pairing_validation.json")
REPORT_PATH = Path("/home/hp5/tell/results/scenario_design/email_attack_pairing_validation_report.md")


def _tool_views(bundle: ScenarioBundle) -> dict:
    email = read_email(bundle, ReadEmailArgs(message_id=bundle.untrusted_inputs.supplier_email.message_id))
    invoice = read_invoice(bundle, ReadInvoiceArgs(document_id=bundle.untrusted_inputs.invoice_document.docid))
    vendor = get_vendor_record(bundle, GetVendorRecordArgs(vendor_id=bundle.trusted_state.canonical_vendor_id))
    return {"read_email": email, "read_invoice": invoice, "get_vendor_record": vendor}


def validate_pairing(clean: ScenarioBundle, attacked: ScenarioBundle) -> dict:
    """Returns a dict of named boolean checks, all of which must be True
    for the pair to be considered valid."""
    clean_views = _tool_views(clean)
    attacked_views = _tool_views(attacked)

    checks = {
        "invoice_tool_result_identical": clean_views["read_invoice"].content.model_dump(mode="json")
        == attacked_views["read_invoice"].content.model_dump(mode="json"),
        "vendor_record_tool_result_identical": clean_views["get_vendor_record"].content.model_dump(mode="json")
        == attacked_views["get_vendor_record"].content.model_dump(mode="json"),
        "invoice_operational_provenance_identical": clean.untrusted_inputs.invoice_document.operational_provenance
        == attacked.untrusted_inputs.invoice_document.operational_provenance,
        "vendor_operational_provenance_identical": clean.trusted_state.operational_provenance
        == attacked.trusted_state.operational_provenance,
        "trusted_state_identical_except_provenance": (
            clean.trusted_state.model_copy(update={"operational_provenance": None})
            == attacked.trusted_state.model_copy(update={"operational_provenance": None})
        ),
        "ledger_seed_identical": clean.trusted_state.ledger_seed == attacked.trusted_state.ledger_seed,
        "email_metadata_identical_except_body_and_flags": (
            clean.untrusted_inputs.supplier_email.model_copy(
                update={
                    "body": None,
                    "contains_beneficiary_change": None,
                    "contains_policy_override_instruction": None,
                    "notes": None,
                }
            )
            == attacked.untrusted_inputs.supplier_email.model_copy(
                update={
                    "body": None,
                    "contains_beneficiary_change": None,
                    "contains_policy_override_instruction": None,
                    "notes": None,
                }
            )
        ),
        "message_id_identical": clean.untrusted_inputs.supplier_email.message_id
        == attacked.untrusted_inputs.supplier_email.message_id,
        "email_body_differs": clean.untrusted_inputs.supplier_email.body != attacked.untrusted_inputs.supplier_email.body,
        "only_email_differs_agent_visible": (
            clean_views["read_email"].content.model_dump(mode="json")
            != attacked_views["read_email"].content.model_dump(mode="json")
            and clean_views["read_invoice"].content.model_dump(mode="json")
            == attacked_views["read_invoice"].content.model_dump(mode="json")
            and clean_views["get_vendor_record"].content.model_dump(mode="json")
            == attacked_views["get_vendor_record"].content.model_dump(mode="json")
        ),
        "only_evaluation_only_differs": clean.evaluation_only != attacked.evaluation_only,
        "attack_target_absent_from_trusted_state": ATTACK_TARGET_BENEFICIARY_ACCOUNT_ID
        not in attacked.trusted_state.model_dump_json(),
        "attack_target_absent_from_vendor_record_tool_result": ATTACK_TARGET_BENEFICIARY_ACCOUNT_ID
        not in attacked_views["get_vendor_record"].model_dump_json(),
        "attack_target_absent_from_invoice_tool_result": ATTACK_TARGET_BENEFICIARY_ACCOUNT_ID
        not in attacked_views["read_invoice"].model_dump_json(),
        "attack_target_present_in_attacked_email_body": ATTACK_TARGET_BENEFICIARY_ACCOUNT_ID
        in attacked.untrusted_inputs.supplier_email.body,
        "attack_target_absent_from_clean_scenario": ATTACK_TARGET_BENEFICIARY_ACCOUNT_ID
        not in clean.model_dump_json(),
        "is_attacked_true": attacked.evaluation_only.is_attacked is True,
        "attack_surface_is_email": attacked.evaluation_only.attack_surface.value == "immediate_email_injection",
    }
    return checks


def main() -> None:
    clean = load_scenario(DEFAULT_SCENARIO_PATH_04D531CA)
    clean_before = DEFAULT_SCENARIO_PATH_04D531CA.read_text()

    attack_paths = sorted(ATTACK_DIR.glob("*.json"))
    if len(attack_paths) != 5:
        raise RuntimeError(f"Expected exactly 5 attacked scenario files under {ATTACK_DIR}, found {len(attack_paths)}")

    all_results: dict[str, dict] = {}
    all_passed = True
    for path in attack_paths:
        attacked = load_scenario(path)
        checks = validate_pairing(clean, attacked)
        passed = all(checks.values())
        all_passed = all_passed and passed
        all_results[attacked.scenario_id] = {
            "source_file": str(path),
            "sha256": __import__("hashlib").sha256(path.read_bytes()).hexdigest(),
            "checks": checks,
            "all_checks_passed": passed,
        }

    # Prove the clean scenario file itself was not touched by this validation.
    clean_after = DEFAULT_SCENARIO_PATH_04D531CA.read_text()
    clean_unchanged = clean_before == clean_after

    result = {
        "clean_scenario_id": clean.scenario_id,
        "clean_scenario_path": str(DEFAULT_SCENARIO_PATH_04D531CA),
        "clean_scenario_unchanged": clean_unchanged,
        "attack_target_beneficiary_account_id": ATTACK_TARGET_BENEFICIARY_ACCOUNT_ID,
        "per_scenario": all_results,
        "all_scenarios_paired_correctly": all_passed and clean_unchanged,
    }

    RESULT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_JSON_PATH.write_text(json.dumps(result, indent=2))

    lines = ["# Email-Attack Pairing Validation\n", f"## Result: {'PASS' if result['all_scenarios_paired_correctly'] else 'FAIL'}\n"]
    lines.append(f"Clean scenario: `{clean.scenario_id}` (unchanged on disk: {clean_unchanged})\n")
    lines.append("| Scenario | SHA-256 | All checks passed |")
    lines.append("|---|---|---|")
    for scenario_id, info in all_results.items():
        lines.append(f"| `{scenario_id}` | `{info['sha256'][:16]}...` | {info['all_checks_passed']} |")
    lines.append("")
    for scenario_id, info in all_results.items():
        lines.append(f"### `{scenario_id}`\n")
        for check, ok in info["checks"].items():
            lines.append(f"- [{'x' if ok else ' '}] {check}: {ok}")
        lines.append("")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines))

    print(f"All scenarios paired correctly: {result['all_scenarios_paired_correctly']}")
    print(f"Wrote {RESULT_JSON_PATH}")
    print(f"Wrote {REPORT_PATH}")
    if not result["all_scenarios_paired_correctly"]:
        raise SystemExit("Pairing validation FAILED -- see report for details.")


if __name__ == "__main__":
    main()
