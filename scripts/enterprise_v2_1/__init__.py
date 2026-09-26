"""Enterprise corpus v2.1 / enterprise benchmark v1.1 protocol correction.

Versioned successor of `scripts/enterprise_v2` (which stays byte-identical
because its hashes are recorded in the frozen v2 manifests). Modules that
did not need to change (docile_profile, ids, selection, economics) are
imported from `enterprise_v2` unchanged; builders/policy/audits/wording
are versioned copies with the v2.1 corrections.
"""
