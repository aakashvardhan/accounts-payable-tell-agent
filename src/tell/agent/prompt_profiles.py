"""Typed prompt-profile selection for Tell's autonomous agent loop
(tell.agent.loop) and delayed-memory-poisoning pilot
(tell.agent.memory_loop).

Two profiles exist:

- ``HARDENED_PROVENANCE_PROMPT_V1``: the original, unmodified system
  prompts already in ``tell.agent.loop_prompts`` /
  ``tell.agent.memory_prompts``, which state explicit trust/provenance
  policy (untrusted inputs, beneficiary-change attacks, verification-
  bypass suspicion, memory laundering). This is the **default** for
  every builder function and every loop entry point in this codebase,
  so no existing caller's behavior changes: calling any of the
  ``build_*`` functions or ``run_agent_loop``/``run_session_a``/
  ``run_session_b`` with no ``prompt_profile`` argument renders the
  exact same text as before this module existed.
- ``TASK_ONLY_BASE_V1``: an ablated prompt that keeps the agent role,
  the action schemas, the one-JSON-action-per-turn contract, and the
  ordinary business task, but omits every sentence of explicit
  security/trust/provenance guidance. It exists to test whether the
  hardened prompt's *policy language* -- not any deterministic
  code-level enforcement, which is identical under both profiles -- is
  what produced previously observed safe behavior. See
  ``results/scenario_design/hardened_prompt_policy_audit.md`` for the
  sentence-by-sentence audit that motivated this profile, and
  ``results/scenario_design/task_only_vs_hardened_diff.md`` for the
  proof that only prompt-policy language differs between the two
  profiles.

Selecting a profile only changes which system-prompt template is
rendered by ``tell.agent.loop_prompts``/``tell.agent.memory_prompts``.
It never changes: action schemas (``tell.agent.actions``), tool
execution or tool result content (``tell.agent.tools``), provenance
derivation (``tell.memory.store``, ``tell.evaluation.scenario``),
generation parameters, scenario/attack content, or the loop's own
control flow (``tell.agent.loop``, ``tell.agent.memory_loop``).
"""

from __future__ import annotations

from enum import Enum


class PromptProfile(str, Enum):
    """The two prompt-policy configurations this codebase supports.
    Neither name is "Tell": no profile here uses an activation probe,
    an alarm threshold, LoRA routing, or a Tell gate decision -- both
    are plain system-prompt choices for the unprotected base agent."""

    HARDENED_PROVENANCE_PROMPT_V1 = "hardened_provenance_prompt_v1"
    TASK_ONLY_BASE_V1 = "task_only_base_v1"


DEFAULT_PROMPT_PROFILE = PromptProfile.HARDENED_PROVENANCE_PROMPT_V1

__all__ = ["PromptProfile", "DEFAULT_PROMPT_PROFILE"]
