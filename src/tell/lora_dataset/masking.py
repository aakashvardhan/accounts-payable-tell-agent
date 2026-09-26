"""Prompt/completion tokenization and loss-mask construction, shared by
the corpus-build/validation script, the smoke-training script, and the
masking tests.

The gold completion is rendered as compact JSON (`json.dumps(...,
separators=(",", ":"))` -- no chain-of-thought, matching exactly what
`tell.agent.actions.parse_agent_action` expects the model to output) and
closed with the tokenizer's own EOS turn-closing token
(`<|im_end|>` for Qwen3), so the model learns both the content and where
to stop.

Masking rule: tokenize the prompt alone (chat template,
`add_generation_prompt=True`) and the prompt+completion+EOS together;
verify the prompt's tokenization is an exact prefix of the combined
tokenization (checked, not assumed -- see `MaskedExample.prefix_verified`),
then label every prompt-side token `-100` and every completion+EOS token
its own token id. Never truncates a completion: an example whose combined
length exceeds `max_seq_len` is rejected outright (`MaskedExample.rejected
= True`), not silently cut.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

IGNORE_INDEX = -100


def render_gold_completion_text(gold_action_dict: dict) -> str:
    return json.dumps(gold_action_dict, separators=(",", ":"))


@dataclass(frozen=True)
class MaskedExample:
    sample_id: str
    input_ids: list[int]
    labels: list[int]
    prompt_token_count: int
    completion_token_count: int  # includes the trailing EOS token
    total_token_count: int
    prefix_verified: bool
    rejected: bool
    reject_reason: str | None


def build_masked_example(
    tokenizer,
    *,
    sample_id: str,
    messages: list[dict],
    gold_action_dict: dict,
    max_seq_len: int,
) -> MaskedExample:
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    completion_text = render_gold_completion_text(gold_action_dict)
    full_text = prompt_text + completion_text + tokenizer.eos_token

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

    prefix_verified = full_ids[: len(prompt_ids)] == prompt_ids
    if not prefix_verified:
        return MaskedExample(
            sample_id=sample_id,
            input_ids=[],
            labels=[],
            prompt_token_count=len(prompt_ids),
            completion_token_count=0,
            total_token_count=len(full_ids),
            prefix_verified=False,
            rejected=True,
            reject_reason="prompt_tokenization_not_a_prefix_of_full_tokenization",
        )

    if len(full_ids) > max_seq_len:
        return MaskedExample(
            sample_id=sample_id,
            input_ids=[],
            labels=[],
            prompt_token_count=len(prompt_ids),
            completion_token_count=len(full_ids) - len(prompt_ids),
            total_token_count=len(full_ids),
            prefix_verified=True,
            rejected=True,
            reject_reason=f"total_token_count {len(full_ids)} exceeds max_seq_len {max_seq_len} (never truncated)",
        )

    labels = [IGNORE_INDEX] * len(prompt_ids) + full_ids[len(prompt_ids) :]
    assert len(labels) == len(full_ids)

    return MaskedExample(
        sample_id=sample_id,
        input_ids=full_ids,
        labels=labels,
        prompt_token_count=len(prompt_ids),
        completion_token_count=len(full_ids) - len(prompt_ids),
        total_token_count=len(full_ids),
        prefix_verified=True,
        rejected=False,
        reject_reason=None,
    )


def decode_non_masked_labels(tokenizer, labels: list[int]) -> str:
    """Decodes only the non-`-100` label tokens -- used by tests/diagnostics
    to prove the loss-bearing span decodes to exactly the gold completion."""
    ids = [t for t in labels if t != IGNORE_INDEX]
    return tokenizer.decode(ids)


__all__ = ["IGNORE_INDEX", "MaskedExample", "render_gold_completion_text", "build_masked_example", "decode_non_masked_labels"]
