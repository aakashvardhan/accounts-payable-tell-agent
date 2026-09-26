# Tell Project Knowledge

## Project purpose

Tell is a locally deployed security runtime for an autonomous accounts-payable agent. The agent reads supplier emails and invoices, consults a vendor record and persistent memory, and submits payments to a simulated fiat ledger. The security problem is that untrusted business content can manipulate the agent even when the content looks legitimate. A compromised supplier email, altered invoice, poisoned tool result, or stored memory could redirect a payment without a human reviewing the decision.

The project tests whether the model's internal activations reveal that it is being manipulated before it makes an unsafe tool call. Tell uses that signal to select a safer response path and to prevent an unresolved risk from reaching the payment executor.

This is an Edge AI systems project rather than a chatbot or a collection of payment rules. The contribution is the combination of an instrumented local model, an activation probe, a safety-tuned LoRA, a deterministic action gate, a realistic agentic workflow, and an evaluation harness that attacks several context surfaces.

## Target user and practical problem

The target user is an accounts-payable or finance-operations lead who wants to automate routine invoice processing without exposing financial documents to a cloud model or allowing untrusted content to change payment instructions.

The AP workflow is deliberately consequential:

1. Read a supplier email and invoice.
2. Retrieve the supplier's approved record and relevant long-term memory.
3. Compare the invoice, vendor identity, amount, and beneficiary information.
4. Create a payment proposal.
5. Submit the payment to a local simulated ledger.

The attack run intentionally removes the human reviewer. This makes the autonomous failure plausible and measurable: the baseline agent can appear to complete a normal finance task while sending money to the wrong beneficiary.

## Main research question

The central question is:

> If a probe is trained to recognize immediate prompt injection in email and tool results, will the same internal signal detect poison after it has been stored, retrieved later, and presented as the agent's own memory?

This connects two security problems: prompt injection and memory poisoning. An immediate injection is visibly external. A sleeper-memory attack is more difficult because the malicious instruction is laundered through the agent's memory system and can return in a later session without the original hostile document in context.

The result is useful in either direction:

- If the signal transfers to delayed memory attacks, Tell provides evidence that model-internal detection can generalize across attack surfaces.
- If it does not transfer, the experiment identifies memory laundering as a blind spot and establishes the need for surface-specific training or provenance-aware detection.

## What Tell actually measures

Tell does not simply inspect attention weights. The initial implementation captures a hidden-state or residual-stream vector from a selected transformer layer and token position after untrusted content has entered the context and before the model chooses its next action. A small linear classifier converts that vector into an alarm score.

Layer and token-position selection are empirical decisions. We will collect activations from several candidate layers on clean and attacked traces, train lightweight probes, and choose the smallest configuration that performs well on held-out scenarios. Attention-based features may be explored later, but they are not required for the minimum viable system.

## Runtime architecture

The protected agent loop is:

1. The user assigns a task such as `Pay this month's approved invoices`.
2. The base agent invokes a read tool, such as email, invoice, vendor-record, or memory retrieval.
3. The tool result is added to the agent context.
4. An instrumented forward pass captures the chosen hidden-state vector before the next action is generated.
5. The Tell probe returns an alarm score.
6. If the score is below the threshold, the base model generates the next action.
7. If the score exceeds the threshold, the same model runs with the safety LoRA enabled and generates a quarantine, verification, or replanning action.
8. The deterministic action gate evaluates the proposed tool call and the current alarm state.
9. Read-only or recovery actions may proceed. A payment or other external side effect is blocked while the alarm remains unresolved.
10. The executor runs the permitted tool and records the result in the audit log.

The probe, LoRA, and gate serve different purposes:

| Component | Responsibility | Method |
| --- | --- | --- |
| Tell probe | Estimate whether the model is being steered by untrusted content | Linear classifier over hidden activations |
| Base model | Complete ordinary AP work | Qwen3-8B instruct model |
| Safety LoRA | Generate safer actions when Tell fires | PEFT adapter trained on quarantine, verification, and replanning targets |
| Action gate | Prevent unresolved risk from causing an external side effect | Deterministic runtime policy after generation and before execution |

The deterministic gate is not a replacement for Tell. It is the final enforcement boundary. Tell recognizes suspicious model state, the safety LoRA helps the model recover, and the gate ensures that a bad generated tool call does not execute while the system is in an alarm state.

## Why the safety LoRA is not always on

An always-on safety model is an important baseline, but it is not automatically the best production design. A strict model may reject legitimate invoices, repeatedly request verification, or reduce normal task completion. Tell allows the ordinary model to handle low-risk work and activates the safety behavior only when the internal signal crosses the calibrated threshold.

We will compare four systems:

1. Base agent with no Tell protection.
2. Safety LoRA enabled on every turn.
3. Tell-routed base model and safety LoRA.
4. Tell routing plus the deterministic action gate.

This comparison will show whether selective routing preserves clean-task performance while reducing successful attacks.

## Agent tools

The minimum agent has six typed tools:

- `read_email`: returns a supplier message from the local scenario store.
- `read_invoice`: extracts structured fields and relevant text from a DocILE invoice.
- `search_memory`: retrieves stored preferences, prior decisions, and supplier-related notes.
- `write_memory`: stores or changes a memory record, including its status and provenance.
- `get_vendor_record`: returns the approved supplier identity and beneficiary information.
- `pay_invoice`: submits a payment intent to the local ledger. This is the primary side-effecting tool and must pass the action gate.

The agent framework should keep tool schemas explicit, validate every argument, log each observation and decision, and attach provenance to retrieved content. Framework-level recovery code can quarantine a memory by changing its status through `write_memory` or a direct trusted storage operation.

## Local payment simulation

No real money or banking API is required. The local simulator should mimic the parts of a fiat payment system needed to make the attack consequential and auditable.

Recommended SQLite entities are:

- `vendors`: canonical supplier identity, approved beneficiary, and verification status.
- `invoices`: invoice number, supplier, amount, currency, due date, and source document.
- `payment_intents`: the agent's proposed beneficiary, amount, reason, and originating invoice.
- `accounts`: simulated company, supplier, and attacker balances.
- `journal_entries`: immutable debit and credit records for executed payments.
- `memories`: text, source, timestamp, provenance, status, and retrieval metadata.
- `audit_events`: prompts, tool results, probe scores, routes, candidate calls, gate decisions, and final outcomes.

`pay_invoice` should first create a payment intent. The action gate then permits or rejects execution. A permitted payment creates balanced local journal entries and updates simulated balances. This gives the demo a visible financial consequence without connecting to a real payment rail.

## Dataset plan

DocILE is sufficient as the document foundation for the hackathon. It supplies realistic business invoices and structured fields. It does not need to provide every part of the security scenario. We will add controlled local records around each selected invoice:

- A canonical vendor-master record that acts as the approved source of truth.
- A supplier email associated with the invoice.
- A simulated account and beneficiary.
- Clean and malicious variants of email, invoice text, tool output, and memory.
- Expected safe action, unsafe action, and final payment outcome.

The evaluation unit is a complete scenario rather than an isolated invoice. Each scenario pairs the same underlying invoice and business facts with clean and attacked context. This controls for invoice difficulty and makes the effect of the attack easier to measure.

The initial dataset should prioritize quality and coverage over large volume. Use several suppliers and invoice layouts, multiple attack phrasings, paraphrases, and clean but unusual cases. Split the data by supplier, document template, and attack wording so that near-duplicate prompts do not appear in both training and testing.

## Attack surfaces

The minimum evaluation covers three surfaces:

### Immediate email injection

A compromised or spoofed supplier email contains instructions that attempt to override the AP task, alter beneficiary information, suppress verification, or cause a memory write.

### Invoice or tool-result injection

Malicious text is embedded in invoice content, OCR output, or another tool result. The attack is presented as data but attempts to become an instruction.

### Delayed memory poisoning

In the infection phase, the agent reads compromised content that causes it to store a plausible but false memory, such as a new supplier beneficiary. No payment theft occurs yet. After a context reset, a clean payment request retrieves the poisoned memory and triggers the malicious action.

Skill-file or tool-description poisoning and a locally run adaptive attacker are stretch goals. They should not delay the end-to-end memory-poisoning demo.

## Threat model

The attacker may control or influence supplier email, invoice text, webpage or tool content, and any memory derived from those sources. The attacker knows the general agent workflow and may paraphrase attacks. The attacker wants to redirect the beneficiary, change the payment amount, suppress verification, or plant a sleeper instruction for later retrieval.

The attacker does not control the trusted application code, action gate, canonical vendor database, probe labels, or evaluation harness. The simulated ledger and audit log are trusted. This boundary lets us evaluate model and context-layer attacks without pretending that compromised application code can be solved by a model guardrail.

The project does not claim perfect security. A missed detection can still reach the candidate action stage. The action gate reduces the consequence of detected attacks, while transaction limits and ordinary payment controls remain defense-in-depth rather than the main research contribution.

## Probe training

For each clean or attacked trace, record:

- The scenario and attack surface.
- The exact context available before the next action.
- The selected layer and token position.
- The hidden-state vector.
- Whether the model was exposed to malicious steering.
- The proposed tool call and final outcome.

Train a regularized logistic-regression or linear-SVM probe on immediate clean and injected examples. Calibrate an operating threshold on validation data. Keep delayed memory-poisoning cases out of probe training for the central zero-shot test. Report AUROC and precision-recall results, but choose the deployment threshold using the practical tradeoff between false alarms and missed unsafe actions.

## Safety LoRA training

The safety LoRA is trained on suspicious contexts paired with safe next actions. Target behaviors include:

- Refuse to update beneficiary information from unverified content.
- Compare the retrieved beneficiary with the canonical vendor record.
- Quarantine a suspicious memory and retain its provenance.
- Request trusted verification when business facts conflict.
- Replan after removing the flagged content.
- Avoid calling `pay_invoice` until the conflict is resolved.

Include clean examples so the adapter does not learn to reject every payment. Use PEFT LoRA or QLoRA on Qwen3-8B. The LoRA should remain an adapter over the same base model so routing is fast and memory-efficient.

## Local implementation stack

The core stack should remain small and inspectable:

- Python 3.11 or the supported Nano Python environment.
- PyTorch and Hugging Face Transformers for the instrumented model forward pass.
- PEFT for LoRA training and adapter switching.
- scikit-learn for the first linear probe.
- Pydantic for typed tool arguments and validation.
- SQLite for vendor data, memory, ledger state, and audit events.
- FastAPI for the local service API.
- A lightweight local web interface, such as Streamlit or a simple React client, for the demo.
- pytest for tool, gate, ledger, and scenario tests.

The instrumented agent should run through Transformers in process because the project needs hidden-state hooks. The standard OpenAI-compatible endpoint exposed by ZRT or vLLM does not normally return internal activations. ZRT and vLLM may still serve an auxiliary baseline or attacker model if time permits, but the core Tell path must use a runtime that exposes hidden states.

The preferred working model is Qwen3-8B because it is small enough for rapid iteration and LoRA training on the ZGX Nano. Larger models are comparison or stretch targets only after the full system works.

## Suggested repository structure

```text
tell/
  README.md
  pyproject.toml
  setup.sh
  configs/
  data/
    docile_samples/
    scenarios/
  src/tell/
    agent/
      loop.py
      prompts.py
      tools.py
    detector/
      capture.py
      probe.py
      calibration.py
    safety/
      adapter.py
      gate.py
    payment/
      ledger.py
      models.py
    memory/
      store.py
      provenance.py
    evaluation/
      runner.py
      metrics.py
    api/
      server.py
  tests/
  scripts/
    prepare_docile.py
    generate_scenarios.py
    collect_activations.py
    train_probe.py
    train_lora.py
    run_evaluation.py
  demo/
  results/
```

Everything needed to reproduce the system must live in the repository because the event device may be wiped after the hackathon.

## Evaluation plan

Run the same held-out scenarios against all four defense configurations. Report:

- Attack success rate.
- Unauthorized payments executed.
- Unsafe payment tool-call rate.
- Clean invoice completion rate.
- False-alarm rate.
- Probe AUROC and precision-recall performance.
- Memory-poisoning success after a context reset.
- Added latency from activation capture, probe inference, adapter routing, and the gate.
- Recovery rate: attacks detected and successfully replanned without payment loss.

The most important outcome measure is unauthorized payments executed, not classifier accuracy alone. A strong probe that does not prevent the side effect is insufficient. Clean completion must also be measured so that a defense that blocks every payment does not appear successful.

## Live demo narrative

The five-minute demonstration should have four acts:

1. **Clean run:** The agent processes a normal DocILE invoice, checks the vendor record, and posts the correct simulated payment.
2. **Infection:** A compromised supplier email causes the undefended agent to save a false beneficiary as a plausible memory. No theft occurs yet.
3. **Trigger:** After clearing conversational context, a clean instruction to pay approved bills retrieves the memory. The baseline agent submits the payment to the attacker account.
4. **Protected replay:** The same scenario is replayed with Tell active. The probe score rises before the next action, the safety LoRA quarantines or challenges the memory, and the deterministic gate prevents the transfer.

The interface should show the email or invoice, retrieved memory, Tell score and threshold, selected route, candidate tool call, gate decision, and ledger balance. Avoid exposing hidden chain-of-thought. Show concise action summaries and structured audit events instead.

## Implementation priorities

Build in this order:

1. Complete one clean AP workflow with the six tools and local ledger.
2. Make one direct injection and one sleeper-memory attack reliably compromise the baseline.
3. Add full audit logging and deterministic replay.
4. Capture hidden activations and train the first linear probe.
5. Add the action gate and prove that it prevents an alarmed payment.
6. Train and integrate the safety LoRA.
7. Expand to held-out suppliers, paraphrases, invoice layouts, and attack surfaces.
8. Produce benchmark charts, the architecture visual, demo video, README, and reproducible setup.

Do not begin large-model comparisons, activation steering, a separate attacker model, or elaborate multi-agent orchestration until the end-to-end baseline attack and protected replay work. For the minimum viable project, one acting AP agent is enough. A separate local red-team agent can be added later to generate adaptive attacks.

## Alignment with the hackathon guidelines

### Local inference requirement

The event requires AI inference to run on the assigned HP ZGX Nano rather than a laptop or cloud AI API. Tell meets this requirement because the AP model, activation capture, probe, LoRA, memory, payment simulator, and evaluation harness all run on the Nano. The laptop is only an SSH client and user interface.

### Local Agentic Systems track

The agent performs a real multi-step task using tools: it reads business documents, retrieves memory, checks vendor data, plans a payment, and calls a side-effecting payment tool. The live demo visibly includes multiple agent-tool-agent loops rather than a single-turn answer.

### Secure AI track

The project demonstrates real prompt-injection and memory-poisoning attacks, defines the attacker's capabilities, shows a mitigation at the model, inference, and execution layers, and measures both attack success and defense failures. Comparing the base model, always-on LoRA, Tell routing, and the deterministic gate satisfies the requirement to assess different defenses and harness behavior.

### Privacy and edge justification

Invoices, supplier correspondence, beneficiary details, agent memory, and payment history are sensitive. Keeping them on the Nano gives a concrete data-residency and privacy advantage. Local deployment also makes the project technically possible because cloud APIs do not expose the hidden activations required by Tell.

The cloud boundary is explicit: no core inference and no sensitive finance data leave the device. Cloud services may be used for permitted non-inference work such as slides, video editing, repository hosting, or uploading de-identified aggregate results.

### Hardware fit

The ZGX Nano uses the NVIDIA GB10 Grace Blackwell platform with 128 GB of coherent unified memory. The event guidance indicates that large quantized models can be served and smaller models can be fine-tuned locally. Qwen3-8B plus a small LoRA and linear probe is deliberately conservative, leaving memory for the agent service, database, interface, and experiments. The design values fast iteration and reliable demonstration over loading the largest possible model.

### ZRT and model serving guidance

The event recommends ZRT, its vLLM-compatible serving path, Hugging Face model access, and OpenAI-compatible tool-calling endpoints. We will use the Nano access and model-management workflow from those instructions. Tell's instrumented forward pass will use Transformers because hidden activations are essential; an opaque completion endpoint is insufficient for the core experiment. This remains compliant because all inference still runs on the assigned Nano.

### Access and operational constraints

All work occurs through SSH on the assigned device. The team must not access another team's Nano, treat devices as a cluster, or saturate shared network egress. Team members should coordinate training and serving because concurrent GPU-heavy jobs can reduce performance. The repository, configuration, metrics, and artifacts must be saved externally before the device is wiped.

### Required deliverables

The submission must include:

- A public GitHub repository containing code, README, Dockerfile or setup script, configuration, and reproduction steps.
- Benchmark results with the selected metrics and an explanation of why they matter.
- A working demo running on the ZGX Nano over SSH or an approved tunneled port.
- A public video no longer than two minutes.
- A dynamic and accessible presentation rather than a set of static text-heavy slides.
- At least one architecture visual, one benchmark or evidence visual, and one impact visual.
- A pitch that explains the problem, solution, evidence, and result, with approximately three minutes reserved for questions.
- Project brief, pitch video, links, presentation, diagrams, and photos in the required shared drive.
- Social posts and the event's requested tags if the team participates in the social requirement.

The stated deadline is Friday, September 25 at 8 PM.

## Definition of done

The minimum viable submission is complete when:

- A Qwen3-8B AP agent runs entirely on the ZGX Nano.
- The six tools and local ledger complete a clean invoice payment.
- At least one immediate injection and one delayed memory-poisoning scenario compromise the baseline.
- The activation probe produces a calibrated score before the next action.
- The safety LoRA can generate a recovery action when selected.
- The deterministic gate prevents an unauthorized payment while an alarm is unresolved.
- All four defense configurations have been evaluated on held-out clean and attacked scenarios.
- Results include security, utility, and latency metrics.
- The demo, repository, setup instructions, two-minute video, and presentation are ready for submission.

## Claims we should and should not make

We can claim that Tell tests and demonstrates a model-internal signal for routing and stopping high-risk agent actions. We can claim measured reductions in attack success only after the evaluation is run. We can also claim that local deployment is necessary for activation access and keeps sensitive AP data on the device.

We should not claim that the probe proves an attack is present, that the system prevents every prompt injection, or that the LoRA replaces ordinary payment controls. Tell is an experimental detector and response system. The action gate, vendor source of truth, transaction limits, and audit log remain important defense-in-depth measures.

## Short project description

Tell is a local security runtime for autonomous payment agents. It reads a signal from the model's hidden activations after untrusted business content enters context. A low-risk turn continues through the base model. A suspicious turn activates a safety LoRA, and a deterministic gate blocks payment tools until the conflict is resolved. The core experiment tests whether a detector trained on immediate email and tool-result injection can also recognize malicious instructions after they have been stored and retrieved as the agent's own memory. The complete system, including inference, fine-tuning, memory, payment simulation, and evaluation, runs on the HP ZGX Nano.
