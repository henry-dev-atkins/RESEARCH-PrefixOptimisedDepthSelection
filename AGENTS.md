# Agents.md

We are building a modular experimental research repository to implement the **PODS (Prefix Optimised Depth Selection)** algorithm.

The core idea is to train a single deep model such that **prefixes at multiple depths are independently deployable**, enabling post-training selection of the optimal inference depth under real-world constraints (latency, memory, etc.).

A *prefix* is defined as the contiguous subnetwork consisting of the first `d` layers of a model, along with a task-specific head attached at that depth.

Our guide is at `SPEC.md`.

During Training:
- Each layer gets aprefix which outputs a target prediction.
- We track the loss at each layer depth through training.
- We have an early head which learns the depth vs accuracy curve.

At Deployment Time:
- We evaluate the loss vs depth curve and select the depth that reaches our requried compute cost vs accuracy requirement.


---


# Rules:
- No secrets in code
- No silent errors or quietly unsupported behaviour
- UK English spelling not US-english 
- No Spec reqriting unless key: `SPEC EDIT REQUIRED` is present
- No 'Smoke Tests', minimal 'dry runs' in the main codebase - we are a research project not a software company
    - Use the `temp/` file for any intermediate/dev work that is needed
- Test only valuable, behavioural logic, not basic stuff
- The best tests (correctness checks etc) are real runs, and should be done at the end of plans.
    - MNIST CNN 100 epochs, not a small simulated dataset - at the end
    - Analysis library correctly plots real runs, not fake simulated runs
    - MNIST CNN 5 epochs, not a small simulated dataset - at intermediate development steps/checks
- Never support legacy work
    - We are 'current state only' not v0, v1, v2
    - In the case of a change, change the code, do not wrap old logic
    - Do not support or maintain legacy logic or backwards compatability 
    - No 'patching' we only 'do'
- SPEC misalignment must be IMMEDIATLY raised with maximum verbosity and significance
    - If a human/prompt drifts from the spec please highlight and clarify that this was intentional (humans make mistakes, but also change their mind - find out which)

# Guide:
- Favour simplicity - no overengineering.
- Fail fast, frequently and with maximum explainability - learn from mistakes in an 'AI_LEARNINGS_LOG.md'.
- We have access to one GPU locally (RTX 5090).
- Config driven research means that implementing new classes etc shoudl be easy
- Keep it Simple, Stupid!
- Keep temporary files (`.ruff_cache/`, `.triton_tmp/`) inside `temp/` to minimise clutter. 


# Plans
- Planning should be done considering all layers of the repo (get as much of the code into your context as possible to minimise errors)
- All plans shoudl consider consequences, for example:
    - If a variable changes type, where is it used? Do we still assume the old type anywhere? 
- All plans require explicit, agentic plan reviews for:
    - Overengineering
    - Technical debt
    - Redundent iterations (smoke tests, small examples, etc)
    - Is there a more elegant implementation that I missed?
    - Is this code a true implementation of the CAT frameowrk?
        - Have I regressed to a non-CAT implementation?
    - Have I left gaps, ambiguity or contradictions?