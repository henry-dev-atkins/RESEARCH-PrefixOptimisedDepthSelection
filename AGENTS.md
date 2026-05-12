# Agents.md
Our guide is at `AI_SPACE/SPEC.md`.


# Rules:
- No secrets in code
- No secrets in logs, configs, handovers or artefacts
- No Spec reqriting unless key: `[SPEC EDIT REQUIRED]` is present
- SPEC misalignment must be IMMEDIATLY raised with maximum verbosity and significance
    - If a human/prompt drifts from the spec please highlight and clarify that this was intentional (humans make mistakes, but also change their mind - find out which)
- Any operations outside the main folder/repo require detailed reports on why you need to do it.
- Never delete any file/folder: All file/folder deletions are not actual deletions, but a move to an `archive/` folder.
- No invented files, paths, modules or commands: check first, then reference
- No silent fallback behaviour
- No silent GPU to CPU fallback
- No ad hoc output sprawl across the repo root
- No editing anything outside the working repo/folder unless given the [EXTERNAL EDIT ALLOWED] flag, even then: NO DELETIONS EVER.
- Forbidden terminology:
    - Canonic (ise/al)
    - Version X
    - Word soup. (e.g. "First Class/Canonical Remote Run Root")
- Your space is `AI_SPACE/`, with folders for:
    - `plans/` store every plan you generate (timestamp in name)
    - `handovers/` context dump of the work for tomorrow
    - `AI_LEARNINGS_LOG.md` store lessons, mistakes and vital information you pick up through work. Keep it clean.
    - `TODO.md` is a rolling store of tasks that you add to, align with `plans/` and use for progress tracking. Do not delete elements. 

# Guide:
- Favour simplicity - no overengineering
- Fail fast, frequently and with maximum explainability - learn from mistakes in an 'AI_LEARNINGS_LOG.md'
- We have access to one GPU locally (RTX 5090)
- Config driven research means that implementing new classes etc should be easy
- Keep it Simple, Stupid!
- Keep temporary files (`.ruff_cache/`, `.triton_tmp/`) inside `temp/` to minimise clutter
- No silent errors or quietly unsupported behaviour
- UK English spelling not US-english
- No 'Smoke Tests', minimal 'dry runs' in the main codebase - we are a research project not a software company
    - Use the `temp/` file for any intermediate/dev work that is needed
- Test only valuable, behavioural logic, not basic stuff
- GPUs for ML, not CPU
- Unsupported GPU, CUDA or device situations must fail loudly and clearly
- Call out expected performance, VRAM or RAM impact when it is materially affected
- The best tests (correctness checks etc) are real runs, and should be done at the end of plans
    - MNIST CNN 100 epochs, not a small simulated dataset - at the end
    - Analysis library correctly plots real runs, not fake simulated runs
    - MNIST CNN 5 epochs, not a small simulated dataset - at intermediate development steps/checks
- Never support legacy work
    - We are 'current state only' not v0, v1, v2
    - In the case of a change, change the code, do not wrap old logic
    - Do not support or maintain legacy logic or backwards compatability
    - No 'patching' we only 'do'
    - Only 'Canonical' work, no 'non-canonical' work.
- Config should be the source of truth for experiment behaviour
- Avoid hardcoded experiment values when config should own them
- New features should slot cleanly into the config structure
- Generated artefacts should go to clear, intentional output locations
- Large datasets and checkpoints should follow repo storage conventions and must not be casually committed
- Every new dependency needs justification
- Prefer repo-standard tools and commands over ad hoc alternatives
- Efficiently design entry points to minimise permission/approval delays
    - Config driven runs may be needed inside a run command, can we pre-approve 'any' config given a runner we trust?
- Do not 'exit', 'return to user', 'apply `<|endoftext|>`' or similar unless [TERMINAL_CONDITION] is met
    - Early exit may be allowed when:
        - something happens to place too much resposibility on the AI 
    - You are expected to continue through the plan -> review plan -> implement -> review implementation -> plan -> ... loop until [TERMINAL_CONDITION] is met.
    - Do not blowup, delete or make widespread edits in this loop
    - Example appliations:
        - Get the X working. Everything is there, we will just have some dugging and you shoudl work through it ratehr than checking in every time.
        - There are long waits for results. Monitor runs, maintain repo, sleep - find things to occupy yourself with while things run.


# Commands
- [SESSION.START]: Teach me the project, philosophy, goals, idea and then note down progress, last steps, current steps, next steps.
- [SESSION.CLEAN]: Move clutter/fluff/noise from repo into the `archive/` folder, or move files into their correct place.
    - Look for old-(handovers, progress reports, intermediate results, etc) 
- [SESSION.END]: Update docs, learnings, to-do's, progress and write a detailed handover note (name = todays date & time) for full reproducibility.
    - Include: what changed, why, exact commands run, results, current state, open issues and next steps
- [SUBPLAN]: We are marking an aside. Name the plan which we were working on before as [PLAN_A] and write a new, subplan [PLAN_B] to temporarily switch to.
    - We will switch back to [PLAN_A] immediatly after [PLAN_B] completion.
- [LEARN]: Review learnings and map them to documentation. 
    - Some learnings may be outdated, so please ask me which lessons to forget.
    - Make surgical doc/spec (request edit permissions) updates of the learnings.
    - Empty the learnings table for next time.
    - Commit this work so we preserve learnings.
- [MONITOR] / [ITERATE]: A script is running and will output logs/pregress/results which you need to await->evaluate->maybe intervene->await->...
    - Spec alignemnt is first priority, do not drift away through iterations.
    - Generally this is needed when the script is 'right' but just needs debugging/diagnosing etc.
- [DIAGNOSE]: What is the underlying cause of this symptom?
    - No band-aid, hacky or workaround fixes
    - You may use agents for diagnostic experiments
    - We are experimentally driven, not speculative
    - Consider high level (philosophy, underlying math, etc), medium level (integration, usage, etc) and low level (bugs, syntax, etc) in different agents.

# SPEC Usage
- Spec will have a walkthrough section with work stages/steps/milestones.


# Repo Understanding
- Before changing code, identify the main entrypoints, main config files, key output paths and the main run/eval/analysis flow
- Prefer repo-standard bootstrap, test and run commands once they are known
- Commands shared with humans should be reproducible and copy-pasteable


# Outputs
- Temporary and intermediate work belongs in `temp/`
- Durable outputs should go in intentional repo-defined output locations
- Plots, logs, checkpoints and exported tables should not be scattered across arbitrary folders
- Serious runs should record the exact config used, the seed, the device and key hyperparameters
- Output artefacts should be reproducible from code and config, not manual steps


# Learnings
- Update `AI_SPACE/AI_LEARNINGS_LOG.md` after
    - failures, 
    - debugging discoveries or 
    - architecture/course corrections
    - mistakes
- Update `AI_SPACE/TODO.md` only with concrete unresolved follow-up work
- Do not fill TODOs with vague ideas, duplicates or already-completed work


# Plans
- Planning should be done considering all layers of the project (get as much of the documentation & code into your context as possible to minimise errors)
- All plans should consider consequences, for example:
    - If a variable changes type, where is it used? Do we still assume the old type anywhere?
- All plans require explicit, agentic plan reviews for:
    - Overengineering
    - Technical debt
    - Redundent iterations (smoke tests, small examples, etc)
    - Is there a more elegant implementation that I missed?
    - Is this code a true implementation of the dream framework?
        - Have I regressed to a non-correct implementation?
    - Have I left gaps, ambiguity or contradictions?
- Plans and final reports must state assumptions, key tradeoffs, unresolved risks and follow-up work
- All plans should end in a short, human facing executive summary of the work.
- Plans are reviewed bottom up by humans scrolling up through the terminal, not top down.
- If a plan implementation will/does end with "X helped only a little, but Y is still the real blocker":
    - it is planning/diagnosis failure so stop, re-[DIAGNOSE], and re-plan before planning/implementing more.


# Done Means
- The code change is complete, coherent and aligned with the spec or explicit human instruction
- Relevant checks or real runs have been completed at the right stage
- Config, docs and outputs have been updated where behaviour changed
- Obsolete assumptions and replaced logic in touched areas have been cleaned up
- Learnings, TODOs and handovers have been updated when relevant
- The remaining limitations and next steps are clearly stated
