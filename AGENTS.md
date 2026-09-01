# PASS Repository Instructions

## Role and priorities

Maintain PASS as a computational beam-dynamics and accelerator-physics codebase.
Use the existing implementation, input schema, documented equations, and runnable
tests as the source of truth. Do not replace repository evidence with assumptions.

Priority order:

1. The user's explicit request.
2. Existing public APIs and input-schema compatibility.
3. The physics conventions in this file.
4. Testing, documentation, and style conventions.

## Test organization

- Human-maintained tests belong in `tests/unit/` or `tests/integration/`.
- Codex-generated tests, exploratory scripts, verification reports, and generated
  result files belong in `tests/codex/`.
- Do not create new tests directly under `tests/`.
- Put reviewed, reusable mocks and fixtures in `tests/support/`.
- Do not modify human-maintained tests merely to make an implementation pass.
  If an intentional behavior or API change requires test changes, explain the reason.

## Test execution protocol

- During development, run a targeted test file or test selection first, for example:
  `python -m pytest tests/unit/test_<component>.py -v` or
  `python -m pytest -k <related_test_name> -v`.
- Run a broader suite only after targeted tests pass, when shared behavior is affected,
  or when the user or CI workflow requests it.
- If pytest discovery is configured, `testpaths` should normally include only
  `tests/unit` and `tests/integration`; run `tests/codex` explicitly when needed.
- Physics tests should use assertions with explicit tolerances and include measured,
  theoretical, and error values in failure messages or test output.
- Run generated-input workflows serially because generated inputs and output files
  may be shared between cases.
- GPU tests may be skipped when CUDA or CuPy is unavailable. RFCavity validation is
  CPU-only unless the implementation explicitly supports another backend.

## Longitudinal-coordinate conventions

- Particle `p.z` stores the continuous bunch-relative coordinate `z_rel`.
- `harmonic_id` is per-bunch grouping metadata, not a particle attribute. It is the
  zero-based bunch slot in `[0, harmonic_number)`, sourced from the Injection bunch
  field `Harmonic ID of this bunch`. If omitted by a high-level input builder, it may
  be assigned from the bunch enumeration index. It must not be recomputed from
  particle coordinates during tracking.
- `z_center = harmonic_id * circumference / harmonic_number`.
- `z_lab = z_rel + z_center`.
- During normal tracking, do not fold or wrap `z_rel`, and do not assign a folded value
  back to `p.z` (for example, do not replace it with `p.z % circumference`).
- Temporary periodic reduction is allowed for local RF-phase evaluation, bunch
  regrouping/sorting, and diagnostic statistics, but it must not replace stored
  continuous `z_rel`.
- RFCavity and Exciter use `z_lab` for arrival-phase or arrival-time calculations.
- `harmonic_number` describes bunch grouping; it must not be used to reject RF
  harmonics. The RF harmonic and grouping harmonic need not be equal or integer multiples.

## Input and schema conventions

- `harmonic_number` is declared at the Injection level and describes the number of
  bunch groups.
- Keep schema aliases, generated JSON layout, CLI options, code, tests, and docstrings
  synchronized after schema changes.
- Do not silently revive obsolete field names or old command-line examples.

## Documentation conventions

- Update the affected technical documentation under both `docs/source/en/` and
  `docs/source/zh/` when behavior, APIs, schemas, equations, examples, or file layout
  changes. Keep corresponding English and Chinese sections synchronized.
- README files do not need to be changed after every code modification. Update
  `README.md` and `README-zh.md` when installation, user workflow, public usage, or
  other README-facing information changes.
- Validate documentation builds and SVG/XML files when documentation is changed.

## Change safety

- Inspect related code, tests, docs, and configuration before editing.
- Keep changes focused on the requested behavior and preserve unrelated user changes.
- Do not delete generated or user files without explicit confirmation.
- Do not use destructive Git commands such as `git reset --hard` or `git clean -fd`
  unless explicitly requested.
- Do not rewrite this `AGENTS.md` merely to resolve a conflict with the codebase.
  Report the conflict and ask the user for clarification.
