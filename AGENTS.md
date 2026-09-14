# PASS Repository Instructions

## Priorities and execution

Maintain PASS as a computational beam-dynamics and accelerator-physics codebase.
Priority: explicit user request > physical correctness and conventions below >
API/schema compatibility > testing, documentation, and style.

- Inspect relevant code, schema, docs, tests, and configuration to establish current behavior.
  Verify physics with stated conventions, equations, and independent checks;
  existing code or test expectations alone do not prove correctness.
- Preserve compatibility by default, but do not retain a demonstrated physics error
  solely for compatibility. Explain any necessary compatibility impact.
- Complete authorized work and relevant validation without repeated confirmation.
  Fix clear implementation deviations; clarify unresolved physics/instruction
  conflicts or API/schema breaks not already authorized. Identify the issue and
  continue work that does not depend on the answer. Respect the file-safety rules below.

## Code organization

- Avoid global variables whenever practical; prefer local variables, explicit
  arguments, or instance attributes for state.
- Keep CPU and GPU implementations of the same component in the same source file.

## Tests

- Ignore all tests in Git; do not upload them. Human-maintained tests belong in
  `tests/unit/` or `tests/integration/`; reviewed mocks/fixtures in `tests/support/`.
- Put Codex-generated tests, exploratory scripts, reports, and results in
  `tests/codex/`. Do not create tests directly under `tests/`.
- Do not change human-maintained tests merely to make code pass. Explain test
  changes required by intentional behavior/API changes.
- Default pytest discovery covers `tests/unit` and `tests/integration` only;
  run `tests/codex` explicitly. Start with a targeted selection, e.g.
  `python -m pytest tests/unit/test_<component>.py -v`.
- Expand tests only for shared behavior, new failures, specific unresolved risks,
  or user/CI requirements. After checks pass, repeat only for changes or new evidence.
  Prose/formatting edits need no runtime tests unless executable content is affected.
- Physics assertions need explicit tolerances and theoretical, measured, and error
  values in output or failure messages.
- Run generated-input workflows serially; cases may share inputs and outputs.
- Use CPU tests as the RFCavity baseline. For GPU or shared CPU/GPU changes, test
  supported GPU paths when CUDA/CuPy are available; otherwise report skipped scope and reason.

## Longitudinal-coordinate conventions

- Particle `p.z` stores continuous bunch-relative `z_rel`; never wrap it during tracking.
- `harmonic_id` is per-bunch metadata in `[0, harmonic_number)`, from Injection's
  `Harmonic ID of this bunch`. Builders may default it to the bunch enumeration index.
  It is not a particle attribute and must not be inferred from tracked coordinates.
- `z_center = harmonic_id * circumference / harmonic_number`.
- `z_lab = z_rel + z_center`.
- Local periodic reduction is allowed for RF phase, regrouping/sorting, and
  statistics; never assign the folded value back to `p.z`.
- RFCavity/Exciter use `t_i = bunch.t0 + arrival_offset_i - z_lab_i / (beta0 * c)`,
  with `beta0 = bunch.beta`; retain nonzero arrival corrections.
- `PASS/core/arrival.py` owns corrections in seconds, independently of wakes.
  Preserve arrival time across zero-length reference-energy changes and regrouping;
  retain `bunch.reference_arrival_offset` in RF reference-phase calculations.
  Equations and lifecycle: [English](docs/source/en/arrival_time.rst),
  [Chinese](docs/source/zh/arrival_time.rst).
- Injection-level `harmonic_number` describes bunch grouping. RF harmonics need not
  equal or be integer multiples of it; do not reject RF harmonics on this basis.

## Schema and documentation

- Synchronize schema aliases, generated JSON, CLI options, code, tests, and docstrings
  after schema changes. Do not revive obsolete fields or command-line examples.
- Keep affected `docs/source/en/` and `docs/source/zh/` sections synchronized for
  behavior, API, schema, equation, example, or file-layout changes.
- Update `README.md` and `README-zh.md` only for installation, workflow, public usage,
  or other README-facing changes.
- Build both languages when Sphinx sources/dependencies change; validate modified
  SVG/XML and affected references. Instruction/README-only edits need Markdown,
  link, and whitespace checks; build Sphinx only if its sources/dependencies are affected.

## Change safety

- Keep changes focused on the requested behavior and preserve unrelated user changes.
- Do not delete generated or user files without explicit confirmation.
- Do not use destructive Git commands such as `git reset --hard` or `git clean -fd`
  unless explicitly requested.
- Do not rewrite this `AGENTS.md` merely to accommodate conflicting implementation.

## Completion reporting

- Report the result, key evidence, actual validation/results, and unverified scope
  or blockers. Distinguish physics theory, measurements, and inferences; never claim
  unrun checks passed.
