# Development Standards

## Language & Version

- **Language**: Rust (latest stable, edition 2021)
- **Package Manager**: cargo (never pip/uv here)

## Code Style

- Linter: `cargo clippy -- -D warnings`
- Formatter: `cargo fmt --check` (rustfmt)
- Line length: 120
- Indentation: spaces (2 for Rust)

## Architecture

- Pattern: single binary, module per pipeline stage
  (`det` DB postprocess, `contours` tracing, `cls` direction,
  `rec` CTC decode, `geometry` primitives, `main` socket loop)
- API style: UNIX socket, length-prefixed frames (see `docs/API.md`)
- Database: none (stateless worker; models loaded from disk)

## Testing

- Framework: `cargo test` (unit tests inline per module)
- Contract tests: `tests/test_socket_contract.py` (framing shared
  with orchestrator clients)
- Coverage target: new code 100% on pure functions
- Models needed for e2e: det ch_PP-OCRv4, cls mobile v2, rec latin
  (`/opt/models/` on the inference host)

## Git Workflow

- Branches: `feature/*` → `develop` → `main`
- Commits: `type(scope): description (#issue)`
- Issues and pull requests live in the GitHub repository

## Deployment

- Delivery: release binary + systemd template
- Binary: `/usr/local/bin/rapidocr-worker`
- systemd template: `configs/systemd/rapidocr-worker@.service`
- CI/CD: GitHub Actions (`.github/workflows/ci.yml`)

## Secrets Scanning

- Pre-commit runs `detect-secrets` against `secrets.baseline` (`.pre-commit-config.yaml`); never bypass it with `--no-verify`.
- Regenerate after intentional changes: `uvx --from detect-secrets==1.5.0 detect-secrets scan --exclude-files 'secrets\.baseline' > secrets.baseline`.

## Documentation

- `README.md` — project overview
- `docs/STANDARDS.md` — this file
- `CHANGELOG.md` — auto-generated from commits
