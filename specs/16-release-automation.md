# Spec 16 — Release Automation

**Status:** Pending | **Effort:** Low | **Module:** `.github/workflows/`

## Context

There is no release pipeline: no publish workflow, no tag automation, and the CHANGELOG
has already drifted from the released version (0.2.0 on crates.io vs breaking changes
sitting in `[Unreleased]`/source). cargo-crap ships releases with automation; a library
crate needs less (no binaries) but still benefits from a one-button, non-drifting release.

## Scope and Invariants

1. `release-plz` (preferred) or `cargo-release` is adopted:
   - release PRs are opened automatically with version bump + CHANGELOG section generated
     from conventional commits;
   - merging the release PR tags and publishes to crates.io via a `CARGO_REGISTRY_TOKEN`
     secret with a GitHub environment gate (manual approval).
2. SemVer discipline is mechanical: the Spec 01 `cargo-semver-checks` job blocks a
   release PR whose bump doesn't match detected API changes.
3. The publish job runs the full `just dev` gate first; a failed gate aborts the release.
4. GitHub Releases are created from the tag with the CHANGELOG section as notes.
5. Version references in README (badges, `Cargo.toml` snippets) update automatically or
   are version-agnostic.
6. The process is documented in `CONTRIBUTING.md` (Spec 15).

## Acceptance Tests

- **Given** commits merged to `main` since the last tag, **when** the automation runs,
  **then** a release PR exists with correct version bump and generated CHANGELOG entries.
- **Given** a breaking API change with only a patch bump proposed, **when** checks run on
  the release PR, **then** `cargo-semver-checks` fails it.
- **Given** the release PR is merged and approved through the environment gate, **then**
  a git tag, a GitHub Release with notes, and a crates.io publish all occur — verified on
  a `0.2.x` dry run (`release-plz --dry-run` output attached to the PR).
- **Given** a red `just dev`, **when** publish is attempted, **then** it aborts before
  `cargo publish`.
