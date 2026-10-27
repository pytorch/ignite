# Instructions for AI agents

<!-- used https://github.com/aaishwarymishra/mlx/blob/388f227f466ef600243bcfce64181ad0d675c476/AGENTS.md as reference -->

AI tools are allowed, but user should understand the code and changes made and they are responsible for the code not the AI. Read [CONTRIBUTING.md](CONTRIBUTING.md) for setup and style.

## Do not

- Open issues or pull requests. Do not run `git push`, `gh issue create`, or
  `gh pr create`.
- Write issue or PR descriptions, commit messages, or reviewer replies for the
  user.
- Work on a task the user cannot explain in their own words.
- Fix a problem with no reproduction and no real user impact.

Don't make code changes for the vague questions like: "fix issue 4432", "make training
faster", "address the review comments". Ask the user to clarify what they want to do and why. If they cannot explain it, do not make the change.

PRs that break these rules are closed without review.

## Bugs and edge cases

A bug report needs a minimal reproduction on current `master`, the actual and
expected result, and real user impact.

Not bugs:

- Inputs the API does not accept or promise to support.
- Wrong use of the API, or an unsupported Python or PyTorch version.
- Theoretical failures with no reproduction.
- Style or naming nits.
- Edge cases that are not reproducible or have no real user impact.

If unsure, open a question first. Do not attach a patch.

## Code standards

- Match the surrounding code.
- Use the standard library and existing dependencies. Do not add a dependency
  for a small helper.
- Comments: 1-2 lines, only where the code is not obvious.
- No defensive code for states that cannot occur.

## Checks

```bash
pre-commit run --all-files   # ruff check, ruff format, whitespace hooks
pytest tests/ignite/<area> -vvv
pyrefly check
```

Full CPU suite: `bash tests/run_cpu_tests.sh`.
