# nightshift

Let a local language model polish your Markdown drafts overnight, without ever being able to break your code, commands or links.

Small models running on a CPU are slow, and they like to "fix" things that were right: a flag in a command, a URL, a version number. nightshift removes that risk by design. Code and links are replaced with placeholders before the model sees the text, and every proposed edit must pass a set of safety checks before it is accepted.

## How it works

```
draft.md
  -> mask      code, links, HTML, front matter  ->  [[CODE_1]], [[URL_1]], [[BLOCK_1]] ...
  -> chunk     split into paragraphs, lists, quotes (headings and tables are left alone)
  -> edit      your model rewrites one block of prose at a time
  -> check     reject the edit if anything it must not change has changed
  -> unmask    put the original code and links back, byte for byte
  -> diff      show what changed, so a human decides what to keep
```

A block is rejected, and the original text kept, when the model:

- lost, duplicated, added or reordered a placeholder
- introduced backticks or a URL of its own
- changed a number (versions, durations, counts)
- changed the length drastically, or broke a list/quote structure, or split a paragraph

Whatever happens, the result is never worse than the input.

## Usage

```python
from nightshift import process_document


def editor(block: str) -> str:
    # `block` is prose only. Code and links appear as placeholders: keep them as they are.
    return my_model(block)


result = process_document(open("draft.md", encoding="utf-8").read(), editor)

print(result.stats)  # e.g. {'edited': 4, 'unchanged': 2, 'rejected': 1, 'skipped': 5}
for block in result.blocks:
    if block.status == "rejected":
        print(block.index, block.reasons)

open("draft.edited.md", "w", encoding="utf-8").write(result.edited)
```

`nightshift.unified_diff` and `nightshift.word_diff` render the differences for review.

## Status

This is a single-user tool. It is meant to run on your own machine or home server, not to be exposed to the internet, and there is no hosted instance.

- [x] M1: core library (masking, chunking, safety checks, diffs, tests)
- [ ] M2: container image, Kustomize manifests, Argo CD application
- [ ] M3: nightly CronJob running local models on CPU, with resource limits
- [ ] M4: CI/CD loop (GitHub Actions, GHCR, Argo CD)
- [ ] M5: model comparison and documentation

Known limits of the Markdown handling: indented (4-space) code blocks are not detected, tables are never edited, and text must be NFC-normalised Unicode for the word diff to be accurate.

## Development

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest
.venv/bin/ruff check .
```

## License

MIT, see [LICENSE](LICENSE).
