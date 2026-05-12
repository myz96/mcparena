# Judge calibration fixtures

`transcripts.jsonl` — hand-labeled MCP-server tool-use transcripts used for
descriptive judge calibration during the pilot. Each line is a JSON object:

```json
{
  "id": "fs_create_file_pass",
  "user_request": "Create a file named hello.txt with contents 'hi'",
  "trajectory": "Reasoning: I need to use write_file. → call write_file(path='hello.txt', content='hi') → tool_output: 'ok'",
  "final_answer": "Done — created hello.txt with 'hi' as contents.",
  "expected_outcome": "A file named hello.txt exists with contents 'hi'",
  "expected": true,
  "tier": "easy",
  "notes": "Clean success — single correct tool call"
}
```

## Conventions
- `expected: true` — the judge should return `success=True`
- `expected: false` — the judge should return `success=False`
- `tier` — `easy` / `medium` / `hard` (for difficulty stratification)
- Roughly balanced: aim for 5 pass + 5 fail in v5.1's 10-transcript fixture
- Real transcripts captured during pilot baseline runs are preferred over synthetic ones

## Per plan v5.1 (R4.5 demoted from gate to descriptive)
The pilot reports `agreement = N/10` in the memo. Not a runtime gate; informational
only. The full 50-task gold set with ≥85% agreement gate is a Phase 1 deliverable.
