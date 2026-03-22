# ADR-001: Pure Text Tool Calling Format for SFT Training

- **Status**: accepted
- **Date**: 2026-03-15
- **Updated**: 2026-03-16
- **Deciders**: @jzq

## Context

We need to train a student model (Qwen3-4B) to perform tool calling via SFT. The student model will be deployed via vLLM, outputting raw text with `<tool_call>` tags that our agent environment parses.

During trajectory collection, we tested two approaches with the teacher model (qwen3.5-plus):

### Approach A: Pure Text Mode (teacher outputs `<tool_call>` tags directly)

**Problems encountered:**
- Garbled output and unexpected truncation
- Inconsistent tag formatting (missing closing tags, malformed JSON)
- Required extensive `clean_assistant_content()` heuristics to fix format issues
- Low collection success rate

### Approach B: API Function Calling Mode (teacher uses structured `tools` parameter)

**Advantages:**
- Stable, well-formed output — API guarantees valid `tool_calls` structure
- Thinking content captured via `enable_thinking=True` parameter
- Higher collection success rate

**Trade-off:**
- API output format differs from SFT target format
- Requires conversion layer: API response → pure text format

## Decision

We adopt a **hybrid approach**:

1. **Teacher (data collection)**: Use DashScope API with `tools` parameter and `enable_thinking=True` for stable, structured output.

2. **Conversion layer**: Transform API responses to pure text SFT format:
   ```
   <think>
   [thinking_content from API]
   </think>
   <tool_call>
   {"name": "tool_name", "arguments": {...}}
   </tool_call>
   ```

3. **Training data format**: Pure text `<tool_call>` tags — this is what the student learns.

4. **Student (inference)**: vLLM outputs raw text; `ToolParser` extracts tool calls.

5. **Parser scope**: `ToolParser` is used **only for student inference**, not for teacher collection (API already provides structured output).

## Consequences

### Positive

- **Stable collection**: API function calling guarantees well-formed tool calls, no garbled output.
- **Complete reasoning preservation**: `enable_thinking=True` captures thinking content alongside tool calls.
- **Clean separation of concerns**:
  - Collection: API handles parsing complexity
  - Conversion: Deterministic transform (API struct → text)
  - Inference: ToolParser handles student output only
- **Debuggability**: Both API responses and converted text are inspectable.

### Negative

- **Conversion layer required**: Must implement `api_response_to_sft_text()` to transform API output.
- **Two code paths**: Teacher uses API mode; student uses text mode — must ensure format consistency.
- **Parser still needed**: ToolParser required for student inference (handles malformed student output).

### Neutral

- Conversion logic is simple (~50 lines) and deterministic.
- ToolParser scope is reduced (student-only), making it easier to maintain.

## Alternatives Considered

### Alternative 1: Pure Text Mode for Teacher (Original ADR-001)

Teacher outputs `<tool_call>` tags directly, no API function calling.

**Rejected because**:
- Garbled output and truncation issues in practice
- Required extensive cleaning heuristics (`clean_assistant_content()`)
- Low collection success rate

### Alternative 2: API Function Calling Without Thinking

Use API function calling but skip thinking capture.

**Rejected because**:
- Loses chain-of-thought reasoning critical for SFT quality
- Student would not learn to reason before tool calls

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA COLLECTION                          │
├─────────────────────────────────────────────────────────────┤
│  Prompt ──► Teacher API (tools=[...], enable_thinking=True) │
│                          │                                  │
│                          ▼                                  │
│              Structured Response:                           │
│              - thinking_content: "..."                      │
│              - tool_calls: [{name, arguments}]              │
│                          │                                  │
│                          ▼                                  │
│              api_response_to_sft_text()                     │
│                          │                                  │
│                          ▼                                  │
│              Pure Text (SFT format):                        │
│              <think>...</think>                             │
│              <tool_call>{"name":...}</tool_call>            │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    STUDENT INFERENCE                        │
├─────────────────────────────────────────────────────────────┤
│  Prompt ──► Student (vLLM) ──► Raw Text                     │
│                                    │                        │
│                                    ▼                        │
│                              ToolParser.parse()             │
│                                    │                        │
│                                    ▼                        │
│                              Execute Tool                   │
└─────────────────────────────────────────────────────────────┘
```

## Related

- [specs/tools.md](../specs/tools.md) — Tool Calling Protocol section
- [specs/training.md](../specs/training.md) — SFT Data Format section
- [plans/phase2.6_trajectory_collection.md](../plans/phase2.6_trajectory_collection.md) — Implementation details
- [src/orchestrator/tool_parser.py](../../src/orchestrator/tool_parser.py) — Student-side parser
