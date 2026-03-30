# Evaluation Dataset

## Golden Dataset Template

The file `golden_dataset.jsonl.template` is a **generic template** with universal questions that work with any PDF corpus. It contains 5 domain-agnostic questions covering:

- General comprehension ("What is the main purpose...")
- Summarization ("Summarize the key points...")
- Findings extraction ("What are the main conclusions...")
- Audience identification ("Who is the intended audience...")
- Methodology ("What methodology is described...")

## Usage

To create your own evaluation dataset:

1. Copy the template:
   ```bash
   cp evals/golden_dataset.jsonl.template evals/my_dataset.jsonl
   ```

2. Replace with domain-specific questions and expected answers:
   ```json
   {"question": "What is the Q3 revenue?", "expected_answer": "$5.2M", "expected_sources": ["report.pdf"], "tags": ["finance"]}
   ```

3. Run evaluation:
   ```bash
   python scripts/run_eval.py --dataset evals/my_dataset.jsonl --collection my_collection
   ```

## Report Output

Evaluation reports are written to `evals/reports/` (JSON format). These are gitignored - reports are generated artifacts, not source code.
