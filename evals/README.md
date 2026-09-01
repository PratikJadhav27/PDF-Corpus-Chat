# Evaluation Fixtures & Golden Dataset

## Overview

| File | Description |
|------|-------------|
| `golden_dataset.jsonl` | **50 verified Q&A pairs** used for CI and local evaluation |
| `golden_dataset.jsonl.template` | Legacy placeholder template (do not use for evaluation) |
| `fixtures/` | PDF fixture files (fetched at setup time, **not committed** to avoid copyright) |
| `reports/` | Auto-generated evaluation report JSON files |

---

## Fixture PDFs

PDFs are **not committed to the repository**. Fetch them at setup/CI time:

```bash
python scripts/download_fixtures.py
```

Idempotent — skips files that already exist at the expected size.

| File | Paper | Source |
|------|-------|--------|
| `attention_is_all_you_need.pdf` | Vaswani et al. 2017 — Attention Is All You Need | [arXiv:1706.03762](https://arxiv.org/abs/1706.03762) |
| `bert_pretraining.pdf` | Devlin et al. 2018 — BERT: Pre-training of Deep Bidirectional Transformers | [arXiv:1810.04805](https://arxiv.org/abs/1810.04805) |

Both are open-access and freely available on arXiv.

---

## Golden Dataset Schema

```json
{
  "id": "attn_fact_001",
  "question": "What is the title of the paper?",
  "expected_answer": "Attention Is All You Need",
  "expected_sources": ["attention_is_all_you_need.pdf"],
  "category": "fact_lookup",
  "difficulty": "easy"
}
```

### Categories (50 questions total)

| Category | Count | Description |
|----------|-------|-------------|
| `fact_lookup` | 15 | Single-document, single-fact retrieval |
| `numeric` | 12 | Exact numbers, parameter counts, scores |
| `multi_hop` | 7 | Requires combining info from multiple chunks |
| `negative` | 5 | Answer is explicitly "No" or a correction of a false premise |
| `cross_document` | 5 | Evidence required from both fixture PDFs |
| `follow_up` | 6 | Deeper reasoning building on earlier topics |

---

## CI Integration

The `eval-gate.yml` workflow:

1. Downloads fixture PDFs via `scripts/download_fixtures.py`
2. Ingests them into a temporary Qdrant collection
3. Runs `scripts/run_eval.py` against `evals/golden_dataset.jsonl`
4. Fails the PR if `avg_faithfulness < 0.85` or `avg_citation_validity_ratio < 0.80`

---

## Validating the Dataset

```bash
python -c "
import json, sys
with open('evals/golden_dataset.jsonl') as f:
    for i, line in enumerate(f, 1):
        line = line.strip()
        if not line: continue
        try:
            r = json.loads(line)
            assert 'id' in r and 'question' in r and 'expected_answer' in r
        except Exception as e:
            print(f'Line {i}: {e}'); sys.exit(1)
print('All lines valid.')
"
```

## Report Output

Evaluation reports are written to `evals/reports/` (JSON format). Reports are generated artifacts and are gitignored.
