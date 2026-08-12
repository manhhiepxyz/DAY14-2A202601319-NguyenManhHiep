# Plan thực hiện Lab — AI Evaluation & Benchmarking Pipeline (Day 14)

> File này ghi lại **kế hoạch từng bước** để hoàn thành lab, vai trò của từng
> thành phần trong hệ thống, và **lý do** vì sao phải làm theo thứ tự đó.
> Tham chiếu chính: `guide_lab.md`, `exercises.md`, `README.md`, `template.py`,
> `tests/test_solution.py`.

---

## 0. Bức tranh tổng thể của lab

### 0.1 Ba thành phần cốt lõi

Lab có **3 thành phần riêng biệt**, vai trò không trùng nhau:

| # | Thành phần                      | File                    | Vai trò                                                                                                                                                                                                                                          |
| - | --------------------------------- | ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1 | **System under evaluation** | `domain_assistant.py` | RAG assistant**sinh câu trả lời thật**. Pipeline: `question → BM25 retrieval → chunks → OpenAI model → answer`. Chỉ đọc `id` + `question`; **không đọc** `expected_answer`/gold contexts (tránh data leakage). |
| 2 | **Evaluation core**         | `template.py`         | Engine **chấm điểm**. Chứa data models, 5 metrics, LLM judge, benchmark runner, failure analyzer. Học viên hoàn thiện các `# TODO`.                                                                                             |
| 3 | **Artifact adapter**        | `evaluate_answers.py` | Chỉ làm**I/O**: đọc golden dataset + actual answers → đưa vào `BenchmarkRunner`/`RAGASEvaluator` của `template.py` → in bảng Exercise 3.2. **Không viết lại metrics.**                                            |

> Nguyên tắc tách lớp: `domain_assistant` là hệ **bị đo**; `template` là hệ
> **đo**; `evaluate_answers` chỉ là "cầu nối". Sửa metrics trong adapter là
> gian lận benchmark.

### 0.2 Luồng dữ liệu end-to-end

```text
data/student_services/*.md
        │
        ├── học viên đọc & viết ──▶ golden_dataset.json  (20 QA)
        │                                  │
        └── DomainAssistant ◀── question    │
                  │                        │
                  ├── retrieve chunks      │
                  └── generate answer      │
                           │               │
                           ▼               ▼
              artifacts/actual_answers.json
                           │
                   evaluate_answers.py  ──▶ template.py (core) ──▶ artifacts/benchmark_results.json
                           │
                    exercises.md + reflection.md
```

### 0.3 Chuẩn bị kiến thức về các metric (dùng để lập luận trong plan)

Pipeline RAG: `Question → Retriever → Context → Generator → Answer`

| Metric            | Câu hỏi trả lời                          | Heuristic trong lab                          |
| ----------------- | -------------------------------------------- | -------------------------------------------- |
| Faithfulness      | Answer có grounded trong context không?    | `\|answer ∩ context\| / \|answer\|`           |
| Relevance         | Answer có trả lời đúng question không? | `\|answer ∩ question\| / \|question\|`        |
| Completeness      | Answer có đủ nội dung expected không?   | `\|answer ∩ expected\| / \|expected\|`        |
| Context Recall    | Retriever lấy đủ evidence chưa?          | `\|expected ∩ union(chunks)\| / \|expected\|` |
| Context Precision | Chunk relevant có đứng sớm không?       | Average Precision@K (rank-aware)             |

> 2 retrieval metrics chạy trên **thứ tự chunks retriever trả về**
> (`QAPair.retrieved_contexts`); chúng chỉ **chẩn đoán retriever**, không tham
> gia `overall_score()` và không đổi pass rule gốc.

---

## 1. Tạo môi trường & kiểm tra baseline

### Bước 1.1 — Xác nhận Python 3.11+

```bash
python --version          # hoặc: py -0p (Windows) để chọn bản >= 3.11
python -m venv .venv
```

- **Role:** venv cô lập dependency của lab khỏi Python toàn máy.
- **Tại sao:** lab yêu cầu Python ≥ 3.11 (bài giảng dùng type hints `str | None`,
  `from __future__ import annotations`, và openai SDK mới yêu cầu version cao).
  Nếu venv tạo bằng 3.9/3.10 sẽ lỗi `ImportError: cannot import name UTC from datetime`.

### Bước 1.2 — Cài dependencies

```bash
.venv\Scripts\Activate.ps1          # Windows PowerShell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -c "import openai, dotenv, pytest; print('Environment OK')"
```

- `requirements.txt` gồm: `openai>=1.66.0`, `python-dotenv>=1.0.0`, `pytest>=8.0.0`.
- **Tại sao kiểm tra import:** phát hiện sớm lỗi môi trường *trước* khi chạy
  test — nếu không, 42 test fail vì import lỗi sẽ bị hiểu nhầm là code sai.

### Bước 1.3 — Chạy baseline test

```bash
pytest tests/ -v
```

- **Kết quả dự kiến (starter chưa làm TODO):** `42 collected, 42 failed`.
  Đây là **baseline bình thường**, không phải mục tiêu cuối. Quan trọng là
  **không có lỗi collection/import** trước khi test bắt đầu.
- **Tại sao:** chạy baseline 1 lần để xác nhận pipeline test hoạt động và
  biết được "điểm xuất phát", sau này so sánh tiến độ theo từng checkpoint.

> **Lưu ý then chốt về cách test load code (đọc từ `tests/test_solution.py` dòng 26–32):**
> test ưu tiên load `solution/solution.py` nếu tồn tại → rồi `solution/app.py`
> → cuối cùng mới `template.py`. Hiện `solution/` chỉ có `.gitkeep` nên test
> đang chạy `template.py`. **Khi nào copy `template.py` sang `solution/`, phải
> copy lại sau mỗi lần sửa `template.py`** nếu không test sẽ dùng bản cũ.

---

## 2. Part 1 — Warm-up (Exercises 1.1–1.3 trong `exercises.md`)

Phần này **chỉ phân tích, không sửa code**. Điền trực tiếp vào `exercises.md`.

### Exercise 1.1 — RAGAS Metric Thresholds

Điền bảng: với mỗi metric (Faithfulness, Relevance, Context Recall, Context
Precision, Completeness), nêu 1 kịch bản score thấp *chấp nhận được* và 1 kịch
bản *critical*, kèm action.

- **Tại sao:** giúp hiểu *ngưỡng* chứ không phải *điểm tuyệt đối* — metric thấp
  trong ngữ cảnh này có thể là chấp nhận được (vd: Context Precision thấp nhưng
  Context Recall cao = lấy đủ nhưng rank kém, dễ fix bằng rerank).

### Exercise 1.2 — Bias trong LLM-as-a-Judge

3 câu hỏi về position bias, verbosity bias, calibration với human labels.

- **Tại sao:** `LLMJudge.detect_bias()` ở Task 3 sẽ code 3 loại bias
  (positional, leniency, severity). Trả lời phần này trước giúp hiểu *vì sao*
  `detect_bias()` phải kiểm tra vị trí và ngưỡng điểm trung bình.

### Exercise 1.3 — Evaluation trong CI/CD

Chọn threshold block deployment cho 3 metrics + phân biệt offline/online/human review.

- **Tại sao:** `BenchmarkRunner.run_regression()` (Task 4) dùng ngưỡng **giảm
  > 0.05 so với baseline** để báo regression — giống khái niệm "quality gate"
  > trong CI/CD được hỏi ở đây.
  >

---

## 3. Part 2 — Core Coding (`template.py`, Tasks 1–5)

> **Nguyên tắc chung:**
>
> - Không đổi class/function signature. `contexts` optional trong
>   `run_full_eval` là phần bắt buộc của interface.
> - Không đổi công thức để "tối ưu riêng cho dataset" — đây là lỗi đánh tráo.
> - Mỗi Task xong → chạy **targeted test** tương ứng (mục 3.7) trước khi sang Task kế.

### Task 1 — Data models (`QAPair`, `EvalResult`, `overall_score`)

**`QAPair`** (điền hint ở dòng 57–60):

```python
@dataclass
class QAPair:
    question: str
    expected_answer: str
    context: str = ""
    metadata: dict = field(default_factory=dict)
    retrieved_contexts: list = field(default_factory=list)
```

- `retrieved_contexts` **giữ thứ tự rank của retriever** — quan trọng cho
  Context Precision (rank-aware). **Tại sao `field(default_factory=...)`:**
  dataclass bất biến default `[]` sẽ bị chia sẻ giữa mọi instance (aliasing bug).

**`EvalResult`** (điền hint ở dòng 93–97):

```python
@dataclass
class EvalResult:
    qa_pair: QAPair
    actual_answer: str
    faithfulness: float
    relevance: float
    completeness: float
    passed: bool
    failure_type: str | None = None
    context_precision: float | None = None
    context_recall: float | None = None
```

- **Tại sao thứ tự field quan trọng:** test dựng `EvalResult(qa, "answer", 0.2, 0.3, 0.1, False, "Hallucination")` **theo vị trí** (dòng 430 test_solution.py). Sai thứ tự → sai ý nghĩa.
- `context_recall`/`context_precision` mặc định `None`; **không** nằm trong
  `overall_score()` (chỉ chẩn đoán retriever).

**`overall_score()`:**

```python
def overall_score(self) -> float:
    return (self.faithfulness + self.relevance + self.completeness) / 3.0
```

**Checkpoint:** `pytest tests/test_solution.py::TestEvalResultOverallScore -v` → **3 passed**.

---

### Task 2 — `RAGASEvaluator` (5 metrics + `run_full_eval`)

Dùng `_tokenize()` đã cung cấp (lowercase, bỏ punctuation + stopwords). Tất cả
kết quả **clamp vào `[0.0, 1.0]`**, tránh chia 0.

**`evaluate_faithfulness(answer, context)`** — answer grounded không?

```python
at = _tokenize(answer); ct = _tokenize(context)
if not at: return 1.0
return max(0.0, min(1.0, len(at & ct) / len(at)))
```

- Nếu answer rỗng → return 1.0 (không có claim sai thì "không bịa").

**`evaluate_relevance(answer, question)`** — answer trả lời đúng câu hỏi không?

```python
at = _tokenize(answer); qt = _tokenize(question)
if not qt: return 1.0
return max(0.0, min(1.0, len(at & qt) / len(qt)))
```

**`evaluate_completeness(answer, expected)`** — answer đủ ý expected chưa?

```python
at = _tokenize(answer); et = _tokenize(expected)
if not et: return 1.0
return max(0.0, min(1.0, len(at & et) / len(et)))
```

**`evaluate_context_recall(contexts, expected)`** — union của chunks cover expected không?

```python
union = set()
for c in contexts: union |= _tokenize(c)
et = _tokenize(expected)
if not et: return 1.0
return max(0.0, min(1.0, len(et & union) / len(et)))
```

- Dùng **union** → thêm/bớt chunk không làm Recall giảm chỉ vì thứ tự.

**`evaluate_context_precision(contexts, expected, relevance_threshold=0.1)`** —
Average Precision@K, rank-aware:

```python
et = _tokenize(expected)
if not et: return 1.0
relevant = [len(_tokenize(c) & et) / len(et) >= relevance_threshold for c in contexts]
total_relevant = sum(relevant)
if total_relevant == 0: return 0.0
precision_at_k = []
seen = 0.0
for k, is_rel in enumerate(relevant, start=1):
    if is_rel: seen += 1.0
    precision_at_k.append(seen / k)
ap = sum(p for p, is_rel in zip(precision_at_k, relevant) if is_rel) / total_relevant
return max(0.0, min(1.0, ap))
```

- **Tại sao rank-aware:** chunk relevant *đứng trước* được điểm cao hơn cùng
  chunk đứng sau → khuyến khích reranker (Exercise 3.5) xếp relevant lên đầu
  **mà không đổi tập chunks** (Recall không đổi).

**`run_full_eval(answer, question, context, expected, contexts=None)`**:

```python
eval_result = EvalResult(
    qa_pair=???,
    actual_answer=answer,
    faithfulness=...,
    relevance=...,
    completeness=...,
    passed=(f>=0.5 and r>=0.5 and c>=0.5),
    failure_type=_pick_failure_type(f, r, c),
    context_recall=... if contexts is not None else None,
    context_precision=... if contexts is not None else None,
)
```

- `failure_type` theo "first match wins" (dòng 253–256 template):
  - `faithfulness < 0.3` → `"hallucination"`
  - `relevance < 0.3` → `"irrelevant"`
  - `completeness < 0.3` → `"incomplete"`
  - ngược lại (fail nhưng không thuộc 3 loại) → `"off_topic"`
- `contexts is None` → 2 retrieval fields là `None`.
- **Retrieval scores KHÔNG đổi pass rule và overall_score.**

> **Lưu ý `qa_pair`:** test `test_run_full_eval_connects_optional_retrieval_metrics`
> (dòng 188–204) không kiểm tra `qa_pair`, nhưng `test_runner_forwards_retrieved_contexts`
> dựng `EvalResult(qa_pair=_make_qa(...), ...)` theo keyword. Với `run_full_eval`
> không có param `qa_pair` — cần tạo một `QAPair` tạm từ question/context/expected
> (không có retrieved_contexts). `overall_score()` của result chỉ cần 3 scores
> nên `qa_pair` ít ảnh hưởng, nhưng cần không-None để `evaluate_answers.py`
> truy cập `result.qa_pair.metadata`/`question` an toàn.

**Checkpoint:** `pytest tests/test_solution.py::TestRAGASEvaluator tests/test_solution.py::TestContextMetrics tests/test_solution.py::TestRetrievalMetricWiring::test_run_full_eval_connects_optional_retrieval_metrics -v` → **14 passed, 1 skipped** (1 skipped = test bonus rerank).

---

### Task 3 — `LLMJudge` (`__init__`, `score_response`, `detect_bias`)

```python
def __init__(self, judge_llm_fn):
    self.judge_llm_fn = judge_llm_fn

def score_response(self, question, answer, rubric):
    prompt = f"Judge the answer...\nQuestion: {question}\nAnswer: {answer}\nRubric: {rubric}"
    raw = self.judge_llm_fn(prompt)
    # parse scores (json dict criterion -> 0-1); fallback 0.5 nếu không parse được
    return {"scores": {...}, "reasoning": raw}

def detect_bias(self, scores_batch):
    # positional: response đầu tiên luôn cao hơn
    # leniency: avg score > 0.8
    # severity: avg score < 0.3
    return {"positional_bias": bool, "leniency_bias": bool, "severity_bias": bool}
```

- **Không gọi API trong test:** test dùng `judge_llm_fn` là mock trả về
  `'{"accuracy": 0.8, "clarity": 0.7}'` (dòng 63–64). Hàm chỉ cần parse đúng
  dict đó.
- **Tại sao fallback 0.5:** LLM có thể trả lời ngoài format JSON; thay vì crash,
  ta gán điểm mặc định trung lập và giữ raw response trong `reasoning`.

**Checkpoint:** `pytest tests/test_solution.py::TestLLMJudge -v` → **4 passed**.

---

### Task 4 — `BenchmarkRunner` (`run`, `generate_report`, `run_regression`, `identify_failures`)

**`run(qa_pairs, agent_fn, evaluator)`:**

```python
results = []
for pair in qa_pairs:
    answer = agent_fn(pair.question)
    results.append(evaluator.run_full_eval(
        answer=answer, question=pair.question,
        context=pair.context, expected=pair.expected_answer,
        contexts=pair.retrieved_contexts,
    ))
return results
```

- **Tại sao phải truyền `pair.retrieved_contexts`:** test
  `test_runner_forwards_retrieved_contexts` (dòng 206–240) dùng một
  `RecordingEvaluator` chặn đối số `contexts` và khẳng định nó bằng
  `pair.retrieved_contexts`. Bỏ qua → test fail và Context Recall/Precision sẽ
  là `None` trong report.
- **`EvalResult.qa_pair`** nên được gắn `pair` gốc (để `evaluate_answers.py` đọc
  `metadata.id`).

**`generate_report(results)`** — trả dict đầy đủ:

```python
{
  "total": len(results),
  "passed": count(result.passed),
  "pass_rate": passed / total,
  "avg_faithfulness": mean(faithfulness),
  "avg_relevance": mean(relevance),
  "avg_completeness": mean(completeness),
  "avg_context_recall": mean(context_recall) if any not None else None,
  "avg_context_precision": mean(context_precision) if any not None else None,
  "failure_types": Counter(type or "passed")  # hoặc chỉ các loại failure
}
```

- **Tại sao average retrieval chỉ trên không-None:** có dataset không có chunks
  (mock demo), report vẫn phải chạy được; `None` tránh báo 0.0 giả.

**`run_regression(new_results, baseline_results)`:**

- Tính avg của 3 metrics cho từng tập.
- Metric regress nếu `baseline_avg - new_avg > 0.05` (giảm hơn 0.05).
- `regressions` = list tên metric bị regress; `passed = len(regressions) == 0`.

**`identify_failures(results, threshold=0.5)`:**

```python
return [r for r in results if min(r.faithfulness, r.relevance, r.completeness) < threshold]
```

- **Tại sao dùng min (không dùng overall trung bình):** test
  `test_identify_failures_returns_subset` dùng result `(0.2, 0.2, 0.2)` → overall
  cũng 0.2, nhưng ý nghĩa là *có metric nào thấp* là đáng xem. Xem xét cả 3 score.

**Checkpoint:** `pytest tests/test_solution.py::TestBenchmarkRunner tests/test_solution.py::TestRunRegression tests/test_solution.py::TestRetrievalMetricWiring::test_runner_forwards_retrieved_contexts tests/test_solution.py::TestRetrievalMetricWiring::test_report_includes_retrieval_averages -v` → **11 passed**.

---

### Task 5 — `FailureAnalyzer` (4 hàm)

**`categorize_failures(failures)`:**

```python
from collections import Counter
return dict(Counter(f.failure_type for f in failures))
```

- **Tại sao `dict(Counter(...))`:** test `test_categorize_failures_empty_list`
  yêu cầu trả về dict (rỗng được); Counter là dict con nhưng `dict()` để chắc.

**`find_root_cause(failure)`** — so 3 score, trả đúng chuỗi trong docstring:

- Thấp nhất là `faithfulness` → `"Context is missing or irrelevant — improve retrieval"`
- Thấp nhất là `relevance` → `"Answer does not address the question — improve prompt clarity"`
- Thấp nhất là `completeness` → `"Answer is missing key information — increase context window or improve generation"`
- Bằng nhau / khó phân biệt → `"Multiple issues detected — review full pipeline"`

**`generate_improvement_suggestions(failures)`** — phân tích theo failure type:

```python
if not failures: return []
# nhóm theo type, map mỗi type với 1-2 gợi ý cụ thể
# ví dụ: hallucination → "Implement hallucination checker..."
#        irrelevant → "Add few-shot examples clarifying the question intent..."
#        incomplete → "Increase chunk size in RAG pipeline..."
# trả list >= 3 nếu có >= 3 loại failure
```

- **Tại sao ≥ 3:** test `test_generate_suggestions_at_least_3` với 3 failures
  khác type yêu cầu `len >= 3`.

**`generate_improvement_log(failures, suggestions)`** — Markdown table:

```markdown
| Failure ID | Type | Root Cause | Suggested Fix | Status |
|------------|------|------------|---------------|--------|
| F001       | Hallucination | ... | ... | Open   |
```

- **Format bắt buộc:** test yêu cầu chuỗi chứa `"Open"`, tên failure type
  (vd `"Hallucination"`), và ký tự `"|"`. `F001` bắt đầu từ 1.
- `find_root_cause(f)` cho mỗi failure; suggestion lấy theo index (nếu
  suggestions ngắn hơn failures, lấy vòng lại hoặc "General improvement").

**Checkpoint:** `pytest tests/test_solution.py::TestFailureAnalyzer tests/test_solution.py::TestGenerateImprovementLog -v` → **9 passed**.

---

### 3.7 Bảng checkpoint đầy đủ

| Checkpoint | Lệnh targeted                                                                                                                                                                                                                                                                            | Mong đợi           |
| ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------- |
| Task 1     | `pytest tests/test_solution.py::TestEvalResultOverallScore -v`                                                                                                                                                                                                                          | 3 passed             |
| Task 2     | `pytest tests/test_solution.py::TestRAGASEvaluator tests/test_solution.py::TestContextMetrics tests/test_solution.py::TestRetrievalMetricWiring::test_run_full_eval_connects_optional_retrieval_metrics -v`                                                                             | 14 passed, 1 skipped |
| Task 3     | `pytest tests/test_solution.py::TestLLMJudge -v`                                                                                                                                                                                                                                        | 4 passed             |
| Task 4     | `pytest tests/test_solution.py::TestBenchmarkRunner tests/test_solution.py::TestRunRegression tests/test_solution.py::TestRetrievalMetricWiring::test_runner_forwards_retrieved_contexts tests/test_solution.py::TestRetrievalMetricWiring::test_report_includes_retrieval_averages -v` | 11 passed            |
| Task 5     | `pytest tests/test_solution.py::TestFailureAnalyzer tests/test_solution.py::TestGenerateImprovementLog -v`                                                                                                                                                                              | 9 passed             |

Full-suite cộng dồn (baseline chuẩn):

| Trạng thái                       | Full suite                      |
| ---------------------------------- | ------------------------------- |
| Chưa làm TODO                    | 0 passed, 42 failed             |
| Xong Task 1                        | 3 passed, 39 failed             |
| Xong Task 2                        | 17 passed, 24 failed, 1 skipped |
| Xong Task 3                        | 21 passed, 20 failed, 1 skipped |
| Xong Task 4                        | 32 passed, 9 failed, 1 skipped  |
| **Xong Task 5 (bắt buộc)** | **41 passed, 1 skipped**  |
| Bonus`rerank_by_overlap()`       | 42 passed                       |

> ⚠️ Không sửa tests để làm bài pass. Test bonus (rerank) skip nếu chưa làm
> Exercise 3.5.

---

## 4. Part 3 — Golden Dataset (Exercise 3.1)

### Bước 4.1 — Đọc corpus trước khi viết QA

- Đọc `data/student_services/manifest.json` → biết **10 documents** + use cases.
- Đọc từng file `.md`. Corpus này **synthetic** nhưng là **source of truth duy
  nhất** — kể cả khi khác kiến thức thực tế, expected answer phải theo corpus.
- **Không sửa corpus** để khớp expected answer với suy đoán.

### Bước 4.2 — Điền `golden_dataset.json` (20 slots có sẵn)

- `E01–E05` (easy), `M01–M07` (medium), `H01–H05` (hard), `A01–A03` (adversarial).
- **Chỉ điền:** `question`, `expected_answer`, `contexts`. **Không đổi** `id`,
  `difficulty`, `attack_type`.
- Mỗi context: `{"source_doc": "tên file md", "text": "đoạn trích nguyên văn"}`.
  `text` phải là **substring nguyên văn** của source (validator kiểm tra chặt).

**Ý nghĩa từng field:**

| Field               | Cách điền                                                           |
| ------------------- | ---------------------------------------------------------------------- |
| `question`        | Câu hỏi tự thiết kế                                               |
| `expected_answer` | Reference answer chuẩn, giữ đủ dates/amounts/conditions/exceptions |
| `contexts`        | Evidence hỗ trợ**toàn bộ** expected answer                   |

**Phân bổ và mục đích (stratified sampling):**

| Nhóm       | Số | Đặc điểm                                                                                |
| ----------- | --- | ------------------------------------------------------------------------------------------- |
| Easy        | 5   | Factual lookup, 1 document                                                                  |
| Medium      | 7   | Multi-step / multi-document (2–3 docs)                                                     |
| Hard        | 5   | Nhiều điều kiện, exception, effective date, ambiguity                                   |
| Adversarial | 3   | out_of_scope, prompt_injection, false_premise trap (bắt buộc dùng`00_system_scope.md`) |

### Bước 4.3 — Yêu cầu toàn dataset

1. Đúng 20 records, đúng thứ tự IDs.
2. Dùng đủ **10 source documents** (ít nhất 1 lần mỗi file).
3. Mọi evidence là substring nguyên văn.
4. Mọi claim trong expected answer được evidence hỗ trợ.
5. Không trùng câu hỏi.
6. Không dùng kiến thức ngoài corpus.
7. Viết bằng tiếng Anh.
8. Expected answer ngắn gọn nhưng đủ dates/amounts/conditions/exceptions.

### Bước 4.4 — Thiết kế 3 adversarial cases

| ID  | Attack type                         | Hành vi cần test                                                                            |
| --- | ----------------------------------- | --------------------------------------------------------------------------------------------- |
| A01 | `out_of_scope`                    | Assistant**từ chối/giới hạn** đúng scope (dẫn chứng từ `00_system_scope.md`) |
| A02 | `prompt_injection`                | **Không tuân** lệnh phá system rules / tiết lộ thông tin                         |
| A03 | `false_premise_or_ambiguous_trap` | **Không xác nhận** premise sai hoặc đoán bừa                                     |

- **Tại sao 3 case này bắt buộc dùng `00_system_scope.md`:** validator khóa
  `required_sources = {"00_system_scope.md"}` cho A01–A03.

### Bước 4.5 — Tự review chất lượng (rubric)

Với từng record: "Có thể trả lời expected answer chỉ từ contexts đã chọn
không?", "Có claim nào không evidence không?", "Difficulty có đúng bản chất
reasoning không?", "Question có lộ nguyên câu answer không?", "Evidence có quá
dài / nhiều noise không?"

- **Tại sao:** validator không kiểm tra *semantic quality* — chỉ kiểm tra cấu
  trúc + provenance. Chất lượng thật phải tự review.

---

## 5. Validate Golden Dataset

```bash
python validate_golden_dataset.py
```

- **Kết quả đạt:** `PASS: dataset structure and evidence provenance are valid.`
- Validator kiểm tra: schema, đủ 20 records, đúng ID/difficulty/attack_type,
  không rỗng, không duplicate question, source tồn tại, evidence là substring
  nguyên văn, dùng đủ 10 docs, 3 adversarial dùng scope evidence.

**Lỗi thường gặp & xử lý:**

| Lỗi                                       | Xử lý                                                                                                 |
| ------------------------------------------ | ------------------------------------------------------------------------------------------------------- |
| `text is not a verbatim substring`       | Copy lại**nguyên văn** từ Markdown; không tự sửa punctuation/spacing/wording               |
| `Dataset must use every source document` | Thiết kế case hợp lý cho file còn thiếu; không thêm evidence vô nghĩa chỉ để đủ coverage |
| `expected attack_type` / `difficulty`  | Khôi phục field theo template                                                                         |

- **Tại sao phải PASS trước khi gọi API:** guide yêu cầu "không gọi API trước
  khi golden dataset validate thành công" — tránh tốn chi phí chạy RAG trên dữ
  liệu hỏng.

---

## 6. Cấu hình OpenAI API

```bash
Copy-Item .env.example .env   # Windows PowerShell
# sửa .env:
# OPENAI_API_KEY=<key thật>
# OPENAI_MODEL=gpt-4o-mini
```

- **Tại sao chỉ `domain_assistant.py` cần key:** Phần core (`template.py`),
  validator và evaluation đều chạy offline; chỉ generation RAG gọi OpenAI.
- `.env` đã trong `.gitignore` — **không commit key.**

---

## 7. Sinh 20 Actual Answers

```bash
python domain_assistant.py
# tương đương: python domain_assistant.py --corpus-dir data/student_services --dataset golden_dataset.json --output artifacts/actual_answers.json --top-k 5
```

- Output: `artifacts/actual_answers.json` — 20 answers + `retrieved_contexts`
  (source_doc, chunk_id, text, score).
- **Tại sao `domain_assistant.py` không bị gold leakage:** nó chỉ đọc `id` +
  `question` từ dataset (hàm `_load_questions`), không bao giờ đọc
  `expected_answer`/gold contexts. Đây là rào cản thiết kế, không phải thói quen.

**Kiểm tra sau khi chạy:** question đúng, `actual_answer` có nội dung,
`retrieved_contexts` đầy đủ + liên quan, `error: null`. Nếu lỗi giữa chừng,
script **dừng** và không ghi artifact giả — sửa lỗi rồi chạy lại.

---

## 8. Chạy Evaluation — Exercise 3.2

**Điều kiện trước khi chạy:**

- Golden dataset đã PASS validator.
- `template.py` đã hoàn thành TODO bắt buộc.
- `artifacts/actual_answers.json` đủ 20 answers.

```bash
python evaluate_answers.py
# tương đương: python evaluate_answers.py --golden golden_dataset.json --actual artifacts/actual_answers.json --output artifacts/benchmark_results.json
```

Adapter thực hiện (đọc code `evaluate_answers.py`):

1. Join golden + actual theo ID.
2. Tạo `QAPair` với gold contexts + retrieved contexts.
3. `recorded_agent(question)` trả về `actual_answer` đã sinh (không gọi lại LLM).
4. Gọi `BenchmarkRunner.run()` trong `template.py`.
5. Gọi `generate_report()` + `FailureAnalyzer`.
6. In bảng Exercise 3.2, lưu `artifacts/benchmark_results.json`.

- Nếu thấy `ERROR: Complete the required TODOs in template.py first` → quay lại
  Part 2. **Không viết metric mới vào `evaluate_answers.py` để bypass.**

---

## 9. Điền Exercises 3.2 & 3.3

### Exercise 3.2 — Bảng benchmark

Điền từ terminal/JSON: Context Recall, Precision, Faithfulness, Relevance,
Completeness, Overall, Passed, Failure Type. Ghi aggregate report + 3 cases
Overall thấp nhất.

**Cách đọc kết quả (đừng chỉ nhìn pass rate):**

| Pattern                             | Chẩn đoán                                     |
| ----------------------------------- | ------------------------------------------------ |
| Recall thấp + Completeness thấp   | Retriever bỏ sót evidence                      |
| Recall cao + Precision thấp        | Lấy đủ nhưng ranking/noise kém              |
| Retrieval tốt + Faithfulness thấp | Generation thêm claim ngoài context            |
| Faithfulness cao + Relevance thấp  | Answer grounded nhưng không trả đúng intent |

### Exercise 3.3 — Rubric 1–5 domain-specific

Rubric phải nêu rõ: điều kiện đạt từng mức, xử lý missing conditions/exceptions,
phạt claim không evidence, xử lý privacy/safety failures, tránh thưởng answer
dài. **Không dùng "5 = tốt, 1 = xấu" mơ hồ.** Kèm bias controls (position,
verbosity, self-preference).

---

## 10. Viết `reflection.md`

Dùng 3 cases thấp nhất từ benchmark. Với mỗi case:

1. Đọc question, expected, actual answer.
2. So sánh gold evidence với retrieved chunks.
3. Xác định symptom.
4. Đi qua 5 Whys đến root cause.
5. So sánh nhận định với `find_root_cause()`.
6. Đề xuất fix cụ thể + metric để verify.

Sau đó **cluster failures** (group theo root cause sửa được, không theo tên
metric) — fix 1 root cause giải quyết nhiều cases.

---

## 11. Bonus (chỉ sau phần bắt buộc)

### Exercise 3.4 (+10) — So sánh 2 evaluation frameworks

RAGAS / DeepEval / TruLens trên cùng dataset. Ghi phương pháp + kết quả trong
`exercises.md`. Không cần tạo file code mới.

### Exercise 3.5 (+5) — Reranking

1. Chọn ≥ 5 cases từ `artifacts/actual_answers.json`.
2. Đo Recall + Precision trước.
3. Implement `rerank_by_overlap(contexts, query)`:

```python
def rerank_by_overlap(contexts, query):
    return sorted(contexts, key=lambda c: len(_tokenize(c) & _tokenize(query)), reverse=True)
```

4. Rerank **cùng tập chunks** → Precision tăng, **Recall không đổi** (vì union giữ nguyên).
5. Đo lại và giải thích.

- **Test bonus:** `test_reranking_improves_or_keeps_precision` (được skip nếu
  `rerank_by_overlap` còn `NotImplementedError`).

---

## 12. Copy Solution & kiểm tra cuối

```powershell
Copy-Item template.py solution/solution.py
pytest tests/ -v
python validate_golden_dataset.py
```

- **Tại sao test ưu tiên `solution/solution.py`:** (xem mục 1.3). Sau khi copy,
  nếu sửa `template.py` phải **copy lại** trước lần test cuối.

**Bốn deliverables bắt buộc:**

```text
solution/solution.py
golden_dataset.json
exercises.md
reflection.md
```

**Artifacts hỗ trợ (không bắt buộc commit):**

```text
artifacts/actual_answers.json
artifacts/benchmark_results.json
```

---

## 13. Commit & push

```bash
git status
git diff --check
git diff
```

- Đảm bảo **không có `.env` hoặc API key** trong diff.

---

## 14. Bảng lỗi thường gặp (trích từ guide Mục 15)

| Triệu chứng                                         | Nguyên nhân                                            | Xử lý                                                       |
| ----------------------------------------------------- | -------------------------------------------------------- | ------------------------------------------------------------- |
| `ImportError: cannot import name UTC from datetime` | Venv tạo bằng 3.9/3.10                                 | Tạo lại venv bằng 3.11+                                    |
| `ModuleNotFoundError: openai/dotenv`                | Chưa activate hoặc chưa cài req                      | Activate +`pip install -r requirements.txt`                 |
| `text is not a verbatim substring`                  | Evidence sửa wording                                    | Copy lại nguyên văn                                        |
| `OPENAI_API_KEY is missing`                         | Thiếu`.env`                                           | Copy`.env.example` → `.env`, điền key                  |
| `Complete the required TODOs`                       | Core còn`NotImplementedError`                         | Làm checkpoint tương ứng                                  |
| Sửa`template.py` nhưng test vẫn cũ              | `solution/solution.py` tồn tại và được ưu tiên | Copy lại rồi test                                           |
| Context Recall/Precision là`None`                  | Chunks chưa truyền vào evaluator                      | Kiểm tra`retrieved_contexts` + Runner truyền `contexts` |

---

## 15. Tóm tắt thứ tự thực hiện (checklist ngắn)

1. [ ] Venv + requirements + import check (Mục 1)
2. [ ] Baseline `pytest tests/ -v` → 42 failed (xác nhận pipeline chạy)
3. [ ] Part 1: Exercises 1.1–1.3 trong `exercises.md`
4. [ ] Task 1 → checkpoint 3 passed
5. [ ] Task 2 → checkpoint 14 passed, 1 skipped
6. [ ] Task 3 → checkpoint 4 passed
7. [ ] Task 4 → checkpoint 11 passed
8. [ ] Task 5 → checkpoint 9 passed
9. [ ] Full suite: **41 passed, 1 skipped**
1. [ ] Điền `golden_dataset.json` (20 QA) → validator PASS
1. [ ] `Copy-Item .env.example .env` + điền key
1. [ ] `python domain_assistant.py` → `artifacts/actual_answers.json`
1. [ ] `python evaluate_answers.py` → bảng 3.2 + `benchmark_results.json`
1. [ ] Điền Exercises 3.2, 3.3; viết `reflection.md`
1. [ ] (Bonus) 3.4 framework comparison, 3.5 rerank → full suite 42 passed
1. [ ] `Copy-Item template.py solution/solution.py`; chạy 2 lệnh kiểm tra cuối
1. [ ] `git diff --check`; commit không kèm `.env`/key
