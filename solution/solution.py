"""
Day 14 — AI Evaluation & Benchmarking Pipeline
AICB-P1: AI Practical Competency Program, Phase 1

Key concepts from lecture:
    - Evaluation = Scientific Method for AI (Hypothesis → Experiment → Measure → Conclude → Iterate)
    - 4 nhóm metrics: Task Completion, Answer Quality, RAG-Specific, Business
    - RAG pipeline metrics: Context Recall → Context Precision → Faithfulness → Answer Relevancy
    - LLM-as-Judge: rubric scoring 1-5, detect bias (positional, verbosity, self-preference)
    - Golden dataset: stratified sampling (5 Easy + 7 Medium + 5 Hard + 3 Adversarial)
    - Failure taxonomy: hallucination, irrelevant, incomplete, off_topic, refusal
    - 5 Whys method for root cause analysis
    - CI/CD integration: eval as quality gate (score < threshold = block deploy)
    - Continuous Improvement Loop: Evaluate → Analyze → Improve → Augment → Repeat

Instructions:
    1. Fill in every required section marked with TODO.
    2. Do NOT change class/function signatures. The optional ``contexts``
       parameter in ``run_full_eval`` is part of the required interface.
    3. Copy this file to solution/solution.py when done.
    4. Run: pytest tests/ -v

The reranking helper is an optional bonus exercise and may remain unimplemented.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Task 1 — Data Models (Golden Dataset + Evaluation Results)
# ---------------------------------------------------------------------------

@dataclass
class QAPair:
    """
    A question-answer pair for evaluation (part of the Golden Dataset).

    From lecture: Golden dataset cần có:
        - question: câu hỏi user
        - ground_truth (expected_answer): expert-written expected answer
        - context: source documents cần retrieve
        - metadata: difficulty (easy/medium/hard), category, source_docs

    Fields:
        question:        The question to answer.
        expected_answer: The reference/ground-truth answer (expert-written).
        context:            Source context (may be empty string if not applicable).
        metadata:           Optional metadata dict (difficulty, category, etc.).
        retrieved_contexts: List of retrieved chunks (ORDER = retriever rank).
                            Used by the retrieval-side metrics (Task 2b).
    """
    question: str
    expected_answer: str
    context: str = ""
    metadata: dict = field(default_factory=dict)
    retrieved_contexts: list = field(default_factory=list)


@dataclass
class EvalResult:
    """
    Evaluation result for a single Q&A pair.

    From lecture - RAG metrics pipeline:
        Question → Retriever → Context → Generator → Answer
        Each step has a metric: Context Recall, Context Precision, Faithfulness, Answer Relevancy

    From lecture - Score interpretation:
        0.8-1.0: Good (Monitor, maintain)
        0.6-0.8: Needs work (Analyze failures, iterate)
        < 0.6: Significant issues (Deep investigation required)

    Fields:
        qa_pair:        The original QAPair.
        actual_answer:  What the agent actually returned.
        faithfulness:   Float 0-1, how grounded the answer is in context.
        relevance:      Float 0-1, how relevant the answer is to the question.
        completeness:   Float 0-1, how complete the answer is vs expected.
        passed:         True if all three scores >= 0.5.
        failure_type:   None if passed, otherwise one of:
                        "hallucination", "irrelevant", "incomplete", "off_topic".
        context_precision: Float 0-1 or None — quality of retrieval ranking.
        context_recall:    Float 0-1 or None — coverage of expected by context.
                        (Both stay None unless retrieved chunks are supplied;
                         they are NOT part of overall_score().)
    """
    # ===== Các biến (fields) của EvalResult =====
    #
    # qa_pair  : bản ghi QAPair gốc (question, expected_answer, context,
    #            metadata, retrieved_contexts). Giữ lại để sau này tra ngược:
    #            biết kết quả này ứng với câu hỏi / độ khó / attack_type nào.
    # actual_answer : câu trả lời THẬT mà agent (domain_assistant) đã sinh ra.
    #            Đây là thứ chúng ta đang chấm điểm.
    #
    # faithfulness  : float 0–1. Answer có "bám" vào context không?
    #            = |answer ∩ context| / |answer|. Low => có claim bịa.
    # relevance     : float 0–1. Answer có trả lời đúng câu hỏi không?
    #            = |answer ∩ question| / |question|. Low => lạc đề.
    # completeness  : float 0–1. Answer có đủ nội dung expected không?
    #            = |answer ∩ expected| / |expected|. Low => thiếu ý.
    #
    # passed     : True nếu CẢ BA score trên >= 0.5 (cổng pass của bài lab).
    # failure_type: None nếu passed, ngược lại là một trong
    #            "hallucination" | "irrelevant" | "incomplete" | "off_topic".
    #
    # context_precision / context_recall: float 0–1 hoặc None — hai metric
    #            CHẨN ĐOÁN RETRIEVER. Mặc định None; chỉ được tính và gán khi
    #            người gọi truyền danh sách chunks (contexts) vào. Chúng KHÔNG
    #            nằm trong overall_score() và không đổi passed rule.
    qa_pair: QAPair
    actual_answer: str
    faithfulness: float
    relevance: float
    completeness: float
    passed: bool
    failure_type: str | None = None
    context_precision: float | None = None
    context_recall: float | None = None

    def overall_score(self) -> float:
        """Tính điểm tổng hợp trung bình của 3 answer-side metrics.

        Vai trò: cho một con số duy nhất để xếp hạng / so sánh các câu trả lời.
        Ý nghĩa: overall_score() = (faithfulness + relevance + completeness) / 3.
        Lưu ý: KHÔNG đưa context_precision / context_recall vào đây — chúng chỉ
        chẩn đoán retriever, không đại diện chất lượng câu trả lời.

        Returns:
            (faithfulness + relevance + completeness) / 3.0
        """
        return (self.faithfulness + self.relevance + self.completeness) / 3.0


# ---------------------------------------------------------------------------
# Task 2 — RAGAS Evaluator (Simplified word-overlap heuristic)
# ---------------------------------------------------------------------------
# In production, replace with actual RAGAS framework:
#   from ragas import evaluate
#   from ragas.metrics import Faithfulness, AnswerRelevancy, ContextRecall, ContextPrecision
#
# Or DeepEval:
#   from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
#   assert_test(test_case, [faithfulness, hallucination])
#
# Or TruLens:
#   from trulens.core import Feedback
#   f_groundedness = Feedback(provider.groundedness_measure_with_cot_reasons)
# ---------------------------------------------------------------------------

# Common English stopwords are ignored so overlap reflects *content* words,
# not filler (otherwise "is"/"a"/"the" inflate every score).
STOPWORDS: set[str] = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "in", "on", "at", "to", "for", "with", "as", "by", "and", "or",
    "it", "its", "this", "that", "these", "those", "from", "into", "than",
}


def _tokenize(text: str) -> set[str]:
    """Lowercase word tokenization, ignoring punctuation and stopwords."""
    if not text:
        return set()
    tokens = re.findall(r"\b\w+\b", text.lower())
    return {t for t in tokens if t not in STOPWORDS}


class RAGASEvaluator:
    """
    Evaluates RAG pipeline outputs using RAGAS-inspired heuristics.

    All metrics use word overlap rather than LLM calls for simplicity.
    Replace with actual LLM-based evaluation in production.
    """

    def evaluate_faithfulness(self, answer: str, context: str) -> float:
        """
        Measure how grounded the answer is in the context.

        Heuristic:
            answer_tokens = _tokenize(answer)
            context_tokens = _tokenize(context)
            faithfulness = |answer_tokens ∩ context_tokens| / |answer_tokens|
            Clamp to [0.0, 1.0]. Return 1.0 if answer is empty.

        Returns:
            float in [0.0, 1.0] — 1.0 = fully grounded in context.
        """
        # ===== Giải thích =====
        # - _tokenize() đưa văn bản về tập token (lowercase, bỏ punctuation &
        #   stopwords) để so sánh "ý" chứ không so chuỗi nguyên văn.
        # - answer_tokens: những khái niệm mà answer nói tới.
        # - context_tokens: những khái niệm có sẵn trong context (chunk truy hồi).
        # - Công thức: phần answer nằm trong context chia cho toàn bộ answer.
        #   Nếu answer chỉ nói những thứ có trong context => 1.0 (grounded).
        #   Answer càng nói thứ không có trong context => càng thấp (bịa).
        answer_tokens = _tokenize(answer)
        context_tokens = _tokenize(context)
        if not answer_tokens:
            return 1.0  # answer rỗng => không có claim nào để "bịa"
        return max(0.0, min(1.0, len(answer_tokens & context_tokens) / len(answer_tokens)))

    def evaluate_relevance(self, answer: str, question: str) -> float:
        """
        Measure how relevant the answer is to the question.

        Heuristic:
            relevance = |answer_tokens ∩ question_tokens| / |question_tokens|
            Clamp to [0.0, 1.0]. Return 1.0 if question is empty.

        Returns:
            float in [0.0, 1.0]
        """
        # ===== Giải thích =====
        # - question_tokens: trọng tâm của câu hỏi (những từ nội dung).
        # - answer_tokens: những gì answer đề cập.
        # - Công thức: phần answer trùng với câu hỏi / toàn bộ câu hỏi.
        #   Điểm cao => answer "chạm" được các ý chính người dùng hỏi.
        #   Điểm thấp => answer lạc đề, không giải quyết câu hỏi.
        # - Mẫu số là |question_tokens| (chứ không phải |answer_tokens|) vì ta
        #   hỏi "câu hỏi có được answer đề cập đủ không".
        answer_tokens = _tokenize(answer)
        question_tokens = _tokenize(question)
        if not question_tokens:
            return 1.0  # question rỗng => không có intent để "trượt"
        return max(0.0, min(1.0, len(answer_tokens & question_tokens) / len(question_tokens)))

    def evaluate_completeness(self, answer: str, expected: str) -> float:
        """
        Measure how well the answer covers the expected answer.

        Heuristic:
            completeness = |answer_tokens ∩ expected_tokens| / |expected_tokens|
            Clamp to [0.0, 1.0]. Return 1.0 if expected is empty.

        Returns:
            float in [0.0, 1.0]
        """
        # ===== Giải thích =====
        # - expected_tokens: các yếu tố bắt buộc mà một câu trả lời chuẩn phải có
        #   (vd: dates, amounts, conditions trong chính sách).
        # - Công thức: phần expected được answer bao phủ / toàn bộ expected.
        #   Điểm cao => answer "đủ ý" so với đáp án chuẩn.
        #   Điểm thấp => answer thiếu thông tin quan trọng (incomplete).
        # - Mẫu số là |expected_tokens|: ta đo mức độ expected được phủ, chứ
        #   không phải answer dài hay ngắn.
        answer_tokens = _tokenize(answer)
        expected_tokens = _tokenize(expected)
        if not expected_tokens:
            return 1.0  # expected rỗng => không có chuẩn nào để thiếu
        return max(0.0, min(1.0, len(answer_tokens & expected_tokens) / len(expected_tokens)))

    # -----------------------------------------------------------------------
    # Task 2b — Retrieval-side metrics (evaluate the GET-CONTEXT step)
    # -----------------------------------------------------------------------
    # From lecture (RAG pipeline): Context Recall → Context Precision →
    #   Faithfulness → Answer Relevancy. The two below score the RETRIEVER,
    #   operating on a LIST of chunks (order = retriever rank).
    # -----------------------------------------------------------------------

    def evaluate_context_recall(self, contexts: list[str], expected: str) -> float:
        """Context Recall — how much of the expected answer is covered by the
        UNION of retrieved chunks.

        Heuristic:
            union_tokens = ⋃ _tokenize(chunk) for chunk in contexts
            recall = |expected_tokens ∩ union_tokens| / |expected_tokens|
            Clamp to [0.0, 1.0]. Return 1.0 if expected is empty.

        Low recall => retriever missed evidence the answer needs.
        """
        # ===== Giải thích =====
        # - union_tokens: HỢP token của tất cả chunks được truy hồi (gom lại).
        #   Dùng union => thứ tự hay số lượng chunk không quan trọng; chỉ cần
        #   evidence NẰM Ở ĐÂU ĐÓ trong các chunk là được.
        # - expected_tokens: những yếu tố answer cần để trả lời đúng.
        # - Công thức: phần expected nằm trong (union của) chunks / toàn bộ expected.
        #   Điểm cao => retriever đã lấy đủ evidence.
        #   Điểm thấp => retriever bỏ sót evidence (retrieval side issue).
        union_tokens: set[str] = set()
        for chunk in contexts:
            union_tokens |= _tokenize(chunk)
        expected_tokens = _tokenize(expected)
        if not expected_tokens:
            return 1.0  # expected rỗng => không có gì cần "cover"
        return max(0.0, min(1.0, len(expected_tokens & union_tokens) / len(expected_tokens)))

    def evaluate_context_precision(
        self,
        contexts: list[str],
        expected: str,
        relevance_threshold: float = 0.1,
    ) -> float:
        """Context Precision — RANK-AWARE Average Precision (AP@K), like RAGAS.
        Rewards retrievers that place RELEVANT chunks BEFORE noise.

        Steps:
            1. A chunk is "relevant" if it covers >= relevance_threshold of the
               expected tokens:  |chunk ∩ expected| / |expected| >= threshold
            2. Precision@k = (#relevant in top-k) / k
            3. AP@K = (1 / #relevant) * Σ_k [ Precision@k · relevant_k ]

        Return 1.0 if expected empty; 0.0 if no chunks or none relevant.
        Reordering relevant chunks earlier (reranking) raises this score.
        """
        # ===== Giải thích (Average Precision @ K) =====
        # Mục đích: đánh giá THỨ TỰ (rank) của chunks chứ không phải coverage.
        # Một chunk "relevant" nếu nó chứa >= relevance_threshold phần token
        # của expected (ngưỡng mặc định 0.1 — chỉ cần chứa một phần nhỏ evidence).
        #
        # Cách tính theo từng vị trí k trong danh sách chunks (theo đúng rank
        # retriever trả về):
        #   - Precision@k = (số chunk relevant trong top-k) / k
        #   - AP@K = (1 / tổng số chunk relevant) * Σ [ Precision@k * relevant_k ]
        #     (chỉ cộng ở các vị trí có relevant_k = True)
        #
        # Ý nghĩa: chunk relevant càng ĐỨNG SỚM thì Precision@k được tính càng
        # sớm và càng lớn => AP cao. Cùng một tập chunks nhưng xếp relevant lên
        # đầu (reranking) sẽ TĂNG điểm này mà KHÔNG đổi tập (Context Recall
        # giữ nguyên) — đây chính là cơ chế của Exercise 3.5.
        expected_tokens = _tokenize(expected)
        if not expected_tokens:
            return 1.0  # expected rỗng => không có gì để "relevant"

        relevant_flags: list[bool] = []
        for chunk in contexts:
            chunk_tokens = _tokenize(chunk)
            if not expected_tokens:
                relevant_flags.append(True)
                continue
            coverage = len(chunk_tokens & expected_tokens) / len(expected_tokens)
            relevant_flags.append(coverage >= relevance_threshold)

        total_relevant = sum(relevant_flags)
        if total_relevant == 0:
            return 0.0  # không chunk nào relevant => precision = 0

        precision_at_k: list[float] = []
        seen_relevant = 0
        for k, is_relevant in enumerate(relevant_flags, start=1):
            if is_relevant:
                seen_relevant += 1
            precision_at_k.append(seen_relevant / k)

        average_precision = sum(
            p for p, is_rel in zip(precision_at_k, relevant_flags) if is_rel
        ) / total_relevant
        return max(0.0, min(1.0, average_precision))

    def run_full_eval(
        self,
        answer: str,
        question: str,
        context: str,
        expected: str,
        contexts: list[str] | None = None,
    ) -> EvalResult:
        """
        Run the three answer-side evaluations and, when ``contexts`` is
        supplied, both retrieval-side evaluations.

        passed = True if all three scores >= 0.5.

        failure_type determination (first match wins):
            faithfulness < 0.3  → "hallucination"
            relevance < 0.3     → "irrelevant"
            completeness < 0.3  → "incomplete"
            otherwise if failed → "off_topic"

        Retrieval wiring:
            contexts is None → context_recall and context_precision stay None
            contexts provided → evaluate and store both retrieval metrics

        The two retrieval metrics diagnose the retriever and do not change the
        three-metric ``passed`` rule or ``overall_score()``.

        Returns:
            EvalResult with all fields populated.
        """
        # ===== Giải thích =====
        # Hàm này là "pipeline thu nhỏ": nạp toàn bộ dữ liệu của MỘT cặp
        # (câu hỏi / câu trả lời / gold context / expected / chunks) vào một
        # EvalResult duy nhất.
        #
        # 1) Luôn tính 3 answer-side metrics (faithfulness, relevance,
        #    completeness) — đây là cốt lõi đánh giá câu trả lời.
        # 2) passed = True chỉ khi CẢ BA >= 0.5 (cổng pass của lab).
        # 3) failure_type theo luật "first match wins" (đọc từ docstring):
        #      faithfulness < 0.3 → "hallucination"  (bịa)
        #      relevance    < 0.3 → "irrelevant"     (lạc đề)
        #      completeness < 0.3 → "incomplete"     (thiếu ý)
        #      còn lại, nếu fail → "off_topic"
        # 4) retrieval metrics: chỉ tính khi người gọi truyền `contexts` (tức
        #    là danh sách chunks thật của retriever). Nếu None → giữ None.
        #    Hai giá trị này KHÔNG đổi passed rule và không nằm trong
        #    overall_score(); chúng chỉ chẩn đoán retriever.
        faithfulness = self.evaluate_faithfulness(answer, context)
        relevance = self.evaluate_relevance(answer, question)
        completeness = self.evaluate_completeness(answer, expected)

        passed = faithfulness >= 0.5 and relevance >= 0.5 and completeness >= 0.5
        if not passed:
            if faithfulness < 0.3:
                failure_type = "hallucination"
            elif relevance < 0.3:
                failure_type = "irrelevant"
            elif completeness < 0.3:
                failure_type = "incomplete"
            else:
                failure_type = "off_topic"
        else:
            failure_type = None

        context_recall = None
        context_precision = None
        if contexts is not None:
            context_recall = self.evaluate_context_recall(contexts, expected)
            context_precision = self.evaluate_context_precision(contexts, expected)

        # qa_pair: tạo tạm một QAPair để các thành phần sau (vd evaluate_answers)
        # truy cập an toàn .question / .expected_answer / .context. Metadata trống
        # vì ở đây chưa có id/difficulty — adapter sẽ tự gắn pair gốc.
        qa_pair = QAPair(
            question=question,
            expected_answer=expected,
            context=context,
        )

        return EvalResult(
            qa_pair=qa_pair,
            actual_answer=answer,
            faithfulness=faithfulness,
            relevance=relevance,
            completeness=completeness,
            passed=passed,
            failure_type=failure_type,
            context_precision=context_precision,
            context_recall=context_recall,
        )


# ---------------------------------------------------------------------------
# Reranking helper (used by Exercise 3.5 — boosting Context Precision)
# ---------------------------------------------------------------------------

def rerank_by_overlap(contexts: list[str], query: str) -> list[str]:
    """A minimal lexical reranker: sort chunks by word overlap with the query,
    most-overlapping first. Stand-in for a real cross-encoder reranker.

    Reordering relevant chunks toward the top increases the rank-aware
    Context Precision WITHOUT changing the retrieved set.

    Hint: sorted(contexts, key=lambda c: len(_tokenize(c) & _tokenize(query)),
                 reverse=True)
    """
    # TODO (Bonus — Exercise 3.5): implement the reranker
    raise NotImplementedError("Implement rerank_by_overlap")


# ---------------------------------------------------------------------------
# Task 3 — LLM Judge
# ---------------------------------------------------------------------------
# From lecture:
#   - Judge LLM nhận: question + agent answer + reference answer + rubric
#   - Judge trả về: Score 1-5 + Rationale
#   - Best practices: multiple judges, randomize order, calibrate against human
#   - Biases: positional, verbosity, self-preference
#   - Rubric template:
#       5 = Correct, complete, well-cited
#       4 = Mostly correct, minor gaps
#       3 = Partially correct, some errors
#       2 = Significant errors or missing info
#       1 = Wrong or irrelevant
# ---------------------------------------------------------------------------

class LLMJudge:
    """
    Uses an LLM to score AI responses according to a rubric.
    """

    def __init__(self, judge_llm_fn: Callable[[str], str]) -> None:
        # ===== Giải thích =====
        # judge_llm_fn: một callable "str → str" — nhận prompt, trả về phản hồi.
        # Dùng callable (thay vì hard-code API) để test cắm mock response mà
        # không cần gọi API thật (như `_mock_judge_llm` trong test_solution.py).
        # Vai trò: đây là "giám khảo LLM" mà chúng ta sẽ hỏi để chấm điểm.
        self.judge_llm_fn = judge_llm_fn

    def score_response(
        self,
        question: str,
        answer: str,
        rubric: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Score an AI response using the judge LLM.

        Args:
            question: The original question.
            answer:   The AI's answer to score.
            rubric:   Dict mapping criterion name → description.
                      Example: {"accuracy": "Is the answer factually correct?",
                                "clarity": "Is the answer clear and well-structured?"}

        Behavior:
            1. Build a judge prompt that includes the question, answer, and rubric.
            2. Call judge_llm_fn(prompt).
            3. Parse the response for scores.

        For simplicity, if the LLM response can't be parsed as JSON scores,
        return a default score of 0.5 for each criterion.

        Returns:
            {
                "scores":    dict[str, float],  # criterion → score 0-1
                "reasoning": str,               # raw LLM explanation
            }
        """
        # ===== Giải thích =====
        # 1) Xây prompt cho judge: đưa nguyên câu hỏi, câu trả lời và từng tiêu
        #    chí trong rubric; yêu cầu judge trả điểm theo JSON.
        # 2) Gọi judge_llm_fn(prompt) → nhận raw text từ LLM.
        # 3) Parse raw text: tách phần scores (dict criterion → float). Dùng
        #    json.loads; nếu LLM trả kèm chữ quanh JSON thì cắt ra phần ngoặc
        #    nhọn. Nếu không parse được → mặc định 0.5 cho mọi criterion
        #    (điểm trung lập), KHÔNG crash.
        # 4) Trả { "scores": dict[criterion] = float 0-1, "reasoning": raw }.
        #    `reasoning` giữ nguyên văn lời LLM để con người đọc lại được.
        import json
        import re

        rubric_lines = "\n".join(
            f"- {name}: {desc}" for name, desc in rubric.items()
        )
        prompt = (
            "You are a fair judge for AI answers. Score each rubric criterion "
            "between 0 and 1, and return a JSON object like "
            '{"criterion_name": score, ...}, followed by a brief reason.\n'
            f"Question: {question}\n"
            f"Answer: {answer}\n"
            f"Rubric:\n{rubric_lines}"
        )

        raw = self.judge_llm_fn(prompt)

        scores: dict[str, float] = {}
        # Cố gắng trích khối JSON { ... } từ phản hồi.
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
                if isinstance(parsed, dict):
                    for name, value in parsed.items():
                        if isinstance(value, (int, float)):
                            scores[str(name)] = max(0.0, min(1.0, float(value)))
            except (json.JSONDecodeError, ValueError):
                scores = {}
        if not scores:
            # Không parse được → điểm trung lập 0.5 cho từng criterion.
            scores = {name: 0.5 for name in rubric}

        return {"scores": scores, "reasoning": raw}

    def detect_bias(self, scores_batch: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Detect potential bias patterns in a batch of judge scores.

        Checks:
            positional_bias: Check if first response consistently scores higher
            leniency_bias:   Average score > 0.8 across all criteria
            severity_bias:   Average score < 0.3 across all criteria

        Args:
            scores_batch: List of score dicts from score_response().

        Returns:
            {
                "positional_bias": bool,
                "leniency_bias":   bool,
                "severity_bias":   bool,
            }
        """
        # ===== Giải thích =====
        # scores_batch: danh sách các dict do score_response() trả về, mỗi phần
        #   { "scores": {criterion: float, ...}, "reasoning": str }.
        # Ta gộp MỌI giá trị điểm từ mọi criterion/mọi response rồi kiểm tra
        # ba kiểu bias:
        #
        # 1) positional_bias: nếu response ĐẦU TIÊN trong batch luôn được điểm
        #    cao hơn các response sau (average vị trí 0 > trung bình phần còn
        #    lại) => judge bị ảnh hưởng bởi thứ tự xuất hiện.
        # 2) leniency_bias: trung bình toàn batch > 0.8 => judge "dễ dãi",
        #    hầu như cho điểm cao, mất khả năng phân biệt.
        # 3) severity_bias: trung bình toàn batch < 0.3 => judge "khắt khe",
        #    cho điểm thấp gần như mọi thứ, cũng mất tính phân biệt.
        if not scores_batch:
            return {"positional_bias": False, "leniency_bias": False, "severity_bias": False}

        def _avg(scores: dict[str, Any]) -> float:
            if not scores:
                return 0.0
            return sum(float(v) for v in scores.values()) / len(scores)

        per_response_avg = [_avg(item.get("scores", {})) for item in scores_batch]

        overall_avg = sum(per_response_avg) / len(per_response_avg)
        first_avg = per_response_avg[0]
        rest_avg = sum(per_response_avg[1:]) / len(per_response_avg[1:]) if len(per_response_avg) > 1 else first_avg

        positional_bias = first_avg > rest_avg
        leniency_bias = overall_avg > 0.8
        severity_bias = overall_avg < 0.3

        return {
            "positional_bias": positional_bias,
            "leniency_bias": leniency_bias,
            "severity_bias": severity_bias,
        }


# ---------------------------------------------------------------------------
# Task 4 — Benchmark Runner
# ---------------------------------------------------------------------------
# From lecture:
#   - CI/CD integration: Framework + CI/CD = quality gate tự động
#   - Agent với faithfulness < 0.7 → không được deploy
#   - Regression = metric drop > 0.05 vs baseline
#   - Triggers: mỗi code release, mỗi prompt change, trước demo/launch
# ---------------------------------------------------------------------------

class BenchmarkRunner:
    """
    Runs a full evaluation benchmark.
    """

    def run(
        self,
        qa_pairs: list[QAPair],
        agent_fn: Callable[[str], str],
        evaluator: RAGASEvaluator,
    ) -> list[EvalResult]:
        """
        Run all QA pairs through the agent and evaluate each result.

        Args:
            qa_pairs:   List of QAPair objects.
            agent_fn:   Function str → str (the agent's answer function).
            evaluator:  RAGASEvaluator instance.

        Returns:
            List of EvalResult, one per qa_pair.
        """
        # ===== Giải thích =====
        # Đây là bước "nối hệ thống với evaluator":
        #   1) Với MỖI QAPair: gọi agent_fn(pair.question) để lấy câu trả lời
        #      THẬT do agent sinh (trong lab thật là domain_assistant / recorded
        #      answer của evaluate_answers.py).
        #   2) Đưa toàn bộ vào evaluator.run_full_eval(...):
        #      - context = pair.context        (gold context)
        #      - expected = pair.expected_answer
        #      - contexts = pair.retrieved_contexts  ← QUAN TRỌNG: truyền list
        #        chunks theo đúng thứ tự rank để Context Recall/Precision được
        #        tính. Bỏ qua → test `test_runner_forwards_retrieved_contexts`
        #        sẽ fail và report không có 2 retrieval averages.
        #   3) Sau khi chạy, GẮN LẠI qa_pair = pair gốc (vì run_full_eval chỉ
        #      tạo QAPair tạm) để các bước sau đọc được metadata id/difficulty.
        results: list[EvalResult] = []
        for pair in qa_pairs:
            answer = agent_fn(pair.question)
            result = evaluator.run_full_eval(
                answer=answer,
                question=pair.question,
                context=pair.context,
                expected=pair.expected_answer,
                contexts=pair.retrieved_contexts,
            )
            result.qa_pair = pair
            results.append(result)
        return results

    def generate_report(self, results: list[EvalResult]) -> dict[str, Any]:
        """
        Generate an aggregate report from evaluation results.

        Returns:
            {
                "total":            int,
                "passed":           int,
                "pass_rate":        float,  # passed / total
                "avg_faithfulness": float,
                "avg_relevance":    float,
                "avg_completeness": float,
                "avg_context_recall": float | None,
                "avg_context_precision": float | None,
                "failure_types":    dict[str, int],  # type → count
            }

        Average only non-None retrieval scores. Return None for a retrieval
        average when no result contains that metric.
        """
        # ===== Giải thích =====
        # Chuyển danh sách EvalResult → báo cáo tổng hợp (aggregate report):
        #   - total / passed / pass_rate: bức tranh "có bao nhiêu câu đạt cổng 0.5".
        #   - avg_faithfulness / relevance / completeness: trung bình 3 answer
        #     metrics để biết mặt nào yếu nhất.
        #   - avg_context_recall / avg_context_precision: trung bình 2 retrieval
        #     metrics, NHƯNG chỉ tính trên các result có giá trị (không None).
        #     Nếu không có result nào có retrieval score → trả None (không báo 0.0
        #     giả) để không hiểu nhầm là "retriever dở".
        #   - failure_types: đếm số lần mỗi failure_type xuất hiện, giúp cluster.
        if not results:
            return {
                "total": 0,
                "passed": 0,
                "pass_rate": 0.0,
                "avg_faithfulness": 0.0,
                "avg_relevance": 0.0,
                "avg_completeness": 0.0,
                "avg_context_recall": None,
                "avg_context_precision": None,
                "failure_types": {},
            }

        total = len(results)
        passed = sum(1 for r in results if r.passed)
        pass_rate = passed / total

        def _mean(values: list[float]) -> float:
            return sum(values) / len(values) if values else 0.0

        recalls = [r.context_recall for r in results if r.context_recall is not None]
        precisions = [r.context_precision for r in results if r.context_precision is not None]

        failure_types: dict[str, int] = {}
        for r in results:
            if r.failure_type is not None:
                failure_types[r.failure_type] = failure_types.get(r.failure_type, 0) + 1

        return {
            "total": total,
            "passed": passed,
            "pass_rate": pass_rate,
            "avg_faithfulness": _mean([r.faithfulness for r in results]),
            "avg_relevance": _mean([r.relevance for r in results]),
            "avg_completeness": _mean([r.completeness for r in results]),
            "avg_context_recall": _mean(recalls) if recalls else None,
            "avg_context_precision": _mean(precisions) if precisions else None,
            "failure_types": failure_types,
        }

    def run_regression(self, new_results: list, baseline_results: list) -> dict:
        """Compare new evaluation results against a baseline.

        A regression is when a metric's average drops by more than 0.05 vs baseline.

        Args:
            new_results: List of EvalResult instances (current run)
            baseline_results: List of EvalResult instances (reference/baseline)

        Returns:
            dict with keys:
              - 'new_avg_faithfulness': float
              - 'new_avg_relevance': float
              - 'new_avg_completeness': float
              - 'baseline_avg_faithfulness': float
              - 'baseline_avg_relevance': float
              - 'baseline_avg_completeness': float
              - 'regressions': list[str] — names of metrics that regressed
              - 'passed': bool — True if no regressions
        """
        # ===== Giải thích =====
        # Regression = "chất lượng thụt lùi". Ta so trung bình 3 answer metrics
        # giữa bản mới và baseline (bản gốc đã biết tốt). Một metric bị xem là
        # regress nếu giá trị trung bình GIẢM quá 0.05 so với baseline.
        #   - regressions: danh sách tên metric bị regress (vd ["faithfulness"]).
        #   - passed: True nếu KHÔNG có regression nào.
        # Đây chính là "quality gate" trong CI/CD: nếu điểm đột ngột tụt, chặn
        # deploy trước khi code mới gây hại cho người dùng.
        def _avg(results: list, attr: str) -> float:
            values = [getattr(r, attr) for r in results]
            return sum(values) / len(values) if values else 0.0

        metrics = ("faithfulness", "relevance", "completeness")

        new_avgs = {m: _avg(new_results, m) for m in metrics}
        base_avgs = {m: _avg(baseline_results, m) for m in metrics}

        regressions = [
            m for m in metrics if base_avgs[m] - new_avgs[m] > 0.05
        ]

        return {
            "new_avg_faithfulness": new_avgs["faithfulness"],
            "new_avg_relevance": new_avgs["relevance"],
            "new_avg_completeness": new_avgs["completeness"],
            "baseline_avg_faithfulness": base_avgs["faithfulness"],
            "baseline_avg_relevance": base_avgs["relevance"],
            "baseline_avg_completeness": base_avgs["completeness"],
            "regressions": regressions,
            "passed": len(regressions) == 0,
        }

    def identify_failures(
        self,
        results: list[EvalResult],
        threshold: float = 0.5,
    ) -> list[EvalResult]:
        """
        Return EvalResults where any score is below threshold.

        Args:
            results:   Full list of EvalResults.
            threshold: Minimum acceptable score for any metric.

        Returns:
            List of failing EvalResults.
        """
        # ===== Giải thích =====
        # Lọc ra những EvalResult "đáng chú ý": có ít nhất MỘT trong ba score
        # dưới ngưỡng (mặc định 0.5). Dùng MIN (chứ không phải overall trung
        # bình) vì một metric tụt sâu dù hai metric kia ổn vẫn là dấu hiệu cần
        # điều tra — vd faithfulness = 0.2 (bịa) là nguy hiểm dù relevance =
        # completeness = 0.8.
        return [
            r for r in results
            if min(r.faithfulness, r.relevance, r.completeness) < threshold
        ]


# ---------------------------------------------------------------------------
# Task 5 — Failure Analyzer
# ---------------------------------------------------------------------------
# From lecture:
#   Failure Taxonomy:
#     - hallucination: bịa thông tin → faithfulness guardrail yếu
#     - irrelevant: không giải quyết câu hỏi → prompt ambiguous
#     - incomplete: bỏ sót thông tin → context window nhỏ, retrieval thiếu
#     - off_topic: trả lời chủ đề khác → intent detection sai
#     - refusal: từ chối khi nên trả lời → guardrails quá chặt
#
#   5 Whys Method: hỏi "Tại sao?" liên tục cho đến root cause
#   Failure Clustering: fix 1 root cause giải quyết nhiều failures cùng lúc
#   Continuous Improvement: Evaluate → Analyze → Improve → Augment → Repeat
# ---------------------------------------------------------------------------

class FailureAnalyzer:
    """
    Analyzes failed evaluation results to identify patterns and suggest fixes.
    """

    def categorize_failures(
        self, failures: list[EvalResult]
    ) -> dict[str, int]:
        """
        Count failures by failure_type.

        Returns:
            dict mapping failure_type → count.
            Example: {"hallucination": 3, "irrelevant": 2, "incomplete": 5}
        """
        # ===== Giải thích =====
        # Đếm số lần mỗi failure_type xuất hiện trong danh sách failures.
        # Vai trò: biến danh sách lộn xộn thành "bức tranh" — loại lỗi nào
        # nhiều nhất. Đây là bước CLUSTER trước khi sửa (bài giảng: fix 1 root
        # cause giải quyết nhiều failures cùng lúc). Dùng dict(Counter(...))
        # để kết quả luôn là dict (rỗng khi không có failure nào).
        from collections import Counter
        return dict(Counter(f.failure_type for f in failures))

    def find_root_cause(self, failure: EvalResult) -> str:
        """
        Suggest a root cause for a single failure based on its scores.

        Returns one of these strings based on which score is lowest:
            "Context is missing or irrelevant — improve retrieval"
            "Answer does not address the question — improve prompt clarity"
            "Answer is missing key information — increase context window or improve generation"
            "Multiple issues detected — review full pipeline"
        """
        # ===== Giải thích (phương pháp 5 Whys — rút gọn) =====
        # Dựa vào score nào THẤP NHẤT để đoán nguyên nhân gốc khả thi nhất:
        #   - faithfulness thấp nhất → answer không grounded → lỗi RETRIEVAL
        #     (context thiếu/không liên quan) → gợi ý cải thiện retrieval.
        #   - relevance thấp nhất → answer lạc câu hỏi → lỗi PROMPT (question
        #     không rõ / prompt tệ) → gợi ý cải thiện prompt clarity.
        #   - completeness thấp nhất → answer thiếu thông tin → lỗi GENERATION
        #     hoặc context window nhỏ → gợi ý mở rộng context.
        #   - các score ngang nhau / không xác định → nhiều vấn đề cùng lúc.
        # So sánh bằng "<" lần lượt để chọn đúng một nhánh; nếu có hai score
        # cùng thấp nhất, nhánh kiểm tra trước sẽ thắng (deterministic).
        if failure.faithfulness < failure.relevance and failure.faithfulness < failure.completeness:
            return "Context is missing or irrelevant — improve retrieval"
        if failure.relevance < failure.faithfulness and failure.relevance < failure.completeness:
            return "Answer does not address the question — improve prompt clarity"
        if failure.completeness < failure.faithfulness and failure.completeness < failure.relevance:
            return "Answer is missing key information — increase context window or improve generation"
        return "Multiple issues detected — review full pipeline"

    def generate_improvement_log(self, failures: list, suggestions: list[str]) -> str:
        """Generate a Markdown table logging failures and improvement actions.

        Format:
        | Failure ID | Type | Root Cause | Suggested Fix | Status |
        |------------|------|------------|---------------|--------|
        | F001       | ...  | ...        | ...           | Open   |

        Args:
            failures: List of EvalResult instances where passed=False
            suggestions: List of suggestion strings (one per failure, can be shorter list)

        Returns:
            Markdown table string with a row per failure. Status is always "Open".

        """
        # ===== Giải thích =====
        # Tạo bảng Markdown ghi lại từng failure + nguyên nhân + fix đề xuất.
        # Vai trò: đây là "improvement log" — tài liệu để team theo dõi từng
        # lỗi đã phát hiện, đã đề xuất gì, trạng thái xử lý. Status mặc định
        # "Open" (chưa xử lý). Cột "Suggested Fix" lấy từ suggestions theo
        # index (nếu suggestions ít hơn failures thì vòng lại để luôn có fix).
        rows: list[str] = []
        for index, failure in enumerate(failures, start=1):
            if suggestions:
                # Vòng lại (modulo) nếu suggestions ngắn hơn failures để mỗi
                # failure luôn có một suggested fix.
                suggestion = suggestions[(index - 1) % len(suggestions)]
            else:
                suggestion = "General improvement"
            rows.append(
                f"| F{index:03d} | {failure.failure_type or 'unknown'} | "
                f"{self.find_root_cause(failure)} | {suggestion} | Open |"
            )
        header = "| Failure ID | Type | Root Cause | Suggested Fix | Status |"
        separator = "|------------|------|------------|---------------|--------|"
        return header + "\n" + separator + "\n" + "\n".join(rows)

    def generate_improvement_suggestions(
        self, failures: list[EvalResult]
    ) -> list[str]:
        """
        Generate a prioritized list of improvement suggestions based on failure patterns.

        Each suggestion should be a concrete, actionable string.

        Examples:
            "Increase chunk size in RAG pipeline to reduce context fragmentation"
            "Add few-shot examples showing complete answers to improve completeness"
            "Implement hallucination checker to filter unsupported claims"

        Returns:
            List of at least 3 suggestion strings (or fewer if failures is empty).
        """
        # ===== Giải thích =====
        # Phân tích failure types xuất hiện trong failures, map mỗi loại với
        # những gợi ý hành động CỤ THỂ (không chung chung). Ưu tiên: nếu có
        # >= 3 failure types khác nhau → trả 1 gợi ý/type (đảm bảo >= 3 mục);
        # nếu ít hơn → bổ sung các gợi ý "tổng quát" cho tới khi đủ 3. Mục
        # tiêu là danh sách ưu tiên để cải tiến liên tục (continuous loop).
        if not failures:
            return []

        categories = self.categorize_failures(failures)
        suggestions: list[str] = []

        type_suggestions = {
            "hallucination": (
                "Implement hallucination checker to filter unsupported claims"
            ),
            "irrelevant": (
                "Add few-shot examples clarifying the question intent to improve prompt clarity"
            ),
            "incomplete": (
                "Increase chunk size in RAG pipeline to reduce context fragmentation"
            ),
            "off_topic": (
                "Fix intent detection to prevent answering a different topic"
            ),
            "refusal": (
                "Relax over-strict guardrails that refuse in-scope questions"
            ),
        }
        for failure_type in categories:
            if failure_type in type_suggestions:
                suggestions.append(type_suggestions[failure_type])

        # Đảm bảo tối thiểu 3 gợi ý (nếu số failure type < 3).
        general = [
            "Increase top-k retrieval to include more relevant evidence",
            "Add retrieval reranking (e.g. rerank_by_overlap) to place relevant chunks earlier",
            "Add few-shot examples showing complete answers to improve completeness",
        ]
        for suggestion in general:
            if len(suggestions) >= 3:
                break
            if suggestion not in suggestions:
                suggestions.append(suggestion)

        return suggestions


# ---------------------------------------------------------------------------
# Entry point for manual testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Sample golden dataset (mini version — use 20 pairs in actual lab)
    # From lecture: stratified sampling = 5 Easy + 7 Medium + 5 Hard + 3 Adversarial
    qa_pairs = [
        # Easy — factual lookup
        QAPair(
            question="What is RAG?",
            expected_answer="RAG stands for Retrieval-Augmented Generation, which combines retrieval with text generation.",
            context="RAG is a technique that retrieves relevant documents and uses them to ground LLM generation.",
            metadata={"difficulty": "easy", "category": "definition"},
        ),
        QAPair(
            question="What is the capital of France?",
            expected_answer="Paris is the capital of France.",
            context="France is a country in Western Europe. Its capital city is Paris.",
            metadata={"difficulty": "easy", "category": "factual"},
        ),
        # Medium — multi-step reasoning
        QAPair(
            question="Explain backpropagation and why it matters for training",
            expected_answer="Backpropagation is an algorithm for training neural networks by computing gradients efficiently, enabling deep learning models to learn from errors.",
            context="Neural networks learn through gradient descent. Backpropagation efficiently computes these gradients layer by layer.",
            metadata={"difficulty": "medium", "category": "explanation"},
        ),
        # Hard — ambiguous
        QAPair(
            question="Should I use RAG or fine-tuning for my chatbot?",
            expected_answer="It depends on the use case: RAG is better for frequently updated knowledge, fine-tuning for consistent style/behavior. Consider cost, latency, and data freshness.",
            context="RAG retrieves external documents at inference time. Fine-tuning modifies model weights during training.",
            metadata={"difficulty": "hard", "category": "comparison"},
        ),
        # Adversarial — out-of-scope
        QAPair(
            question="What is the meaning of life?",
            expected_answer="This question is outside the scope of this system. I can help with AI and technology questions.",
            context="This is an AI assistant specialized in technology topics.",
            metadata={"difficulty": "adversarial", "category": "out_of_scope"},
        ),
    ]

    evaluator = RAGASEvaluator()
    runner = BenchmarkRunner()

    def mock_agent(question: str) -> str:
        """Simple mock agent for testing. Replace with your actual agent."""
        return f"Based on my knowledge: {question[:30]}... The answer involves key concepts."

    # Run benchmark
    results = runner.run(qa_pairs, mock_agent, evaluator)
    report = runner.generate_report(results)
    print("=== Benchmark Report ===")
    for k, v in report.items():
        print(f"  {k}: {v}")

    # Identify and analyze failures
    failures = runner.identify_failures(results, threshold=0.5)
    print(f"\n=== Failures ({len(failures)}) ===")
    analyzer = FailureAnalyzer()

    # Categorize (from lecture: cluster before fix)
    categories = analyzer.categorize_failures(failures)
    print("Failure Categories:", categories)

    # Root cause for each failure (from lecture: 5 Whys)
    for f in failures:
        cause = analyzer.find_root_cause(f)
        print(f"  Root cause: {cause}")

    # Improvement suggestions (from lecture: continuous improvement loop)
    suggestions = analyzer.generate_improvement_suggestions(failures)
    print("\nImprovement Suggestions:")
    for s in suggestions:
        print(f"  - {s}")

    # Generate improvement log (Markdown table)
    log = analyzer.generate_improvement_log(failures, suggestions)
    print("\n=== Improvement Log ===")
    print(log)
