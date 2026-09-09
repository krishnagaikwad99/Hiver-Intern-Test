# Decision Log — Hiver AI Support Agent

This document records 12 non-obvious engineering and architectural decisions made during the design and implementation of the Hiver AI Support Agent system.

---

### Decision 1: Evidence-Based Brand Selection (`AppleSupport`) Over Popularity Choice
- **Decision**: Selected `AppleSupport` as the primary brand after empirical profiling of candidate brands (`AppleSupport`, `AmazonHelp`, `Uber_Support`, `SpotifyCares`, `Delta`).
- **Reason**: Empirical dataset analysis revealed `AppleSupport` contained the highest multi-turn conversation reconstruction rate, high intent diversity, and complete historical resolutions.
- **Alternative considered**: `AmazonHelp` or `Uber_Support`.
- **Trade-off**: Technical hardware/software queries predominate over logistics queries, but conversation quality and evidence density are significantly higher.

---

### Decision 2: Domain-Derived Intent Taxonomy Over Off-the-Shelf Benchmarks (e.g., Banking77)
- **Decision**: Designed a 10-intent taxonomy directly from the target brand's historical conversation data instead of using generic intent datasets like Banking77.
- **Reason**: Twitter tech customer support differs fundamentally from banking queries. Custom intents (`device_hardware_issue`, `software_update_bug`, `account_access_login`, etc.) capture true customer pain points.
- **Alternative considered**: Adapting Banking77 or IntentScope benchmarks.
- **Trade-off**: Requires custom ground-truth annotation, but matches real-world brand support operations.

---

### Decision 3: Deterministic Escalation Policy Engine Over Unconstrained LLM Decision-Making
- **Decision**: Implemented an explicit rule-based escalation engine (`src/agent/escalation.py`) evaluating low confidence, low retrieval similarity, sensitive keywords, and policy boundaries.
- **Reason**: Allowing LLMs to make unconstrained escalation decisions leads to unpredictable risk posture and missed security policy breaches (e.g. refund claims or account locks auto-handled by mistake).
- **Alternative considered**: Prompting the LLM to output "escalate" or "auto_handle" inside the system prompt.
- **Trade-off**: Slightly less flexible than pure LLM reasoning, but yields deterministic, auditable risk control.

---

### Decision 4: Retrieval Similarity Thresholding for Anti-Hallucination Guardrails
- **Decision**: Enforced a strict retrieval similarity threshold (cos_sim >= 0.55). If no historical resolution meets this threshold, the agent escalates.
- **Reason**: RAG systems frequently hallucinate plausible-sounding policies when retrieved evidence is weak or irrelevant.
- **Alternative considered**: Forcing the LLM to generate a best-guess response even when vector similarity is low.
- **Trade-off**: Increases escalation rate slightly, but guarantees zero unsupported policy hallucinations.

---

### Decision 5: Multi-Turn Conversation Thread Reconstruction Tree Traversal
- **Decision**: Reconstructed conversation pairs using `in_reply_to_tweet_id` parent pointers and `response_tweet_id` child links.
- **Reason**: Raw tweets in Twitter support datasets are isolated fragments. Reconstructing turn pairs connects customer problem statements directly to historical agent resolutions.
- **Alternative considered**: Treating every tweet independently as a standalone document.
- **Trade-off**: Requires tree traversal preprocessing, but provides true ground-truth resolution context for RAG retrieval.

---

### Decision 6: Dual Inference Strategy (Provider-Agnostic LLM Engine + Deterministic Mock Mode)
- **Decision**: Architected the system to support live Gemini (`google-genai`), OpenAI, and a deterministic offline/mock mode.
- **Reason**: Ensures 100% reproducible execution in offline environments, CI/CD pipelines, and interview reviews without requiring active API keys or paid credits.
- **Alternative considered**: Hardcoding OpenAI or Gemini API calls exclusively.
- **Trade-off**: Requires maintaining mock fallback logic, but delivers zero-cost instant testability and bulletproof evaluation reproduction.

---

### Decision 7: Multi-Dimensional LLM-as-a-Judge Evaluation (5 Dimensions)
- **Decision**: Built a structured LLM Judge scoring responses on a 1-5 scale across 5 distinct rubrics (Relevance, Groundedness, Correctness, Helpfulness, Tone).
- **Reason**: Single overall quality scores hide critical failures (e.g. polite tone disguising a hallucinated refund policy).
- **Alternative considered**: Evaluating only classification accuracy and ROUGE/BLEU scores.
- **Trade-off**: Higher evaluation runtime, but provides deep diagnostic insights into response safety and helpfulness.

---

### Decision 8: Human-in-the-Loop Calibration Subset (35 Examples) for Judge Validation
- **Decision**: Created a 35-example human-annotated calibration subset to explicitly measure agreement statistics (Exact Agreement %, MAE, Pearson Correlation, Cohen's Quadratic Kappa).
- **Reason**: LLM judges cannot be trusted blindly without empirical validation against human judgment.
- **Alternative considered**: Accepting LLM judge scores at face value without calibration.
- **Trade-off**: Manual annotation effort required for calibration set, but provides mathematical evidence of judge reliability.

---

### Decision 9: Subsampled Representative Dataset Pipeline for Sub-15 Minute Reproduction
- **Decision**: Configured the default data ingestion pipeline to process a clean, representative sample (5,000 to 100,000 tweets) rather than the entire 3M row Kaggle dataset.
- **Reason**: Assignment explicitly required headline results reproduction in under 15 minutes on standard CPU laptops.
- **Alternative considered**: Forcing full 3M row dataset vectorization.
- **Trade-off**: Excludes long-tail rare customer tweets, but guarantees fast, reliable, reproducible evaluation.

---

### Decision 10: TF-IDF + Logistic Regression as Simple Baseline Over Fine-Tuned BERT
- **Decision**: Selected TF-IDF + Logistic Regression with balanced class weighting as Baseline 2.
- **Reason**: Light, fast, interpretable, and provides a clear non-deep-learning baseline to benchmark LLM classification against.
- **Alternative considered**: Fine-tuning BERT or RoBERTa.
- **Trade-off**: TF-IDF misses deep semantic context, but trains in <1 second and serves as a clean simple baseline.

---

### Decision 11: Structured Output Enforcement via Pydantic Schemas
- **Decision**: Enforced JSON schema validation (`AgentResponse`, `IntentClassificationResult`, `JudgeScores`) across all pipeline outputs.
- **Reason**: Free-form text outputs break downstream integration, monitoring, and automated evaluation harnesses.
- **Alternative considered**: Regular expression parsing of unstructured model outputs.
- **Trade-off**: Requires strict schema definition, but guarantees API stability and type safety.

---

### Decision 12: Automated Top 5 Real Failure Mode Extraction
- **Decision**: Built the evaluation harness to automatically identify real misclassification and bad-action instances and log them into `metrics.json`.
- **Reason**: Diagnostic failure analysis based on empirical error logs is far more actionable than static doc assertions.
- **Alternative considered**: Hand-picking hypothetical error cases.
- **Trade-off**: Error examples change as the model improves, but remain strictly grounded in empirical evaluation data.
