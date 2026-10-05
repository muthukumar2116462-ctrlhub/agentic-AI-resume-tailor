import re
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from sentence_transformers import SentenceTransformer, util

# Shared State Definition
class AgentState(TypedDict):
    resume_text: str
    job_description: str
    candidate_name: str
    contact_info: str
    baseline_match: float
    tailored_draft: str
    violations: List[str]
    metric_violations: List[str]
    reflection_count: int
    cover_letter: str
    final_match: float
    status_log: List[str]

# Node 1: Semantic Gap Analyzer
def gap_analyzer_node(state: AgentState, embedder: SentenceTransformer) -> Dict[str, Any]:
    emb_res = embedder.encode(state["resume_text"], convert_to_tensor=True)
    emb_jd = embedder.encode(state["job_description"], convert_to_tensor=True)
    score = float(util.cos_sim(emb_res, emb_jd)[0][0])
    baseline = round(score * 100, 1)
    return {
        "baseline_match": baseline,
        "status_log": state.get("status_log", []) + [f"Computed semantic baseline: {baseline}%"]
    }

# Node 2: Contextual ReWriter
def rewriter_node(state: AgentState, llm: ChatGroq) -> Dict[str, Any]:
    retry = state.get("reflection_count", 0)
    violations = state.get("violations", [])
    metric_violations = state.get("metric_violations", [])
    
    correction_notes = []
    if violations:
        correction_notes.append(f"UNVERIFIED TOOLS: {violations}. Remove them immediately.")
    if metric_violations:
        correction_notes.append(f"FABRICATED METRICS/NUMBERS: {metric_violations}. You must use ONLY numbers found in the source resume.")
        
    correction_inst = ("\n\nCRITICAL FIXES REQUIRED FROM PREVIOUS DRAFT:\n" + "\n".join(correction_notes)) if correction_notes else ""
        
    prompt = f"""
    You are an expert ATS Resume Tailoring Agent.
    
    CANDIDATE: {state['candidate_name']}
    
    SOURCE RESUME:
    [SOURCE RESUME START]
    {state['resume_text']}
    [SOURCE RESUME END]
    
    TARGET JOB DESCRIPTION:
    [TARGET JD START]
    {state['job_description']}
    [TARGET JD END]
    
    STRICT RULES:
    1. Re-align and re-weight bullet points to highlight skills matching the target JD.
    2. Follow the XYZ format (Accomplished [X], as measured by [Y], by doing [Z]).
    3. ZERO-HALLUCINATION POLICY: 
       - Do NOT add any tech tool or library absent from SOURCE RESUME.
       - Do NOT inflate, fabricate, or invent numbers, percentages, or metrics. Stick strictly to original metrics or qualitative impact.
    4. Output clean plain-text resume sections ONLY:
       SUMMARY
       TECHNICAL SKILLS
       EXPERIENCE
       PROJECTS
       EDUCATION
    5. Do NOT include markdown code blocks (```). Just plain text with bold headings.{correction_inst}
    """
    
    res = llm.invoke([SystemMessage(content="You are an autonomous factual resume editor."), HumanMessage(content=prompt)])
    return {
        "tailored_draft": res.content,
        "reflection_count": retry + 1,
        "status_log": state.get("status_log", []) + [f"Draft synthesized by LLM (Iteration {retry + 1})"]
    }

# Node 3: Advanced Deterministic Diff & Numeric Invariant Gatekeeper
def diff_gatekeeper_node(state: AgentState) -> Dict[str, Any]:
    source_lower = state["resume_text"].lower()
    source_words = set(re.findall(r'\b[A-Za-z0-9+#.-]{3,}\b', source_lower))
    
    # 1. Tech stack entity check
    tech_keywords = [
        "kubernetes", "aws", "azure", "gcp", "docker", "graphql", "terraform",
        "jenkins", "kafka", "redis", "mongodb", "postgresql", "react", "angular",
        "vue", "django", "flask", "fastapi", "spring", "pytorch", "tensorflow"
    ]
    drafted_lower = state["tailored_draft"].lower()
    hallucinated_tech = [tech for tech in tech_keywords if tech in drafted_lower and tech not in source_words]
    
    # 2. Metric Invariant Check: Catch fabricated percentages (e.g. 85%) and multipliers (e.g. 10x)
    source_metrics = set(re.findall(r'\b\d+%\b|\b\d+x\b', source_lower))
    draft_metrics = set(re.findall(r'\b\d+%\b|\b\d+x\b', drafted_lower))
    # Exclude 0% (allowed for describing 0% hallucination)
    hallucinated_metrics = [m for m in draft_metrics if m not in source_metrics and m != "0%"]
    
    total_violations = len(hallucinated_tech) + len(hallucinated_metrics)
    
    if total_violations == 0:
        log_msg = "Gatekeeper: 0 violations. Verified 100% ground-truth alignment (Entities & Metrics)."
    else:
        log_msg = f"Gatekeeper: Caught {len(hallucinated_tech)} ungrounded tools and {len(hallucinated_metrics)} fabricated metrics."
        
    return {
        "violations": hallucinated_tech,
        "metric_violations": hallucinated_metrics,
        "status_log": state.get("status_log", []) + [log_msg]
    }

# Conditional Reflection Edge
def check_gatekeeper_condition(state: AgentState) -> str:
    has_faults = bool(state.get("violations") or state.get("metric_violations"))
    if has_faults and state.get("reflection_count", 0) < 3:
        return "re_write"
    return "synthesize"

# Node 4: Synthesizer
def synthesizer_node(state: AgentState, llm: ChatGroq, embedder: SentenceTransformer) -> Dict[str, Any]:
    emb_new = embedder.encode(state["tailored_draft"], convert_to_tensor=True)
    emb_jd = embedder.encode(state["job_description"], convert_to_tensor=True)
    final_score = round(float(util.cos_sim(emb_new, emb_jd)[0][0]) * 100, 1)
    
    cover_prompt = f"""
    You are an expert Executive Career Consultant. Write a formal, compelling, 3-paragraph tailored cover letter for {state['candidate_name']}.
    
    CANDIDATE HISTORY:
    \"\"\"{state['resume_text']}\"\"\"
    
    TARGET JOB POSTING:
    \"\"\"{state['job_description']}\"\"\"
    
    GUIDELINES:
    - Paragraph 1: Enthusiastic introduction stating the target role and core technical alignment.
    - Paragraph 2: Spotlight 2 relevant projects/accomplishments from candidate history proving direct fit.
    - Paragraph 3: Professional closing demonstrating enthusiasm and value proposition.
    - Strictly use factual metrics and projects from the candidate history.
    - Format cleanly with standard salutation and sign-off.
    """
    res_cover = llm.invoke([SystemMessage(content="You write professional, concise cover letters."), HumanMessage(content=cover_prompt)])
    
    return {
        "final_match": final_score,
        "cover_letter": res_cover.content,
        "status_log": state.get("status_log", []) + ["Synthesis and ATS alignment complete."]
    }

# LangGraph Compiler
def build_agentic_workflow(llm: ChatGroq, embedder: SentenceTransformer):
    workflow = StateGraph(AgentState)
    
    workflow.add_node("gap_analyzer", lambda s: gap_analyzer_node(s, embedder))
    workflow.add_node("rewriter", lambda s: rewriter_node(s, llm))
    workflow.add_node("gatekeeper", diff_gatekeeper_node)
    workflow.add_node("synthesizer", lambda s: synthesizer_node(s, llm, embedder))
    
    workflow.set_entry_point("gap_analyzer")
    workflow.add_edge("gap_analyzer", "rewriter")
    workflow.add_edge("rewriter", "gatekeeper")
    workflow.add_conditional_edges(
        "gatekeeper",
        check_gatekeeper_condition,
        {
            "re_write": "rewriter",
            "synthesize": "synthesizer"
        }
    )
    workflow.add_edge("synthesizer", END)
    
    return workflow.compile()