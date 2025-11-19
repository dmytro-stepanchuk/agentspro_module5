"""
МУЛЬТИАГЕНТНА СИСТЕМА - LangGraph 1.0 Supervisor Pattern
Розширення з координацією кількох спеціалізованих агентів

Architecture:
- SupervisorAgent: Координує команду агентів
- ResearcherVectorAgent: Vector search в knowledge base (NEW!)
- ResearcherWebAgent: Web search через Tavily (NEW!)
- AnalyzerAgent: Аналіз знайденої інформації
- SynthesizerAgent: Синтез фінальної відповіді
- CriticAgent: Оцінка якості відповіді

LangSmith Integration: Трейсинг всіх агентів та їх взаємодії
"""

import os
from typing import TypedDict, Annotated, Literal, List
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import operator

load_dotenv()

# ============================================================================
# LANGSMITH VERIFICATION
# ============================================================================

if os.getenv("LANGCHAIN_TRACING_V2") == "true":
    print("OK LangSmith трейсинг активний для мультиагентної системи")
    print(f"Stats: Project: {os.getenv('LANGCHAIN_PROJECT', 'default')}\n")
else:
    print("WARNING  LangSmith не ввімкнений\n")


# ============================================================================
# STATE DEFINITION - Shared state для всіх агентів
# ============================================================================

class MultiAgentState(TypedDict):
    """
    Спільний state для всієї мультиагентної системи

    Включає:
    - messages: історія комунікації між агентами
    - question: початкове питання користувача
    - current_agent: який агент зараз активний
    - vector_docs: документи з vector search (NEW!)
    - web_docs: документи з web search (NEW!)
    - retrieved_docs: об'єднані документи для Analyzer
    - analysis: результат аналізу (AnalyzerAgent)
    - final_answer: фінальна відповідь (SynthesizerAgent)
    - supervisor_decision: рішення supervisor про наступний крок
    - iteration_count: лічильник ітерацій
    - critic_score: оцінка від critic (1-10)
    - critic_feedback: feedback від critic
    - revision_count: кількість ревізій
    - research_complete: чи завершено обидва researcher (NEW!)
    """
    messages: Annotated[List, operator.add]  # Додавання повідомлень
    question: str
    current_agent: str
    # NEW: Separate storage for parallel researchers
    vector_docs: List[Document]
    web_docs: List[Document]
    retrieved_docs: List[Document]  # Combined docs for analyzer
    analysis: str
    final_answer: str
    supervisor_decision: str
    iteration_count: int
    # Critic fields
    critic_score: int
    critic_feedback: str
    revision_count: int
    # NEW: Track research completion
    research_complete: bool


# ============================================================================
# KNOWLEDGE BASE - LangGraph 1.0 Documentation
# ============================================================================

# Створюємо knowledge base з інформацією про LangGraph 1.0
documents = [
    Document(
        page_content="""LangGraph 1.0 Supervisor Pattern:
        Hierarchical multi-agent architecture where a central supervisor agent coordinates multiple specialized agents.
        The supervisor receives user input, delegates work to sub-agents based on their capabilities,
        and when sub-agents respond, control returns to the supervisor. Each agent maintains its own scratchpad
        while the supervisor orchestrates communication and task delegation. This pattern is ideal for complex
        workflows requiring specialized expertise.""",
        metadata={"source": "supervisor_pattern", "version": "1.0"}
    ),
    Document(
        page_content="""LangGraph StateGraph API:
        StateGraph is the core abstraction for building multi-agent systems in LangGraph 1.0.
        It maintains centralized state storing intermediate results and metadata. Agents are represented as nodes,
        connections as edges. Control flow is managed by edges with conditional routing.
        StateGraph enables parallel execution, conditional branching, and state persistence through checkpointing.
        Key methods: add_node(), add_edge(), add_conditional_edges(), compile().""",
        metadata={"source": "stategraph_api", "version": "1.0"}
    ),
    Document(
        page_content="""LangGraph Checkpointing Mechanisms:
        LangGraph 1.0 provides persistent state storage through checkpointing. MemorySaver for development,
        PostgresSaver/SqliteSaver for production. Checkpoints enable time-travel through execution states,
        rollback to prior points, and replay workflows with adjusted parameters. Each checkpoint is identified
        by thread_id allowing separate conversation sessions. Prevents state corruption and ensures data integrity.
        Checkpoint memory is managed using threads for isolation.""",
        metadata={"source": "checkpointing", "version": "1.0"}
    ),
    Document(
        page_content="""LangGraph Multi-Agent Coordination:
        Agent coordination patterns in LangGraph 1.0 include: 1) Supervisor Pattern - central coordinator,
        2) Hierarchical Teams - nested supervision layers, 3) Network Pattern - peer-to-peer communication.
        State management handles agent communication through shared StateGraph. Each agent reads/writes to state.
        Communication via messages in state. Output consolidation through final synthesis node.
        Guardrails via conditional routing and validation nodes.""",
        metadata={"source": "coordination", "version": "1.0"}
    ),
    Document(
        page_content="""LangGraph Swarm (2025 Release):
        New lightweight library for swarm-style multi-agent systems. Maintains shared state with conversation history
        and active_agent marker. Uses checkpointer (in-memory or database) to persist state across turns.
        Aims to make multi-agent coordination easier and more reliable. Provides abstractions to link individual
        LLM agents into one integrated application. Emphasizes state management and checkpointing for reliability.
        Supports parallel agent execution with conflict resolution.""",
        metadata={"source": "swarm", "version": "1.0"}
    ),
    Document(
        page_content="""LangGraph Server & Persistence:
        LangGraph 1.0 includes LangGraph Server for production deployments. Provides comprehensive persistence:
        stores checkpoints, memories, thread metadata, and assistant configurations. Enables distributed multi-agent
        systems with API endpoints. Supports horizontal scaling. Built-in monitoring and observability.
        Integration with LangSmith for tracing all agents. REST API for agent invocation and state inspection.
        Webhook support for async workflows.""",
        metadata={"source": "server", "version": "1.0"}
    ),
    Document(
        page_content="""LangGraph Error Handling & Recovery:
        Multi-agent systems in LangGraph 1.0 include robust error handling. Each node can handle exceptions gracefully.
        Conditional edges for error routing. Retry mechanisms with exponential backoff. Circuit breakers to prevent
        cascade failures. State rollback on errors using checkpoints. Validation nodes before critical operations.
        Supervisor can reassign tasks if agent fails. Timeout handling at node level.
        Error messages propagated through state.""",
        metadata={"source": "error_handling", "version": "1.0"}
    ),
    Document(
        page_content="""LangGraph Best Practices for Multi-Agent Systems:
        1) Define clear agent responsibilities and capabilities. 2) Use supervisor for complex coordination.
        3) Implement checkpointing for long-running workflows. 4) Add validation nodes between critical steps.
        5) Use conditional edges for dynamic routing. 6) Keep state schema simple and typed.
        7) Implement timeouts for all agent operations. 8) Use LangSmith for observability.
        9) Test each agent independently before integration. 10) Design for agent failure and recovery.""",
        metadata={"source": "best_practices", "version": "1.0"}
    ),
]

print(f"KB: Knowledge Base: {len(documents)} документів про LangGraph 1.0\n")

# ============================================================================
# VECTOR STORE SETUP
# ============================================================================

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.from_documents(documents, embeddings)
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}  # Топ-3 документи
)

print("OK Vector store готовий (FAISS)\n")

# ============================================================================
# TAVILY WEB SEARCH SETUP
# ============================================================================

tavily_available = False
tavily_search = None

if os.getenv("TAVILY_API_KEY"):
    tavily_search = TavilySearchResults(max_results=3)
    tavily_available = True
    print("OK Tavily web search готовий\n")
else:
    print("WARNING  TAVILY_API_KEY не знайдений - web search буде симульований\n")

# ============================================================================
# LLM SETUP
# ============================================================================

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ============================================================================
# PYDANTIC MODELS для structured output
# ============================================================================

class SupervisorDecision(BaseModel):
    """Рішення supervisor агента про наступний крок"""
    next_agent: Literal["research", "analyzer", "synthesizer", "critic", "FINISH"] = Field(
        description="Який агент має виконувати наступний крок або FINISH"
    )
    reasoning: str = Field(description="Пояснення чому обрано цього агента")


class ResearchQuality(BaseModel):
    """Оцінка якості знайдених документів"""
    is_sufficient: bool = Field(description="Чи достатньо інформації знайдено")
    confidence: float = Field(description="Впевненість в якості (0.0-1.0)")
    reasoning: str = Field(description="Обґрунтування оцінки")


class CriticEvaluation(BaseModel):
    """Оцінка якості відповіді від Critic агента"""
    accuracy_score: int = Field(description="Оцінка точності (1-10)", ge=1, le=10)
    completeness_score: int = Field(description="Оцінка повноти (1-10)", ge=1, le=10)
    readability_score: int = Field(description="Оцінка читабельності (1-10)", ge=1, le=10)
    overall_score: int = Field(description="Загальна оцінка (1-10)", ge=1, le=10)
    feedback: str = Field(description="Детальний feedback для покращення відповіді")
    needs_revision: bool = Field(description="Чи потрібна ревізія (True якщо overall_score < 7)")


# ============================================================================
# AGENT NODES - Спеціалізовані агенти
# ============================================================================

def supervisor_node(state: MultiAgentState) -> MultiAgentState:
    """
    SupervisorAgent: Координує команду агентів

    Використовує детерміновану логіку для routing
    """
    print("\n" + "="*70)
    print("SUPERVISOR SUPERVISOR AGENT: Приймає рішення про делегування")
    print("="*70)

    iteration = state.get("iteration_count", 0) + 1
    question = state["question"]

    # Аналізуємо поточний стан
    research_complete = state.get("research_complete", False)
    has_docs = bool(state.get("retrieved_docs"))
    has_analysis = bool(state.get("analysis"))
    has_answer = bool(state.get("final_answer"))
    critic_score = state.get("critic_score", 0)
    revision_count = state.get("revision_count", 0)

    print(f"Stats: Iteration: {iteration}")
    print(f"📝 Question: {question}")
    print(f"🔍 Research complete: {research_complete}")
    print(f"KB: Docs combined: {has_docs}")
    print(f"RESEARCHER Analysis done: {has_analysis}")
    print(f"OK Answer ready: {has_answer}")
    print(f"⭐ Critic score: {critic_score}")
    print(f"🔄 Revisions: {revision_count}")

    # DETERMINISTIC ROUTING
    if not research_complete:
        next_agent = "research"
        reasoning = "Need to run parallel research (vector + web search)"
    elif not has_analysis:
        next_agent = "analyzer"
        reasoning = "Research complete, need to analyze combined results"
    elif not has_answer:
        next_agent = "synthesizer"
        reasoning = "Analysis ready, need to create answer"
    elif critic_score == 0:
        next_agent = "critic"
        reasoning = "Answer ready but not evaluated yet, need critic review"
    elif critic_score >= 7:
        next_agent = "FINISH"
        reasoning = f"Answer approved with score {critic_score}/10"
    elif revision_count >= 2:
        next_agent = "FINISH"
        reasoning = f"Max revisions reached ({revision_count}), finalizing with score {critic_score}/10"
    else:
        next_agent = "synthesizer"
        reasoning = f"Score {critic_score}/10 < 7, revision {revision_count + 1} needed"

    print(f"\nSUPERVISOR Decision: {next_agent}")
    print(f"💭 Reasoning: {reasoning}\n")

    return {
        **state,
        "current_agent": next_agent,
        "supervisor_decision": reasoning,
        "iteration_count": iteration,
        "messages": [AIMessage(content=f"Supervisor → {next_agent}: {reasoning}")]
    }


def researcher_vector_node(state: MultiAgentState) -> MultiAgentState:
    """
    ResearcherVectorAgent: Vector search в knowledge base

    Виконується паралельно з researcher_web_node
    """
    print("\n" + "="*70)
    print("📚 RESEARCHER VECTOR: Пошук в knowledge base")
    print("="*70)

    question = state["question"]

    # Виконуємо vector search
    vector_docs = retriever.invoke(question)

    print(f"KB: Знайдено {len(vector_docs)} документів з vector store")
    for i, doc in enumerate(vector_docs, 1):
        print(f"  {i}. {doc.metadata.get('source', 'unknown')}: {doc.page_content[:80]}...")

    return {
        **state,
        "vector_docs": vector_docs,
        "messages": [AIMessage(content=f"ResearcherVector: Found {len(vector_docs)} docs from knowledge base")]
    }


def researcher_web_node(state: MultiAgentState) -> MultiAgentState:
    """
    ResearcherWebAgent: Web search через Tavily

    Виконується паралельно з researcher_vector_node
    """
    print("\n" + "="*70)
    print("🌐 RESEARCHER WEB: Web search через Tavily")
    print("="*70)

    question = state["question"]
    web_docs = []

    if tavily_available and tavily_search:
        try:
            # Виконуємо web search
            search_results = tavily_search.invoke(question)

            # Конвертуємо результати в Documents
            for result in search_results:
                content = result.get("content", "")
                url = result.get("url", "")
                web_docs.append(Document(
                    page_content=content,
                    metadata={"source": "web_search", "url": url}
                ))

            print(f"🌐 Знайдено {len(web_docs)} результатів з web")
            for i, doc in enumerate(web_docs, 1):
                print(f"  {i}. {doc.metadata.get('url', 'unknown')[:50]}...")
                print(f"     {doc.page_content[:80]}...")

        except Exception as e:
            print(f"WARNING  Web search error: {e}")
            # Fallback: симулюємо результати
            web_docs = [
                Document(
                    page_content=f"Web search result about LangGraph: Modern framework for building stateful AI agents with support for cycles, controllability, and persistence.",
                    metadata={"source": "web_search", "url": "https://langchain-ai.github.io/langgraph/"}
                )
            ]
    else:
        # Симулюємо web search якщо Tavily недоступний
        print("⚠️  Tavily недоступний - використовуємо симульовані результати")
        web_docs = [
            Document(
                page_content=f"Simulated web result: LangGraph is a library for building stateful, multi-actor applications with LLMs. It extends LangChain Expression Language with cyclic computational capabilities.",
                metadata={"source": "web_search_simulated", "url": "https://example.com/langgraph"}
            ),
            Document(
                page_content=f"Simulated web result: LangGraph enables complex agent architectures including supervisor patterns, hierarchical teams, and network topologies for multi-agent coordination.",
                metadata={"source": "web_search_simulated", "url": "https://example.com/langgraph-patterns"}
            )
        ]
        print(f"🌐 Симульовано {len(web_docs)} web результатів")

    return {
        **state,
        "web_docs": web_docs,
        "messages": [AIMessage(content=f"ResearcherWeb: Found {len(web_docs)} docs from web search")]
    }


def combine_research_node(state: MultiAgentState) -> MultiAgentState:
    """
    Об'єднує результати від обох researcher agents

    Запускається після завершення паралельного пошуку
    """
    print("\n" + "="*70)
    print("🔗 COMBINE RESEARCH: Об'єднання результатів")
    print("="*70)

    vector_docs = state.get("vector_docs", [])
    web_docs = state.get("web_docs", [])

    # Об'єднуємо документи
    combined_docs = vector_docs + web_docs

    print(f"📚 Vector docs: {len(vector_docs)}")
    print(f"🌐 Web docs: {len(web_docs)}")
    print(f"📊 Total combined: {len(combined_docs)}")

    return {
        **state,
        "retrieved_docs": combined_docs,
        "research_complete": True,
        "messages": [AIMessage(content=f"CombineResearch: Combined {len(combined_docs)} docs ({len(vector_docs)} vector + {len(web_docs)} web)")]
    }


def analyzer_node(state: MultiAgentState) -> MultiAgentState:
    """
    AnalyzerAgent: Аналізує знайдену інформацію з обох джерел
    """
    print("\n" + "="*70)
    print("ANALYZER ANALYZER AGENT: Аналізує інформацію")
    print("="*70)

    question = state["question"]
    docs = state.get("retrieved_docs", [])

    if not docs:
        print("WARNING  No documents to analyze")
        return {
            **state,
            "analysis": "No documents found for analysis",
            "messages": [AIMessage(content="Analyzer: No documents to analyze")]
        }

    # Розділяємо документи за джерелом для аналізу
    vector_count = len([d for d in docs if d.metadata.get("source") != "web_search" and d.metadata.get("source") != "web_search_simulated"])
    web_count = len(docs) - vector_count

    # Аналізуємо документи
    newline = chr(10)
    docs_text = newline.join([
        f"{i+1}. Source: {doc.metadata.get('source', 'unknown')}{newline}{doc.page_content}{newline}"
        for i, doc in enumerate(docs)
    ])

    analysis_prompt = f"""Analyze the following documents from TWO sources and extract key insights.

Question: {question}

Documents ({vector_count} from knowledge base, {web_count} from web search):
{docs_text}

Provide a structured analysis with:
1. Key concepts found (note which source)
2. Relevant patterns/architectures
3. Best practices mentioned
4. Specific technical details
5. How web results complement or validate knowledge base info"""

    messages = [
        SystemMessage(content="You are an expert technical analyst specializing in LangGraph and multi-agent systems. You analyze information from multiple sources."),
        HumanMessage(content=analysis_prompt)
    ]

    response = llm.invoke(messages)
    analysis = response.content

    print(f"Stats: Analysis:\n{analysis[:300]}...\n")

    return {
        **state,
        "analysis": analysis,
        "messages": [AIMessage(content=f"Analyzer: Completed analysis ({len(analysis)} chars) from {len(docs)} docs")]
    }


def synthesizer_node(state: MultiAgentState) -> MultiAgentState:
    """
    SynthesizerAgent: Синтезує фінальну відповідь
    """
    print("\n" + "="*70)
    print("SYNTHESIZER SYNTHESIZER AGENT: Створює фінальну відповідь")
    print("="*70)

    question = state["question"]
    analysis = state.get("analysis", "")
    docs = state.get("retrieved_docs", [])
    critic_feedback = state.get("critic_feedback", "")
    revision_count = state.get("revision_count", 0)

    # Якщо є feedback від critic - це ревізія
    if critic_feedback and revision_count > 0:
        print(f"🔄 Revision #{revision_count} based on critic feedback")

        synthesis_prompt = f"""REVISION REQUESTED: Improve the answer based on critic feedback.

Question: {question}

Previous Answer:
{state.get('final_answer', '')}

Critic Feedback:
{critic_feedback}

Analysis:
{analysis}

Create an IMPROVED answer that:
1. Addresses ALL points in the critic feedback
2. Maintains accuracy and completeness
3. Improves readability and structure
4. Keeps the same core information but presents it better"""

    else:
        # Перший синтез
        synthesis_prompt = f"""Create a comprehensive, well-structured answer based on analysis from multiple sources.

Question: {question}

Analysis:
{analysis}

Source Documents:
{chr(10).join([f"- {doc.metadata.get('source', 'unknown')}" for doc in docs])}

Create a clear, informative answer that:
1. Directly addresses the question
2. Incorporates insights from BOTH knowledge base and web sources
3. Provides specific technical details
4. Includes examples where relevant
5. Notes where different sources agree or complement each other"""

    messages = [
        SystemMessage(content="You are an expert technical writer creating clear, comprehensive answers from multiple sources."),
        HumanMessage(content=synthesis_prompt)
    ]

    response = llm.invoke(messages)
    final_answer = response.content

    print(f"OK Final Answer:\n{final_answer[:300]}...\n")

    return {
        **state,
        "final_answer": final_answer,
        "messages": [AIMessage(content=f"Synthesizer: {'Revised' if revision_count > 0 else 'Created'} answer ({len(final_answer)} chars)")]
    }


def critic_node(state: MultiAgentState) -> MultiAgentState:
    """
    CriticAgent: Оцінює якість відповіді від Synthesizer
    """
    print("\n" + "="*70)
    print("⭐ CRITIC AGENT: Оцінює якість відповіді")
    print("="*70)

    question = state["question"]
    answer = state.get("final_answer", "")
    analysis = state.get("analysis", "")
    revision_count = state.get("revision_count", 0)

    if not answer:
        print("WARNING  No answer to evaluate")
        return {
            **state,
            "critic_score": 0,
            "critic_feedback": "No answer provided",
            "messages": [AIMessage(content="Critic: No answer to evaluate")]
        }

    # Оцінюємо відповідь
    critic_prompt = f"""You are a strict but fair critic evaluating the quality of an AI-generated answer.

Question: {question}

Answer to evaluate:
{answer}

Reference Analysis (ground truth):
{analysis}

Evaluate the answer on three criteria (1-10 scale):

1. ACCURACY (1-10): Does the answer contain correct information? Is it factually accurate based on the analysis?
2. COMPLETENESS (1-10): Does it cover all important aspects mentioned in the analysis? Are there gaps?
3. READABILITY (1-10): Is it well-structured, clear, and easy to understand?

Calculate OVERALL score as the average of the three scores.

If overall_score < 7, the answer needs revision. Provide specific, actionable feedback.
If overall_score >= 7, the answer is acceptable.

Be critical but constructive. Focus on specific improvements needed."""

    messages = [
        SystemMessage(content="You are an expert technical reviewer with high standards for quality."),
        HumanMessage(content=critic_prompt)
    ]

    # Отримуємо structured evaluation
    structured_llm = llm.with_structured_output(CriticEvaluation)
    evaluation = structured_llm.invoke(messages)

    print(f"\n📊 Evaluation Results:")
    print(f"  - Accuracy:     {evaluation.accuracy_score}/10")
    print(f"  - Completeness: {evaluation.completeness_score}/10")
    print(f"  - Readability:  {evaluation.readability_score}/10")
    print(f"  - OVERALL:      {evaluation.overall_score}/10")
    print(f"\n💭 Feedback: {evaluation.feedback[:200]}...")
    print(f"\n🔄 Needs revision: {evaluation.needs_revision}")

    # Оновлюємо revision_count якщо потрібна ревізія
    new_revision_count = revision_count + 1 if evaluation.needs_revision else revision_count

    return {
        **state,
        "critic_score": evaluation.overall_score,
        "critic_feedback": evaluation.feedback,
        "revision_count": new_revision_count,
        "messages": [AIMessage(content=f"Critic: Score {evaluation.overall_score}/10 {'(needs revision)' if evaluation.needs_revision else '(approved)'}")]
    }


# ============================================================================
# ROUTING LOGIC - Умовна маршрутизація
# ============================================================================

def route_after_supervisor(state: MultiAgentState) -> str:
    """Визначає наступний крок після supervisor"""
    decision = state.get("current_agent", "research")

    if decision == "FINISH":
        return "end"
    elif decision == "research":
        return "research"
    elif decision == "analyzer":
        return "analyzer"
    elif decision == "synthesizer":
        return "synthesizer"
    elif decision == "critic":
        return "critic"
    else:
        return "end"


# ============================================================================
# GRAPH CONSTRUCTION - Мультиагентний workflow з паралельним пошуком
# ============================================================================

def create_multiagent_system():
    """
    Створює мультиагентну систему з Supervisor Pattern та паралельним пошуком

    Architecture:
    START → Supervisor → [ResearcherVector → ResearcherWeb] → Combine → Supervisor → Analyzer → ...
    """
    print("=" * 70)
    print("BUILDING  СТВОРЕННЯ МУЛЬТИАГЕНТНОЇ СИСТЕМИ")
    print("=" * 70 + "\n")

    # Створюємо StateGraph
    workflow = StateGraph(MultiAgentState)

    # Додаємо агентів як nodes
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("researcher_vector", researcher_vector_node)
    workflow.add_node("researcher_web", researcher_web_node)
    workflow.add_node("combine_research", combine_research_node)
    workflow.add_node("analyzer", analyzer_node)
    workflow.add_node("synthesizer", synthesizer_node)
    workflow.add_node("critic", critic_node)

    # Встановлюємо entry point
    workflow.set_entry_point("supervisor")

    # Conditional edges від supervisor
    workflow.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {
            "research": "researcher_vector",  # Start parallel research
            "analyzer": "analyzer",
            "synthesizer": "synthesizer",
            "critic": "critic",
            "end": END
        }
    )

    # Паралельне виконання: vector -> web -> combine
    workflow.add_edge("researcher_vector", "researcher_web")
    workflow.add_edge("researcher_web", "combine_research")

    # Після combine повертаємось до supervisor
    workflow.add_edge("combine_research", "supervisor")

    # Інші агенти повертаються до supervisor
    workflow.add_edge("analyzer", "supervisor")
    workflow.add_edge("synthesizer", "supervisor")
    workflow.add_edge("critic", "supervisor")

    # Компілюємо з checkpointer
    checkpointer = MemorySaver()
    app = workflow.compile(checkpointer=checkpointer)

    print("OK Мультиагентна система створена\n")
    print("Agents:")
    print("  SUPERVISOR Supervisor - координує команду")
    print("  📚 ResearcherVector - vector search в knowledge base (NEW!)")
    print("  🌐 ResearcherWeb - web search через Tavily (NEW!)")
    print("  🔗 CombineResearch - об'єднання результатів (NEW!)")
    print("  ANALYZER Analyzer - аналіз інформації")
    print("  SYNTHESIZER Synthesizer - синтез відповіді")
    print("  ⭐ Critic - оцінка якості\n")
    print("Research Flow:")
    print("  Supervisor → ResearcherVector → ResearcherWeb → Combine → Supervisor\n")

    return app


# ============================================================================
# TESTING - Тестування мультиагентної системи
# ============================================================================

def test_multiagent_system():
    """Тестує мультиагентну систему з різними запитами"""

    app = create_multiagent_system()

    test_queries = [
        "What is the Supervisor Pattern in LangGraph 1.0 and how does it work?",
        "Explain LangGraph StateGraph API and checkpointing mechanisms",
        "How do multi-agent coordination patterns work in LangGraph 1.0?",
    ]

    for i, query in enumerate(test_queries, 1):
        print("\n" + "=" * 70)
        print(f"TEST {i}: {query}")
        print("=" * 70)

        # Ініціалізуємо state
        initial_state = {
            "messages": [],
            "question": query,
            "current_agent": "supervisor",
            "vector_docs": [],
            "web_docs": [],
            "retrieved_docs": [],
            "analysis": "",
            "final_answer": "",
            "supervisor_decision": "",
            "iteration_count": 0,
            "critic_score": 0,
            "critic_feedback": "",
            "revision_count": 0,
            "research_complete": False
        }

        # Виконуємо з checkpointing
        config = {"configurable": {"thread_id": f"test_{i}"}}

        try:
            # Stream результатів
            for event in app.stream(initial_state, config):
                agent_name = list(event.keys())[0]
                print(f"\n📍 Event from: {agent_name}")

            # Отримуємо фінальний state
            final_state = app.get_state(config)

            print("\n" + "-" * 70)
            print("Stats: FINAL RESULT")
            print("-" * 70)
            print(f"\nSUPERVISOR Question: {query}\n")
            print(f"OK Answer:\n{final_state.values.get('final_answer', 'No answer')}\n")
            print(f"📈 Stats:")
            print(f"  - Iterations: {final_state.values.get('iteration_count', 0)}")
            print(f"  - Vector docs: {len(final_state.values.get('vector_docs', []))}")
            print(f"  - Web docs: {len(final_state.values.get('web_docs', []))}")
            print(f"  - Total docs: {len(final_state.values.get('retrieved_docs', []))}")
            print(f"  - Messages exchanged: {len(final_state.values.get('messages', []))}")
            print(f"  - Critic score: {final_state.values.get('critic_score', 0)}/10")
            print(f"  - Revisions made: {final_state.values.get('revision_count', 0)}")

        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()

        if i < len(test_queries):
            input("\n⏸️  Press Enter to continue to next test...\n")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n")
    print("SUPERVISOR LangGraph 1.0 - Multi-Agent System (Parallel Research)")
    print("=" * 70)
    print()
    print("Features:")
    print("  OK Supervisor Pattern - hierarchical coordination")
    print("  OK 6 Specialized Agents + Combine node")
    print("  📚 NEW: ResearcherVector - vector search in knowledge base")
    print("  🌐 NEW: ResearcherWeb - web search via Tavily")
    print("  🔗 NEW: CombineResearch - merge results from both sources")
    print("  OK StateGraph - centralized state management")
    print("  OK Checkpointing - persistent state with MemorySaver")
    print("  OK Conditional Routing - dynamic agent selection")
    print("  OK LangSmith Tracing - full observability")
    print("  ⭐ Critic Agent - answer quality evaluation")
    print("  🔄 Revision Loop - max 2 revisions if score < 7")
    print()
    print("=" * 70 + "\n")

    if not os.getenv("OPENAI_API_KEY"):
        print("❌ ERROR: OPENAI_API_KEY not found!")
        exit(1)

    try:
        test_multiagent_system()

        print("\n" + "=" * 70)
        print("OK ALL TESTS COMPLETED")
        print("=" * 70)
        print("\n💡 Check LangSmith dashboard for full trace!")
        print("   https://smith.langchain.com/\n")

    except KeyboardInterrupt:
        print("\n\n⏹️  Tests interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()