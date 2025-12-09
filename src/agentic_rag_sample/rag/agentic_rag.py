from typing import TypedDict, List, Dict, Any
from urllib.parse import unquote
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from pydantic import BaseModel, Field
from pinecone import Pinecone
from tavily import TavilyClient
from langchain_core.output_parsers import StrOutputParser
import logging
from langgraph.graph import StateGraph, END

# Prompt constants from specification (in Japanese as provided)
EVAL_SYSTEM = """
あなたは、検索された文書が与えられた質問に関連しているかどうかを評価する専門の評価者です。
評価を行う際は、以下の指示に従ってください。
文書に質問に関連するキーワードまたは意味が含まれている場合、関連性があると評価してください。
あなたの最終出力はJSON形式で、以下のみを出力してください。
{{"relevant": "yes"}} または {{"relevant": "no"}}"""

EVAL_HUMAN = """
検索された文書:
{document}
与えられた質問:
{question}"""

OPT_SYSTEM = """
質問の再構成者として以下のタスクを実行してください。
以下の質問を、ウェブ検索向けに最適化されたより良いバージョンに改善する。
質問を分析し、その背後にある意図／意味を推論する。"""

OPT_HUMAN = """
質問:
{question}
改善された質問を作成してください。
改善された質問だけを最大で300字で出力してください。"""

ANS_SYSTEM = """
あなたは質問応答タスクのアシスタントです。
質問に答えるために、取得した以下の文脈情報を使用してください。
文脈情報が存在しない場合、または答えがわからない場合は、答えがわからないと伝えてください。
提供された文脈にない限り、答えをでっち上げないでください。
質問に対して詳細かつ的を射た回答を提供してください。"""

ANS_USER = """
質問:
{question}
文脈:
{context}
回答:"""

class DocRelevance(BaseModel):
    """Evaluation of document relevance."""
    relevant: str = Field(description="'yes' if the document is relevant to the question (contains related keywords or meaning), 'no' otherwise.")

class AgentState(TypedDict):
    question: str
    docs: List[Document]
    relevant_docs: List[Document]
    needs_web_search: bool
    web_query: str
    web_results: List[Document]
    answer: str


class AgenticRAGError(Exception):
    """Custom exception for AgenticRAG errors."""


class AgenticRAG:
    def __init__(
        self,
        openai_api_key: str,
        pinecone_api_key: str,
        pinecone_index_name: str,
        tavily_api_key: str,
        base_url: str = "https://openrouter.ai/api/v1",
        llm_model: str = "tngtech/deepseek-r1t2-chimera:free",
        embedding_model: str = "openai/text-embedding-3-small",
        score_threshold: float = 0.3,
        k: int = 5,
        web_k: int = 3,
        log_level: int = logging.INFO,
        verbose_output: bool = True,
    ):
        """
        Agentic RAG using LangGraph for workflow management.
        
        Assumes Pinecone index exists with vectors, metadata including 'title' and 'article_id'.
        """
        
        self.llm = ChatOpenAI(
            model=llm_model,
            api_key=openai_api_key,
            base_url=base_url,
            temperature=0,
        )
        
        self.embeddings = OpenAIEmbeddings(
            model=embedding_model,
            api_key=openai_api_key,
            base_url=base_url,
        )
        
        pc = Pinecone(api_key=pinecone_api_key)
        index = pc.Index(pinecone_index_name)
        self.vectorstore = PineconeVectorStore(index=index, embedding=self.embeddings)
        self.tavily = TavilyClient(api_key=tavily_api_key)
        self.web_k = web_k
        self.score_threshold = score_threshold
        self.k = k
        self.verbose_output = verbose_output
        
        self.logger = logging.getLogger(__class__.__name__)
        self.logger.setLevel(log_level)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            handler.setLevel(self.logger.level)
            self.logger.addHandler(handler)
        self.logger.propagate = False
        
        self._setup_prompts()
        self.graph = self._build_graph()
    
    def _setup_prompts(self):
        self.eval_prompt = ChatPromptTemplate.from_messages([
            ("system", EVAL_SYSTEM),
            ("human", EVAL_HUMAN),
        ])
        
        self.opt_prompt = ChatPromptTemplate.from_messages([
            ("system", OPT_SYSTEM),
            ("human", OPT_HUMAN),
        ])
        
        self.ans_prompt = ChatPromptTemplate.from_messages([
            ("system", ANS_SYSTEM),
            ("user", ANS_USER),  # As per spec
        ])
    
    def retrieve(self, state: AgentState) -> Dict[str, List[Document]]:
        """Retrieve top-k docs above score_threshold from Pinecone."""
        raw_results = self.vectorstore.similarity_search_with_score(state["question"], k=self.k)
        docs = []
        for doc, score in raw_results:
            if score >= self.score_threshold:
                doc.metadata['score'] = score  # Cosine similarity score (higher = more similar)
                docs.append(doc)
                if len(docs) >= self.k:
                    break
        
        if self.verbose_output:
            self.logger.info(f"Retrieved {len(docs)} documents")
            for i, doc in enumerate(docs, 1):
                score = doc.metadata.get('score', 0)
                content_preview = doc.page_content[:100].replace('\n', ' ') + '...' 
                meta = doc.metadata
                self.logger.info(f"[{i}/{len(docs)}] Score: {score:.4f}, Title: {meta.get('title', 'N/A')}\nPreview: {content_preview}")
        
        return {"docs": docs}
    
    def evaluate_docs(self, state: AgentState) -> Dict[str, List[Document]]:
        """Evaluate relevance of each doc using LLM (yes/no)."""
        docs = state["docs"]
        relevant = []
        for doc in docs:
            chain = self.eval_prompt | self.llm.with_structured_output(DocRelevance)
            resp = chain.invoke({
                "document": doc.page_content,
                "question": state["question"]
            })
            if resp.relevant == "yes":
                relevant.append(doc)
        
        if self.verbose_output:
            self.logger.info(f"Document Evaluation: {len(relevant)}/{len(docs)} documents relevant")
            for i, doc in enumerate(relevant, 1):
                score = doc.metadata.get('score', 0)
                content_preview = doc.page_content[:100].replace('\n', ' ') + '...' 
                meta = doc.metadata
                self.logger.info(f"[{i}/{len(relevant)}] Score: {score:.4f}, Title: {meta.get('title', 'N/A')}\nPreview: {content_preview}")
        
        return {"relevant_docs": relevant}
    
    def should_web_search(self, state: AgentState) -> Dict[str, bool]:
        """Decide if web search needed: no docs or some docs filtered."""
        needs_web = len(state["docs"]) == 0 or len(state["relevant_docs"]) < len(state["docs"])
        self.logger.debug(f"TRACE should_web_search: needs_web_search={needs_web}")
        return {"needs_web_search": needs_web}
    
    def optimize_query(self, state: AgentState) -> Dict[str, str]:
        """Optimize question for web search using LLM."""
        chain = self.opt_prompt | self.llm | StrOutputParser()
        original_question = state["question"]
        resp = chain.invoke({"question": original_question})
        optimized = resp.strip()
        
        if self.verbose_output:
            self.logger.info(f"Query Optimization:\nOriginal: {original_question}\nOptimized: {optimized}")
        
        return {"web_query": optimized}
    
    def web_search(self, state: AgentState) -> Dict[str, List[Document]]:
        """Perform web search with Tavily."""
        query = state["web_query"]
        if self.verbose_output:
            self.logger.info(f"Web Search : Query: {query}")
        
        results = self.tavily.search(query, max_results=self.web_k).get("results", [])
        
        web_docs = []
        for r in results:
            doc = Document(
                page_content=r["content"],
                metadata={
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                }
            )
            web_docs.append(doc)
        
        if self.verbose_output:
            self.logger.info(f"Web Search: Retrieved {len(web_docs)} results")
            for i, doc in enumerate(web_docs, 1):
                title = doc.metadata.get('title', 'No title')
                raw_url = doc.metadata.get('url', '')
                decoded_url = unquote(raw_url) if raw_url else 'No URL'
                self.logger.info(f"[{i}/{len(web_docs)}] Title: {title}\nURL: {decoded_url}")
        
        return {"web_results": web_docs}
    
    def generate_answer(self, state: AgentState) -> Dict[str, str]:
        """Generate answer from relevant docs + web results."""
        relevant_docs = state.get("relevant_docs", [])
        web_docs = state.get("web_results", [])
        all_docs = relevant_docs + web_docs
        all_contents = [d.page_content for d in all_docs]
        
        context = "\n\n".join(all_contents) if all_contents else "No relevant context available."
        
        chain = self.ans_prompt | self.llm | StrOutputParser()
        resp = chain.invoke({
            "question": state["question"],
            "context": context
        })
        self.logger.debug(f"TRACE generate_answer: Generated answer (len={len(resp)}): {resp[:100]}...")
        return {"answer": resp.strip()}
    
    def _build_graph(self):
        """Build LangGraph workflow."""
        workflow = StateGraph(AgentState)
        
        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("evaluate_docs", self.evaluate_docs)
        workflow.add_node("should_web_search", self.should_web_search)
        workflow.add_node("optimize_query", self.optimize_query)
        workflow.add_node("web_search", self.web_search)
        workflow.add_node("generate_answer", self.generate_answer)
        
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "evaluate_docs")
        workflow.add_edge("evaluate_docs", "should_web_search")
        
        def route_to_web(state: AgentState):
            return "optimize_query" if state["needs_web_search"] else "generate_answer"
        
        workflow.add_conditional_edges(
            "should_web_search",
            route_to_web,
            {
                "optimize_query": "optimize_query",
                "generate_answer": "generate_answer",
            },
        )
        workflow.add_edge("optimize_query", "web_search")
        workflow.add_edge("web_search", "generate_answer")
        
        return workflow.compile()
    
    def __call__(self, question: str) -> AgentState:
        """Run the agentic RAG pipeline."""
        initial_state = {"question": question}
        result = self.graph.invoke(initial_state)
        return result

        
    def query(self, question: str) -> Dict[str, Any]:
        """Public method to perform RAG query and return answer with sources."""
        result = self(question)
        sources = []
        
        # Collect sources from relevant_docs
        if 'relevant_docs' in result:
            for doc in result['relevant_docs']:
                title = doc.metadata.get('title', '')
                if title:
                    sources.append(title)
        
        # Collect sources from web_results
        if 'web_results' in result:
            for doc in result['web_results']:
                source = doc.metadata.get('title', doc.metadata.get('url', ''))
                if source:
                    sources.append(source)
        
        return {
            'answer': result.get('answer', ''),
            'sources': list(set(sources))  # Remove duplicates
        }


# Example usage (commented; requires API keys and existing Pinecone index):
# rag = AgenticRAG(
#     openai_api_key="your_openrouter_key",
#     pinecone_api_key="your_pinecone_key",
#     pinecone_index_name="your_index",
#     tavily_api_key="your_tavily_key",
#     # base_url="https://openrouter.ai/api/v1",  # optional
# )
# result = rag("Your question here")
# print(result["answer"])
