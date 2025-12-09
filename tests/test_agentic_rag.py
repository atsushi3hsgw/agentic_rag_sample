import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document
from agentic_rag_sample.rag.agentic_rag import AgenticRAG, AgentState

# モック用のデータ定義
def get_mock_docs():
    """テストごとに新しいインスタンスを返す（副作用を防ぐため）"""
    return [
        Document(page_content="Relevant content", metadata={"title": "Doc A", "score": 0.9}),
        Document(page_content="Irrelevant content", metadata={"title": "Doc B", "score": 0.8})
    ]

MOCK_WEB_RESULT = {"title": "Web Hit", "url": "http://example.com", "content": "Web content"}

@pytest.fixture
def mock_dependencies():
    """外部APIへの依存を全てモックする"""
    with patch("agentic_rag_sample.rag.agentic_rag.ChatOpenAI") as mock_llm_cls, \
         patch("agentic_rag_sample.rag.agentic_rag.OpenAIEmbeddings") as mock_emb_cls, \
         patch("agentic_rag_sample.rag.agentic_rag.PineconeVectorStore") as mock_vs_cls, \
         patch("agentic_rag_sample.rag.agentic_rag.Pinecone") as mock_pc_cls, \
         patch("agentic_rag_sample.rag.agentic_rag.TavilyClient") as mock_tavily_cls:
        
        yield {
            "llm": mock_llm_cls.return_value,
            "vectorstore": mock_vs_cls.return_value,
            "tavily": mock_tavily_cls.return_value,
        }

@pytest.fixture
def rag_engine(mock_dependencies):
    """テスト対象のAgenticRAGインスタンス"""
    return AgenticRAG(
        openai_api_key="fake",
        pinecone_api_key="fake",
        pinecone_index_name="fake_index",
        tavily_api_key="fake",
        score_threshold=0.5,
        k=2
    )

class TestAgenticRAGLogic:
    """各ノードのロジックテスト"""

    def test_retrieve_filters_low_score(self, rag_engine, mock_dependencies):
        """retrieve: スコアが閾値以下のドキュメントがフィルタリングされるか"""
        docs = get_mock_docs()
        # Pineconeが返す生の結果 (ドキュメント, スコア)
        mock_dependencies["vectorstore"].similarity_search_with_score.return_value = [
            (docs[0], 0.9),       # 閾値(0.5)以上 -> 採用
            (docs[1], 0.1)        # 閾値(0.5)未満 -> 棄却
        ]

        state = {"question": "test query"}
        result = rag_engine.retrieve(state)

        assert len(result["docs"]) == 1
        assert result["docs"][0].page_content == "Relevant content"
        assert result["docs"][0].metadata["score"] == 0.9

    def test_evaluate_docs_logic(self, rag_engine, mock_dependencies):
        """evaluate_docs: LLMがNoと言ったドキュメントが弾かれるか"""
        docs = get_mock_docs()
        state = {"question": "test", "docs": docs}

        # 【重要】パイプライン演算子(|)を成功させるためのモック設定
        # 1. eval_prompt をモック化する
        rag_engine.eval_prompt = MagicMock()
        
        # 2. 実行されるチェーン(mock_chain)を用意する
        mock_chain = MagicMock()
        
        # 3. "prompt | llm" の演算結果が mock_chain になるように設定する
        rag_engine.eval_prompt.__or__.return_value = mock_chain

        # 4. チェーンの実行結果(invoke)の挙動を設定
        # 1回目(Doc A) -> yes, 2回目(Doc B) -> no
        mock_result_yes = MagicMock()
        mock_result_yes.relevant = "yes"
        mock_result_no = MagicMock()
        mock_result_no.relevant = "no"
        mock_chain.invoke.side_effect = [mock_result_yes, mock_result_no]

        # 実行
        result = rag_engine.evaluate_docs(state)

        # 検証: Doc B (no判定) は消え、Doc A (yes判定) だけ残るはず
        assert len(result["relevant_docs"]) == 1
        assert result["relevant_docs"][0].metadata["title"] == "Doc A"

    @pytest.mark.parametrize("retrieved_count, relevant_count, expected_web_search", [
        (0, 0, True),   # 検索結果ゼロ -> Web検索必要
        (3, 0, True),   # 全て関連性なし -> Web検索必要
        (3, 2, True),   # フィルタリングされた -> Web検索必要
        (3, 3, False),  # 全て関連性あり -> Web検索不要
    ])
    def test_should_web_search_logic(self, rag_engine, retrieved_count, relevant_count, expected_web_search):
        """should_web_search: 分岐ロジックの網羅テスト"""
        # 単なる数合わせのダミーデータ
        dummy_docs = [Document(page_content="x")] * retrieved_count
        dummy_rel = [Document(page_content="x")] * relevant_count
        
        state = {
            "docs": dummy_docs,
            "relevant_docs": dummy_rel
        }
        result = rag_engine.should_web_search(state)
        assert result["needs_web_search"] == expected_web_search

    def test_web_search_execution(self, rag_engine, mock_dependencies):
        """web_search: Tavilyの結果がDocument形式に変換されるか"""
        mock_dependencies["tavily"].search.return_value = {
            "results": [MOCK_WEB_RESULT]
        }
        
        state = {"web_query": "optimized query"}
        result = rag_engine.web_search(state)
        
        docs = result["web_results"]
        assert len(docs) == 1
        assert isinstance(docs[0], Document)
        assert docs[0].page_content == "Web content"
        assert docs[0].metadata["title"] == "Web Hit"
        assert docs[0].metadata["url"] == "http://example.com"
        
    def test_query_integration(self, rag_engine, mock_dependencies):
        """query: エンドツーエンドの戻り値フォーマット確認"""
        
        mock_final_state = {
            "answer": "This is the final answer.",
            "relevant_docs": get_mock_docs()[:1], # Doc A
            "web_results": [
                Document(page_content="Web", metadata={"title": "WebSite", "url": "http://web.com"})
            ]
        }
        
        rag_engine.graph = MagicMock()
        rag_engine.graph.invoke.return_value = mock_final_state

        output = rag_engine.query("User question")

        assert output["answer"] == "This is the final answer."
        assert any("Doc A" in s for s in output["sources"])
        assert any("WebSite" in s for s in output["sources"])
