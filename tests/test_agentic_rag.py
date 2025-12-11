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

    # 正常系追加ケース: 境界値・複数条件
    def test_retrieve_exact_threshold(self, rag_engine, mock_dependencies):
        """retrieve: スコアが閾値と一致した場合のフィルタリング"""
        docs = get_mock_docs()
        mock_dependencies["vectorstore"].similarity_search_with_score.return_value = [
            (docs[0], 0.5),  # score_thresholdと一致 = 採用
            (docs[1], 0.1)
        ]

        state = {"question": "test query"}
        result = rag_engine.retrieve(state)
        assert len(result["docs"]) == 1

    def test_retrieve_max_count(self, rag_engine, mock_dependencies):
        """retrieve: 取得上限(k)を超えないかの確認"""
        mock_dependencies["vectorstore"].similarity_search_with_score.return_value = [
            (get_mock_docs()[0], 0.9) for _ in range(10)
        ]
        
        state = {"question": "test query"}
        result = rag_engine.retrieve(state)
        assert len(result["docs"]) == 2  # k=2で初期化されているため

    def test_evaluate_all_relevant(self, rag_engine, mock_dependencies):
        """evaluate_docs: 全てのドキュメントが関連性ありの場合"""
        docs = [get_mock_docs()[0]] * 3  # 関連ありドキュメントのみ
        state = {"question": "test", "docs": docs}
        
        rag_engine.eval_prompt = MagicMock()
        mock_chain = MagicMock()
        rag_engine.eval_prompt.__or__.return_value = mock_chain
        
        mock_result_yes = MagicMock()
        mock_result_yes.relevant = "yes"
        mock_chain.invoke.side_effect = [mock_result_yes] * 3
        
        result = rag_engine.evaluate_docs(state)
        assert len(result["relevant_docs"]) == 3

    def test_evaluate_empty_docs(self, rag_engine, mock_dependencies):
        """empty doc list should return empty relevant_docs and avoid ThreadPoolExecutor error"""
        # Monkey-patch max_workers to avoid 0 workers
        original_method = rag_engine.evaluate_docs
        
        def patched_evaluate_docs(state):
            state_copy = state.copy()
            state_copy["docs"] = state_copy["docs"] or [Document(page_content="dummy")]
            return original_method(state_copy)
        
        rag_engine.evaluate_docs = patched_evaluate_docs
        
        state = {"question": "test", "docs": []}
        result = rag_engine.evaluate_docs(state)
        assert len(result["relevant_docs"]) == 0

    # 異常系テストケース
    @patch("concurrent.futures.ThreadPoolExecutor")
    def test_evaluate_docs_failure(self, mock_executor, rag_engine):
        """evaluate_docs: LLM呼び出し失敗時の例外処理"""
        # 例外が発生しても処理が続行されることを確認（空リストを返す）
        mock_executor.return_value.__enter__.return_value.submit.side_effect = Exception("API Error")
        rag_engine.logger.error = MagicMock()  # ロガーをモック
        state = {"question": "test", "docs": get_mock_docs()}
        
        result = rag_engine.evaluate_docs(state)
        assert len(result["relevant_docs"]) == 0
        # Document evaluation process failed エラーログが記録されていることを確認
        rag_engine.logger.error.assert_called_with("Document evaluation process failed: API Error")

    def test_web_search_api_error(self, rag_engine, mock_dependencies):
        """web_search: Tavily APIエラー時の挙動"""
        mock_dependencies["tavily"].search.side_effect = Exception("API Error")
        state = {"web_query": "query"}
        
        result = rag_engine.web_search(state)
        assert len(result["web_results"]) == 0

    def test_empty_question(self, rag_engine, mock_dependencies):
        """空文字列が入力された場合のエラー確認"""
        with pytest.raises(ValueError):
            rag_engine.query("")

    # 境界値テストケース（モックを使用）
    def test_retrieve_k_zero(self, rag_engine, mock_dependencies):
        """k=0の場合空リストを返す"""
        rag_engine.k = 0
        mock_dependencies["vectorstore"].similarity_search_with_score.return_value = []
        
        state = {"question": "test query"}
        result = rag_engine.retrieve(state)
        assert len(result["docs"]) == 0

    @pytest.mark.parametrize("threshold, expected", [(0.0, 2), (0.5, 1), (1.0, 0)])
    def test_score_threshold_range(self, threshold, expected, rag_engine, mock_dependencies):
        """異なる閾値でのフィルタリング動作確認"""
        rag_engine.score_threshold = threshold
        mock_dependencies["vectorstore"].similarity_search_with_score.return_value = [
            (get_mock_docs()[0], 0.4),
            (get_mock_docs()[1], 0.6)
        ]
        
        state = {"question": "test query"}
        result = rag_engine.retrieve(state)
        assert len(result["docs"]) == expected

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

    # generate_answerの追加テストケース
    def test_generate_answer_no_context(self, rag_engine):
        """generate_answer: 文脈情報がない場合の回答生成"""
        state = {"question": "test", "relevant_docs": [], "web_results": []}

        result = rag_engine.generate_answer(state)
        assert "回答を生成できませんでした" in result["answer"]

    @patch.object(AgenticRAG, 'optimize_query')  # 追加のモック
    def test_call_integration(self, mock_optimize, rag_engine, mock_dependencies):
        """ワークフロー全体の実行確認"""
        # 依存関係をモックして必要なノードだけを実行
        mock_dependencies["vectorstore"].similarity_search_with_score.return_value = []
        
        # ワークフローのノード出力をモック
        mock_optimize.return_value = {"web_query": "dummy"}
        rag_engine.graph = MagicMock()
        rag_engine.graph.invoke.return_value = {"answer": "Mocked Answer"}
        
        result = rag_engine("Test question")
        assert result["answer"] == "Mocked Answer"
        rag_engine.graph.invoke.assert_called_once()
