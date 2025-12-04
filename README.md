# 📘Agentic RAG Sample

LangGraph × Pinecone × OpenRouter (OpenAI Embeddings) × Tavily を使用して構築した
自律型（Agentic）RAG システム のサンプル実装です。

検索結果が不十分な場合は LLM が自律的に判断して Web 検索へ移行し、
より正確な回答を生成することが特徴です。

- Python で動作

- Pinecone に自動でインデックス作成

- JSONL データをロードして自動チャンク＋ベクトル化

- CLI で RAG 質問実行

- LangGraph による Agentic フロー制御

- Mermaid でフロー図を可視化可能

---
## 📂 プロジェクト構成

```bash
src/
├── rag/
│   └── agentic_rag.py            # Agentic RAG のコア実装（LangGraph）
└── cmd/
    ├── load2vector_cli.py        # JSONL → Pinecone ベクトル登録ツール
    └── agentic_rag_cli.py        # RAG 質問 CLI
```
---
## 🚀 機能
### ✔ Agentic RAG

- LLM による 関連性判定

- LLM による Web 検索の要否判断

- 質問最適化による Web クエリ生成

- Pinecone + Tavily のハイブリッド検索

### ✔ CLI から実行可能

- agentic_rag_cli.py により対話形式 QA

- load2vector_cli.py により JSONL → Pinecone 自動登録

### ✔ LangGraph によるフロー制御

メリット：

- ステップごとに明確な状態遷移

- 条件分岐しやすい

- Mermaid による可視化が容易

---
## 🧠 Agentic RAG フロー（LangGraph）

本システムは LangGraph によって次のように制御されています：

```mermaid
graph TD;
        __start__([<p>__start__</p>]):::first
        retrieve(retrieve)
        evaluate_docs(evaluate_docs)
        should_web_search(should_web_search)
        optimize_query(optimize_query)
        web_search(web_search)
        generate_answer(generate_answer)
        __end__([<p>__end__</p>]):::last
        __start__ --> retrieve;
        evaluate_docs --> should_web_search;
        optimize_query --> web_search;
        retrieve --> evaluate_docs;
        should_web_search -.-> generate_answer;
        should_web_search -.-> optimize_query;
        web_search --> generate_answer;
        generate_answer --> __end__;
        classDef default line-height:1.2
        classDef first fill-opacity:0
        classDef last fill-opacity:0
```
---

## 🔍 各ステップの説明

1. retrieve（ベクトル検索）

    Pinecone から関連ドキュメントを取得。

2. evaluate_docs（評価）

    LLM（構造化出力）で「本当に関連あるか」を Yes/No で判定。

3. should_web_search（Web 検索要否）

    - ドキュメントが不足している

    - 外部情報が必要

    と判断すれば optimize_query に遷移。
    不要なら generate_answer へ直接進む。

4. optimize_query（質問最適化）

    Web 検索向けにクエリを LLM が変換。

5. web_search（外部検索）

    Tavily API でインターネット検索し、結果を Document 化。

6. generate_answer（最終回答）

    関連ドキュメントと Web 結果を統合し、
    信頼性の高い最終回答を生成。

---
## 🛠 セットアップ

1. 依存パッケージインストール

```bash
pip install -r requirements.txt
```

2. 環境変数の設定（.env）

```bash
OPENAI_API_KEY=xxxx
PINECONE_API_KEY=xxxx
TAVILY_API_KEY=xxxx

PINECONE_INDEX_NAME=agentic-rag-index
BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL=tngtech/deepseek-r1t2-chimera:free
EMBEDDING_MODEL=openai/text-embedding-3-small

SCORE_THRESHOLD=0.3
K=5
WEB_K=3

CHUNK_SIZE=2000
CHUNK_OVERLAP=300
```

---
## 📥 JSONL → Pinecone への登録（load2vector_cli）

### JSONL 形式
```json
{"id": "001", "title": "サンプル文書", "paragraphs": ["文章1", "文章2", "文章3"]}
{"id": "002", "title": "別の文書", "paragraphs": ["内容A", "内容B"]}
```

### 実行

```bash
python src/cmd/load2vector_cli.py data/articles.jsonl
```

（インデックスが無ければ自動作成）

---
### ❓ RAG 質問実行（agentic_rag_cli）

```bash
python src/cmd/agentic_rag_cli.py "LLM はどのように学習されますか？"
```
---
#### オプション例

--k 8
--web_k 5
--score_threshold 0.25
--log_level DEBUG
--no-verbose

---
### 📊 LangGraph フロー図だけ出力する

```bash
python src/cmd/agentic_rag_cli.py --dump_graph
```

README に貼れる Mermaid が生成されます。

---
### 📜 サンプル出力

```bash
Answer:
LLM（大規模言語モデル）は大量のテキストデータを学習し...

Sources:
- サンプル文書
- https://example.com/llm
```

---

### 🧭 今後の拡張アイデア

- 🔥 Retrieval のフィードバックループ追加

- 🧪 Web 結果の信頼度分析

- 🧱 ローカル LLM モデル対応

- 🧩 マルチエージェント化

- 📎 PDF / Web ページ自動 ingestion

---

## ⭐️ ライセンス

MIT License（必要なら変更可能）
