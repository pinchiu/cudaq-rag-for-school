# Retrieval-Augmented Generation (RAG) 實作流程解析

本專案將 RAG 的實作流程分為兩大階段：**建立知識庫 (Indexing Phase)** 與 **提問與生成 (Retrieval & Generation Phase)**。

---

## 🛠️ 第一階段：建立知識庫時 (Indexing Phase)
這個階段的目標是將企業內部或外部的原始資料，轉換成系統能夠快速搜尋的「向量字典」。這通常是在背景自動執行或定期更新的。

### 1. 資料收集與提取 (Data Extraction)
從各個來源收集資料，例如 PDF、Word、Notion 頁面、網頁或內部資料庫，將這些不同格式的檔案統一萃取出純文字。
> 🔗 **程式碼映射**：此步驟在 [`cudaq_craw_and_Split.py`](file:///home/poo/Desktop/scrap/cudaq/cudaq_craw_and_Split.py#L32-L61) 中的 `scrape_docs()` 函式完成。我們使用了 `WebBaseLoader` 與 `BeautifulSoup` 爬取官方文件並萃取純文字內容。

### 2. 文本切塊 (Text Chunking)
因為大型語言模型（LLM）和向量模型都有每次處理字數的限制，且太長的文章會讓語意模糊。系統會將長篇文章切分成較小的「文字區塊（Chunks）」，並且前後區塊通常會保留一點重疊（Overlap），以免切斷上下文的語意。
> 🔗 **程式碼映射**：此步驟在 [`cudaq_craw_and_Split.py`](file:///home/poo/Desktop/scrap/cudaq/cudaq_craw_and_Split.py#L63-L123) 中的 `process_and_split_documents()` 函式完成。我們使用 `RecursiveCharacterTextSplitter` 將文章切分為大小為 1000 的區塊，重疊 200 字元。

### 3. 文本向量化 (Embedding)
這是最核心的轉換步驟。系統會把每一個切好的文字區塊，送進「嵌入模型（Embedding Model）」中。模型會將這些人類語言轉換成幾百到幾千維度的「數字矩陣（向量）」，這代表了該段文字的深層語意。
> 🔗 **程式碼映射**：此步驟在 [`embedding.py`](file:///home/poo/Desktop/scrap/cudaq/embedding.py#L5-L7) 實作。使用了 Ollama 的 `qwen3-embedding:8b` 將我們先前切塊的文本轉為向量。

### 4. 存入向量資料庫 (Vector Database Storage)
將轉換好的「向量資料」以及對應的「原始文字區塊」和「後設資料（Metadata，如切塊檔案來源）」，一起存入向量資料庫（如 ChromaDB）。建立索引以便未來能進行極速的相似度比對。
> 🔗 **程式碼映射**：此步驟在 [`embedding.py`](file:///home/poo/Desktop/scrap/cudaq/embedding.py#L36-L42) 的 `Chroma.from_texts()` 函式中完成。我們將文本、向量，以及含有 `source` 標籤的 `metadatas` 存入 `cuda_quantum_chroma_db` 資料夾。

---

## 💬 第二階段：提問時 (Retrieval & Generation Phase)
當知識庫建立好後，系統就能隨時準備迎接使用者的問題。這個階段要求的是即時性與準確度。所有此階段的實作皆在 [`query.py`](file:///home/poo/Desktop/scrap/cudaq/query.py) 腳本中。

### 1. 使用者提問與處理 (Query Processing)
接收使用者的自然語言問題進入互動迴圈 (`while True:`)。
> 🔗 **程式碼映射**：此步驟在 [`query.py`](file:///home/poo/Desktop/scrap/cudaq/query.py#L55-L61) 的 `user_input = input(...)` 中處理。

### 2. 問題向量化 & 檢索與召回 (Query Embedding & Retrieval)
將使用者的問題，透過與第一階段完全相同的嵌入模型轉換成「問題向量」。接著拿這個「問題向量」去向量資料庫裡進行相似度搜尋，找出語意最接近的候選文字區塊。
> 🔗 **程式碼映射**：此步驟在 [`query.py`](file:///home/poo/Desktop/scrap/cudaq/query.py#L23-L24) 中透過 `base_retriever` 執行。

### 3. 精細重排 (Reranking)
由於向量搜尋主要依賴語意空間的距離，有時不夠精確。我們引入了 Cross-Encoder Reranker 模型，針對初步檢索出的候選名單與使用者問題進行一對一的深度評分，挑選出最相關的 Top-4 內容。
> 🔗 **程式碼映射**：此步驟在 [`query.py`](file:///home/poo/Desktop/scrap/cudaq/query.py#L31-L42) 實作。使用了 `BAAI/bge-reranker-base` 模型搭配 `ContextualCompressionRetriever` 進行重新評分。

### 4. 建構提示詞 (Prompt Construction)
系統會寫一段隱藏的 Prompt 給 LLM，提示模型扮演 AI 助手，只能根據給定的上下文來回答。
> 🔗 **程式碼映射**：此步驟對應於 [`query.py`](file:///home/poo/Desktop/scrap/cudaq/query.py#L48-L57) 中的 `template` 定義。

### 5. 生成最終回答 (Generation)
大型語言模型（如 `qwen3:14b-q4_K_M`）接收到上述包含「參考資料」與「問題」的 Prompt 後，進行閱讀理解與總結，最後生成流暢、準確的回答。
> 🔗 **程式碼映射**：此步驟透過 LangChain 的 LCEL 語法在 [`query.py`](file:///home/poo/Desktop/scrap/cudaq/query.py#L64-L69) 中建構。