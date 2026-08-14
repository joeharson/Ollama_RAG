# 🤖 Local RAG with Ollama + LangChain

> **Learning Goal:** Understand how to build a **Retrieval-Augmented Generation (RAG)** pipeline that runs 100% locally — no OpenAI API key, no cloud, no cost.

---

## 📌 What Is This?

This notebook (`start.ipynb`) walks through building a **local RAG system** step by step.  
The final outcome: you can ask questions about a PDF document, and a **locally-running LLM** answers using only what is inside that document.

**Tech used:**
| Tool | Role |
|---|---|
| [Ollama](https://ollama.com/) | Runs LLMs locally on your machine |
| `tinyllama` | The local LLM model (small, fast) |
| LangChain | Orchestration framework — connects all the pieces |
| HuggingFace Embeddings (`all-MiniLM-L6-v2`) | Converts text chunks into vectors |
| FAISS | Vector database — stores & searches those vectors |
| PyPDFLoader | Reads and extracts text from PDF files |

---

## 🧠 Core Concept: What is RAG?

Without RAG, an LLM only knows what it was trained on. Ask it about **your** PDF? It has no idea.

**RAG fixes this** by giving the LLM relevant context *at query time*:

```
Your PDF ──► Split into chunks ──► Convert to vectors ──► Store in FAISS
                                                                │
User Question ──► Convert to vector ──► Find similar chunks ◄──┘
                                                │
                              Chunks + Question ──► LLM ──► Answer
```

The LLM does not "read" your PDF. It reads **the most relevant pieces** retrieved by vector search, then generates a grounded answer.

---

## 🔢 Step-by-Step Breakdown

### Step 1 — Basic LLM Setup (Cells 1 & 2)

```python
from langchain_community.llms import Ollama
llm = Ollama(model="tinyllama")
response = llm.generate(["Write a short story about a robot learning to love."])
```

**What happens here?**
- Ollama is running in the background as a local server.
- LangChain's `Ollama` class connects to it and sends a prompt.
- `llm.generate([...])` returns a `LLMResult` object. You access the text via `response.generations[0][0].text`.

> ⚠️ **Deprecation Note:** `langchain_community.llms.Ollama` is deprecated. The modern way is:
> `from langchain_ollama import OllamaLLM`

---

### Step 2 — Direct Question (Cell 3)

```python
question = "Who is Sachin Tendulkar?"
response = llm.invoke(question)
print(response)
```

**What happens here?**
- `llm.invoke()` is the simpler, modern alternative to `llm.generate()`. It returns a plain string directly.
- The model answered with **hallucinated info** (called him a golfer!) — this is expected because `tinyllama` has limited training data.
- This is exactly *why* RAG exists — to stop hallucinations by grounding answers in real documents.

---

### Step 3 — Loading & Chunking the PDF (Cell 5)

```python
pdf_reader = PyPDFLoader("LangChain.pdf")
documents = pdf_reader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
chunks = text_splitter.split_documents(documents)
```

**What happens here?**
- `PyPDFLoader` reads the PDF page by page and returns a list of `Document` objects (each with `.page_content` and `.metadata`).
- The PDF is too large to feed into the LLM at once (LLMs have a **context window** limit).
- `RecursiveCharacterTextSplitter` breaks it into ~1000 character chunks. It tries to split at paragraph → sentence → word boundaries to keep chunks semantically meaningful.
- `chunk_overlap=0` means no overlap between chunks (increase this if answers feel cut off).

---

### Step 4 — Embeddings (Cell 8)

```python
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```

**What happens here?**
- An **embedding model** converts text into a list of numbers (a vector) that captures its *meaning*.
- `all-MiniLM-L6-v2` is a lightweight sentence transformer — downloads once, runs locally.
- Two semantically similar sentences will produce vectors that are **close** to each other in vector space.
- CUDA was verified available (Cell 6) — so the embedding model runs on GPU for speed.

---

### Step 5 — Building the Vector Store (Cell 10)

```python
db = FAISS.from_documents(documents=chunks, embedding=embeddings)
```

**What happens here?**
- Every chunk from the PDF is passed through the embedding model → turned into a vector.
- FAISS stores all these vectors in an **in-memory index**.
- FAISS = **Facebook AI Similarity Search** — ultra-fast library for finding nearest neighbors in vector space.
- Now `db` can accept a question, embed it, and return the **top-k most similar chunks** from the PDF.

---

### Step 6 — Conversational RAG Chain (Cell 12)

```python
qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=db.as_retriever(),
    condense_question_prompt=CONDENSE_QUESTION_PROMPT,
    return_source_documents=True,
    verbose=True
)

result = qa.invoke({"question": "What is LangChain?", "chat_history": []})
print("Answer:", result["answer"])
```

**What happens here?**
- `db.as_retriever()` wraps FAISS so LangChain can use it as a retrieval step.
- `ConversationalRetrievalChain` is a two-step chain:
  1. **Condense** — If there is chat history, rephrase the follow-up question into a standalone question.
  2. **QA** — Retrieve relevant chunks, stuff them into the LLM prompt, generate an answer.
- `CONDENSE_QUESTION_PROMPT` handles multi-turn conversations — so follow-up questions like *"tell me more"* get resolved with prior context.
- `verbose=True` shows the full internal chain trace so you can see exactly what the LLM receives.

---

## ❓ Questions to Test Your Understanding

Use these to quiz yourself before looking at the answers below.

**Conceptual**
1. What problem does RAG solve that a plain LLM cannot?
2. Why do we need to split the PDF into chunks instead of sending the whole thing to the LLM?
3. What is the difference between an **embedding model** and an **LLM**?
4. Why did the model give a wrong answer about Sachin Tendulkar in Step 2?

**Technical**
5. What does `chunk_size=1000` refer to — characters, words, or tokens?
6. What does FAISS actually *search* for when you ask a question?
7. What is the difference between `llm.generate()` and `llm.invoke()`?
8. What does `return_source_documents=True` return and why is it useful?
9. If you set `chunk_overlap=200`, what would change and when would that be helpful?
10. Why did we need to install `pyarrow` and `faiss-cpu` mid-notebook?

**Think Deeper**
11. The LLM used here is `tinyllama`. What trade-offs did we make by choosing it over a larger model?
12. How would this pipeline change if the document was a `.txt` file instead of a PDF?
13. Why is `sentence-transformers/all-MiniLM-L6-v2` used for embeddings instead of the Ollama model itself?

---

## 💡 Answers

<details>
<summary>Click to reveal answers (try answering first!)</summary>

**1.** RAG gives the LLM access to **custom, up-to-date documents** it was never trained on — preventing hallucinations on domain-specific queries.

**2.** LLMs have a **context window limit** (e.g., 2048–4096 tokens). A full PDF will not fit. Chunking also makes retrieval precise — only *relevant* chunks are sent.

**3.** An **embedding model** converts text to vectors (no text generation). An **LLM** generates new text. They are different models with different jobs.

**4.** `tinyllama` has limited training data and may not know about Sachin Tendulkar — or confuses him with someone else. This is called a **hallucination**.

**5.** `chunk_size=1000` refers to **characters** (default for `RecursiveCharacterTextSplitter`).

**6.** FAISS searches for vectors **closest in cosine/euclidean distance** to the query vector — i.e., semantically similar chunks.

**7.** `llm.generate([...])` returns an `LLMResult` object (batch use). `llm.invoke(...)` returns a plain string — simpler for single queries.

**8.** It returns the actual document chunks used to generate the answer. Useful for **citations, debugging, and transparency**.

**9.** `chunk_overlap=200` means 200 characters are shared between adjacent chunks — prevents answers from being split across chunk boundaries, at the cost of slightly more storage.

**10.** `faiss-cpu` and `pyarrow` are not pre-installed in the environment — they are required at runtime for the vector store to function.

**11.** `tinyllama` is fast and runs on modest hardware, but has limited reasoning ability. Trade-off: **speed and resource efficiency vs. answer quality**.

**12.** Replace `PyPDFLoader` with `TextLoader`. The rest of the pipeline (splitting, embedding, FAISS, chain) stays exactly the same.

**13.** The Ollama LLM is a **generative** model optimized for text output. `all-MiniLM-L6-v2` is a **bi-encoder** specifically trained to produce meaningful semantic vectors — much better suited for similarity search tasks.

</details>

---

## 🚀 How to Run

**Prerequisites:**
```bash
# 1. Install Ollama and pull the model
ollama pull tinyllama

# 2. Install Python dependencies
pip install langchain langchain-community langchain-ollama langchain-huggingface
pip install langchain-text-splitters langchain-classic
pip install faiss-cpu pyarrow pypdf sentence-transformers torch
```

**Run:**
1. Make sure Ollama is running (`ollama serve` or it starts automatically on install).
2. Place `LangChain.pdf` in the same directory as the notebook.
3. Open `start.ipynb` in Jupyter and run cells top to bottom.

---

## 📁 Project Files

```
Ollama_RAG/
├── start.ipynb      ← The main learning notebook
├── LangChain.pdf    ← Source document used for RAG Q&A
└── README.md        ← You are here
```

---

## 🔗 Further Reading

- [Ollama Official Docs](https://ollama.com/docs)
- [LangChain RAG Concepts](https://python.langchain.com/docs/concepts/#retrieval)
- [FAISS — Facebook AI Similarity Search](https://github.com/facebookresearch/faiss)
- [Sentence Transformers](https://www.sbert.net/)
- [all-MiniLM-L6-v2 on HuggingFace](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
