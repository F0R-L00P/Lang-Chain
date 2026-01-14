# Copilot Instructions for Lang-Chain Learning Repository

## Project Overview
This is an educational repository exploring **LangChain** and **LanGraph** frameworks for building LLM applications. The project is structured as progressive learning modules with hands-on examples demonstrating core concepts.

### Architecture
```
LangChain/              # LangChain framework examples
├── Foundation/        # Base concepts (Module0.py - reserved for foundational imports)
└── Templates/         # Progressive learning modules with increasing complexity
    ├── 101_ChatBotMechanics      # Basic prompt templates, chains, chat roles
    ├── 102_ChainsandAgents        # Sequential chains, agents, tool integration
    └── 103_DocumentLoader         # RAG pipeline: PDFs → chunks → embeddings → retrieval

LangGraph/            # Graph-based agent orchestration (placeholder for future development)
```

## Key Patterns & Conventions

### 1. **LCEL (LangChain Expression Language) Chaining**
- Use the **pipe operator `|`** to chain components:
  ```python
  chain = prompt_template | llm | output_parser
  sequential = {"context": retriever | format_docs} | prompt | llm | parser
  ```
- Each pipe creates a runnable that flows data left-to-right
- Dictionary inputs allow parallel execution and variable mapping (see 102)

### 2. **Prompt Templates Pattern**
- **PromptTemplate**: Basic string interpolation with variables
  ```python
  template = PromptTemplate.from_template("Explain {concept} in simple terms")
  ```
- **ChatPromptTemplate**: Multi-turn conversations with roles (system/human/ai)
  ```python
  messages = [("system", "You are..."), ("human", "{input}")]
  ```
- **FewShotPromptTemplate**: Task examples before the query (see 101 for structure)

### 3. **RAG (Retrieval-Augmented Generation) Workflow**
Implemented in [LangChain103_DocumentLoader.py](LangChain/Templates/LangChain103_DocumentLoader.py):
1. **Load**: `PyPDFLoader` → raw documents with metadata
2. **Split**: `RecursiveCharacterTextSplitter` with separators: `["\n\n", "\n", ". ", " ", ""]`
3. **Embed**: `HuggingFaceEmbeddings` (all-MiniLM-L6-v2 model)
4. **Store**: `Chroma` vector database with persistence
5. **Retrieve**: `vectorstore.as_retriever(search_type="similarity", k=3)`
6. **Format**: Custom `format_docs()` function that preserves page numbers `(p. X)`
7. **Chain**: `retriever → prompt → llm → StrOutputParser`

### 4. **LLM Model Selection**
- **Local/lightweight**: `HuggingFacePipeline` with Qwen/Qwen3-0.6B (examples 101-102)
- **Production/powerful**: `ChatWatsonx` with Llama-4-Maverick-17B (RAG + agents)
- Models configured via environment variables (`.env` file): `API_KEY`, `ENDPOINT`, `PROJECT_ID`

### 5. **Agent & Tool Pattern**
From [LangChain102_ChainsandAgents.py](LangChain/Templates/LangChain102_ChinsandAgents.py):
```python
tools = load_tools(["llm-math"], llm=llm)
agent = create_agent(model=llm, tools=tools)
result = agent.invoke({"messages": [HumanMessage(content=question)]})
```
- Tools must have `.name`, `.description`, and `.return_direct` attributes
- Use `@tool` decorator for custom tools with verbose descriptions

### 6. **Environment & Configuration**
- Load `.env` from workspace root: `load_dotenv(dotenv_path=os.path.abspath(".env"))`
- Never hardcode API keys; use environment variables
- For vector DB persistence: `Chroma(..., persist_directory="chroma_cp_theory")`

## Development Workflow

### Running Examples
Each template is self-contained and executable:
```bash
python LangChain/Templates/LangChain101_ChatBotMechanics.py
python LangChain/Templates/LangChain102_ChainsandAgents.py
python LangChain/Templates/LangChain103_DocumentLoader.py
```

### File Dependencies
- **Module0.py**: Reserved for foundational imports (currently empty)
- **Templates are independent**: No cross-template imports; each is a learning module
- **External files**: Examples reference `CPTheory.pdf` (not in repo; provide locally)

### Common Issues
- **Missing PDF**: Update file path in 103 to match your PDFs
- **API failures**: Verify `.env` has correct `API_KEY`, `ENDPOINT`, `PROJECT_ID` for Watson
- **Import errors**: Install `langchain`, `langchain-community`, `langchain-huggingface`, `sentence-transformers`, `chromadb`

## Adding New Content

### Template Naming Convention
- Use `LangChain{XXX}_{TopicName}.py` pattern
- Start with `from dotenv import load_dotenv` if using external APIs
- Include comments separating sections: `# ------- Topic ------- #`
- Document LCEL chains inline (e.g., what each pipe does)

### When Expanding RAG
- Reuse `format_docs()` function for consistent source citations
- Test retrieval with `retriever.invoke()` before chaining to LLM
- Add chunk overlap (150-200 chars) to preserve context across splits

## Integration Points

- **Chroma DB**: Persists to `chroma_*/` directories; managed via `persist_directory` parameter
- **HuggingFace Hub**: Models loaded dynamically; requires internet on first run
- **IBM Watson**: Enterprise LLM endpoint; credentials in `.env`
- **LanGraph** (future): Will orchestrate complex agent workflows; currently a placeholder

---
**Last updated**: January 2026 | Scope: Educational repository for LLM application patterns
