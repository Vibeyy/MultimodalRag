"""Lazy-loading Streamlit UI - all imports deferred."""

import streamlit as st
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Page configuration
st.set_page_config(
    page_title="Multimodal RAG Assistant",
    page_icon="📄",
    layout="wide",
)

st.title("Multimodal RAG Assistant")

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'last_query' not in st.session_state:
    st.session_state.last_query = ""
if 'last_answer' not in st.session_state:
    st.session_state.last_answer = ""

# Sidebar
with st.sidebar:
    st.header("Settings")
    expand_query = st.checkbox("Query Expansion", value=False)
    
    st.divider()
    st.subheader("Tracing")
    enable_langfuse = st.checkbox("Enable Langfuse", value=False, help="Track queries in Langfuse Cloud")
    if enable_langfuse:
        st.caption("Traces will appear in Langfuse dashboard")
        # Set environment variable and reinitialize tracer
        import os
        os.environ["LANGFUSE_ENABLED"] = "true"
        # Force tracer reinitialization
        if st.session_state.initialized:
            try:
                from multimodal_rag.utils import tracing
                tracing._tracer = None  # Reset global instance
                tracer = tracing.get_tracer()
                if tracer._enabled:
                    st.caption("Connected to Langfuse")
            except Exception as e:
                st.error(f"Langfuse init failed: {e}")
    else:
        import os
        os.environ["LANGFUSE_ENABLED"] = "false"
    
    st.divider()
    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.success("Chat cleared!")

# Main content
tab1, tab2, tab3 = st.tabs(["Chat", "Ingest", "Evaluation"])

with tab1:
    st.header("Chat with Documents")
    
    # Initialize components (lazy loading with deferred imports)
    if not st.session_state.initialized:
        st.info("Click 'Initialize System' to load components (may take 5-10 min first time)")
        
        if st.button("Initialize System", type="primary"):
            progress = st.progress(0, text="Starting initialization...")
            status = st.empty()
            
            try:
                status.info("Importing modules...")
                progress.progress(5, text="Importing modules...")
                
                # Import only when needed
                from multimodal_rag.retrieval.vector_store import QdrantStore
                from multimodal_rag.retrieval.retrievers import HybridRetriever, DenseRetriever
                from multimodal_rag.ingestion.embedder import TextEmbedder
                from multimodal_rag.ingestion.pipeline import IngestionPipeline
                from multimodal_rag.generation.generator import GeminiGenerator
                from multimodal_rag.orchestration.agents import RAGAgent
                
                status.info("Connecting to Qdrant...")
                progress.progress(10, text="Connecting to Qdrant...")
                vector_store = QdrantStore()
                
                status.warning("Downloading embedding model (1.3GB - this may take 5-10 minutes)...")
                progress.progress(30, text="Loading text embedder (downloading model if first run)...")
                text_embedder = TextEmbedder()
                
                status.info("Setting up retrievers...")
                progress.progress(50, text="Setting up retrievers...")
                dense_retriever = DenseRetriever(vector_store, text_embedder)
                hybrid_retriever = HybridRetriever(dense_retriever)
                
                status.info("Initializing Gemini generator...")
                progress.progress(70, text="Initializing Gemini generator...")
                generator = GeminiGenerator()
                
                status.info("Building RAG agent...")
                progress.progress(85, text="Building RAG agent...")
                rag_agent = RAGAgent(hybrid_retriever, generator, None)
                
                status.info("Setting up ingestion pipeline...")
                progress.progress(95, text="Setting up ingestion pipeline...")
                pipeline = IngestionPipeline()
                
                progress.progress(100, text="Complete!")
                status.success("All components loaded!")
                
                st.session_state.vector_store = vector_store
                st.session_state.rag_agent = rag_agent
                st.session_state.pipeline = pipeline
                st.session_state.initialized = True
                
                st.balloons()
                st.success("System ready! You can now ask questions.")
                st.rerun()
                
            except Exception as e:
                status.empty()
                progress.empty()
                st.error(f"Failed to initialize: {e}")
                import traceback
                with st.expander("Show full error"):
                    st.code(traceback.format_exc())
        st.stop()
    
    # Query interface (only shown after initialization)
    st.success("System is ready")
    
    query = st.text_input("Ask a question:", placeholder="What do you want to know?")
    
    if st.button("Submit", type="primary") and query:
        with st.spinner("Thinking..."):
            try:
                result = st.session_state.rag_agent.run(
                    query=query,
                    expand_query=expand_query,
                    check_hallucination=False
                )
                
                # Flush Langfuse traces if enabled
                if enable_langfuse:
                    try:
                        from multimodal_rag.utils.tracing import get_tracer
                        tracer = get_tracer()
                        tracer.flush()
                        st.sidebar.success("Trace sent to Langfuse")
                    except Exception as e:
                        st.sidebar.warning(f"Langfuse flush failed: {e}")
                
                # Save last query/answer for evaluation tab
                st.session_state.last_query = query
                st.session_state.last_answer = result['answer']
                
                # Display answer
                st.markdown("### Answer")
                st.write(result['answer'])
                
                # Display sources
                if result.get('citations'):
                    st.markdown("### Sources")
                    for i, cit in enumerate(result['citations'], 1):
                        st.markdown(f"{i}. **{cit['source_file']}** (Page {cit['page_num']})")
                
                # Save to history
                st.session_state.chat_history.append({
                    'query': query,
                    'answer': result['answer'],
                    'citations': result.get('citations', [])
                })
                
            except Exception as e:
                st.error(f"Query failed: {e}")
                import traceback
                with st.expander("Show error"):
                    st.code(traceback.format_exc())
    
    # Show history
    if st.session_state.chat_history:
        st.divider()
        st.markdown("### Recent History")
        for i, item in enumerate(reversed(st.session_state.chat_history[-3:]), 1):
            with st.expander(f"Q: {item['query'][:60]}..."):
                st.markdown(f"**A:** {item['answer']}")

with tab2:
    st.header("Ingest Documents")
    
    if not st.session_state.initialized:
        st.warning("System not initialized. Go to Chat tab and click 'Initialize System' first.")
    else:
        uploaded_file = st.file_uploader("Upload a document", type=['pdf', 'txt', 'png', 'jpg', 'jpeg'])
        tags_input = st.text_input("Tags (comma-separated)", value="uploaded")
        
        if uploaded_file and st.button("Ingest Document", type="primary"):
            with st.spinner("Processing document..."):
                try:
                    # Save uploaded file
                    save_path = Path("data/raw") / uploaded_file.name
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    save_path.write_bytes(uploaded_file.read())
                    
                    # Process document
                    tags = [t.strip() for t in tags_input.split(",")] if tags_input else []
                    result = st.session_state.pipeline.process_document(save_path, tags=tags)
                    
                    # Store in vector store
                    chunks = result.get('chunks', [])
                    if chunks:
                        st.session_state.vector_store.insert_chunks(chunks)
                        st.success(f"Ingested {result['total_chunks']} chunks from {uploaded_file.name}!")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Chunks", result['total_chunks'])
                        with col2:
                            st.metric("Text Chunks", result.get('text_chunks', 0))
                        with col3:
                            st.metric("Image Chunks", result.get('image_chunks', 0))
                    else:
                        st.warning("No chunks extracted from document.")
                    
                except Exception as e:
                    st.error(f"Ingestion failed: {e}")
                    import traceback
                    with st.expander("Show error"):
                        st.code(traceback.format_exc())

with tab3:
    st.header("Evaluation")
    
    if not st.session_state.initialized:
        st.warning("System not initialized. Go to Chat tab and click 'Initialize System' first.")
    elif not st.session_state.last_query:
        st.info("No query found. Please run a query in the Chat tab first.")
    else:
        st.markdown("""
        Evaluate RAG quality using LLM-based metrics powered by **gpt-4o-mini**.
        
        **3 Metrics (Self-Evaluation):**
        - Answer Relevancy: Does the answer address the question?
        - Faithfulness: Is the answer grounded in retrieved context?
        - Context Precision: Do the contexts support the answer?
        
        **4 Metrics (With Ground Truth):**
        - Add expected answer below to also get Context Recall
        """)
        
        st.divider()
        
        # Display current query and answer from chat
        st.subheader("Current Query & Answer")
        st.write(f"**Question:** {st.session_state.last_query}")
        st.write(f"**Answer:** {st.session_state.last_answer}")
        
        st.divider()
        
        # Optional ground truth
        with st.expander("Add Expected Answer (Optional)"):
            st.caption("Provide an expected answer to compare against ground truth and enable Context Recall metric.")
            expected_answer = st.text_area("Expected Answer:", height=100)
        
        if st.button("Run Evaluation", type="primary"):
            with st.spinner("Running evaluation..."):
                try:
                    from multimodal_rag.evaluation.evaluator import RAGEvaluator
                    
                    # Get the last executed result from chat
                    result = st.session_state.rag_agent.run(
                        query=st.session_state.last_query,
                        expand_query=False,
                        check_hallucination=False
                    )
                    
                    # Run RAGAS evaluation
                    evaluator = RAGEvaluator()
                    st.info("Running RAGAS metrics (may take 30-60 seconds)...")
                    
                    # Format retrieved contexts
                    contexts = [
                        chunk.get('text', chunk.get('content', ''))
                        for chunk in result.get('retrieved_chunks', [])
                    ]
                    
                    # RAGAS expects lists
                    metrics = evaluator.evaluate_generation(
                        queries=[st.session_state.last_query],
                        answers=[st.session_state.last_answer],
                        contexts=[contexts],
                        ground_truths=[expected_answer] if expected_answer else None
                    )
                    
                    st.markdown("### RAGAS Metrics")
                    
                    # Show metrics
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Answer Relevancy", f"{metrics.get('answer_relevancy', 0):.3f}", 
                                 help="How relevant is the answer to the question?")
                        st.metric("Context Precision", f"{metrics.get('context_precision', 0):.3f}",
                                 help="Do the contexts support the answer?")
                    with col2:
                        st.metric("Faithfulness", f"{metrics.get('faithfulness', 0):.3f}",
                                 help="Is the answer grounded in the retrieved context?")
                        if expected_answer and 'context_recall' in metrics:
                            st.metric("Context Recall", f"{metrics.get('context_recall', 0):.3f}",
                                     help="Did we retrieve all info from the ground truth?")
                    
                    if expected_answer:
                        st.success("Evaluated against your expected answer (4 metrics)")
                    else:
                        st.info("3 metrics available. Add expected answer for Context Recall.")
                        
                except ImportError:
                    st.warning("RAGAS not installed. Install with: pip install ragas datasets")
                except Exception as e:
                    st.warning(f"RAGAS evaluation skipped: {e}")
                    import traceback
                    with st.expander("Show error"):
                        st.code(traceback.format_exc())

