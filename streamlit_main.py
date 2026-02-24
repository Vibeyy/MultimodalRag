"""Lazy-loading Streamlit UI - all imports deferred."""

import streamlit as st
from pathlib import Path
import sys
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

# Add src to path (correct path: MultimodalRag/src/)
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Page configuration
st.set_page_config(
    page_title="DoberGirl - Multimodal RAG Assistant",
    page_icon="📄",
    layout="wide",
)

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'user_email' not in st.session_state:
    st.session_state.user_email = None
if 'current_session_id' not in st.session_state:
    st.session_state.current_session_id = None
if 'sessions' not in st.session_state:
    st.session_state.sessions = []
if 'firebase_manager' not in st.session_state:
    # Initialize Firebase Manager once
    from multimodal_rag.utils.firebase_auth import get_firebase_manager
    st.session_state.firebase_manager = get_firebase_manager()

# ============================================================================
# AUTHENTICATION PAGE
# ============================================================================

if not st.session_state.authenticated:
    st.title("DoberGirl - Multimodal RAG Assistant")
    st.markdown("### Login or Create Account")
    
    # Check if Firebase is available
    if not st.session_state.firebase_manager.initialized:
        st.error("⚠️ Firebase not configured. Please set up Firebase credentials in .env file.")
        st.info("""
        Required environment variables:
        - FIREBASE_PROJECT_ID
        - FIREBASE_PRIVATE_KEY
        - FIREBASE_CLIENT_EMAIL
        - FIREBASE_WEB_API_KEY
        """)
        st.stop()
    
    # Login form
   
    st.info("💡 Users must be created directly in Firebase Console")
    
    with st.form("login_form"):
        login_email = st.text_input("Email", key="login_email")
        login_password = st.text_input("Password", type="password", key="login_password")
        login_submit = st.form_submit_button("Login", type="primary", use_container_width=True)
        
        if login_submit:
            if not login_email or not login_password:
                st.error("Please enter both email and password")
            else:
                with st.spinner("Logging in..."):
                    result = st.session_state.firebase_manager.sign_in(login_email, login_password)
                    
                    if result['success']:
                        st.session_state.authenticated = True
                        st.session_state.user_id = result['user_id']
                        st.session_state.user_email = result['email']
                        
                        # Load sessions from Firestore
                        st.session_state.sessions = st.session_state.firebase_manager.get_sessions(
                            result['user_id']
                        )
                        
                        # Create first session if none exist
                        if not st.session_state.sessions:
                            new_session = st.session_state.firebase_manager.create_session(
                                result['user_id'],
                                "New Chat"
                            )
                            if new_session['success']:
                                st.session_state.current_session_id = new_session['session_id']
                                st.session_state.sessions = [{
                                    'id': new_session['session_id'],
                                    'title': 'New Chat',
                                    'created_at': datetime.utcnow().isoformat(),
                                    'updated_at': datetime.utcnow().isoformat(),
                                    'message_count': 0
                                }]
                        else:
                            # Set current session to most recent
                            st.session_state.current_session_id = st.session_state.sessions[0]['id']
                        
                        # Load messages for current session
                        st.session_state.chat_history = st.session_state.firebase_manager.get_session_messages(
                            result['user_id'],
                            st.session_state.current_session_id
                        )
                        
                        st.success(result['message'])
                        st.rerun()
                    else:
                        st.error(result['message'])
    
    st.stop()  # Stop rendering if not authenticated

# ============================================================================
# MAIN APPLICATION (Only shown after authentication)
# ============================================================================

st.title("DoberGirl")
st.caption(f"👤 Logged in as: **{st.session_state.user_email}**")

# Sidebar
with st.sidebar:
    st.header("Settings")
    expand_query = st.checkbox("Query Expansion", value=False)
    
    st.divider()
    st.subheader("💬 Chat Sessions")
    
    # New Chat button
    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        result = st.session_state.firebase_manager.create_session(
            st.session_state.user_id,
            "New Chat"
        )
        if result['success']:
            st.session_state.current_session_id = result['session_id']
            st.session_state.chat_history = []
            # Refresh sessions list
            st.session_state.sessions = st.session_state.firebase_manager.get_sessions(
                st.session_state.user_id
            )
            st.success("New chat created!")
            st.rerun()
    
    st.caption(f"💭 {len(st.session_state.sessions)} total sessions")
    
    # Display sessions
    if st.session_state.sessions:
        st.markdown("#### Recent Chats")
        for session in st.session_state.sessions[:10]:  # Show latest 10
            col1, col2 = st.columns([4, 1])
            
            with col1:
                # Highlight active session
                is_active = session['id'] == st.session_state.current_session_id
                button_type = "primary" if is_active else "secondary"
                
                # Session button
                title_display = session['title'][:30] + "..." if len(session['title']) > 30 else session['title']
                if st.button(
                    f"{'🟢' if is_active else '⚪'} {title_display}",
                    key=f"session_{session['id']}",
                    use_container_width=True,
                    type=button_type
                ):
                    # Switch to this session
                    st.session_state.current_session_id = session['id']
                    st.session_state.chat_history = st.session_state.firebase_manager.get_session_messages(
                        st.session_state.user_id,
                        session['id']
                    )
                    st.rerun()
            
            with col2:
                # Delete button
                if st.button("🗑️", key=f"delete_{session['id']}", help="Delete session"):
                    if session['id'] == st.session_state.current_session_id:
                        # If deleting current session, switch to another
                        other_sessions = [s for s in st.session_state.sessions if s['id'] != session['id']]
                        if other_sessions:
                            st.session_state.current_session_id = other_sessions[0]['id']
                            st.session_state.chat_history = st.session_state.firebase_manager.get_session_messages(
                                st.session_state.user_id,
                                other_sessions[0]['id']
                            )
                        else:
                            # Create new session if this was the last one
                            new_session = st.session_state.firebase_manager.create_session(
                                st.session_state.user_id,
                                "New Chat"
                            )
                            st.session_state.current_session_id = new_session['session_id']
                            st.session_state.chat_history = []
                    
                    # Delete the session
                    st.session_state.firebase_manager.delete_session(
                        st.session_state.user_id,
                        session['id']
                    )
                    # Refresh sessions list
                    st.session_state.sessions = st.session_state.firebase_manager.get_sessions(
                        st.session_state.user_id
                    )
                    st.rerun()
    
    st.divider()
    st.subheader("👤 Account")
    
    if st.button("🚪 Logout", use_container_width=True, type="secondary"):
        # Clear all session state
        st.session_state.authenticated = False
        st.session_state.user_id = None
        st.session_state.user_email = None
        st.session_state.chat_history = []
        st.session_state.sessions = []
        st.session_state.current_session_id = None
        st.session_state.initialized = False
        st.success("Logged out successfully!")
        st.rerun()

# Main content
tab1, tab2 = st.tabs(["Bark", "Eat"])

with tab1:
    st.header("💬 BowBow(chat) with Documents")
    
    # Initialize components automatically (lazy loading with deferred imports)
    if not st.session_state.initialized:
        with st.spinner("Initializing system components... This may take a moment on first load."):
            try:
                # Import only when needed
                from multimodal_rag.retrieval.vector_store import QdrantStore
                from multimodal_rag.retrieval.retrievers import HybridRetriever, DenseRetriever
                from multimodal_rag.ingestion.embedder import TextEmbedder
                from multimodal_rag.ingestion.pipeline import IngestionPipeline
                from multimodal_rag.generation.generator import OpenAIGenerator
                from multimodal_rag.orchestration.agents import RAGAgent
                
                vector_store = QdrantStore()
                text_embedder = TextEmbedder()
                dense_retriever = DenseRetriever(vector_store, text_embedder)
                hybrid_retriever = HybridRetriever(dense_retriever)
                generator = OpenAIGenerator()
                rag_agent = RAGAgent(hybrid_retriever, generator, None)
                pipeline = IngestionPipeline()
                
                st.session_state.vector_store = vector_store
                st.session_state.rag_agent = rag_agent
                st.session_state.pipeline = pipeline
                st.session_state.initialized = True
                
                st.success("✅ System initialized successfully!")
                st.rerun()
                
            except Exception as e:
                st.error(f"Failed to initialize: {e}")
                import traceback
                with st.expander("Show full error"):
                    st.code(traceback.format_exc())
                st.stop()
    
    # Query interface (only shown after initialization)
    st.success("✅ System is ready")
    
    # Display chat history first (like ChatGPT)
    st.markdown("---")
    
    # Show all chat messages
    for msg in st.session_state.chat_history:
        # User message
        with st.chat_message("user"):
            st.write(msg['query'])
        
        # Assistant message
        with st.chat_message("assistant"):
            # Source indicator
            answer_source = msg.get('answer_source', 'retrieval')
            has_citations = len(msg.get('citations', [])) > 0
            
            if has_citations:
                st.caption(f"📚 From your documents ({len(msg['citations'])} sources)")
            else:
                st.caption("💡 General knowledge")
            
            st.write(msg['answer'])
            
            # Show citations if available
            if msg.get('citations'):
                with st.expander(f"📎 {len(msg['citations'])} Sources"):
                    for i, cit in enumerate(msg['citations'], 1):
                        st.markdown(f"{i}. **{cit['source_file']}** - Page {cit['page_num']}")
    
    # Chat input at bottom
    query = st.chat_input("Ask a question about your documents...")
    
    if query:
        # Show user message immediately
        with st.chat_message("user"):
            st.write(query)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    from multimodal_rag.utils.config import get_config
                    config = get_config()
                    
                    result = st.session_state.rag_agent.run(
                        query=query,
                        expand_query=expand_query,
                        check_hallucination=False,
                        allow_general_knowledge=config.allow_general_knowledge,
                    )
                    
                    # Determine actual source based on answer_source and citations
                    answer_source = result.get('answer_source', 'retrieval')
                    has_citations = len(result.get('citations', [])) > 0
                    
                    # Show source indicator
                    if has_citations:
                        # Has citations = definitely from documents
                        st.caption(f"📚 From your documents ({len(result['citations'])} sources)")
                    elif answer_source == 'general_knowledge':
                        # Explicitly marked as general knowledge
                        st.caption("💡 General knowledge")
                    else:
                        # Context provided but no citations = model used general knowledge
                        st.caption("💡 General knowledge")
                    
                    # Show answer
                    st.write(result['answer'])
                    
                    # Show citations if available
                    if result.get('citations'):
                        with st.expander(f"📎 {len(result['citations'])} Sources"):
                            for i, cit in enumerate(result['citations'], 1):
                                st.markdown(f"{i}. **{cit['source_file']}** - Page {cit['page_num']}")
                    
                    # Save last query/answer for evaluation tab
                    # Save to history with proper source detection
                    has_citations = len(result.get('citations', [])) > 0
                    final_answer_source = 'retrieval' if has_citations else 'general_knowledge'
                    
                    message_data = {
                        'query': query,
                        'answer': result['answer'],
                        'citations': result.get('citations', []),
                        'answer_source': final_answer_source,
                    }
                    
                    # Save to session state
                    st.session_state.chat_history.append(message_data)
                    
                    # Save to Firestore session
                    st.session_state.firebase_manager.save_message_to_session(
                        st.session_state.user_id,
                        st.session_state.current_session_id,
                        message_data
                    )
                    
                    # Force rerun to show in history
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Query failed: {e}")
                    import traceback
                    with st.expander("Show error details"):
                        st.code(traceback.format_exc())

with tab2:
    st.header("Ingest Documents")
    
    if not st.session_state.initialized:
        st.info("⏳ System is initializing... Please wait or switch to Chat tab.")
    else:
        uploaded_file = st.file_uploader("Upload a document", type=['pdf', 'txt', 'png', 'jpg', 'jpeg'])
        tags_input = st.text_input("Tags (comma-separated)", value="uploaded")
        
        st.divider()
        st.subheader("Processing Options")
        
        enable_image_extraction = st.checkbox(
            "Enable Image Extraction (for PDFs with diagrams/charts)",
            value=False,
            help="⚠️ Expensive & slow! Only enable for PDFs with important images. Text-only PDFs should leave this OFF."
        )
        
        if enable_image_extraction:
            st.warning("⚠️ Image extraction will convert every page to an image and process with Vision API. This is slow and costs ~$0.01 per page.")
        else:
            st.info("ℹ️ Fast text extraction mode. Vision API only used for pages with minimal text (<100 chars).")
        
        st.divider()
        
        if uploaded_file and st.button("Ingest Document", type="primary"):
            with st.spinner("Processing document..."):
                try:
                    # Save uploaded file
                    save_path = Path("data/raw") / uploaded_file.name
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    save_path.write_bytes(uploaded_file.read())
                    
                    # Process document
                    tags = [t.strip() for t in tags_input.split(",")] if tags_input else []
                    result = st.session_state.pipeline.process_document(
                        save_path,
                        enable_image_extraction=enable_image_extraction,
                        tags=tags
                    )
                    
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

