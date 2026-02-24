"""Firebase Authentication and Firestore integration for chat history."""

import os
import json
from typing import Optional, Dict, List, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

try:
    import firebase_admin
    from firebase_admin import credentials, auth, firestore
    from google.cloud.firestore import Increment
    import requests
    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False
    logger.warning("firebase-admin not installed. Install with: pip install firebase-admin requests")

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("openai not installed. Install with: pip install openai")


class FirebaseManager:
    """
    Manages Firebase Authentication and Firestore operations.
    
    Features:
    - User signup/login with email/password
    - Chat history storage per user
    - Auto-save and auto-load functionality
    """
    
    def __init__(self):
        """Initialize Firebase Admin SDK and Firestore client."""
        self._initialized = False
        self._db = None
        self._web_api_key = None
        self._openai_client = None
        
        if not FIREBASE_AVAILABLE:
            logger.error("Firebase libraries not available")
            return
        
        try:
            # Get Firebase config from environment
            project_id = os.getenv("FIREBASE_PROJECT_ID")
            private_key = os.getenv("FIREBASE_PRIVATE_KEY")
            client_email = os.getenv("FIREBASE_CLIENT_EMAIL")
            self._web_api_key = os.getenv("FIREBASE_WEB_API_KEY")
            
            if not all([project_id, private_key, client_email, self._web_api_key]):
                logger.error("Missing Firebase credentials in environment variables")
                return
            
            # Initialize OpenAI client if available
            if OPENAI_AVAILABLE:
                openai_api_key = os.getenv("OPENAI_API_KEY")
                if openai_api_key:
                    self._openai_client = OpenAI(api_key=openai_api_key)
                    logger.info("OpenAI client initialized for title generation")
            
            # Fix newlines in private key (environment variables escape them)
            private_key = private_key.replace('\\n', '\n')
            
            # Initialize Firebase Admin SDK (if not already initialized)
            if not firebase_admin._apps:
                # Create complete service account credential dictionary
                # Firebase Admin SDK requires all these fields
                cred_dict = {
                    "type": "service_account",
                    "project_id": project_id,
                    "private_key_id": "placeholder",  # Not validated by SDK
                    "private_key": private_key,
                    "client_email": client_email,
                    "client_id": "placeholder",  # Not validated by SDK
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                    "client_x509_cert_url": f"https://www.googleapis.com/robot/v1/metadata/x509/{client_email}"
                }
                cred = credentials.Certificate(cred_dict)
                firebase_admin.initialize_app(cred)
                logger.info("Firebase Admin SDK initialized")
            
            # Get Firestore client
            self._db = firestore.client()
            self._initialized = True
            logger.info("Firestore client ready")
            
        except Exception as e:
            logger.error(f"Failed to initialize Firebase: {e}")
            self._initialized = False
    
    @property
    def initialized(self) -> bool:
        """Check if Firebase is properly initialized."""
        return self._initialized
    
    def sign_up(self, email: str, password: str) -> Dict[str, Any]:
        """
        Create a new user account using Firebase Auth REST API.
        
        Args:
            email: User email
            password: User password (min 6 characters)
            
        Returns:
            dict: {"success": bool, "message": str, "user_id": str (optional)}
        """
        if not self._initialized:
            return {"success": False, "message": "Firebase not initialized"}
        
        try:
            # Use Firebase Auth REST API for signup
            url = f"https://identitytoolkit.googleapis.com/v1/accounts:signUp?key={self._web_api_key}"
            payload = {
                "email": email,
                "password": password,
                "returnSecureToken": True
            }
            
            response = requests.post(url, json=payload)
            data = response.json()
            
            if response.status_code == 200:
                user_id = data.get('localId')
                logger.info(f"User created: {email}")
                
                # Initialize empty chat history in Firestore
                self._db.collection('users').document(user_id).set({
                    'email': email,
                    'created_at': datetime.utcnow().isoformat()
                })
                
                return {
                    "success": True,
                    "message": "Account created successfully!",
                    "user_id": user_id,
                    "id_token": data.get('idToken')
                }
            else:
                error_msg = data.get('error', {}).get('message', 'Unknown error')
                # Translate Firebase error codes
                if 'EMAIL_EXISTS' in error_msg:
                    error_msg = "Email already registered. Please login instead."
                elif 'WEAK_PASSWORD' in error_msg:
                    error_msg = "Password must be at least 6 characters."
                elif 'INVALID_EMAIL' in error_msg:
                    error_msg = "Invalid email format."
                
                return {"success": False, "message": error_msg}
                
        except Exception as e:
            logger.error(f"Signup failed: {e}")
            return {"success": False, "message": f"Signup error: {str(e)}"}
    
    def sign_in(self, email: str, password: str) -> Dict[str, Any]:
        """
        Sign in existing user using Firebase Auth REST API.
        
        Args:
            email: User email
            password: User password
            
        Returns:
            dict: {"success": bool, "message": str, "user_id": str (optional)}
        """
        if not self._initialized:
            return {"success": False, "message": "Firebase not initialized"}
        
        try:
            # Use Firebase Auth REST API for signin
            url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={self._web_api_key}"
            payload = {
                "email": email,
                "password": password,
                "returnSecureToken": True
            }
            
            response = requests.post(url, json=payload)
            data = response.json()
            
            if response.status_code == 200:
                user_id = data.get('localId')
                logger.info(f"User signed in: {email}")
                
                return {
                    "success": True,
                    "message": "Login successful!",
                    "user_id": user_id,
                    "email": email,
                    "id_token": data.get('idToken')
                }
            else:
                error_msg = data.get('error', {}).get('message', 'Unknown error')
                # Translate Firebase error codes
                if 'INVALID_PASSWORD' in error_msg or 'EMAIL_NOT_FOUND' in error_msg:
                    error_msg = "Invalid email or password."
                elif 'USER_DISABLED' in error_msg:
                    error_msg = "Account has been disabled."
                
                return {"success": False, "message": error_msg}
                
        except Exception as e:
            logger.error(f"Signin failed: {e}")
            return {"success": False, "message": f"Login error: {str(e)}"}
    
    def create_session(self, user_id: str, title: str = "New Chat") -> Dict[str, Any]:
        """
        Create a new chat session.
        
        Args:
            user_id: Firebase user ID
            title: Session title (default: "New Chat")
            
        Returns:
            dict: {"success": bool, "session_id": str, "message": str}
        """
        if not self._initialized:
            return {"success": False, "message": "Firebase not initialized"}
        
        try:
            session_data = {
                'title': title,
                'created_at': datetime.utcnow().isoformat(),
                'updated_at': datetime.utcnow().isoformat(),
                'message_count': 0
            }
            
            # Create session document
            session_ref = self._db.collection('users').document(user_id).collection('sessions').document()
            session_ref.set(session_data)
            
            logger.info(f"Created session {session_ref.id} for user {user_id}")
            return {
                "success": True,
                "session_id": session_ref.id,
                "message": "Session created successfully"
            }
            
        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            return {"success": False, "message": str(e)}
    
    def get_sessions(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Get all chat sessions for a user.
        
        Args:
            user_id: Firebase user ID
            
        Returns:
            list: List of session dictionaries with id, title, created_at, updated_at
        """
        if not self._initialized:
            logger.error("Cannot get sessions: Firebase not initialized")
            return []
        
        try:
            sessions_ref = self._db.collection('users').document(user_id).collection('sessions')
            sessions = sessions_ref.order_by('updated_at', direction='DESCENDING').stream()
            
            session_list = []
            for session in sessions:
                data = session.to_dict()
                data['id'] = session.id
                session_list.append(data)
            
            logger.info(f"Loaded {len(session_list)} sessions for user {user_id}")
            return session_list
            
        except Exception as e:
            logger.error(f"Failed to get sessions: {e}")
            return []
    
    def _is_greeting(self, text: str) -> bool:
        """
        Check if the text is a simple greeting.
        
        Args:
            text: The message text
            
        Returns:
            bool: True if it's a greeting
        """
        greetings = [
            'hi', 'hello', 'hey', 'greetings', 'good morning', 'good afternoon',
            'good evening', 'howdy', 'hola', 'sup', 'what\'s up', 'yo',
            'hi there', 'hello there', 'hey there'
        ]
        
        text_lower = text.lower().strip().rstrip('!.,?')
        
        # Check if the text is just a greeting (or very short)
        if text_lower in greetings or len(text.split()) <= 3:
            return any(greeting in text_lower for greeting in greetings)
        
        return False
    
    def _generate_title_summary(self, query: str) -> str:
        """
        Generate a concise title for the session using GPT.
        
        Args:
            query: The user's question
            
        Returns:
            str: Generated title (max 50 chars)
        """
        if not self._openai_client:
            # Fallback to truncation if OpenAI not available
            title = query[:50]
            if len(query) > 50:
                title += '...'
            return title
        
        try:
            response = self._openai_client.chat.completions.create(
                model="gpt-4o-mini",  # Using mini for cost efficiency
                messages=[
                    {
                        "role": "system",
                        "content": "Generate a very brief, concise title (max 6 words) that summarizes the user's question. Return only the title, no quotes or punctuation."
                    },
                    {
                        "role": "user",
                        "content": query
                    }
                ],
                max_tokens=20,
                temperature=0.7
            )
            
            title = response.choices[0].message.content.strip()
            
            # Ensure title isn't too long
            if len(title) > 50:
                title = title[:47] + "..."
            
            logger.info(f"Generated title: {title}")
            return title
            
        except Exception as e:
            logger.error(f"Failed to generate title with GPT: {e}")
            # Fallback to truncation
            title = query[:50]
            if len(query) > 50:
                title += '...'
            return title
    
    def save_message_to_session(
        self,
        user_id: str,
        session_id: str,
        message: Dict[str, Any]
    ) -> bool:
        """
        Save a chat message to a specific session.
        
        Args:
            user_id: Firebase user ID
            session_id: Session ID
            message: dict with keys: query, answer, citations, answer_source
            
        Returns:
            bool: Success status
        """
        if not self._initialized:
            logger.error("Cannot save message: Firebase not initialized")
            return False
        
        try:
            # Add timestamp
            message['timestamp'] = datetime.utcnow().isoformat()
            
            # Save message to session
            self._db.collection('users').document(user_id).collection('sessions').document(session_id).collection('messages').add(message)
            
            # Update session metadata
            session_ref = self._db.collection('users').document(user_id).collection('sessions').document(session_id)
            session_ref.update({
                'updated_at': datetime.utcnow().isoformat(),
                'message_count': Increment(1)
            })
            
            # Auto-generate title with intelligent greeting detection
            session_doc = session_ref.get()
            if session_doc.exists:
                session_data = session_doc.to_dict()
                query = message.get('query', '')
                
                # For first message
                if session_data.get('message_count', 0) == 1 and session_data.get('title') == 'New Chat':
                    if not self._is_greeting(query):
                        # Not a greeting - generate title using GPT
                        title = self._generate_title_summary(query)
                        session_ref.update({'title': title})
                    # If it's a greeting, keep "New Chat" and wait for next message
                
                # For second message (if first was a greeting)
                elif session_data.get('message_count', 0) == 2 and session_data.get('title') == 'New Chat':
                    # First message was likely a greeting, use second message for title
                    title = self._generate_title_summary(query)
                    session_ref.update({'title': title})
            
            logger.debug(f"Message saved to session {session_id} for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save message: {e}")
            return False
    
    def get_session_messages(self, user_id: str, session_id: str) -> List[Dict[str, Any]]:
        """
        Get all messages from a specific session.
        
        Args:
            user_id: Firebase user ID
            session_id: Session ID
            
        Returns:
            list: Chat messages sorted by timestamp
        """
        if not self._initialized:
            logger.error("Cannot load messages: Firebase not initialized")
            return []
        
        try:
            messages_ref = (self._db.collection('users').document(user_id)
                           .collection('sessions').document(session_id)
                           .collection('messages'))
            messages = messages_ref.order_by('timestamp').stream()
            
            message_list = []
            for msg in messages:
                data = msg.to_dict()
                # Remove timestamp from display
                data.pop('timestamp', None)
                message_list.append(data)
            
            logger.info(f"Loaded {len(message_list)} messages from session {session_id}")
            return message_list
            
        except Exception as e:
            logger.error(f"Failed to load session messages: {e}")
            return []
    
    def delete_session(self, user_id: str, session_id: str) -> bool:
        """
        Delete a chat session and all its messages.
        
        Args:
            user_id: Firebase user ID
            session_id: Session ID
            
        Returns:
            bool: Success status
        """
        if not self._initialized:
            logger.error("Cannot delete session: Firebase not initialized")
            return False
        
        try:
            # Delete all messages in session
            messages_ref = (self._db.collection('users').document(user_id)
                           .collection('sessions').document(session_id)
                           .collection('messages'))
            
            batch = self._db.batch()
            for msg in messages_ref.stream():
                batch.delete(msg.reference)
            batch.commit()
            
            # Delete session document
            self._db.collection('users').document(user_id).collection('sessions').document(session_id).delete()
            
            logger.info(f"Deleted session {session_id} for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete session: {e}")
            return False
    
    def update_session_title(self, user_id: str, session_id: str, title: str) -> bool:
        """
        Update session title.
        
        Args:
            user_id: Firebase user ID
            session_id: Session ID
            title: New title
            
        Returns:
            bool: Success status
        """
        if not self._initialized:
            logger.error("Cannot update session: Firebase not initialized")
            return False
        
        try:
            session_ref = self._db.collection('users').document(user_id).collection('sessions').document(session_id)
            session_ref.update({'title': title})
            logger.info(f"Updated session {session_id} title to: {title}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update session title: {e}")
            return False
    
    # Legacy methods for backward compatibility (deprecated)
    def save_chat_message(self, user_id: str, message: Dict[str, Any]) -> bool:
        """
        Save a chat message to Firestore.
        
        Args:
            user_id: Firebase user ID
            message: dict with keys: query, answer, citations, answer_source
            
        Returns:
            bool: Success status
        """
        if not self._initialized:
            logger.error("Cannot save message: Firebase not initialized")
            return False
        
        try:
            # Add timestamp
            message['timestamp'] = datetime.utcnow().isoformat()
            
            # Save to Firestore: users/{user_id}/chat_history/{auto_id}
            self._db.collection('users').document(user_id).collection('chat_history').add(message)
            logger.debug(f"Message saved for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save message: {e}")
            return False
    
    def load_chat_history(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Load all chat history for a user.
        
        Args:
            user_id: Firebase user ID
            
        Returns:
            list: Chat messages sorted by timestamp
        """
        if not self._initialized:
            logger.error("Cannot load history: Firebase not initialized")
            return []
        
        try:
            # Get all messages from Firestore
            messages_ref = self._db.collection('users').document(user_id).collection('chat_history')
            messages = messages_ref.order_by('timestamp').stream()
            
            chat_history = []
            for msg in messages:
                data = msg.to_dict()
                # Remove timestamp from display (internal use only)
                data.pop('timestamp', None)
                chat_history.append(data)
            
            logger.info(f"Loaded {len(chat_history)} messages for user {user_id}")
            return chat_history
            
        except Exception as e:
            logger.error(f"Failed to load chat history: {e}")
            return []
    
    def clear_chat_history(self, user_id: str) -> bool:
        """
        Delete all chat history for a user.
        
        Args:
            user_id: Firebase user ID
            
        Returns:
            bool: Success status
        """
        if not self._initialized:
            logger.error("Cannot clear history: Firebase not initialized")
            return False
        
        try:
            # Delete all messages in subcollection
            messages_ref = self._db.collection('users').document(user_id).collection('chat_history')
            
            # Batch delete (Firestore doesn't have deleteCollection method in Python SDK)
            batch = self._db.batch()
            docs = messages_ref.stream()
            
            count = 0
            for doc in docs:
                batch.delete(doc.reference)
                count += 1
            
            batch.commit()
            logger.info(f"Cleared {count} messages for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear chat history: {e}")
            return False


# Global instance
_firebase_manager = None


def get_firebase_manager() -> FirebaseManager:
    """Get or create global FirebaseManager instance."""
    global _firebase_manager
    if _firebase_manager is None:
        _firebase_manager = FirebaseManager()
    return _firebase_manager
