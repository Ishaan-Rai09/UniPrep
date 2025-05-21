"""
PDF to University MCQ Generator - Configuration File
Copyright (c) 2024 Ishaan Rai

All rights reserved.
"""

import os
import streamlit as st

def get_groq_api_key():
    """
    Get the Groq API key from session state, environment variables, or Streamlit secrets.
    Returns None if no key is found.
    """
    # First check if user has provided an API key in the session
    if "groq_api_key" in st.session_state and st.session_state.groq_api_key:
        return st.session_state.groq_api_key
    
    # Try environment variables
    api_key = os.getenv("GROQ_API_KEY")
    
    # If not in environment variables, try Streamlit secrets without the warning
    if not api_key:
        try:
            api_key = st.secrets.get("GROQ_API_KEY")
        except (FileNotFoundError, KeyError, Exception) as e:
            # Silently ignore missing secrets file errors
            if isinstance(e, FileNotFoundError):
                pass
            elif isinstance(e, KeyError):
                pass
            api_key = None
    
    return api_key

def get_openai_api_key():
    """
    Get the OpenAI API key from session state, environment variables, or Streamlit secrets.
    Returns None if no key is found.
    """
    # First check if user has provided an API key in the session
    if "openai_api_key" in st.session_state and st.session_state.openai_api_key:
        return st.session_state.openai_api_key
    
    # Try environment variables
    api_key = os.getenv("OPENAI_API_KEY")
    
    # If not in environment variables, try Streamlit secrets
    if not api_key:
        try:
            api_key = st.secrets.get("OPENAI_API_KEY")
        except (FileNotFoundError, KeyError, Exception) as e:
            # Silently ignore missing secrets file errors
            api_key = None
    
    return api_key

def get_ollama_host():
    """
    Get the Ollama host URL from session state, environment variables, or Streamlit secrets.
    Returns default localhost URL if no host is found.
    """
    # First check if user has provided a host in the session
    if "ollama_host" in st.session_state and st.session_state.ollama_host:
        return st.session_state.ollama_host
    
    # Try environment variables
    host = os.getenv("OLLAMA_HOST")
    
    # If not in environment variables, try Streamlit secrets
    if not host:
        try:
            host = st.secrets.get("OLLAMA_HOST")
        except (FileNotFoundError, KeyError, Exception) as e:
            # Silently ignore missing secrets file errors
            host = None
    
    # Default to localhost if not specified
    if not host:
        host = "http://localhost:11434"
    
    return host

# For backward compatibility
def get_api_key():
    """Legacy function for backward compatibility"""
    return get_groq_api_key() 