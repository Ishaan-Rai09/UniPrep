"""
PDF to University MCQ Generator - Configuration File
Copyright (c) 2024 [Your Full Name]
https://github.com/[your-username]/[your-repo]

All rights reserved.
"""

import os
import streamlit as st

def get_api_key():
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