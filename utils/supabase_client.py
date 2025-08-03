# In utils/supabase_client.py

from supabase import create_client, Client
from supabase.lib.client_options import ClientOptions
from typing import Optional
import os
from dotenv import load_dotenv
import logging

# Load environment variables from .env if present
load_dotenv()

log = logging.getLogger(__name__)

def get_supabase_client(access_token: Optional[str] = None) -> Optional[Client]:
    """
    Initializes and returns a Supabase client.
    Can be used with an access_token for an authenticated user,
    or with the anon key for public access.
    """
    url: str = os.environ.get("SUPABASE_URL")
    key: str = os.environ.get("SUPABASE_ANON_KEY")

    if not url or not key:
        log.error("SUPABASE_URL or SUPABASE_ANON_KEY not set in environment variables!")
        return None

    # --- START OF FIX ---

    # The 'key' parameter in create_client handles the 'apikey' header automatically.
    # We only need to manage the 'Authorization' header.
    headers = {}
    if access_token:
        # If we have a user's token, set the Authorization header.
        headers["Authorization"] = f"Bearer {access_token}"

    # Pass the headers dictionary to ClientOptions. It will be empty for anon users
    # and contain the Authorization header for authenticated users.
    options = ClientOptions(headers=headers)

    # Initialize the client. The `key` parameter will set the `apikey` header,
    # and the `options` will add the `Authorization` header if it exists.
    supabase: Client = create_client(url, key, options=options)

    # --- END OF FIX ---

    return supabase

def get_supabase_admin_client() -> Client:
    """
    Initializes and returns a Supabase client with the service key (admin privileges).
    This client bypasses Row Level Security (RLS).
    """
    url: str = os.environ.get("SUPABASE_URL")
    key: str = os.environ.get("SUPABASE_SERVICE_KEY")
    if not url or not key:
        raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in environment variables for admin client.")

    # For debugging purposes, print the loaded URL and a partial key
    print(f"DEBUG ADMIN: SUPABASE_URL loaded: '{url}'")
    print(f"DEBUG ADMIN: SUPABASE_SERVICE_KEY loaded (first 10 chars): '{key[:10]}...' Length: {len(key)}")

    return create_client(url, key)

def refresh_supabase_session(refresh_token: str) -> Optional[dict]:
    """
    Attempts to refresh a Supabase session using a refresh token.
    Returns the new session dictionary if successful, None otherwise.
    """
    url: str = os.environ.get("SUPABASE_URL")
    key: str = os.environ.get("SUPABASE_ANON_KEY")
    
    if not url or not key:
        log.error("SUPABASE_URL or SUPABASE_ANON_KEY not set for refreshing session.")
        return None
    
    supabase_anon_client = create_client(url, key)
    try:
        # Use the refresh_session method from the Supabase client's auth object
        response = supabase_anon_client.auth.refresh_session(refresh_token)
        log.info(f"Supabase session refreshed successfully for user: {response.user.id}")
        return response.session.dict() # Return the new session data as a dictionary
    except Exception as e:
        log.error(f"Error refreshing Supabase session: {e}", exc_info=True)
        return None

