"""
Streamlit frontend application for AI blog post generation.

This script provides a web interface using Streamlit that communicates with a backend service
to generate blog posts on specified topics.
"""
import base64
import json
import logging
import os
import requests
import streamlit as st
from dotenv import load_dotenv
from io import StringIO
from subprocess import run, PIPE

def load_dotenv_from_azd():
    """
    Load environment variables from Azure Developer CLI (azd) or fallback to .env file.
    
    Attempts to retrieve environment variables using the 'azd env get-values' command.
    If unsuccessful, falls back to loading from a .env file.
    """
    result = run("azd env get-values", stdout=PIPE, stderr=PIPE, shell=True, text=True)
    if result.returncode == 0:
        logging.info(f"Found AZD environment. Loading...")
        load_dotenv(stream=StringIO(result.stdout))
    else:
        logging.info(f"AZD environment not found. Trying to load from .env file...")
        load_dotenv()


# Initialize environment
load_dotenv_from_azd()

def get_principal_id():
    """
    Retrieve the current user's principal ID from request headers.
    If the application is running in Azure Container Apps, and is configured for authentication, 
    the principal ID is extracted from the 'x-ms-client-principal-id' header.
    If the header is not present, a default user ID is returned.
    
    Returns:
        str: The user's principal ID if available, otherwise 'default_user_id'
    """
    result = st.context.headers.get('x-ms-client-principal-id')
    logging.info(f"Retrieved principal ID: {result if result else 'default_user_id'}")
    return result if result else "default_user_id"

def get_principal_display_name():
    """
    Get the display name of the current user from the request headers.
    
    Extracts user information from the 'x-ms-client-principal' header used in 
    Azure Container Apps authentication.
    
    Returns:
        str: The user's display name if available, otherwise 'Default User'
        
    See https://learn.microsoft.com/en-us/azure/container-apps/authentication#access-user-claims-in-application-code for more information.
    """
    default_user_name = "Default User"
    principal = st.context.headers.get('x-ms-client-principal')
    if principal:
        principal = json.loads(base64.b64decode(principal).decode('utf-8'))
        claims = principal.get("claims", [])
        return next((claim["val"] for claim in claims if claim["typ"] == "name"), default_user_name)
    else:
        return default_user_name

def is_valid_json(json_string): 
    """
    Validate if a string is properly formatted JSON.
    
    Args:
        json_string (str): The string to validate as JSON
        
    Returns:
        bool: True if string is valid JSON, False otherwise
    """
    try: 
        json.loads(json_string) 
        return True 
    except json.JSONDecodeError: 
        return False
    
    
def show_client_list():
    clients = st.session_state.clients
    st.markdown("<h2>Clients</h2>", unsafe_allow_html=True)
    if not clients:
        st.info("No clients found from the API.")
        return

    # ... dark themed table code from before ...
    hdr_cols = st.columns([2, 3, 2, 2, 2])
    hdr_cols[0].markdown("**Client ID**")
    hdr_cols[1].markdown("**Full Name**")
    hdr_cols[2].markdown("**DOB**")
    hdr_cols[3].markdown("**Status**")
    hdr_cols[4].markdown("**Action**")

    for i, p in enumerate(clients):
        row_cols = st.columns([2, 3, 2, 2, 2])
        row_cols[0].write(p.get('clientID', ''))
        row_cols[1].write(p.get('fullName', ''))
        row_cols[2].write(p.get('dateOfBirth', ''))
        row_cols[3].write(p.get('status', ''))

        if row_cols[4].button("Show Details", key=f"show_{i}"):
            st.session_state.selected_client = p
            st.session_state.view = "detail"
            st.rerun()

def show_client_details():
    """
    Display client data and agentic evaluation response
    """
    client = st.session_state.selected_client
    if not client:
        st.warning("No client selected.")
        return

    st.markdown(f"### Client ID: **{client['clientID']}**")

    st.subheader("KYC SoW Checks - Evaluation")

    response = run_agents_in_backend(client, get_principal_id())
    # Parse the JSON string from the response
    data = json.loads(response[0])

    # Extract the "content" field
    content = data["content"]

    # Display the content as markdown in the Streamlit app
    st.markdown(content)



def fetch_clients():
    payload = {"user_id": get_principal_id()}  
    try:
        response = requests.post(os.getenv('BACKEND_URL')+"/clients", json=payload)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, str):
            data = json.loads(data)
        return data if data else []
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching clients from backend: {e}")
        return []


def run_agents_in_backend(client_data: dict, user_id: str = "default_user"):
    """
    Calls the FastAPI endpoint /run_agents to update the status of the client.
    Returns the response data 
    """
    payload = {
        "user_id": user_id,
        # The backend expects client_data as a JSON string
        "client_data": json.dumps(client_data)
    }
    try:
        resp = requests.post(os.getenv('BACKEND_URL')+"/run_kyc_checks", json=payload)
        resp.raise_for_status()
        # The endpoint returns a JSON string or None
        data = resp.json()
        # If data is a JSON string, parse it
        if isinstance(data, str):
            data = json.loads(data)
        return data  
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to run ao agentic process in backend: {e}")
        return None


def main():
    st.set_page_config(page_title="KYC SoW Agentic", layout="wide")
    st.title("Agentic KYC SoW Checks")

    # Setup sidebar with user information and logout link
    st.sidebar.write(f"Welcome, {get_principal_display_name()}!")
    st.sidebar.markdown(
        '<a href="/.auth/logout" target = "_self">Sign Out</a>', unsafe_allow_html=True
    )

    if "view" not in st.session_state:
        st.session_state.view = "list"

    if "clients" not in st.session_state:
        st.session_state.clients = fetch_clients()

    if "selected_client" not in st.session_state:
        st.session_state.selected_client = None

    # Routing
    if st.session_state.view == "list":
        show_client_list()
    elif st.session_state.view == "detail":
        show_client_details()
    

if __name__ == "__main__":
    main()
