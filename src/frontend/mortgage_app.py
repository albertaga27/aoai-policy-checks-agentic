"""
Streamlit frontend application for Mortgage-Origination Agentic demo.

This UI lets advisors:
1. View the list of mortgage prospects already submitted.
2. Inspect an individual prospect’s details & status.
3. Create a **new** mortgage request: fill the financial form & drag-drop
   supporting PDFs/scans.  The payload is posted to a backend that will be
   implemented separately.

The code re-uses the authentication helpers from the reference KYC sample but
swaps out the business logic and endpoints.
"""

from __future__ import annotations

import base64
import json
import logging
import os
from io import StringIO
from subprocess import PIPE, run
from typing import List, Tuple

import requests
import streamlit as st
from dotenv import load_dotenv

from uuid import uuid4
from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential

###############################################################################
# ── Environment helpers ──────────────────────────────────────────────────────
###############################################################################

def _load_dotenv_from_azd() -> None:
    """Populate ``os.environ`` either from *azd* CLI or fallback to *.env*."""

    result = run("azd env get-values", stdout=PIPE, stderr=PIPE, shell=True, text=True)
    if result.returncode == 0:
        logging.info("Found AZD environment. Loading variables from current azd env …")
        load_dotenv(stream=StringIO(result.stdout))
    else:
        logging.info("No azd context detected – falling back to .env file …")
        load_dotenv()


def _get_principal_id() -> str:
    """Return Azure AD object ID if app is fronted by ACA auth, else dummy."""

    return st.context.headers.get("x-ms-client-principal-id", "default_user_id")


def _get_principal_display_name() -> str:
    """Pretty name for the sidebar welcome message."""

    principal = st.context.headers.get("x-ms-client-principal")
    if principal:
        decoded = json.loads(base64.b64decode(principal).decode("utf-8"))
        for claim in decoded.get("claims", []):
            if claim["typ"] == "name":
                return claim["val"]
    return "Default User"


def _require_backend_url() -> str:
    url = os.getenv("BACKEND_URL")
    if not url:
        st.error("BACKEND_URL environment variable is not set – cannot continue.")
        st.stop()
    return url.rstrip("/")

def _get_blob_client() -> BlobServiceClient:
    return BlobServiceClient(
        account_url=os.getenv("BLOB_ACCOUNT_URL"),
        credential=DefaultAzureCredential(),
    )

def _upload_files_to_blob(files: list[st.runtime.uploaded_file_manager.UploadedFile],
                         request_id: str) -> dict[str, str]:
    """
    Streams files to the container at `<request_id>/<filename>`
    and returns {filename: blob_url}.
    """
    container = os.getenv("BLOB_CONTAINER")
    bc = _get_blob_client().get_container_client(container)

    urls = {}
    for file in files:
        blob_path = f"{request_id}/{file.name}"
        bc.upload_blob(blob_path, file, overwrite=True)
        urls[file.name] = f"{bc.url}/{blob_path}"

    return urls

###############################################################################
# ── Backend integration stubs ────────────────────────────────────────────────
###############################################################################

_BACKEND_URL = None  # filled at runtime in main()


def fetch_prospects(user_id: str) -> List[dict]:
    """Retrieve list of mortgage requests associated with *user_id*.

    Expected backend endpoint: ``POST <BACKEND_URL>/prospects`` returning
    ``[{ "request_id": "…", "full_name": "…", "status": "…", "created": "…" }, …]``
    """

    try:
        resp = requests.post(f"{_BACKEND_URL}/prospects", json={"user_id": user_id}, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        return json.loads(data) if isinstance(data, str) else data
    except requests.RequestException as exc:
        st.error(f"Error fetching prospects from backend: {exc}")
        return []


def submit_mortgage_request(form_data: dict, files: List[Tuple[str, bytes]], user_id: str) -> dict | None:
    """Send a new mortgage dossier to the backend.

    *form_data*  – dict with numeric/text fields.
    *files*      – list of (filename, raw_bytes).
    Returns backend JSON or *None* on error.
    """

    # Base64-encode docs so we can send everything as JSON.
    encoded_docs = [
        {"filename": name, "b64": base64.b64encode(content).decode()} for name, content in files
    ]
    payload = {
        "user_id": user_id,
        "request_data": form_data,
        "documents": encoded_docs,
    }

    try:
        resp = requests.post(f"{_BACKEND_URL}/mortgage_requests", json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return json.loads(data) if isinstance(data, str) else data
    except requests.RequestException as exc:
        st.error(f"Failed to submit request: {exc}")
        return None


def fetch_request_detail(request_id: str, user_id: str) -> dict | None:
    """Fetch a single prospect’s detail/decision payload."""
    try:
        resp = requests.get(
            f"{_BACKEND_URL}/mortgage_requests/{request_id}", params={"user_id": user_id}, timeout=15
        )
        resp.raise_for_status()
        data = resp.json()
        return json.loads(data) if isinstance(data, str) else data
    except requests.RequestException as exc:
        st.error(f"Could not retrieve request {request_id}: {exc}")
        return None


###############################################################################
# ── UI fragments ─────────────────────────────────────────────────────────────
###############################################################################

def prospect_list_ui() -> None:
    """Render the table of existing prospects."""

    prospects: List[dict] = st.session_state.prospects

    st.markdown("<h2>Mortgage Prospects</h2>", unsafe_allow_html=True)

    if not prospects:
        st.info("No prospects found – submit a new request using the *New* button above.")
        return

    header_cols = st.columns([2, 3, 2, 3, 2])
    header_cols[0].markdown("**Request ID**")
    header_cols[1].markdown("**Full Name**")
    header_cols[2].markdown("**Created**")
    header_cols[3].markdown("**Status**")
    header_cols[4].markdown("**Action**")

    for i, p in enumerate(prospects):
        row = st.columns([2, 3, 2, 3, 2])
        row[0].write(p.get("request_id", "—"))
        row[1].write(p.get("full_name", "—"))
        row[2].write(p.get("created", "—"))
        row[3].write(p.get("status", "—"))
        if row[4].button("Show", key=f"show_{i}"):
            st.session_state.selected_request_id = p["request_id"]
            st.session_state.view = "detail"
            st.rerun()


def prospect_detail_ui() -> None:
    """Show extracted data, policy result, and documents for a single request."""

    request_id: str | None = st.session_state.selected_request_id
    if request_id is None:
        st.warning("No prospect selected.")
        return

    data = fetch_request_detail(request_id, _get_principal_id())
    if not data:
        st.error("No data returned from backend.")
        return

    st.markdown(f"### Request ID: **{request_id}** – Status: **{data.get('status', '—')}**")

    st.subheader("Decision & Metrics")
    st.markdown(data.get("decision_markdown", "_No decision available yet…_"))

    with st.expander("🔍 Extracted Fields"):
        st.json(data.get("extracted", {}), expanded=False)

    with st.expander("📎 Uploaded Documents"):
        docs = data.get("documents", [])
        for d in docs:
            st.write(f"• {d.get('filename')}")
            if st.button("Download", key=f"dl_{d['filename']}"):
                # Provide a link or binary download once backend supports it.
                st.info("Download functionality to be implemented.")


###############################################################################
# ── Form to create a new request ─────────────────────────────────────────────
###############################################################################

def new_request_form_ui() -> None:
    st.header("Submit a Mortgage Request")

    # ─ Basic financial inputs ───────────────────────────────────────────────
    col1, col2 = st.columns(2)
    property_value = col1.number_input(
        "Property value (CHF)", min_value=0.0, step=1000.0, key="prop_val"
    )
    income = col2.number_input(
        "Net yearly income (CHF)", min_value=0.0, step=1000.0, key="income"
    )

    liabilities = col1.number_input(
        "Yearly liabilities (CHF)", min_value=0.0, step=100.0, key="liabilities"
    )
    down_payment_pc = col2.slider(
        "Down‑payment (%)", 0.0, 100.0, 20.0, key="down_pm_pc"
    )

    # ─ Pillar pledge widgets (dynamic) ─────────────────────────────────────
    pledge_p2 = st.checkbox("Pledge 2nd pillar", key="pledge2")
    pledge_p2_amt = 0.0
    if pledge_p2:
        pledge_p2_amt = st.number_input(
            "Amount to pledge for 2nd pillar (CHF)",
            min_value=0.0,
            step=5000.0,
            format="%.2f",
            key="pledge2_amt",
        )

    pledge_p3 = st.checkbox("Pledge 3rd pillar", key="pledge3")
    pledge_p3_amt = 0.0
    if pledge_p3:
        pledge_p3_amt = st.number_input(
            "Amount to pledge for 3rd pillar (CHF)",
            min_value=0.0,
            step=5000.0,
            format="%.2f",
            key="pledge3_amt",
        )

    # ─ File uploader ────────────────────────────────────────────────────────
    files = st.file_uploader(
        "Upload documents (PDF / JPG / PNG, multiple allowed)",
        accept_multiple_files=True,
        type=["pdf", "jpg", "jpeg", "png"],
        key="support_docs",
    )

    # ─ Submit button ────────────────────────────────────────────────────────
    if st.button("Submit request"):
        if not files:
            st.error("Please upload at least one document.")
            return

        request_id = str(uuid4())

        with st.spinner("Uploading documents …"):
            try:
                urls = _upload_files_to_blob(files, request_id)
            except Exception as ex:
                st.error(f"Upload failed: {ex}")
                return

        payload = {
            "user_id": _get_principal_id(),
            "request_id": request_id,
            "request_data": {
                "property_value": property_value,
                "income": income,
                "liabilities": liabilities,
                "down_payment_pc": down_payment_pc,
                "pledge_p2": pledge_p2,
                "pledge_p2_amount": pledge_p2_amt,
                "pledge_p3": pledge_p3,
                "pledge_p3_amount": pledge_p3_amt,
            },
            "documents": urls,  # blob URLs only
        }

        try:
            resp = requests.post(
                os.getenv("BACKEND_URL") + "/mortgage_requests",
                json=payload,
                timeout=30,
            )
            resp.raise_for_status()
            st.success("Request submitted successfully!")
            st.balloons()
            # Switch back to list view
            st.session_state.view = "list"
        except requests.RequestException as e:
            st.error(f"Backend error: {e}")


###############################################################################
# ── Main entrypoint ──────────────────────────────────────────────────────────
###############################################################################

def _init_session_state() -> None:
    if "view" not in st.session_state:
        st.session_state.view = "list"
    if "prospects" not in st.session_state:
        st.session_state.prospects = fetch_prospects(_get_principal_id())
    if "selected_request_id" not in st.session_state:
        st.session_state.selected_request_id = None


def main() -> None:
    global _BACKEND_URL  # noqa: WPS420, pylint: disable=global-statement

    _load_dotenv_from_azd()
    _BACKEND_URL = _require_backend_url()

    st.set_page_config(page_title="Mortgage Origination – Agentic", layout="wide")
    st.title("🏡 Mortgage Origination – Agentic Frontend")

    # ─ Sidebar ──────────────────────────────────────────────────────────────
    st.sidebar.write(f"Logged-in as: {_get_principal_display_name()}")
    if st.sidebar.button("🔄 Refresh list"):
        st.session_state.prospects = fetch_prospects(_get_principal_id())
    if st.sidebar.button("➕ New request"):
        st.session_state.view = "new"

    st.sidebar.markdown('<a href="/.auth/logout" target="_self">Sign Out</a>', unsafe_allow_html=True)

    _init_session_state()

    # ─ Routing ──────────────────────────────────────────────────────────────
    view = st.session_state.view
    if view == "list":
        prospect_list_ui()
    elif view == "detail":
        prospect_detail_ui()
    elif view == "new":
        new_request_form_ui()
    else:
        st.error(f"Unknown view: {view}")


if __name__ == "__main__":
    main()
