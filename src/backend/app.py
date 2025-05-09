"""
FastAPI backend application using AI orchestration.

This module initializes a FastAPI application that exposes endpoints for...
"""
import json
import logging
import os
from uuid import uuid4
import base64
from fastapi import FastAPI, Body, HTTPException
from fastapi.responses import JSONResponse

from patterns.kyc_team import KYCTeamOrchestrator
from patterns.mortgage_orchestrator import Orchestrator, RequestPayload, RequestData, Document, _download_blob_to_bytes

from typing import Any, Dict
from pydantic import BaseModel, Field
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobClient   

from utils.util import load_dotenv_from_azd, set_up_tracing, set_up_metrics, set_up_logging

load_dotenv_from_azd()
#set_up_tracing()
#set_up_metrics()
#set_up_logging()

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:   %(name)s   %(message)s',
)
logger = logging.getLogger(__name__)
logging.getLogger('azure.core.pipeline.policies.http_logging_policy').setLevel(logging.WARNING)
logging.getLogger('azure.monitor.opentelemetry.exporter.export').setLevel(logging.WARNING)


kyc_orchestrator = KYCTeamOrchestrator()

app = FastAPI()

logger.info("Diagnostics: %s", os.getenv('SEMANTICKERNEL_EXPERIMENTAL_GENAI_ENABLE_OTEL_DIAGNOSTICS'))

@app.post("/run_kyc_checks")
async def run_kyc_checks(request_body: dict = Body(...)):
    """
    Run the agentic KYC checks
    
    Args:
        request_body (dict): JSON body containing:
            - client data (str): client profile to run KYC check against
            - user_id (str): Identifier for the user making the request. Defaults to 'default_user'.
    
    Returns:
        the final response of the workflow run
    """
    logger.info('API request received with body %s', request_body)

    client_profile = request_body.get('client_data', '')
    user_id = request_body.get('user_id', 'kyc_default_user')
    content = client_profile

    conversation_messages = []
    conversation_messages.append({'role': 'user', 'name': 'user', 'content': content})

    try:
        reply_items = []
        async for item in kyc_orchestrator.process_conversation(user_id, conversation_messages):
            reply_items.append(item)

        return JSONResponse(
            content=reply_items,
            status_code=200
)
    except Exception as e:  
        logging.error(f"Error in agentic workflow: {e}")  
        raise HTTPException(status_code=400, detail="agent-error")  



@app.post("/clients")
def get_all_clients(request: dict = Body(...)):
    """
    Return all records from Cosmos DB whose clientID starts with 'PRO'.
    The request body must include a user_id for demonstration/authorization purposes.
    """
     
    logging.info('KYC Checks - <POST get_all_clients> triggered...')

    # Extract parameters from the request body  
    user_id = request.get('user_id')
    # Validate required parameters
    if not user_id:
        raise HTTPException(status_code=400, detail="<user_id> is required!")
   
    try:
        clients = []
        current_path = os.getcwd()
        file_path = os.path.join(current_path, 'agents/kyc_review/client_profile.json')
        client_profile = ""
        with open(file_path, 'r') as file:
            client_profile = file.read()

        clients.append(json.loads(client_profile))
        return json.dumps(clients) if clients else None

    except Exception as e:
        logging.error(f"Error in load_all_clients: {str(e)}")
        return json.dumps({"error": f"load_all_clients failed with error: {str(e)}"})
    



mortgage_orchestrator = Orchestrator()

@app.post("/mortgage_requests")
async def new_request(payload: RequestPayload):
    # 0. Resolve / create a request ID
    request_id = payload.request_id

    try:
        # 1. Retrieve every blob *now* so we have them locally for the first agent
        docs_bytes: Dict[str, bytes] = {
            fname: _download_blob_to_bytes(url)
            for fname, url in payload.documents.items()
        }

        # 2. Persist a *RECEIVED* row in Cosmos DB
        #    (the orchestrator takes care of container/database creation)
        mortgage_orchestrator.create_request(
            user_id=payload.user_id,
            form_data=payload.request_data.model_dump(),
            documents=payload.documents,
            request_id=request_id,        
        )

        # 3. Kick off the heavy ingestion pipeline in a background thread
        #    We pass the raw bytes so the classifier agent can work immediately.
        await mortgage_orchestrator.run_ingestion_pipeline(
            request_id=request_id,
            user_id= payload.user_id
        )

        # ───────── alternative patterns ─────────
        # • FastAPI's BackgroundTasks
        # • push (user_id, request_id) on Service Bus and let an Azure Function
        #   do mortgage_orchestrator.run_ingestion_pipeline

        # 4. Tell the caller we have the dossier
        return {"request_id": request_id, "status": "RECEIVED"}
    
    except Exception as e:  
        logging.error(f"Error in the mortgage agentic workflow: {e}")  
        raise HTTPException(status_code=400, detail=str(e))  

#TODO
#``POST <BACKEND_URL>/prospects`` returning
#    ``[{ "request_id": "…", "full_name": "…", "status": "…", "created": "…" }, …]``

#TODO
# ``POST <BACKEND_URL>/mortgage_requests/{request_id}", params={"user_id": user_id}