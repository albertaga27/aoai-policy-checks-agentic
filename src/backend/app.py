"""
FastAPI backend application using AI orchestration.

This module initializes a FastAPI application that exposes endpoints for...
"""
import json
import logging
import os
from fastapi import FastAPI, Body, HTTPException
from fastapi.responses import JSONResponse

from patterns.kyc_team import KYCTeamOrchestrator

from azure.identity import DefaultAzureCredential  

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


orchestrator = KYCTeamOrchestrator()

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
        async for item in orchestrator.process_conversation(user_id, conversation_messages):
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
