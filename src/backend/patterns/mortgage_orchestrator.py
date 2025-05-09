"""Backend *Agentic Orchestrator* for mortgage‑origination requests.

This module is framework‑agnostic – you can import it from a FastAPI route,
Azure Functions HTTP trigger, or a Service Bus trigger.  All orchestration
state lives in **Azure Cosmos DB**; heavy binaries (PDFs, images) are assumed to
be stored by the caller in **Azure Blob Storage** with their blob URLs included
in the *request_payload*.

"""

from __future__ import annotations

import datetime as _dt
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional, List
from fastapi import APIRouter, UploadFile, File, BackgroundTasks
from pydantic import BaseModel, Field, constr
from dataclasses import asdict
from typing import Dict
import threading     
import logging
from datetime import datetime as dt 

from azure.cosmos import CosmosClient, PartitionKey, exceptions  # type: ignore
import azure.ai.inference.aio as aio_inference
import azure.identity.aio as aio_identity
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.identity import DefaultAzureCredential  # type: ignore
from azure.storage.blob import BlobClient

import asyncio, re, requests

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:   %(name)s   %(message)s',
)
logger = logging.getLogger(__name__)

###############################################################################
# ── Models & helpers ─────────────────────────────────────────────────────────
###############################################################################

RequestStatus = Literal[
    "RECEIVED",         # initial write from frontend
    "INGESTION",        # docs uploaded to Blob, ready for classification
    "CLASSIFIED",       # Classifier/Splitter finished; extracted JSON available
    "PROCESSING",       # downstream policy & registry agents running
    "COMPLETED",        # final decision written
    "FAILED",           # unrecoverable error – see ``error`` field
]


@dataclass
class MortgageRequest:
    request_id: str
    data: dict[str, Any]
    documents: dict[str, str]
    status: str = "RECEIVED"
    created_utc: str = field(default_factory=lambda: dt.utcnow().isoformat())
    updated_utc: str = field(default_factory=lambda: dt.utcnow().isoformat())
    error: str | None = None
    classification: dict[str, Any] | None = None
    decision: dict[str, Any] | None = None


###############################################################################
# ── Orchestrator implementation ──────────────────────────────────────────────
###############################################################################

class Orchestrator:
    """Lightweight orchestrator that persists state & invokes specialised agents."""

    def __init__(self) -> None:
        # ─ CosmosDB ───────────────────────────────────────────────────────────
        endpoint = os.environ["COSMOSDB_ENDPOINT"]
        #key = os.environ.get("COSMOS_KEY")  # if None, MSI/AAD auth is attempted
        credential = DefaultAzureCredential()
        self.cosmos = CosmosClient(endpoint, credential)  # type: ignore[arg-type]

        db_name = os.environ.get("COSMOSDB_DATABASE_NAME")
        container_name = os.environ.get("COSMOSDB_CONTAINER_MORTGAGE_NAME")

        self.db = self.cosmos.create_database_if_not_exists(db_name)
        self.container = self.db.create_container_if_not_exists(
            id=container_name, partition_key=PartitionKey(path="/user_id"), offer_throughput=400
        )

        # ─ OpenAI client (will be used by sub‑agents) ─────────────────────────
        endpoint_name = os.getenv("AZURE_OPENAI_ENDPOINT")
        deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        credential = aio_identity.DefaultAzureCredential()
        llm_client = aio_inference.ChatCompletionsClient(
            endpoint=f"{endpoint_name.strip('/')}/openai/deployments/{deployment_name}",
            credential=credential,
            credential_scopes=["https://cognitiveservices.azure.com/.default"],
        )

        self._openai_client = llm_client
        self._deployment = os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"]


    # ─ Low‑level persistence helpers ─────────────────────────────────────────

    def _load_user_doc(self, user_id: str) -> dict[str, Any]:
        """
        Fetch the user document. If it does not exist, create an empty shell that
        *already* contains user_id so the partition‑key is valid.
        """
        try:
            query = "SELECT * FROM c WHERE c.id=@userId"
            parameters = [{"name": "@userId", "value": user_id}]
            items = list(self.container.query_items(
                query=query,
                parameters=parameters,
                enable_cross_partition_query=True
            ))

            if items:
                item_dict = items[0]
                # values_dict = {key : value for key, value in item_dict.items() if not key.startswith('_')}
                return item_dict
            else:
                logging.error(f"Warining, this user has no requests saved yet, creating basic structure...")
                doc = {
                    "id": user_id,
                    "user_id": user_id,        # PK field
                    "requests": [],
                    "created_utc": dt.utcnow().isoformat(),
                    "updated_utc": dt.utcnow().isoformat(),
                }

                # upsert_item(item, partition_key)  ← positional
                self.container.upsert_item(doc, user_id)
                return doc
        except Exception as e:
            logging.error(f"_load_user_doc> An error occurred: {e}")
            return None
        
    def _save_user_doc(self, doc: dict[str, Any]) -> None:
        """Upsert with the correct partition key every time."""
        doc["updated_utc"] = dt.utcnow().isoformat()
        self.container.upsert_item(doc, doc["user_id"])   # positional


    def _get_request_ref(self, user_doc: dict[str, Any], request_id: str) -> dict[str, Any]:
        """Return *reference* to the dict inside the list so edits mutate in‑place."""
        for req in user_doc["requests"]:
            if req["request_id"] == request_id:
                return req
        raise ValueError(f"Request {request_id} not found for user {user_doc['id']}")
    


    def _to_blob_path(url: str) -> str:
        _BLOB_PATH_RE = re.compile(r"https?://[^/]+/(.+)")
        m = _BLOB_PATH_RE.match(url)
        if not m:
            raise ValueError(f"Unexpected blob URL: {url!r}")
        return m.group(1)


    async def _extract_doc(
        self,
        blob_path: str,
        dataset_type: str = "default-dataset",
    ) -> Dict[str, Any]:
        """
        Calls the extraction micro-service using `requests.post` but inside
        `asyncio.to_thread` so our coroutine doesn't block the event loop.
        """

        def _post() -> Dict[str, Any]:
            BACKEND_ARGUS_URL = os.environ.get("BACKEND_ARGUS_URL")
            resp = requests.post(
                f"{BACKEND_ARGUS_URL.rstrip('/')}/process",
                json={"blob_path": blob_path, "dataset_type": dataset_type},
                timeout=60,
            )
            resp.raise_for_status()
            return resp.json()

        return await asyncio.to_thread(_post)


    # ─ Public API ────────────────────────────────────────────────────────────

    def create_request(
        self,
        user_id: str,
        form_data: dict,
        documents: dict[str, str],
        request_id: str | None = None,
    ) -> str:
        
        logging.info(f"Creating new request {request_id}, for user_id={user_id}...")
        request_id = request_id

        new_req = MortgageRequest(
            request_id=request_id,
            data=form_data,
            documents=documents,
        )

        user_doc = self._load_user_doc(user_id)
        logging.info(f"Appending new request {request_id}, to user_id={user_id} structure...")
        user_doc["requests"].append(asdict(new_req))     # append once
        self._save_user_doc(user_doc)                    # persist once
        return request_id
    
        
    async def run_ingestion_pipeline(self, request_id: str, *, user_id: str) -> None:
        """Drive the full pipeline for *request_id* synchronously.

        In a production system you would likely break this across Durable
        Functions with activities/fan‑out, but keeping it linear here makes the
        control‑flow easy to read.

        """
        logging.info(f"run_ingestion_pipeline> request_id={request_id}, user_id={user_id}")
        user_doc = self._load_user_doc(user_id)
        req_dict = self._get_request_ref(user_doc, request_id)

        # status: INGESTION
        req_dict["status"] = "INGESTION"
        self._save_user_doc(user_doc)

        # turn dict -> dataclass so the agent logic stays unchanged
        req_obj = MortgageRequest(**req_dict)

        # ─ Classifier Agent ───────────────────────────────────────────
        try:
            req_dict["classification"] = await self._run_classifier_agent(req_obj)
            req_dict["status"] = "CLASSIFIED"
            self._save_user_doc(user_doc)
        except Exception as exc:
            req_dict["status"] = "FAILED"
            req_dict["error"] = str(exc)
            self._save_user_doc(user_doc)
            raise

        # ─ Document-extraction Agent ───────────────────────────────────────────
        req_dict["status"] = "PROCESSING"
        self._save_user_doc(user_doc)

        req_dict["extracted"] = await self._run_doc_extraction_agent(
            docs=req_dict["documents"],
            classification=req_dict["classification"],
        )
        self._save_user_doc(user_doc)

        time.sleep(3)
        req_dict["decision"] = {"approved": False, "reason": "Down‑payment < 20 %"}
        req_dict["status"] = "COMPLETED"
        self._save_user_doc(user_doc)



    # ─ Agent invocation stubs ────────────────────────────────────────────────

    async def _run_classifier_agent(self, req: MortgageRequest) -> dict[str, Any]:
        """Naïve GPT‑4 call that classifies each document by type.

        Input: list of blob URLs.  For brevity we *don’t* download the file
        content – we just ask the model to reason on filenames.  Replace with
        Azure Document Intelligence for real OCR & page splitting.
        """
        system_prompt = (
            "You are a file‑routing assistant.  Based only on the filenames, map each"
            " document to one of: identity, salary_statement, tax_return, property_valuation,"
            " other. Reply with valid JSON: {\"classification\": {<filename>: <label>, …}}."
        )

        user_prompt = json.dumps({"filenames": list(req.documents.keys())})

        response = await self._openai_client.complete(
                messages=[
                    SystemMessage(content=system_prompt),
                    UserMessage(content=user_prompt),
                ]
        )
        content = response.choices[0].message.content  # type: ignore[index]
        try:
            return json.loads(content)
        except json.JSONDecodeError:  # model hallucinated; mark failure
            raise RuntimeError(f"Classifier returned non‑JSON: {content}")
        

    async def _run_doc_extraction_agent(
        self,
        docs: Dict[str, str],
        classification: Dict[str, Any],
    ) -> Dict[str, Any]:
        tasks = []
        for fname, url in docs.items():
            blob_path = self._to_blob_path(url)
            # TODO: set dataset_type based on classification if needed
            tasks.append(self._extract_one(blob_path, "default-dataset"))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        extracted = {}
        for (fname, _), res in zip(docs.items(), results):
            if isinstance(res, Exception):
                logging.error("Extraction failed for %s: %s", fname, res)
                extracted[fname] = {"error": str(res)}
            else:
                extracted[fname] = res
        return extracted


class Document(BaseModel):
    filename: str
    b64: str = Field(..., description="Base-64 content of the file")


class RequestData(BaseModel):
   
    property_value: float              # CHF
    income: float                      # yearly net CHF
    liabilities: float                 # yearly CHF
    down_payment_pc: float             # e.g. 20.0 for 20 %
    # pillar pledges
    pledge_p2: bool
    pledge_p2_amount: float = Field(0.0, ge=0)
    pledge_p3: bool
    pledge_p3_amount: float = Field(0.0, ge=0)

    
class RequestPayload(BaseModel):
    user_id: str
    request_id: str | None = Field(
        None,
        description="Optional GUID created client‑side; server will generate if omitted",
    )

    request_data: RequestData                # ← now **typed**, no loose Dict
    documents: Dict[str, str]                # {"salary.pdf": "https://…", …}

from uuid import uuid4
import base64


# ─────────────────────────────────────────────────────────────────────────────
# Helper: download a blob from its URL into memory
# ─────────────────────────────────────────────────────────────────────────────
def _download_blob_to_bytes(blob_url: str) -> bytes:
    """
    Uses the URL (with HTTPS scheme) that Streamlit gave us.
    Works whether or not the URL already contains an SAS token.
    """
    
    blob_client = BlobClient.from_blob_url(blob_url, credential=DefaultAzureCredential())
    return blob_client.download_blob().readall()
    