import json
import os
import logging

from typing import Annotated
from semantic_kernel.functions import kernel_function


class CRMFacade:
    """ 
    The class acts as an facade for the crm_store.

    Reads a mock client profile from file rather than implmeenting a full access CRUD to a database or a CRM sytem

    """

    @kernel_function(
        name="load_from_crm_by_client_fullname",
        description="Load insured client data from the CRM from the given full name")
    def get_customer_profile_by_full_name(self,
                                          full_name: Annotated[str,"The customer full name to search for"]) -> Annotated[str, "The output is a customer profile"]:
        
        current_path = os.getcwd()
        file_path = os.path.join(current_path, 'agents/kyc_review/client_profile.json')
        client_profile = ""
        with open(file_path, 'r') as file:
            client_profile = file.read()

        return client_profile
        
