import json
import os
import logging

from typing import Annotated
from semantic_kernel.functions import kernel_function
from openai import AzureOpenAI

class KYCPolicyEvaulator:

    def __init__(self, api_key, api_version, azure_endpoint, azure_deployment):
        self.aoai_client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=azure_endpoint,
            azure_deployment=azure_deployment)


    @kernel_function(
        name="evaluate_kyc_policy",
        description="Evaluate a client or prospect data profile against a specified policy")
    def evaluate_kyc_policy(self, client_profile: Annotated[str,"The client or prospect data json"]) -> Annotated[str, "The output is a json object"]:
        current_path = os.getcwd()
        file_path = os.path.join(current_path, 'agents/kyc_review/individual_policy.txt')
        business_logic = ""
        with open(file_path, 'r') as file:
            business_logic = file.read()
        
        prompt_template = """
        Your goal is to analyze and compare client/prospect data represented in json format against a policy document to verify compliancy.
        Read carefully the policy: {policy}.
        Analyze the data of the client or prospect: {client_profile}.
        Provide as your response a json output that contains the analysis of the client profile data against the policy, in particular the
        json output should contains fields like, required documents, missing documents (if any), unclear data or partial data and
        and a 'kyc_policy_evaluation_status' field containing the result of the analisys:
        - 'cleared' if all the required documents and fields are present in the client profile according to what the policy states;
        - 'missing_info' if one or more data field or documents are missing in the client profile compared to what the policy states;
        """
         
        prompt = prompt_template.replace("{policy}",str(business_logic)).replace("{client_profile}",str(client_profile))
     
        response = self.aoai_client .chat.completions.create(
            model=os.getenv("AOAI_OPENAI_DEPLOYMENT_NAME"),
            messages=[{'role': 'user', 'content': prompt}]
        )
        
        print(f"📟 Response evaluate_kyc_policy: {response.choices[0].message.content}")
        return response.choices[0].message.content if response else None


    