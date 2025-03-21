import os
import json
import logging
from typing import ClassVar
import datetime

from semantic_kernel.kernel import Kernel
from semantic_kernel.agents import AgentGroupChat
from semantic_kernel.agents.strategies.termination.termination_strategy import TerminationStrategy
from semantic_kernel.agents.strategies import KernelFunctionSelectionStrategy
from semantic_kernel.connectors.ai.open_ai import AzureChatPromptExecutionSettings

from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.contents.utils.author_role import AuthorRole
from semantic_kernel.core_plugins.time_plugin import TimePlugin
from semantic_kernel.functions import KernelPlugin, KernelFunctionFromPrompt, KernelArguments

from semantic_kernel.connectors.ai.azure_ai_inference import AzureAIInferenceChatCompletion
from azure.ai.inference.aio import ChatCompletionsClient
from azure.identity import DefaultAzureCredential  

from opentelemetry.trace import get_tracer

from pydantic import Field
from utils.util import create_agent_from_yaml
from agents.skills.crm_facade import CRMFacade
from agents.skills.policy_evaluator import KYCPolicyEvaulator


class KYCTeamOrchestrator:
    """
    Orchestrates a KYC semi-fixed flow to support SoW corroboration checks
    """
    
    # --------------------------------------------
    # Constructor
    # --------------------------------------------
    def __init__(self):
        """
        Creates the orchestrator with necessary services and kernel configurations.
        
        Sets up Azure OpenAI connections for both executor and utility models, 
        configures Semantic Kernel, and prepares execution settings for the agents.
        """
        
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.info("Semantic Orchestrator Handler init")

        credential = DefaultAzureCredential()

        self.crm = CRMFacade()

        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION")
        executor_deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        utility_deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        aoai_api_key=os.getenv("AZURE_OPENAI_API_KEY")

        self.kyc_policy = KYCPolicyEvaulator(
            api_key=aoai_api_key,
            api_version=api_version,
            azure_endpoint=endpoint,
            azure_deployment=executor_deployment_name
        )
        
        
        
        # Multi model setup - a service is an LLM in SK terms
        # Executor - gpt-4o 
        # Utility  - gpt-4o-mini
        executor_service = AzureAIInferenceChatCompletion(
            ai_model_id="executor",
            service_id="executor",
            client=ChatCompletionsClient(
                endpoint=f"{str(endpoint).strip('/')}/openai/deployments/{executor_deployment_name}",
                api_version=api_version,
                credential=credential,
                credential_scopes=["https://cognitiveservices.azure.com/.default"],
            ))
        
        utility_service = AzureAIInferenceChatCompletion(
            ai_model_id="utility",
            service_id="utility",
            client=ChatCompletionsClient(
                endpoint=f"{str(endpoint).strip('/')}/openai/deployments/{utility_deployment_name}",
                api_version=api_version,
                credential=credential,
                credential_scopes=["https://cognitiveservices.azure.com/.default"],
            ))
        
        self.kernel = Kernel(
            services=[executor_service, utility_service],
            plugins=[
                KernelPlugin.from_object(plugin_instance=self.crm, plugin_name="crm"),
                KernelPlugin.from_object(plugin_instance=self.kyc_policy, plugin_name="kyc_policy"),

            ])
        
        self.settings_executor = AzureChatPromptExecutionSettings(service_id="executor", temperature=0)
        self.settings_utility = AzureChatPromptExecutionSettings(service_id="utility", temperature=0)
        
        #self.resourceGroup = os.getenv("AZURE_RESOURCE_GROUP")

    # --------------------------------------------
    # Create Agent Group Chat
    # --------------------------------------------
    def create_agent_group_chat(self):
        """
        Create the KYC Agent Team chat
        
        Returns:
            AgentGroupChat: A configured group chat with specialized agents, 
                           selection strategy and termination strategy.
        """
        
        self.logger.debug("Creating Agents gropup's chat")
        
        crm_agent = create_agent_from_yaml(service_id="executor",
                                        kernel=self.kernel,
                                        definition_file_path="agents/kyc_review/client.yaml")
        kyc_policy_agent = create_agent_from_yaml(service_id="executor",
                                        kernel=self.kernel,
                                        definition_file_path="agents/kyc_review/kyc_policy.yaml")
        responder_agent = create_agent_from_yaml(service_id="executor",
                                      kernel=self.kernel,
                                      definition_file_path="agents/kyc_review/responder.yaml")
        agents=[crm_agent, kyc_policy_agent, responder_agent]

        agent_group_chat = AgentGroupChat(
                agents=agents,
                selection_strategy=self.create_selection_strategy(agents, crm_agent),
                termination_strategy = self.create_termination_strategy(
                                         agents,
                                         responder_agent,
                                         maximum_iterations=3))

        return agent_group_chat
        
    # --------------------------------------------
    # Run the agent conversation
    # --------------------------------------------
    async def process_conversation(self, user_id, conversation_messages):
        """
        Processes a request containing client current snapshot and orchestrate a partial KYC review sub step
        focusing on SoW and SoF checks
        
        Manages the entire flow, from initializing the agent group chat to
        collecting and returning responses. Uses OpenTelemetry for tracing.
        
        Args:
            user_id: Unique identifier for the user, used in session tracking.
            conversation_messages: List of dictionaries with role, name and content
                                  representing the conversation history.
                                  
        Yields:
            Status updates during processing and the final response in JSON format.
        """
        
        agent_group_chat = self.create_agent_group_chat()
       
        # Load chat history
        chat_history = [
            ChatMessageContent(
                role=AuthorRole(d.get('role')),
                name=d.get('name'),
                content=d.get('content')
            ) for d in filter(lambda m: m['role'] in ("assistant", "user"), conversation_messages)
        ]

        await agent_group_chat.add_chat_messages(chat_history)

        tracer = get_tracer(__name__)
        
        # UNIQUE SESSION ID is a must for AI Foundry Tracing
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        session_id = f"{user_id}-{current_time}"
        
        messages = []
        
        with tracer.start_as_current_span(session_id):
            async for a in agent_group_chat.invoke():
                self.logger.info("Agent: %s", a.to_dict())
                messages.append(a.to_dict())

        response = list(reversed([item async for item in agent_group_chat.get_chat_messages()]))

        # Last writer response
        reply = [r for r in response if r.name == "ResponderAgent"][-1].to_dict()
        
        # Final message is formatted as JSON to indicate the final response
        yield json.dumps(reply)
        
    # --------------------------------------------
    # Speaker Selection Strategy
    # --------------------------------------------
    # Using executor model since we need to process context - cognitive task
    def create_selection_strategy(self, agents, default_agent):
        """
        Creates a strategy to determine which agent speaks next in the conversation.
        
        Uses the executor model to analyze conversation context and select the most 
        appropriate next speaker based on the conversation history.
        
        Args:
            agents: List of available agents in the conversation.
            default_agent: The fallback agent to use if selection fails.
            
        Returns:
            KernelFunctionSelectionStrategy: A strategy for selecting the next speaker.
        """
        definitions = "\n".join([f"{agent.name}: {agent.description}" for agent in agents])
        
        selection_function = KernelFunctionFromPrompt(
                function_name="SpeakerSelector",
                prompt_execution_settings=self.settings_executor,
                prompt=fr"""
You are the next speaker selector.

- You MUST return ONLY agent name from the list of available agents below.
- You MUST return the agent name and nothing else.
- The agent names are case-sensitive and should not be abbreviated or changed.
- Check the history, and decide WHAT agent is the best next speaker
- YOU MUST OBSERVE AGENT USAGE INSTRUCTIONS.                    

# AVAILABLE AGENTS

{definitions}

# AGENTS INVOCATION SEQUENCE

Always follow these rules when selecting the next participant:
- After user input, it is ClientProfileAgent's turn.
- After ClientProfileAgent replies, it is KYCPolicyAgent's turn.
- After KYCPolicyAgent provides its analysis, it is ResponderAgent's turn.

Each agent should be invoked in that sequence and only once. In case you cannot select the approriate
agent or in case of any error invoke the ResponderAgent as failover instructing it to terminate the flow.

# CHAT HISTORY

{{{{$history}}}}
""")

        # Could be lambda. Keeping as function for clarity
        def parse_selection_output(output):
            self.logger.info("------- Speaker selected: %s", output)
            if output.value is not None:
                return output.value[0].content
            return default_agent.name

        return KernelFunctionSelectionStrategy(
                    kernel=self.kernel,
                    function=selection_function,
                    result_parser=parse_selection_output,
                    agent_variable_name="agents",
                    history_variable_name="history")

    # --------------------------------------------
    # Termination Strategy
    # --------------------------------------------
    def create_termination_strategy(self, agents, final_agent, maximum_iterations):
        """
        Create a chat termination strategy that terminates when the final agent is reached.
        params:
            agents: List of agents to trigger termination evaluation
            final_agent: The agent that should trigger termination
            maximum_iterations: Maximum number of iterations before termination
        """
        class CompletionTerminationStrategy(TerminationStrategy):
            async def should_agent_terminate(self, agent, history):
                """Terminate if the last actor is the Responder Agent."""
                logging.getLogger(__name__).debug(history[-1])
                return (agent.name == final_agent.name)

        return CompletionTerminationStrategy(agents=agents,
                                             maximum_iterations=maximum_iterations)