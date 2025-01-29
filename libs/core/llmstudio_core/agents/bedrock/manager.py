import os
from typing import Optional

import openai
from llmstudio_core.exceptions import AgentError
from llmstudio_core.agents.manager import AgentManager, agent_manager
from llmstudio_core.agents.bedrock.data_models import BedrockAgent, BedrockRun, BedrockResult
import boto3

SERVICE= 'bedrock-agent'

@agent_manager
class BedrockAgentManager(AgentManager):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._client = boto3.client(
            SERVICE,
            region_name=self.region if self.region else os.getenv("BEDROCK_REGION"),
            aws_access_key_id=self.access_key
            if self.access_key
            else os.getenv("BEDROCK_ACCESS_KEY"),
            aws_secret_access_key=self.secret_key
            if self.secret_key
            else os.getenv("BEDROCK_SECRET_KEY"),
        )


    @staticmethod
    def _provider_config_name():
        return "bedrock"
    
    def _validate_create_request(self, request):
        raise NotImplementedError("Agents need to implement the method")
    
    def _validate_run_request(self, request):
        raise NotImplementedError("Agents need to implement the method")
    
    def _validate_create_request(self, request):
        raise NotImplementedError("Agents need to implement the method")

    
    def create_agent(self, **kargs) -> BedrockAgent:
        """
        Creates a new instance of the agent.
        """

        raise NotImplementedError("Agents need to implement the 'create' method.")
    
    def run_agent(self, **kwargs) -> BedrockRun:
        """
        Runs the agent
        """
        raise NotImplementedError("Agents need to implement the 'create_thread_and_run' method.")
   
    
    def retrieve_result(self, **kwargs) -> BedrockResult:
        """
        Retrieves an existing agent.
        """
        raise NotImplementedError("Agents need to implement the 'retrieve' method.")
    