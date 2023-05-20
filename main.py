import os
import requests
import json
import re
from langchain.schema import AgentAction, AgentFinish
from typing import List, Tuple, Any, Union
from langchain.agents import LLMSingleActionAgent
from langchain.llms import OpenAI,LanguageModel
from langchain.agents import AgentExecutor



#my_secret = os.environ['OPENAI_API_KEY']

# First, let's load the language model we're going to use to control the agent.
llms = OpenAI(openai_api_key='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', temperature=0)
#llm = OpenAI(temperature=0)

def call_external_api(params: str):
    # Parse the input string to extract rss_feed and email
    rss_feed = re.search(r'rss_feed=(.*?)&', params).group(1)
    email = re.search(r'email=(.*)', params).group(1)

    url = f"https://streamlinedpodcasts.com/version-test/api/1.1/wf/coach?rss_feed={rss_feed}&email={email}"
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",             
    }
    response = requests.post(url, headers=headers)

    if response.status_code in [200, 201]:
        return response.json()
    else:
        print(f"Response content: {response.content}")
        raise Exception(f"Request failed with status code {response.status_code}")

class Tool:
    def __init__(self, name, description, func):
        self.name = name
        self.description = description
        self.func = func

custom_tool = Tool(
    name="Custom API",
    description="This tool is used to analyze the audio quality of a podcast. To use this tool, first ask the user for the rss_feed and the user's email. Do not use the example parameters. Stop the chain and ask the user for the rss_feed and email.",
    func=call_external_api
)

tools = [custom_tool]

class CustomAgent(LLMSingleActionAgent):
    """Custom Agent."""

    def __init__(self, llms: List[LanguageModel]):
        # Call the parent constructor
        super().__init__(llms=llms)

        # Initialize class variables
        self.intermediate_steps = []
        self.llm_chain = [llm for llm in llms if isinstance(llm, LanguageModel)]

    @property
    def required_features(self):
        return

    def plan(
        self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any
    ) -> Union[List[AgentAction], AgentFinish]:
        """Given input, decide what to do.

        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with observations
            **kwargs: User inputs.

        Returns:
            Action specifying what tool to use.
        """
        if len(intermediate_steps) == 0:
            return [
                AgentAction(tool="Custom API", tool_input=kwargs["input"], log=""),
            ]
        else:
            rss_feed = input("Please enter the RSS feed of your podcast: ")
            email = input("Please enter your email address: ")
            return [
                AgentAction(tool="Custom API", tool_input=f"rss_feed={rss_feed}&email={email}", log=""),
            ]

# Finally, let's initialize an agent with the tools, the language model, and the type of agent we want to use.
agent_executor = AgentExecutor.from_agent_and_tools(CustomAgent(llms=[llms]), tools=tools, verbose=True)

# Run the agent
result = agent_executor.run(f"Please analyze the audio quality of my podcast. If the response is status success, then simply tell the user that the results will be emailed to them.")

print(result)
