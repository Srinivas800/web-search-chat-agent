import os
import datetime
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from memory import ChronologicalMemory

class WebSearchChatAgent:
    def __init__(self, api_key: str = None):
        self.search_tool = TavilySearchResults(max_results=3)
        self.tools = [self.search_tool]
        
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, streaming=True)
        
        self.memory = ChronologicalMemory(window_size=10)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", 
             "You are a fast, factual, and helpful AI assistant equipped with ONE tool: web_search.\n"
             "Current UTC Time: {current_time}\n\n"
             "RULES:\n"
             "1. If answering requires fresh facts, recent events, or verification, ALWAYS use web_search.\n"
             "2. Provide direct, concise answers with inline source citations format:.\n"
             "3. Remember what the user asked and when based on conversation history."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        agent = create_openai_functions_agent(self.llm, self.tools, self.prompt)
        self.executor = AgentExecutor(agent=agent, tools=self.tools, verbose=False)

    def chat(self, user_input: str) -> str:
        now_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        history_messages = self.memory.get_messages()
        
        response = self.executor.invoke({
            "input": user_input,
            "current_time": now_str,
            "chat_history": history_messages
        })
        
        output_text = response["output"]
        
        self.memory.add_user_message(user_input, timestamp=now_str)
        self.memory.add_ai_message(output_text, timestamp=now_str)
        
        return output_text

if __name__ == "__main__":
    agent = WebSearchChatAgent()
    print(agent.chat("What were the top tech headlines today?"))