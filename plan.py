from app.agent.planning import PlanningAgent
from app.tool import all_tools

async def main():
    request = 'I want produce a state of the art agent system for video understanding. Can you generate a comprehensive list of 30 tools in table format (name, description, sample input, sample out input, implementation suggestion) for me to implement the tool call to the agent with ease? Thanks'
    agent = PlanningAgent(available_tools=all_tools)
    response = await agent.run(request)
    print(response)

if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
