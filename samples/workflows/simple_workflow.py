# Copyright (c) Microsoft. All rights reserved.

import asyncio
import os

from agent_framework import ChatMessage, ChatRole
from agent_framework._agents import ChatClientAgent
from agent_framework.workflow import (
    AgentRunEvent,
    WorkflowBuilder,
    WorkflowCompletedEvent,
)
from agent_framework_foundry._chat_client import FoundryChatClient
from agent_framework_workflow._executor import (
    AgentExecutor,
    AgentExecutorRequest,
    AgentExecutorResponse,
    handler,
)
from agent_framework_workflow._workflow_context import WorkflowContext
from azure.ai.projects.aio import AIProjectClient
from azure.identity.aio import AzureCliCredential

"""
The following sample demonstrates a basic workflow with two executors
where one executor guesses a number and the other executor judges the
guess iteratively. This version uses AgentExecutor with specific 
instructions to implement the logic through natural language rather 
than hardcoded algorithms.
"""


class GuessAgentExecutor(AgentExecutor):
    """Custom AgentExecutor for the guessing agent that can handle both requests and responses."""

    @handler(output_types=[AgentExecutorResponse])
    async def handle_start_message(self, message: str, ctx: WorkflowContext) -> None:
        """Handle the initial start message and convert it to a request for the guesser."""

        chat_message = ChatMessage(ChatRole.USER, text=message)
        request = AgentExecutorRequest(messages=[chat_message], should_respond=True)
        await self.run(request, ctx)

    @handler(output_types=[AgentExecutorResponse])
    async def handle_judge_response(
        self, response: AgentExecutorResponse, ctx: WorkflowContext
    ) -> None:
        """Handle response from the judge and convert it to a request for the guesser."""

        messages = response.agent_run_response.messages
        if messages and messages[-1].text.lower().strip() == "correct":
            await ctx.add_event(
                WorkflowCompletedEvent(
                    data="Number guessing game completed successfully!"
                )
            )
            return

        request = AgentExecutorRequest(messages=messages, should_respond=True)
        await self.run(request, ctx)


class JudgeAgentExecutor(AgentExecutor):
    """Custom AgentExecutor for the judging agent that can handle both requests and responses."""

    @handler(output_types=[AgentExecutorResponse])
    async def handle_guess_response(
        self, response: AgentExecutorResponse, ctx: WorkflowContext
    ) -> None:
        """Handle response from the guesser and convert it to a request for the judge."""

        messages = response.agent_run_response.messages
        request = AgentExecutorRequest(messages=messages, should_respond=True)
        await self.run(request, ctx)


async def main():
    """Main function to run the workflow."""

    credential = AzureCliCredential()
    client = AIProjectClient(
        endpoint=os.environ["FOUNDRY_PROJECT_ENDPOINT"], credential=credential
    )
    guess_agent = await client.agents.create_agent(
        model=os.environ["FOUNDRY_MODEL_DEPLOYMENT_NAME"], name="GuessAgent"
    )

    judge_agent = await client.agents.create_agent(
        model=os.environ["FOUNDRY_MODEL_DEPLOYMENT_NAME"], name="JudgeAgent"
    )

    try:
        # Step 1: Create agent-based executors with specific instructions
        guess_number_executor = GuessAgentExecutor(
            ChatClientAgent(
                chat_client=FoundryChatClient(client=client, agent_id=guess_agent.id),
                instructions=(
                    "You are a number guessing agent. Your task is to guess a number between 1 and 100 using binary search strategy. "
                    "Binary search strategy: "
                    "1. Initial range: lower_bound=1, upper_bound=100 "
                    "2. Always guess the midpoint of the current range: (lower_bound + upper_bound) // 2 "
                    "3. Adjust the range based on feedback: "
                    "   - If guess is 'too low' or 'below target', set lower_bound = current_guess + 1 "
                    "   - If guess is 'too high' or 'above target', set upper_bound = current_guess - 1 "
                    "   - If guess is 'correct' or 'matched', you've found the answer! "
                    "4. Repeat until you find the correct answer "
                    "When you receive 'start' or any initial message, make your first guess as 50 (midpoint of 1-100). "
                    "Always respond with just the integer number you're guessing."
                ),
            ),
            id="guesser",
        )

        judge_number_executor = JudgeAgentExecutor(
            ChatClientAgent(
                chat_client=FoundryChatClient(client=client, agent_id=judge_agent.id),
                instructions=(
                    "You are a number judging agent. Your target number is 30. "
                    "When you receive a guessed number, compare it to your target (30) and respond with exactly: "
                    "- 'correct' if the guess equals 30 "
                    "- 'too low' if the guess is less than 30 "
                    "- 'too high' if the guess is greater than 30 "
                    "Always respond with only these exact phrases, nothing more."
                ),
            ),
            id="judge",
        )

        # Step 2: Build the workflow with the defined edges.
        workflow = (
            WorkflowBuilder()
            .add_edge(guess_number_executor, judge_number_executor)
            .add_edge(judge_number_executor, guess_number_executor)
            .set_start_executor(guess_number_executor)
            .build()
        )

        # Step 3: Run the workflow and let agents decide when to complete.
        async for event in workflow.run_streaming("start"):
            if isinstance(event, AgentRunEvent):
                print(f"{event.executor_id}: {event.data}")
            elif isinstance(event, WorkflowCompletedEvent):
                print(f"🎉 {event.data}")
                break

    finally:
        # Cleanup: Delete agents and close clients
        try:
            if guess_agent:
                await client.agents.delete_agent(guess_agent.id)
            if judge_agent:
                await client.agents.delete_agent(judge_agent.id)
            await client.close()
            await credential.close()
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(main())
