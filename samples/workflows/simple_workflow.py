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
                instructions="""You are a number guessing agent playing a guessing game. I need to find a number between 1 and 100.

                IMPORTANT RULES:
                1. NEVER repeat the same guess twice
                2. Use binary search strategy to be efficient
                3. Always respond with ONLY the number, nothing else
                4. Keep track of what you've learned from previous guesses

                BINARY SEARCH STRATEGY:
                - Start with middle of current range
                - If 'too low': the number is higher, so guess higher
                - If 'too high': the number is lower, so guess lower
                - Always eliminate half the remaining possibilities

                EXAMPLE SEQUENCE (target is 30):
                Range 1-100: guess 50 → 'too high' → range becomes 1-49
                Range 1-49: guess 25 → 'too low' → range becomes 26-49
                Range 26-49: guess 37 → 'too high' → range becomes 26-36
                Range 26-36: guess 31 → 'too high' → range becomes 26-30
                Range 26-30: guess 28 → 'too low' → range becomes 29-30
                Range 29-30: guess 30 → 'correct'

                CRITICAL: If you just guessed 25 and got 'too high', your next guess must be LOWER than 25!
                Think step by step about your range and pick the middle of the valid range.""",
            ),
            id="guesser",
        )

        judge_number_executor = JudgeAgentExecutor(
            ChatClientAgent(
                chat_client=FoundryChatClient(client=client, agent_id=judge_agent.id),
                instructions="""You are a number judging agent. The secret number you're thinking of is 30.
                Your job is to compare each guess to 30 and give feedback.

                RESPONSE RULES - respond with EXACTLY these phrases:
                • If the guess is less than 30: say 'too low'
                • If the guess is greater than 30: say 'too high'
                • If the guess equals 30: say 'correct'

                EXAMPLES:
                Guess 15 → '15 < 30' → respond 'too low'
                Guess 25 → '25 < 30' → respond 'too low'
                Guess 35 → '35 > 30' → respond 'too high'
                Guess 45 → '45 > 30' → respond 'too high'
                Guess 30 → '30 = 30' → respond 'correct'

                IMPORTANT: Only respond with the three exact phrases above. Nothing else.""",
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
