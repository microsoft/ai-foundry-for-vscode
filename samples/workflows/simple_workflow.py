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


class StudentAgentExecutor(AgentExecutor):
    @handler(output_types=[AgentExecutorResponse])
    async def handle_teacher_question(
        self, response: AgentExecutorResponse, ctx: WorkflowContext
    ) -> None:
        messages = response.agent_run_response.messages
        request = AgentExecutorRequest(messages=messages, should_respond=True)
        await self.run(request, ctx)


class TeacherAgentExecutor(AgentExecutor):
    def __init__(self, agent, id="teacher"):
        super().__init__(agent, id=id)
        self.turn_count = 0

    @handler(output_types=[AgentExecutorResponse])
    async def handle_start_message(self, message: str, ctx: WorkflowContext) -> None:
        chat_message = ChatMessage(ChatRole.USER, text=message)
        request = AgentExecutorRequest(messages=[chat_message], should_respond=True)
        await self.run(request, ctx)

    @handler(output_types=[AgentExecutorResponse])
    async def handle_student_answer(
        self, response: AgentExecutorResponse, ctx: WorkflowContext
    ) -> None:
        self.turn_count += 1

        if self.turn_count >= 5:
            await ctx.add_event(
                WorkflowCompletedEvent(
                    data="Student-teacher conversation completed after 5 turns!"
                )
            )
            return

        messages = response.agent_run_response.messages
        if messages and "[COMPLETE]" in messages[-1].text.upper():
            await ctx.add_event(
                WorkflowCompletedEvent(
                    data="Student-teacher conversation completed by teacher!"
                )
            )
            return

        request = AgentExecutorRequest(messages=messages, should_respond=True)
        await self.run(request, ctx)


async def main():
    credential = AzureCliCredential()
    client = AIProjectClient(
        endpoint=os.environ["FOUNDRY_PROJECT_ENDPOINT"], credential=credential
    )
    student_agent = await client.agents.create_agent(
        model=os.environ["FOUNDRY_MODEL_DEPLOYMENT_NAME"], name="StudentAgent"
    )
    teacher_agent = await client.agents.create_agent(
        model=os.environ["FOUNDRY_MODEL_DEPLOYMENT_NAME"], name="TeacherAgent"
    )

    try:
        student_executor = StudentAgentExecutor(
            ChatClientAgent(
                chat_client=FoundryChatClient(client=client, agent_id=student_agent.id),
                instructions="""You are Jamie, a student. Your role is to answer the teacher's questions briefly and clearly.

                IMPORTANT RULES:
                1. Answer questions directly and concisely
                2. Keep responses short (1-2 sentences maximum)
                3. Do NOT ask questions back""",
            ),
            id="student",
        )

        teacher_executor = TeacherAgentExecutor(
            ChatClientAgent(
                chat_client=FoundryChatClient(client=client, agent_id=teacher_agent.id),
                instructions="""You are Dr. Smith, a teacher. Your role is to ask the student different, simple questions to test their knowledge.

                IMPORTANT RULES:
                1. Ask ONE simple question at a time
                2. NEVER repeat the same question twice
                3. Ask DIFFERENT topics each time (science, math, history, geography, etc.)
                4. Keep questions short and clear
                5. Do NOT provide explanations - only ask questions""",
            ),
            id="teacher",
        )

        workflow = (
            WorkflowBuilder()
            .add_edge(teacher_executor, student_executor)
            .add_edge(student_executor, teacher_executor)
            .set_start_executor(teacher_executor)
            .build()
        )

        async for event in workflow.run_streaming("Start the quiz session."):
            if isinstance(event, AgentRunEvent):
                agent_name = event.executor_id
                print(f"\n{agent_name}: {event.data}")
            elif isinstance(event, WorkflowCompletedEvent):
                print(f"\n🎉 {event.data}")
                break

    finally:
        try:
            if student_agent:
                await client.agents.delete_agent(student_agent.id)
            if teacher_agent:
                await client.agents.delete_agent(teacher_agent.id)
            await client.close()
            await credential.close()
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(main())
