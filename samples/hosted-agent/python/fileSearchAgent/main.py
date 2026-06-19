"""File Search Agent using Azure AI Foundry and Agent Framework."""

import asyncio
import contextlib
import os
from pathlib import Path
from datetime import date

from dotenv import load_dotenv

load_dotenv(override=True)

from agent_framework import HostedFileSearchTool, HostedVectorStoreContent
from agent_framework.azure import AzureAIAgentClient
from azure.ai.projects.aio import AIProjectClient
from azure.ai.agentserver.agentframework import from_agent_framework
from azure.identity.aio import DefaultAzureCredential

PROJECT_ENDPOINT = os.getenv(
    "PROJECT_ENDPOINT"
)  # e.g., "https://<project>.services.ai.azure.com"
MODEL_DEPLOYMENT_NAME = os.getenv(
    "MODEL_DEPLOYMENT_NAME", "gpt-4.1-mini"
) 
SEARCH_FILE_PATH = str(Path(__file__).parent / "employee.pdf")

async def main():
    """Run a file-search-enabled agent as an HTTP server."""

    vector_store = None

    async with (
        DefaultAzureCredential() as credential,
        AIProjectClient(endpoint=PROJECT_ENDPOINT, credential=credential) as project_client,
        AzureAIAgentClient(
            project_endpoint=PROJECT_ENDPOINT,
            model_deployment_name=MODEL_DEPLOYMENT_NAME,
            credential=credential,
        ) as client,
    ):
        openai_client = project_client.get_openai_client()

        try:
            vector_store = await openai_client.vector_stores.create(name="filesearch-store")
            print(f"Created vector store: {vector_store.id}")

            with open(SEARCH_FILE_PATH, "rb") as file_handle:
                uploaded = await openai_client.vector_stores.files.upload_and_poll(
                    vector_store_id=vector_store.id,
                    file=file_handle,
                )
            print(f"Uploaded search file: {uploaded.id}")

            file_search_tool = HostedFileSearchTool(
                inputs=[HostedVectorStoreContent(vector_store_id=vector_store.id)],
                max_results=5,
                description="Search the uploaded employee document for grounded answers.",
            )

            agent = client.create_agent(
                name="FileSearchAgent",
                instructions=(
                    "You are a helpful assistant that answers questions using the uploaded file content. "
                    "Prefer grounded answers based on the file search results. "
                    "If the file does not contain enough information, say so clearly. "
                ),
                tool_choice="auto",
                tools=[file_search_tool],
            )

            print("File Search Agent Server running on http://localhost:8088")
            server = from_agent_framework(agent, credentials=credential)
            await server.run_async()
        finally:
            if vector_store is not None:
                with contextlib.suppress(Exception):
                    await openai_client.vector_stores.delete(vector_store.id)


if __name__ == "__main__":
    asyncio.run(main())
