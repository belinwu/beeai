# Copyright 2025 IBM Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import random
import string
from collections.abc import Awaitable, Callable
from inspect import isfunction
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, InstanceOf

from beeai_framework.agents.base import AnyAgent, BaseAgent
from beeai_framework.agents.tool_calling.agent import ToolCallingAgent
from beeai_framework.agents.tool_calling.types import ToolCallingAgentRunOutput, ToolCallingAgentTemplates
from beeai_framework.agents.types import (
    AgentExecutionConfig,
)
from beeai_framework.backend.chat import ChatModel
from beeai_framework.backend.message import AnyMessage
from beeai_framework.context import Run
from beeai_framework.memory import BaseMemory, ReadOnlyMemory, UnconstrainedMemory
from beeai_framework.template import PromptTemplateInput
from beeai_framework.tools.tool import AnyTool
from beeai_framework.utils.asynchronous import ensure_async
from beeai_framework.workflows.types import WorkflowRun
from beeai_framework.workflows.workflow import Workflow

AgentFactory = Callable[[ReadOnlyMemory], AnyAgent | Awaitable[AnyAgent]]


class AgentFactoryInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str
    role: str | None = None
    llm: ChatModel
    instructions: str | None = None
    tools: list[InstanceOf[AnyTool]] | None = None
    execution: AgentExecutionConfig | None = None


class AgentWorkflowInput(BaseModel):
    prompt: str
    context: str | None = None
    expected_output: str | type[BaseModel] | None = None

    @classmethod
    def from_message(cls, message: AnyMessage) -> Self:
        return cls(prompt=message.text)


class Schema(BaseModel):
    inputs: list[InstanceOf[AgentWorkflowInput]]
    context: str | None = None
    final_answer: str | None = None
    new_messages: list[AnyMessage] = []


class AgentWorkflow:
    def __init__(self, name: str = "AgentWorkflow") -> None:
        self.workflow = Workflow(name=name, schema=Schema)

    def run(
        self, messages: list[AnyMessage] | None = None, *, inputs: list[AgentWorkflowInput] | None = None
    ) -> Run[WorkflowRun[Any, Any]]:
        if not messages and not inputs:
            raise ValueError("At least one of messages or inputs must be provided")

        schema = Schema(
            inputs=list(inputs) if inputs else [AgentWorkflowInput.from_message(msg) for msg in messages or []]
        )
        return self.workflow.run(schema)

    def del_agent(self, name: str) -> "AgentWorkflow":
        self.workflow.delete_step(name)
        return self

    def add_agent(
        self,
        agent: (
            AnyAgent | Callable[[ReadOnlyMemory], AnyAgent | asyncio.Future[AnyAgent]] | AgentFactoryInput | None
        ) = None,
        /,
        **kwargs: Any,
    ) -> "AgentWorkflow":
        if not agent:
            if not kwargs:
                raise ValueError("An agent object or keyword arguments must be provided")
            elif "agent" in kwargs:
                agent = kwargs.get("agent")
            else:
                agent = AgentFactoryInput.model_validate(kwargs, strict=False, from_attributes=True)
        elif kwargs:
            raise ValueError("Agent object required or keyword arguments required but not both")
        assert agent is not None

        if isinstance(agent, BaseAgent):

            async def factory(memory: ReadOnlyMemory) -> AnyAgent:
                instance: AnyAgent = await ensure_async(agent)(memory.as_read_only()) if isfunction(agent) else agent
                instance.memory = memory
                return instance

            return self._add(agent.meta.name, factory)

        random_string = "".join(random.choice(string.ascii_letters) for _ in range(4))
        name = agent.name if not callable(agent) else f"Agent{random_string}"
        return self._add(name, agent if callable(agent) else self._create_factory(agent))

    def _create_factory(self, agent_input: AgentFactoryInput) -> AgentFactory:
        def factory(memory: BaseMemory) -> ToolCallingAgent:
            def customizer(config: PromptTemplateInput[Any]) -> PromptTemplateInput[Any]:
                new_config = config.model_copy()
                new_config.defaults["instructions"] = agent_input.instructions or config.defaults.get("instructions")
                new_config.defaults["role"] = agent_input.role or config.defaults.get("role")
                return new_config

            templates = ToolCallingAgentTemplates()
            templates.system = templates.system.fork(customizer=customizer)

            return ToolCallingAgent(
                llm=agent_input.llm,
                tools=agent_input.tools or [],
                memory=memory,
                templates=templates,
            )

        return factory

    def _add(self, name: str, factory: AgentFactory) -> Self:
        async def step(state: Schema) -> None:
            agent = await ensure_async(factory)(UnconstrainedMemory())
            input = state.inputs.pop(0)
            run_output: ToolCallingAgentRunOutput = await agent.run(**input.model_dump())

            if state.inputs:
                state.inputs[0].context = "\n\n".join([input.context or "", run_output.result.text])

            state.final_answer = run_output.result.text
            state.new_messages.append(run_output.result)

        self.workflow.add_step(name, step)
        return self
