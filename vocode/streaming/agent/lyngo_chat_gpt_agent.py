import asyncio
import logging

from typing import Any, Dict, List, Optional, Tuple, Union

import openai
from openai import AsyncOpenAI, AsyncAzureOpenAI

from typing import AsyncGenerator, Optional, Tuple

import logging
from pydantic import BaseModel
from services.cliniko import ClinikoAPI

from vocode import getenv
from vocode.streaming.action.factory import ActionFactory
from vocode.streaming.agent.base_agent import RespondAgent
from vocode.streaming.models.actions import FunctionCall, FunctionFragment
from vocode.streaming.models.agent import LyngoChatGPTAgentConfig
from vocode.streaming.agent.utils import (
    format_openai_chat_messages_from_transcript,
    collate_response_async,
    openai_get_tokens,
    vector_db_result_to_openai_chat_message,
)
from vocode.streaming.models.events import Sender
from vocode.streaming.models.transcript import Transcript
from vocode.streaming.telephony.config_manager.redis_config_manager import RedisConfigManager
from vocode.streaming.vector_db.factory import VectorDBFactory


class LyngoChatGPTAgent(RespondAgent[LyngoChatGPTAgentConfig]):
    def __init__(
        self,
        agent_config: LyngoChatGPTAgentConfig,
        action_factory: ActionFactory = ActionFactory(),
        logger: Optional[logging.Logger] = None,
        openai_api_key: Optional[str] = None,
        vector_db_factory=VectorDBFactory(),
    ):
        super().__init__(
            agent_config=agent_config, action_factory=action_factory, logger=logger
        )
        if agent_config.azure_params:
            print("IN AZURE!!!")
            api_key = getenv("AZURE_OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY must be set in environment or passed in")
            self.aclient = AsyncAzureOpenAI(
            api_key=api_key,
            azure_endpoint=getenv("AZURE_OPENAI_API_BASE"),
            api_version=agent_config.azure_params.api_version)
        else:
            print("IN OPENAI!!!")
            api_key = openai_api_key or getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY must be set in environment or passed in")
            self.aclient = AsyncOpenAI(api_key=api_key)
        self.first_response = (
            self.create_first_response(agent_config.expected_first_prompt)
            if agent_config.expected_first_prompt
            else None
        )
        self.is_first_response = True

        if self.agent_config.vector_db_config:
            self.vector_db = vector_db_factory.create_vector_db(
                self.agent_config.vector_db_config
            )
        self.get_patient_details_task = asyncio.create_task(self.get_patient_details())

    async def get_patient_details(self):
        config = await RedisConfigManager().get_config(self.agent_config.conversation_id)
        # Scrapping this for now as fixed on vonage end
        # phone_number = self.ensure_country_code(config.to_phone, self.agent_config.customer.timezone)
        phone_number = config.to_phone

        patient_data = await ClinikoAPI(self.agent_config.customer.cliniko.api_key).get_patient_data(phone_number,
                                                                                self.agent_config.customer.timezone,
                                                                                self.agent_config.conversation_id,
                                                                                self.agent_config.customer)
        # Update agent system prompt
        self.agent_config.prompt_preamble = self.agent_config.prompt_preamble.format(patient_data=patient_data)
        return True

    def get_country_code(self, timezone):
        australia_timezones = [
            'Australia/Perth',      # AWST
            'Australia/Adelaide',   # ACST
            'Australia/Darwin',     # ACST
            'Australia/Brisbane',   # AEST
            'Australia/Sydney',     # AEST
            'Australia/Melbourne',  # AEST
            'Australia/Hobart',     # AEST
            'Australia/Lord_Howe',  # LHST/LHDT
            'Australia/Eucla'       # ACWST
        ]

        newzealand_timezones = [
            'Pacific/Auckland',
            'Pacific/Chatham'
        ]
        if timezone in australia_timezones:
            return "61"
        elif timezone in newzealand_timezones:
            return "64"
        else:
            return ""

    def ensure_country_code(self, phone_number, timezone):
        country_code = self.get_country_code(timezone)
        # Check if the phone number already starts with the country code
        if not phone_number.startswith(country_code):
            # If not, prepend the country code
            phone_number = country_code + phone_number
        return phone_number

    async def ten_seconds_task(self):
        print("Task started, waiting for 10 seconds...")

        await asyncio.sleep(10)
        self.agent_config.prompt_preamble = "im am ai assistant called tim"
        print("Task completed after 10 seconds!!!!!!!!!")
         # Waits for 10 seconds asynchronously


    def get_functions(self):
        assert self.agent_config.actions
        if not self.action_factory:
            return None
        return [
            self.action_factory.create_action(action_config).get_openai_function()
            for action_config in self.agent_config.actions
        ]

    def get_chat_parameters(self, messages: Optional[List] = None):
        assert self.transcript is not None
        messages = messages or format_openai_chat_messages_from_transcript(
            self.transcript, self.agent_config.prompt_preamble
        )
        # print("MESSAGES")
        # print(messages)
        parameters: Dict[str, Any] = {
            "messages": messages,
            "max_tokens": self.agent_config.max_tokens,
            "temperature": self.agent_config.temperature,
        }
        print("TEMP", self.agent_config.temperature)

        if self.agent_config.azure_params is not None:
            parameters["model"] = self.agent_config.azure_params.model
        else:
            parameters["model"] = self.agent_config.model_name

        if self.functions:
            updated_functions_list = [{"type": "function", "function": func} for func in self.functions]
            parameters["tools"] = updated_functions_list

        # parameters["seed"] = 1234
        return parameters

    def create_first_response(self, first_prompt):
        messages = [
            (
                [{"role": "system", "content": self.agent_config.prompt_preamble}]
                if self.agent_config.prompt_preamble
                else []
            )
            + [{"role": "user", "content": first_prompt}]
        ]

        parameters = self.get_chat_parameters(messages)
        return self.aclient.chat.completions.create(**parameters)

    def attach_transcript(self, transcript: Transcript):
        self.transcript = transcript

    async def respond(
        self,
        human_input,
        conversation_id: str,
        is_interrupt: bool = False,
    ) -> Tuple[str, bool]:
        assert self.transcript is not None
        if is_interrupt and self.agent_config.cut_off_response:
            cut_off_response = self.get_cut_off_response()
            return cut_off_response, False
        self.logger.debug("LLM responding to human input")
        if self.is_first_response and self.first_response:
            self.logger.debug("First response is cached")
            self.is_first_response = False
            text = self.first_response
        else:
            chat_parameters = self.get_chat_parameters()
            chat_completion = await self.aclient.chat.completions.create(**chat_parameters)
            text = chat_completion.choices[0].message.content
        self.logger.debug(f"LLM response: {text}")
        return text, False

    async def generate_response(
        self,
        human_input: str,
        conversation_id: str,
        is_interrupt: bool = False,
    ) -> AsyncGenerator[Tuple[Union[str, FunctionCall], bool], None]:
        if is_interrupt and self.agent_config.cut_off_response:
            cut_off_response = self.get_cut_off_response()
            yield cut_off_response, False
            return
        assert self.transcript is not None

        chat_parameters = {}
        if self.agent_config.vector_db_config:
            try:
                docs_with_scores = await self.vector_db.similarity_search_with_score(
                    self.transcript.get_last_user_message()[1]
                )
                docs_with_scores_str = "\n\n".join(
                    [
                        "Document: "
                        + doc[0].metadata["source"]
                        + f" (Confidence: {doc[1]})\n"
                        + doc[0].lc_kwargs["page_content"].replace(r"\n", "\n")
                        for doc in docs_with_scores
                    ]
                )
                vector_db_result = f"Found {len(docs_with_scores)} similar documents:\n{docs_with_scores_str}"
                messages = format_openai_chat_messages_from_transcript(
                    self.transcript, self.agent_config.prompt_preamble
                )
                messages.insert(
                    -1, vector_db_result_to_openai_chat_message(vector_db_result)
                )
                chat_parameters = self.get_chat_parameters(messages)
            except Exception as e:
                self.logger.error(f"Error while hitting vector db: {e}", exc_info=True)
                chat_parameters = self.get_chat_parameters()
        else:
            chat_parameters = self.get_chat_parameters()
        chat_parameters["stream"] = True
        stream = await self.aclient.chat.completions.create(**chat_parameters)
        async for message in collate_response_async(
            openai_get_tokens(stream), get_functions=True
        ):
            yield message, True
