from copy import deepcopy
import re
from typing import (
    Dict,
    Any,
    AsyncGenerator,
    AsyncIterable,
    Callable,
    List,
    Literal,
    Optional,
    TypeVar,
    Union,
)

from vocode.streaming.models.actions import FunctionCall, FunctionFragment
from vocode.streaming.models.events import Sender
from vocode.streaming.models.transcript import (
    ActionFinish,
    ActionStart,
    EventLog,
    Message,
    Transcript,
)

SENTENCE_ENDINGS = [".", "!", "?", "\n"]


async def collate_response_async(
    gen: AsyncIterable[Union[str, FunctionFragment]],
    sentence_endings: List[str] = SENTENCE_ENDINGS,
    get_functions: Literal[True, False] = False,
) -> AsyncGenerator[Union[str, FunctionCall], None]:
    sentence_endings_pattern = "|".join(map(re.escape, sentence_endings))
    list_item_ending_pattern = r"\n"
    buffer = ""
    function_name_buffer = ""
    function_args_buffer = ""
    prev_ends_with_money = False
    prev_starts_with_whitespace = True
    
    async for token in gen:
        if not token:
            continue
        if isinstance(token, str):
            token_starts_with_whitespace = token.startswith(" ")
            
            if prev_ends_with_money and token_starts_with_whitespace:
                yield buffer.strip()
                buffer = ""
            # If the previous token didn't have leading whitespace and the current does:
            elif not prev_starts_with_whitespace and token_starts_with_whitespace:
                # Assume last word in buffer is an email address
                potential_email_text = buffer.split()[-1]
                converted_email_text = convert_email_characters(potential_email_text)
                # # If it was an email
                # if converted_email_text != potential_email_text:
                #     buffer = buffer.replace(potential_email_text, converted_email_text)
                if bool(re.findall(sentence_endings_pattern, buffer)):
                    yield buffer.strip()
                    buffer = ""

            buffer += token
                
            possible_list_item = bool(re.match(r"^\d+[ .]", buffer))
            ends_with_money = bool(re.findall(r"\$\d+.$", buffer))
            if re.findall(
                list_item_ending_pattern
                if possible_list_item
                else sentence_endings_pattern,
                token,
            ):
                if not ends_with_money and token_starts_with_whitespace:
                    to_return = buffer.strip()
                    if to_return:
                        yield to_return
                    buffer = ""
            prev_ends_with_money = ends_with_money
            prev_starts_with_whitespace = token_starts_with_whitespace
        elif isinstance(token, FunctionFragment):
            function_name_buffer += token.name
            function_args_buffer += token.arguments
            
    if not prev_starts_with_whitespace:
        potential_email_text = buffer.split()[-1]
        buffer = buffer.replace(potential_email_text, convert_email_characters(potential_email_text))
        
    to_return = buffer.strip()
    if to_return:
        yield to_return
    if function_name_buffer and get_functions:
        yield FunctionCall(name=function_name_buffer, arguments=function_args_buffer)
        
        
def convert_email_characters(message: str):
    email_regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    if not re.findall(email_regex, message):
        return message
    
    last_char_equals_dot = message.endswith(".")
    message = message.removesuffix(".")

    special_char_dict = {'-' : " dash ", 
                        '_' : " underscore ", 
                        '.' : " dot ",
                        '@' : " at "}
    converted_message = ""
    for char in message:
        converted_message += special_char_dict.get(char, char)
    
    if last_char_equals_dot:
        converted_message += "."
            
    return converted_message



async def openai_get_tokens(gen) -> AsyncGenerator[Union[str, FunctionFragment], None]:
    async for event in gen:
        choices = event.choices
        if len(choices) == 0:
            continue
        choice = choices[0]
        if choice.finish_reason:
            break
        delta = choice.delta
        if hasattr(delta, 'text') and delta.text is not None:
            token = delta.text
            yield token
        if hasattr(delta, 'content') and delta.content is not None:
            token = delta.content
            yield token
        elif hasattr(delta,'tool_calls') and delta.tool_calls is not None:
            for tool_call in delta.tool_calls:
                if hasattr(tool_call, 'function') and tool_call.function is not None:
                    yield FunctionFragment(
                        name=tool_call.function.name if (hasattr(tool_call.function, 'name') and tool_call.function.name is not None) else "",
                        arguments=tool_call.function.arguments if (hasattr(tool_call.function, 'arguments') and tool_call.function.arguments is not None) else ""
                    )


def find_last_punctuation(buffer: str) -> Optional[int]:
    indices = [buffer.rfind(ending) for ending in SENTENCE_ENDINGS]
    if not indices:
        return None
    return max(indices)


def get_sentence_from_buffer(buffer: str):
    last_punctuation = find_last_punctuation(buffer)
    if last_punctuation:
        return buffer[: last_punctuation + 1], buffer[last_punctuation + 1 :]
    else:
        return None, None


def format_openai_chat_messages_from_transcript(
    transcript: Transcript, prompt_preamble: Optional[str] = None
) -> List[dict]:
    chat_messages: List[Dict[str, Optional[Any]]] = (
        [{"role": "system", "content": prompt_preamble}] if prompt_preamble else []
    )

    # merge consecutive bot messages
    new_event_logs: List[EventLog] = []
    idx = 0
    while idx < len(transcript.event_logs):
        bot_messages_buffer: List[Message] = []
        current_log = transcript.event_logs[idx]
        while isinstance(current_log, Message) and current_log.sender == Sender.BOT:
            bot_messages_buffer.append(current_log)
            idx += 1
            try:
                current_log = transcript.event_logs[idx]
            except IndexError:
                break
        if bot_messages_buffer:
            merged_bot_message = deepcopy(bot_messages_buffer[-1])
            merged_bot_message.text = " ".join(
                event_log.text for event_log in bot_messages_buffer
            )
            new_event_logs.append(merged_bot_message)
        else:
            new_event_logs.append(current_log)
            idx += 1

    for event_log in new_event_logs:
        if isinstance(event_log, Message):
            chat_messages.append(
                {
                    "role": "assistant" if event_log.sender == Sender.BOT else "user",
                    "content": event_log.text,
                }
            )
        elif isinstance(event_log, ActionStart):
            chat_messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "function_call": {
                        "name": event_log.action_type,
                        "arguments": event_log.action_input.params.json(),
                    },
                }
            )
        elif isinstance(event_log, ActionFinish):
            chat_messages.append(
                {
                    "role": "function",
                    "name": event_log.action_type,
                    "content": event_log.action_output.response.json(),
                }
            )
    return chat_messages


def vector_db_result_to_openai_chat_message(vector_db_result):
    return {"role": "user", "content": vector_db_result}
