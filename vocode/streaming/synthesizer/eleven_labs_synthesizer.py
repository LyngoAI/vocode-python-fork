import asyncio
import logging
import time
from typing import Any, AsyncGenerator, Optional, Tuple, Union
import wave
import aiohttp
from opentelemetry.trace import Span
import re

from vocode import getenv
from vocode.streaming.synthesizer.base_synthesizer import (
    BaseSynthesizer,
    SynthesisResult,
    encode_as_wav,
    tracer,
)
from vocode.streaming.models.synthesizer import (
    ElevenLabsSynthesizerConfig,
    SynthesizerType,
)
from vocode.streaming.agent.bot_sentiment_analyser import BotSentiment
from vocode.streaming.models.message import BaseMessage
from vocode.streaming.utils.mp3_helper import decode_mp3
from vocode.streaming.synthesizer.miniaudio_worker import MiniaudioWorker


ADAM_VOICE_ID = "pNInz6obpgDQGcFmaJgB"
ELEVEN_LABS_BASE_URL = "https://api.elevenlabs.io/v1/"


class ElevenLabsSynthesizer(BaseSynthesizer[ElevenLabsSynthesizerConfig]):
    def __init__(
        self,
        synthesizer_config: ElevenLabsSynthesizerConfig,
        logger: Optional[logging.Logger] = None,
        aiohttp_session: Optional[aiohttp.ClientSession] = None,
    ):
        super().__init__(synthesizer_config, aiohttp_session)

        import elevenlabs

        self.elevenlabs = elevenlabs

        self.api_key = synthesizer_config.api_key or getenv("ELEVEN_LABS_API_KEY")
        self.voice_id = synthesizer_config.voice_id or ADAM_VOICE_ID
        self.stability = synthesizer_config.stability
        self.similarity_boost = synthesizer_config.similarity_boost
        self.model_id = synthesizer_config.model_id
        self.optimize_streaming_latency = synthesizer_config.optimize_streaming_latency
        self.words_per_minute = 150
        self.experimental_streaming = synthesizer_config.experimental_streaming

    async def create_speech(
        self,
        message: BaseMessage,
        chunk_size: int,
        bot_sentiment: Optional[BotSentiment] = None,
    ) -> SynthesisResult:
        
        # Returns input string with email portions spelt out in caps and punctuation replaced with their respective words.
        def spell_email_addresses(message: str):
            # Retain any trailing dots.
            last_char_equals_dot = message.endswith(".")
            message = message.removesuffix(".")

            special_char_dict = {
                '-' : 'dash', 
                '_' : 'underscore', 
                '.' : 'dot',
                '@' : 'at',
                '!' : 'exclamation mark',
                '#' : 'hash sign',
                '$' : 'dollar sign',
                '%' : 'percent sign',
                '&' : 'ampersand',
                "'" : 'apostrophe',
                '*' : 'asterisk',
                '+' : 'plus sign',
                '/' : 'forward slash',
                '=' : 'equals sign',
                '?' : 'question mark',
                '^' : 'caret',
                '`' : 'backtick',
                '{' : 'left curly brace',
                '|' : 'vertical bar',
                '}' : 'right curly brace',
                '~' : 'tilde'
            }
            converted_message = ""
            for char in message:
                # Replace it if it exists in the dict and add additional formatting.
                converted_message += " [" + special_char_dict.get(char, char.upper()) + "],"
            
            # Remove placed leading whitespace and trailing comma.
            converted_message = converted_message.removeprefix(" ").removesuffix(",")
            # If email had a trailing dot, add it back.
            if last_char_equals_dot:
                converted_message += "."
            
            print("CONVERTED EMAIL SPEECH: ", converted_message)
                    
            return converted_message
        
        
        # Initialize voice object
        voice = self.elevenlabs.Voice(voice_id=self.voice_id)
        
        # Configure voice settings if stability and similarity boost are provided
        if self.stability is not None and self.similarity_boost is not None:
            voice.settings = self.elevenlabs.VoiceSettings(
                stability=self.stability, similarity_boost=self.similarity_boost
            )
            
        # Construct API endpoint URL
        url = ELEVEN_LABS_BASE_URL + f"text-to-speech/{self.voice_id}"

        if self.experimental_streaming:
            url += "/stream"

        if self.optimize_streaming_latency:
            url += f"?optimize_streaming_latency={self.optimize_streaming_latency}"


        # Email checks
        message_with_spelt_email = message.text
        email_regex = r'\b[A-Za-z0-9\!\$\#\%\&\'\*\+\/\=\?\^\`\{\}\|\~._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'   
        # Loops over all email match objects found. Should not occur but problems arise if multiple emails share the same message.
        for email_match in re.finditer(email_regex, message_with_spelt_email):
            # Replace the portion of the message at the match indices with the converted email address.
            message_with_spelt_email = (
                message_with_spelt_email[:email_match.start()].removesuffix("\"") + 
                spell_email_addresses(email_match.group()) + 
                message_with_spelt_email[email_match.end():].removeprefix("\"")
            )

        # Prepare request headers
        headers = {"xi-api-key": self.api_key}
        
        # Prepare request body
        body = {
            "text": message_with_spelt_email,
            "voice_settings": voice.settings.dict() if voice.settings else None,
        }
        if self.model_id:
            body["model_id"] = self.model_id

        # Start span for tracing
        create_speech_span = tracer.start_span(
            f"synthesizer.{SynthesizerType.ELEVEN_LABS.value.split('_', 1)[-1]}.create_total",
        )

        # Initialize aiohttp session
        session = self.aiohttp_session

        # Make asynchronous POST request to API endpoint
        response = await session.request(
            "POST",
            url,
            json=body,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=15),
        )
        # Handle Response
        if not response.ok:
            raise Exception(f"ElevenLabs API returned {response.status} status code")
        
        # If experimental streaming is enabled, return streaming synthesis result
        if self.experimental_streaming:
            return SynthesisResult(
                self.experimental_mp3_streaming_output_generator(
                    response, chunk_size, create_speech_span
                ),  # should be wav
                lambda seconds: self.get_message_cutoff_from_voice_speed(
                    message, seconds, self.words_per_minute
                ),
            )
            
        # If not experimental streaming, read audio data and process it
        else:
            audio_data = await response.read()
            create_speech_span.end()
            
            # Decode MP3 audio data
            convert_span = tracer.start_span(
                f"synthesizer.{SynthesizerType.ELEVEN_LABS.value.split('_', 1)[-1]}.convert",
            )
            output_bytes_io = decode_mp3(audio_data)

            # Create synthesis result from WAV audio data
            result = self.create_synthesis_result_from_wav(
                synthesizer_config=self.synthesizer_config,
                file=output_bytes_io,
                message=message,
                chunk_size=chunk_size,
            )
            convert_span.end()

            return result
