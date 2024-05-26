import asyncio
import io
import json
import base64
import logging
import re
import wave
import aiohttp
import websockets
from typing import Optional, AsyncGenerator
from vocode import getenv
from vocode.streaming.models.audio_encoding import AudioEncoding
from vocode.streaming.synthesizer.base_synthesizer import (
    BaseSynthesizer,
    SynthesisResult,
    tracer,
)
from vocode.streaming.models.synthesizer import ElevenLabsSynthesizerConfig, SynthesizerType
from vocode.streaming.agent.bot_sentiment_analyser import BotSentiment
from vocode.streaming.models.message import BaseMessage
from vocode.streaming.utils.mp3_helper import decode_mp3

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

        def spell_email_addresses(message: str):
            last_char_equals_dot = message.endswith(".")
            message = message.removesuffix(".")

            special_char_dict = {'-' : "dash", 
                                '_' : "underscore", 
                                '.' : "dot",
                                '@' : "at"}
            converted_message = ""
            for char in message:
                converted_message += " *" + special_char_dict.get(char, char.upper()) + "*."
            
            converted_message = converted_message.removeprefix(" ").removesuffix(",")
            if last_char_equals_dot:
                converted_message += "."
            
            return converted_message

        voice = self.elevenlabs.Voice(voice_id=self.voice_id)
        
        if self.stability is not None and self.similarity_boost is not None:
            voice.settings = self.elevenlabs.VoiceSettings(
                stability=self.stability, similarity_boost=self.similarity_boost
            )

        url = f"wss://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}/stream-input?model_id=eleven_turbo_v2&output_format=pcm_16000"

        message_with_spelt_email = message.text
        email_regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'   
        for email_match in re.finditer(email_regex, message_with_spelt_email):
            message_with_spelt_email = message_with_spelt_email[:email_match.start()] + spell_email_addresses(email_match.group()) + message_with_spelt_email[email_match.end():]

        headers = {"xi-api-key": self.api_key}
        body = {
            "text": message_with_spelt_email,
            "voice_settings": voice.settings.dict() if voice.settings else None,
        }
        if self.model_id:
            body["model_id"] = self.model_id

        create_speech_span = tracer.start_span(
            f"synthesizer.{SynthesizerType.ELEVEN_LABS.value.split('_', 1)[-1]}.create_total",
        )

        async with websockets.connect(url, extra_headers=headers) as ws:
            await ws.send(json.dumps({
                "text": " ",
                "voice_settings": voice.settings.dict() if voice.settings else None,
            }))

            def encode_as_wav(chunk: bytes, synthesizer_config: ElevenLabsSynthesizerConfig) -> bytes:
                output_bytes_io = io.BytesIO()
                in_memory_wav = wave.open(output_bytes_io, "wb")
                in_memory_wav.setnchannels(1)
                assert synthesizer_config.audio_encoding == AudioEncoding.LINEAR16
                in_memory_wav.setsampwidth(2)
                in_memory_wav.setframerate(synthesizer_config.sampling_rate)
                in_memory_wav.writeframes(decode_mp3(bytes(chunk)))
                output_bytes_io.seek(0)
                return output_bytes_io.read()


            async def sender():
                await ws.send(json.dumps(body))
                print("Sent text to ElevenLabs")

            async def receiver() -> AsyncGenerator[bytes, None]:
                print("received message")
                try:
                    while True:
                        message = await ws.recv()
                        print("received message")
                        data = json.loads(message)
                        if data.get("audio"):
                            yield base64.b64decode(data["audio"])
                            
                            # encode_as_wav(data["audio"], self.synthesizer_config)
                            
                            # self.experimental_mp3_streaming_output_generator(
                            #     data["audio"], chunk_size, create_speech_span
                            # ),
                            
                            # experimental_mp3_streaming_output_generator
                            
                            # 
                        elif data.get('isFinal'):
                            break
                except websockets.exceptions.ConnectionClosed as e:
                    print(f"Connection closed with exception: {e}")

            await sender()
            audio_generator = receiver()
            # create_speech_span.end()

            return SynthesisResult(
                audio_generator,
                lambda seconds: self.get_message_cutoff_from_voice_speed(
                    message, seconds, self.words_per_minute
                ),
            )
