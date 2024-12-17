import asyncio
import datetime
import os
import re
from dataclasses import dataclass
from typing import List, Tuple

import azure.cognitiveservices.speech as speechsdk
from pydub import AudioSegment


@dataclass
class SpeakerConfig:
    voice_name: str
    style: str = "chat"
    style_degree: float = 1.0


SPEAKER_CONFIGS = {
    "Host": SpeakerConfig(voice_name="en-US-JasonNeural", style="chat"),
    "Learner": SpeakerConfig(voice_name="en-US-JennyNeural", style="friendly"),
    "Expert": SpeakerConfig(voice_name="en-US-GuyNeural", style="professional"),
}


class PodcastGenerator:
    def __init__(
        self, subscription_key: str, region: str, base_dir: str = "./podcasts"
    ):
        self.speech_config = speechsdk.SpeechConfig(
            subscription=subscription_key, region=region
        )
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    def _generate_audio(self, text: str, speaker: str, output_path: str) -> str:
        """Generate audio for a single piece of dialogue."""
        config = SPEAKER_CONFIGS[speaker]

        # Configure voice and style
        self.speech_config.speech_synthesis_voice_name = config.voice_name
        ssml_text = f"""
        <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis">
            <voice name="{config.voice_name}">
                <mstts:express-as style="{config.style}" styledegree="{config.style_degree}" xmlns:mstts="http://www.w3.org/2001/mstts">
                    {text}
                </mstts:express-as>
            </voice>
        </speak>
        """

        # Create speech synthesizer
        audio_config = speechsdk.AudioConfig(filename=output_path)
        speech_synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=self.speech_config, audio_config=audio_config
        )

        # Generate audio
        result = speech_synthesizer.speak_ssml_async(ssml_text).get()
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            return output_path
        else:
            raise Exception(f"Speech synthesis failed: {result.reason}")

    async def _generate_audio_async(
        self, segments: List[Tuple[str, str, int]]
    ) -> List[str]:
        """Generate audio files concurrently."""
        loop = asyncio.get_event_loop()
        tasks = []
        for speaker, text, timestamp in segments:
            output_path = f"{self.output_dir}/{speaker.lower()}_{timestamp}.mp3"
            tasks.append(
                loop.run_in_executor(
                    None, self._generate_audio, text, speaker, output_path
                )
            )
        return await asyncio.gather(*tasks)

    def _merge_audio_files(self, audio_files: List[str], output_file: str) -> str:
        """Merge audio files with crossfade."""
        merged = AudioSegment.empty()
        sorted_files = sorted(
            audio_files, key=lambda x: int(re.search(r"(\d{10})", x).group(1))
        )

        for file in sorted_files:
            audio = AudioSegment.from_mp3(file)
            if len(merged) > 0:
                merged = merged.append(audio, crossfade=50)
            else:
                merged = audio

        merged.export(output_file, format="mp3", bitrate="192k")
        return output_file

    async def generate_podcast(self, script: str) -> str:
        """Main method to generate podcast from script."""
        # Create output directory
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        self.output_dir = f"{self.base_dir}/podcast_{timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)

        # Parse script and generate segments
        segments = []
        matches = re.findall(
            r"(Host|Learner|Expert):\s*(.*?)(?=(Host|Learner|Expert|$))",
            script,
            re.DOTALL,
        )

        for speaker, text, _ in matches:
            timestamp = int(datetime.datetime.now().timestamp())
            segments.append((speaker, text.strip(), timestamp))

        # Generate audio concurrently
        audio_files = await self._generate_audio_async(segments)

        # Merge audio files
        output_file = f"podcast_{int(datetime.datetime.now().timestamp())}.mp3"
        return self._merge_audio_files(audio_files, output_file)
