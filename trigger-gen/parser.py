"""
Parser class, taking as input the audio of the book, and returns the dict of words,
their start and end time, and the text of the word.


Example of the output of the transcribe function:
{
  "text": " Bonjour! Est-ce que vous allez bien?",
  "segments": [
    {
      "id": 0,
      "seek": 0,
      "start": 0.5,
      "end": 1.2,
      "text": " Bonjour!",
      "tokens": [ 25431, 2298 ],
      "temperature": 0.0,
      "avg_logprob": -0.6674491882324218,
      "compression_ratio": 0.8181818181818182,
      "no_speech_prob": 0.10241222381591797,
      "confidence": 0.51,
      "words": [
        {
          "text": "Bonjour!",
          "start": 0.5,
          "end": 1.2,
          "confidence": 0.51
        }
      ]
    },
    {
      "id": 1,
      "seek": 200,
      "start": 2.02,
      "end": 4.48,
      "text": " Est-ce que vous allez bien?",
      "tokens": [ 50364, 4410, 12, 384, 631, 2630, 18146, 3610, 2506, 50464 ],
      "temperature": 0.0,
      "avg_logprob": -0.43492694334550336,
      "compression_ratio": 0.7714285714285715,
      "no_speech_prob": 0.06502953916788101,
      "confidence": 0.595,
      "words": [
        {
          "text": "Est-ce",
          "start": 2.02,
          "end": 3.78,
          "confidence": 0.441
        },
        {
          "text": "que",
          "start": 3.78,
          "end": 3.84,
          "confidence": 0.948
        },
        {
          "text": "vous",
          "start": 3.84,
          "end": 4.0,
          "confidence": 0.935
        },
        {
          "text": "allez",
          "start": 4.0,
          "end": 4.14,
          "confidence": 0.347
        },
        {
          "text": "bien?",
          "start": 4.14,
          "end": 4.48,
          "confidence": 0.998
        }
      ]
    }
  ],
  "language": "fr"
}
"""

import os
import json
import logging
import pydantic
from pathlib import Path
import pandas as pd

from typing import Dict, List, Tuple
from pydantic import BaseModel

import whisper_timestamped as whisper

from defaults import DATA_DIR, OUTPUT_DIR, TEXT_DIR, WAV_DIR

class Parser:

    output_path: Path
    audio_path: Path
    
    def __init__(self, output_path: Path = DATA_DIR):
        self.default_path = output_path

    def parse_audio(self, filename) -> List[dict]:
        """
        Parse the audio file and return a list of words with their start and end time.
        """

        audio_path = Path(self.audio_path) / filename
        audio = whisper.load_audio(audio_path)

        model = whisper.load_model("tiny", device="cpu")

        result = whisper.transcribe(model, audio, language="fr")

        # Parse the json output in result, and build words
        words = []
        for segment in result["segments"]:
            for word in segment["words"]:
                words.append(dict(start=word["start"], end=word["end"], text=word["text"]))

        # Create a df from the words, and save it to the output path
        df = pd.DataFrame(words)
        df.to_csv(self.output_path / f"{filename}.csv")

        return words
    