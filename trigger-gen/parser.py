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

from typing import Dict, List, Tuple
from pydantic import BaseModel

import whisper_timestamped as whisper


class Word(BaseModel):
    start: float
    end: float
    text: str

class Parser:

    default_path: str
    
    def __init__(self, default_path: str):
        self.default_path = default_path

    def parse_audio(self, file) -> List[Word]:
        """
        Parse the audio file and return a list of words with their start and end time.
        """

        audio_path = Path(self.default_path) / file
        audio = whisper.load_audio(audio_path)

        model = whisper.load_model("tiny", device="cpu")

        result = whisper.transcribe(model, audio, language="fr")

        # Parse the json output in result, and build words
        words = []
        for segment in result["segments"]:
            for word in segment["words"]:
                words.append(Word(start=word["start"], end=word["end"], text=word["text"]))
        return words
    