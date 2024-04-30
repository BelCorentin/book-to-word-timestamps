"""
Class to generate audio from text
using the *** Text-to-Speech API
"""

import os
import json
import logging
import pydantic

from typing import Dict, List, Tuple
from pydantic import BaseModel

# Audio import

class AudioFile(BaseModel):
    path: str

    def __init__(self, path: str):
        """
        Load the text file from the path.
        """
        self.path = path
        self.text = self.load_text()

    def load_text(self) -> str:
        """
        Load the text from the file.
        """
        with open(self.path, 'r') as f:
            return f.read()


    def generate_audio(self, text: str) -> None:
        """
        Generate audio from text and save it to the path.
        """
        raise NotImplementedError

class Book(BaseModel):
    title: str
    text: str
    path: str

    def __init__(self, title: str, text: str):
        self.title = title
        self.text = self.parse_text()

    def parse_text(self) -> str:
        """
        Parse the text and return the text in a format suitable for the audio generation.
        """
        raise NotImplementedError
    
    

