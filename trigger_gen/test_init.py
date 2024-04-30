import whisper_timestamped
from .parser import Parser


def test_init1():
    assert help(whisper_timestamped) == None

def test_first_book():
    parser = Parser(language="en")
    words = parser.parse_audio("hello.wav")
    assert words.head(1).text[0] == "Hello."

def test_second_book():
    parser = Parser(language="fr")
    words = parser.parse_audio("bonjour.mp3")
    assert words.head(1).text[0] == "Bonjour !"

def test_lpp():
    parser = Parser(language="fr")
    words = parser.parse_audio("ch1-3.wav")
    assert words.head(1).text[0] == "Lorsque"