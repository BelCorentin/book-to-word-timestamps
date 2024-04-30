import whisper_timestamped
from trigger_gen import Parser

def test_init1():
    assert help(whisper_timestamped) != None

def test_first_book():
    parser = Parser()
    words = parser.parse_audio("hello.wav")
    assert words.head(1).text == "I"