import whisper_timestamped


def test_init1():
    assert help(whisper_timestamped) != None

def test_first_book():
    parser = whisper_timestamped.Parser()
    words = parser.parse_audio("hello.wav")
    assert words.head(1).text == "I"