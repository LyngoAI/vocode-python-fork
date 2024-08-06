from vocode.streaming.utils.deepgram_keyword_encoder import urlencode_keywords


def test_deepgram_keyword_encoder():
    keywords = ["Tanmey:1", "hooking:50"]
    url_string = urlencode_keywords(keywords)
    assert url_string == 'keywords=Tanmey%3A1&keywords=hooking%3A50'