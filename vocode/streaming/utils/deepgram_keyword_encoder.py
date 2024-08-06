
from urllib.parse import urlencode
def urlencode_keywords(keywords):
    url_strings = "&".join([urlencode({"keywords": keyword}) for keyword in keywords])
    return url_strings