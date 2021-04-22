import requests
from bs4 import BeautifulSoup
import datetime

API_URL = "https://wikipedia.org/w/api.php"
def get_wikipedia_revision(title, date, diagnostic=False):
    session = requests.Session()
    parameters = {
        "action": "query",
        "prop": "revisions",
        "titles": title,
        "rvprop": "ids|timestamp|comment",
        "rvslots": "main",
        "rvstart": date.isoformat(),
        "rvlimit": "1",
        "formatversion" : "2",
        "format": "json"
    }
    
    revision_response = session.get(url=API_URL, params=parameters)
    revision_response.raise_for_status()
    
    # revision data succeeded
    revision_data = revision_response.json()
    try:
        parameters = {
            "title" : title,
            "oldid" : revision_data['query']['pages'][0]['revisions'][0]['revid']
        }
    except (IndexError,KeyError) as e:
        if not diagnostic:
            return None, None
        else:
            raise e
    
    page_response = session.get(url=API_URL, params=parameters)
    
    return page_response.text, page_response.url
