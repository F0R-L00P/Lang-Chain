from urllib.request import Request, urlopen
import json


def get_genres():
    genre_url = "https://api.deezer.com/genre"

    headers = {
        "Content-Type": "application/json"
    }

    request = Request(genre_url, headers=headers)

    with urlopen(request, timeout=10) as response:
        data = json.load(response)

    genres = data.get("data", [])

    return genres

def get_artists_by_genre(genre_id):
    artist_url = "https://api.deezer.com/genre/" + str(genre_id) + "/artists"

    headers = {
        "Content-Type": "application/json"
    }

    request = Request(artist_url, headers=headers)

    with urlopen(request, timeout=10) as response:
        data = json.load(response)

    artists = data.get("data", [])

    return artists