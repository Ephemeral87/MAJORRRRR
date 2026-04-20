from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


def get_video_comments(api_key, video_id, limit=100):
    youtube = build("youtube", "v3", developerKey=api_key)
    comments = []

    try:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            textFormat="plainText",
            maxResults=100,
            order="time"
        )

        while request and len(comments) < limit:
            response = request.execute()

            for item in response.get("items", []):
                snippet = item["snippet"]["topLevelComment"]["snippet"]
                comments.append({
                    "comment": snippet.get("textDisplay", ""),
                    "author": snippet.get("authorDisplayName", "Unknown"),
                    "published_at": snippet.get("publishedAt", "")
                })

                if len(comments) >= limit:
                    break

            request = youtube.commentThreads().list_next(request, response)

        return comments[:limit]

    except HttpError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": str(e)}
