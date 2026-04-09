"""
Ops component — send a Microsoft Teams notification.

Image: data-cpu
"""

from flytekit import Resources, task


@task(
    retries=3,
    requests=Resources(cpu="0.25", mem="256Mi"),
    cache=False,
)
def notify_teams(
    webhook_url: str,
    title: str,
    message: str,
    success: bool = True,
) -> int:
    """Send a Microsoft Teams Adaptive Card notification.

    Args:
        webhook_url: Incoming webhook URL for the target Teams channel.
        title: Card title text.
        message: Card body text (Markdown supported).
        success: Whether the workflow succeeded (controls card colour).

    Returns:
        HTTP status code returned by Teams (``200`` indicates success).
    """
    import requests

    color = "Good" if success else "Attention"
    payload = {
        "type": "message",
        "attachments": [
            {
                "contentType": "application/vnd.microsoft.card.adaptive",
                "content": {
                    "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                    "type": "AdaptiveCard",
                    "version": "1.4",
                    "msteams": {"width": "Full"},
                    "body": [
                        {
                            "type": "TextBlock",
                            "text": title,
                            "weight": "Bolder",
                            "size": "Medium",
                            "color": color,
                        },
                        {"type": "TextBlock", "text": message, "wrap": True},
                    ],
                },
            }
        ],
    }
    resp = requests.post(webhook_url, json=payload, timeout=10)
    return resp.status_code
