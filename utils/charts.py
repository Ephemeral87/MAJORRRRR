import plotly.express as px


def sentiment_pie(df):
    counts = df["sentiment"].value_counts().reset_index()
    counts.columns = ["sentiment", "count"]

    fig = px.pie(
        counts,
        names="sentiment",
        values="count",
        hole=0.45,
        color="sentiment",
        color_discrete_map={
            "Positive": "#22c55e",
            "Neutral": "#94a3b8",
            "Negative": "#ef4444",
            "Other": "#f59e0b"
        }
    )

    fig.update_layout(template="plotly_dark")
    return fig


def sentiment_by_language(df):
    counts = df.groupby(["language", "sentiment"]).size().reset_index(name="count")

    fig = px.bar(
        counts,
        x="language",
        y="count",
        color="sentiment",
        barmode="group",
        color_discrete_map={
            "Positive": "#22c55e",
            "Neutral": "#94a3b8",
            "Negative": "#ef4444",
            "Other": "#f59e0b"
        }
    )

    fig.update_layout(template="plotly_dark")
    return fig
