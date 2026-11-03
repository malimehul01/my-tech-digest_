
import logging
import os
import re
import smtplib
import sys
import time
import xml.etree.ElementTree as ET

from datetime import datetime
from html import unescape
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.utils import formatdate

import requests


# ==========================================
# 1. CONFIGURATION
# ==========================================

FEEDS = {
    "Android Studio & Gradle": [
        "https://androidstudio.googleblog.com/feeds/posts/default?alt=rss",
        "https://android-developers.googleblog.com/feeds/posts/default?alt=rss",
    ],

    "Kotlin Multiplatform (KMP)": [
        "https://blog.jetbrains.com/kotlin/feed/",
    ],

    "Global AI Market & Tools": [
        "https://techcrunch.com/category/artificial-intelligence/feed/",
        "https://www.theverge.com/ai-artificial-intelligence/rss/index.xml",
    ],
}

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

MAX_ITEMS_PER_FEED = 5
REQUEST_TIMEOUT = 25

session = requests.Session()
session.headers.update({
    "User-Agent": "MehulDailyTechDigest/1.0"
})

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)


# ==========================================
# 2. RSS CLEANING AND PARSING
# ==========================================

def clean_text(value):
    if not value:
        return ""

    value = unescape(value)
    value = re.sub(r"<[^>]*>", " ", value)
    value = re.sub(r"\s+", " ", value)

    return value.strip()


def local_name(tag):
    return tag.rsplit("}", 1)[-1]


def get_child_text(element, names):

    for child in element:

        if local_name(child.tag) in names:

            return "".join(child.itertext()).strip()

    return ""


def parse_feed(content, category):

    root = ET.fromstring(content)

    entries = [
        item for item in root.iter()
        if local_name(item.tag) in ("item", "entry")
    ]

    articles = []

    for entry in entries[:MAX_ITEMS_PER_FEED]:

        title = clean_text(
            get_child_text(entry, {"title"})
        )

        description = clean_text(
            get_child_text(
                entry,
                {"description", "summary", "content", "encoded"}
            )
        )

        published = clean_text(
            get_child_text(
                entry,
                {"pubDate", "published", "updated", "date"}
            )
        )

        link = ""

        for child in entry:

            if local_name(child.tag) == "link":

                link = (
                    child.attrib.get("href")
                    or (child.text or "").strip()
                )

                if link:
                    break

        if title:

            articles.append({
                "category": category,
                "title": title,
                "description": description[:900],
                "published": published,
                "link": link
            })

    return articles


# ==========================================
# 3. COLLECT GLOBAL TECH NEWS
# ==========================================

def collect_news():

    all_articles = []
    seen_articles = set()

    for category, feed_urls in FEEDS.items():

        category_articles = []

        for feed_url in feed_urls:

            try:

                logging.info("Fetching RSS: %s", feed_url)

                response = session.get(
                    feed_url,
                    timeout=REQUEST_TIMEOUT
                )

                response.raise_for_status()

                category_articles = parse_feed(
                    response.content,
                    category
                )

                if category_articles:
                    break

                logging.warning(
                    "No articles found: %s",
                    feed_url
                )

            except (
                requests.RequestException,
                ET.ParseError,
                ValueError
            ) as error:

                logging.warning(
                    "RSS feed failed: %s",
                    error
                )

        logging.info(
            "%s: %d articles",
            category,
            len(category_articles)
        )

        for article in category_articles:

            unique_key = (
                article["link"] or article["title"]
            )

            if unique_key not in seen_articles:

                seen_articles.add(unique_key)
                all_articles.append(article)

    return all_articles


# ==========================================
# 4. BUILD GEMINI AI PROMPT
# ==========================================

def build_prompt(articles):

    news_context = "\n\n".join(

        f"""
CATEGORY: {article['category']}

TITLE: {article['title']}

PUBLISHED: {article['published'] or 'Not provided'}

SUMMARY: {article['description'] or 'No summary available'}

SOURCE: {article['link'] or 'Not available'}
"""
        for article in articles
    )

    today = datetime.now().strftime("%d %B %Y")

    prompt = f"""
You are Mehul's personal premium Global Tech AI Agent.

Your task is to create a high-quality daily technology digest.

LANGUAGE:
Write in professional, clean and natural Hinglish.
Hindi must be written using English letters.

DATE:
{today}

USE EXACTLY THESE FOUR SECTIONS:

1. 🤖 Android Studio & Gradle

Focus on:
- Android Studio updates
- Stable releases
- Gradle improvements
- Developer productivity
- Performance tools
- Android development ecosystem

2. 🚀 Kotlin Multiplatform (KMP)

Focus on:
- KMP stability
- Kotlin releases
- iOS integration
- Desktop integration
- Compose Multiplatform
- Important ecosystem libraries

3. 📈 Global AI Market Trends

Focus on:
- Major global AI companies
- OpenAI, Google, Microsoft, Anthropic
- Important business transformations
- AI market developments
- Major corporate announcements
- Verified financial and stock developments

4. 🛠️ Popular AI Tools

Focus on:
- New AI tools
- Important tool upgrades
- Developer AI tools
- AI coding assistants
- New capabilities
- Important product launches

STRICT RULES:

- Only report facts supported by the supplied articles.
- Do not invent announcements or release versions.
- Do not include local Indian regional news.
- Prioritize important international developments.
- Avoid duplicate stories.
- Use short and punchy bullet points.
- Explain why an update matters to developers.
- Include source URLs wherever available.
- Do not output code, JSON or technical logs.
- Do not add a long generic introduction.

If a section has no relevant confirmed update, write:

"Aaj collected feeds mein is section ka
koi confirmed major update nahi mila."

Make the digest informative, useful and easy to scan.

RAW NEWS DATA:

{news_context}
"""

    return prompt.strip()


# ==========================================
# 5. GEMINI AI SUMMARY GENERATION
# ==========================================

def generate_summary(prompt):

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:

        raise RuntimeError(
            "GEMINI_API_KEY GitHub Secret missing."
        )

    endpoint = (
        "https://generativelanguage.googleapis.com/"
        f"v1beta/models/{GEMINI_MODEL}:generateContent"
    )

    response = session.post(

        endpoint,

        params={
            "key": api_key
        },

        json={

            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ],

            "generationConfig": {
                "maxOutputTokens": 3500
            }
        },

        timeout=75
    )

    if not response.ok:

        raise RuntimeError(
            f"Gemini API Error {response.status_code}: "
            f"{response.text[:1000]}"
        )

    data = response.json()

    candidates = data.get("candidates", [])

    if not candidates:

        raise RuntimeError(
            "Gemini returned no candidates."
        )

    parts = (
        candidates[0]
        .get("content", {})
        .get("parts", [])
    )

    summary = "\n".join(

        part.get("text", "")

        for part in parts

        if part.get("text")
    ).strip()

    if not summary:

        raise RuntimeError(
            "Gemini generated an empty summary."
        )

    return summary


# ==========================================
# 6. SEND EMAIL USING GMAIL SMTP
# ==========================================

def send_email(body):

    sender = os.getenv("SENDER_EMAIL")
    password = os.getenv("SENDER_PASSWORD")
    receiver = os.getenv("RECEIVER_EMAIL")

    required_secrets = {
        "SENDER_EMAIL": sender,
        "SENDER_PASSWORD": password,
        "RECEIVER_EMAIL": receiver
    }

    missing = [
        name
        for name, value in required_secrets.items()
        if not value
    ]

    if missing:

        raise RuntimeError(
            "Missing GitHub Secrets: "
            + ", ".join(missing)
        )

    message = MIMEMultipart()

    message["From"] = sender
    message["To"] = receiver

    message["Subject"] = (
        "Mehul's Morning Global Tech Digest 🚀"
    )

    message["Date"] = formatdate(localtime=True)

    message.attach(
        MIMEText(
            body,
            "plain",
            "utf-8"
        )
    )

    last_error = None

    for attempt in range(1, 4):

        smtp = None

        try:

            logging.info(
                "Connecting to Gmail SMTP (Attempt %d/3)",
                attempt
            )

            # Correct Gmail SMTP hostname
            smtp = smtplib.SMTP_SSL(
                "smtp.gmail.com",
                465,
                timeout=30
            )

            smtp.login(
                sender,
                password
            )

            smtp.sendmail(
                sender,
                [receiver],
                message.as_string()
            )

            logging.info(
                "EMAIL_SENT successfully to %s",
                receiver
            )

            return

        except (
            smtplib.SMTPException,
            OSError
        ) as error:

            last_error = error

            logging.exception(
                "Email attempt %d failed",
                attempt
            )

            if attempt < 3:
                time.sleep(6)

        finally:

            if smtp is not None:

                try:
                    smtp.quit()

                except Exception:
                    pass

    raise RuntimeError(
        f"Email failed after 3 attempts: {last_error}"
    )


# ==========================================
# 7. MAIN AGENT
# ==========================================

def main():

    logging.info(
        "========== DAILY TECH DIGEST START =========="
    )

    logging.info(
        "Gemini model: %s",
        GEMINI_MODEL
    )

    # Step 1: Collect news
    articles = collect_news()

    logging.info(
        "TOTAL_ARTICLES_COLLECTED: %d",
        len(articles)
    )

    if not articles:

        raise RuntimeError(
            "No RSS articles collected. "
            "Refusing to send an empty digest."
        )

    # Step 2: Build prompt
    prompt = build_prompt(articles)

    # Step 3: Generate AI summary
    summary = generate_summary(prompt)

    logging.info(
        "GEMINI_SUMMARY_GENERATED: %d characters",
        len(summary)
    )

    # Step 4: Send email
    send_email(summary)

    logging.info(
        "========== DAILY TECH DIGEST COMPLETED =========="
    )


# ==========================================
# 8. EXECUTION
# ==========================================

if __name__ == "__main__":

    try:

        main()

    except Exception:

        logging.exception(
            "DAILY TECH DIGEST FAILED"
        )

        sys.exit(1)
