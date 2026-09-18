import os
import json
import requests

from dotenv import load_dotenv
from groq import Groq


# =========================================================
# LOAD ENVIRONMENT VARIABLES
# =========================================================

load_dotenv()

RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")


# =========================================================
# GROQ CLIENT
# =========================================================

groq_client = Groq(
    api_key=GROQ_API_KEY
)


# =========================================================
# CONVERSATION MEMORY
# =========================================================

conversation_history = []


def get_recent_history(limit=6):
    """
    Return the most recent user messages.

    This allows the AI to understand follow-up requests
    such as:

    "Show me 5 Webflow jobs"
    "Only remote ones"
    "Not from Upwork"
    """

    return conversation_history[-limit:]



# =========================================================
# PARSE USER REQUEST
# =========================================================

def parse_user_request(user_request):

    recent_history = get_recent_history()


    history_text = "\n".join(
        f"- {message}"
        for message in recent_history
    )


    prompt = f"""
You are an AI job search request parser.

Convert the user's request into structured JSON.

The user may ask for:
- quantity
- technology/stack
- posting date
- location
- remote/full-time/part-time/contract
- excluded job sources
- follow-up modifications

IMPORTANT:
If the current request is a follow-up, use the recent conversation
history to preserve previous requirements.


Return ONLY valid JSON.


Required JSON structure:

{{
    "quantity": number,
    "stack": string or null,
    "posted_within_days": number or null,
    "location": string or null,
    "employment_type": string or null,
    "excluded_sources": []
}}


Recent conversation:

{history_text}


Current user request:

{user_request}
"""


    try:

        response = groq_client.chat.completions.create(

            model="openai/gpt-oss-120b",

            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            temperature=0
        )


        content = response.choices[0].message.content.strip()


        if content.startswith("```"):

            content = content.replace(
                "```json",
                ""
            )

            content = content.replace(
                "```",
                ""
            )

            content = content.strip()


        return json.loads(content)


    except Exception as error:

        print(
            "\nERROR: Could not parse user request."
        )

        print(
            f"Details: {error}"
        )


        return {

            "quantity": 5,

            "stack": user_request,

            "posted_within_days": None,

            "location": None,

            "employment_type": None,

            "excluded_sources": []

        }



# =========================================================
# DISPLAY SEARCH REQUIREMENTS
# =========================================================

def display_search_requirements(requirements):

    print(
        "\n================ SEARCH REQUIREMENTS ================\n"
    )


    print(
        f"Quantity: {requirements.get('quantity')}"
    )


    print(
        f"Stack: {requirements.get('stack')}"
    )


    print(
        f"Posted within days: {requirements.get('posted_within_days')}"
    )


    print(
        f"Location: {requirements.get('location')}"
    )


    print(
        f"Employment type: {requirements.get('employment_type')}"
    )


    print(
        f"Excluded sources: {requirements.get('excluded_sources')}"
    )


    print(
        "\n======================================================\n"
    )



# =========================================================
# BUILD JSEARCH QUERY
# =========================================================

def build_search_query(requirements):

    stack = requirements.get("stack")

    location = requirements.get("location")

    employment_type = requirements.get(
        "employment_type"
    )


    query_parts = []


    if stack:

        query_parts.append(
            stack
        )


    query_parts.append(
        "jobs"
    )


    if location:

        query_parts.append(
            f"in {location}"
        )


    if employment_type:

        employment_lower = employment_type.lower()


        if employment_lower == "remote":

            query_parts.append(
                "remote"
            )


        elif employment_lower == "full-time":

            query_parts.append(
                "full time"
            )


        elif employment_lower == "part-time":

            query_parts.append(
                "part time"
            )


        elif employment_lower == "contract":

            query_parts.append(
                "contract"
            )


    return " ".join(query_parts)



# =========================================================
# SEARCH JOBS - JSEARCH
# =========================================================

def search_jobs(query, num_pages=1):

    """
    Search jobs using the current JSearch API.

    Current endpoint:
    https://jsearch.p.rapidapi.com/search-v2
    """


    API_URL = (
        "https://jsearch.p.rapidapi.com/search-v2"
    )


    API_HOST = (
        "jsearch.p.rapidapi.com"
    )


    headers = {

        "x-rapidapi-key": RAPIDAPI_KEY,

        "x-rapidapi-host": API_HOST

    }


    params = {

        "query": query,

        "num_pages": str(num_pages),

        "country": "us",

        "date_posted": "all"

    }


    response = None


    try:


        response = requests.get(

            API_URL,

            headers=headers,

            params=params,

            timeout=60

        )


        # Debug information

        print(
            "\nJSearch Status:",
            response.status_code
        )


        print(
            "JSearch URL:",
            response.url
        )


        response.raise_for_status()


        data = response.json()


        jobs = data.get(

            "data",

            {}

        ).get(

            "jobs",

            []

        )


        print(
            f"JSearch returned {len(jobs)} jobs."
        )


        return jobs


    except requests.exceptions.RequestException as error:


        print(
            "\nERROR: JSearch request failed."
        )


        print(
            f"Details: {error}"
        )


        if response is not None:

            print(
                "API Response:",
                response.text[:2000]
            )


        return []

# =========================================================
# FILTER JOBS BY DATE
# =========================================================

def filter_jobs_by_date(jobs, max_days):

    """
    Keep only jobs posted within the requested number of days.

    Example:

    max_days = 3

    Keeps:
    - 1 day ago
    - 2 days ago
    - 3 days ago
    - today
    - recent hours

    This is a simple learning implementation.
    """


    if not max_days:

        return jobs


    filtered_jobs = []


    for job in jobs:


        posted_text = job.get(
            "job_posted_at",
            ""
        ).lower()


        if "hour" in posted_text:

            filtered_jobs.append(job)

            continue



        if "today" in posted_text:

            filtered_jobs.append(job)

            continue



        if "day" in posted_text:


            try:

                days = int(
                    posted_text.split()[0]
                )


                if days <= max_days:

                    filtered_jobs.append(job)


            except (ValueError, IndexError):

                continue


    return filtered_jobs




# =========================================================
# FILTER JOBS BY SOURCE
# =========================================================

def filter_jobs_by_source(
    jobs,
    excluded_sources
):

    """
    Remove jobs from excluded sources.

    Example:

    excluded_sources = ["Upwork"]

    Jobs published by Upwork will be removed.
    """


    if not excluded_sources:

        return jobs


    filtered_jobs = []


    excluded_sources_lower = [

        source.lower()

        for source in excluded_sources

    ]


    for job in jobs:


        publisher = job.get(
            "job_publisher",
            ""
        ).lower()


        employer = job.get(
            "employer_name",
            ""
        ).lower()



        should_exclude = any(

            source in publisher

            or source in employer

            for source in excluded_sources_lower

        )


        if not should_exclude:

            filtered_jobs.append(job)


    return filtered_jobs





# =========================================================
# AI JOB ANALYSIS
# =========================================================

def analyze_job(job, search_query):

    """
    Use AI to analyze whether a job is actually relevant
    to the requested technology/stack.
    """


    title = job.get(
        "job_title",
        ""
    )


    company = job.get(
        "employer_name",
        ""
    )


    description = job.get(
        "job_description",
        ""
    )



    prompt = f"""

You are an AI job relevance analyzer.

Analyze this job against the user's search query.

User search:

{search_query}


Job title:

{title}


Company:

{company}


Job description:

{description}



Return ONLY valid JSON in this format:

{{
    "requirements": [],
    "relevance_score": 0,
    "reason": ""
}}


Rules:

1. Extract actual technical skills, tools,
technologies and professional requirements.

2. Do not invent requirements.

3. Ignore salary, location, availability,
portfolio requirements and generic soft skills.

4. Score relevance from 0 to 100.

"""


    try:


        response = groq_client.chat.completions.create(

            model="openai/gpt-oss-120b",

            messages=[

                {

                    "role": "user",

                    "content": prompt

                }

            ],

            temperature=0

        )



        content = response.choices[0].message.content.strip()



        if content.startswith("```"):

            content = content.replace(
                "```json",
                ""
            )

            content = content.replace(
                "```",
                ""
            )

            content = content.strip()



        return json.loads(content)



    except Exception as error:


        print(
            "\nERROR: Could not analyze job."
        )


        print(
            f"Details: {error}"
        )


        return {

            "requirements": [],

            "relevance_score": 0,

            "reason": "AI analysis failed."

        }






# =========================================================
# ANALYZE ALL JOBS
# =========================================================

def analyze_jobs(jobs, search_query):


    analyzed_jobs = []


    print(
        f"\nAnalyzing {len(jobs)} jobs with AI...\n"
    )


    for index, job in enumerate(

        jobs,

        start=1

    ):


        title = job.get(

            "job_title",

            "Unknown title"

        )


        print(

            f"Analyzing {index}/{len(jobs)}: {title}"

        )


        analysis = analyze_job(

            job,

            search_query

        )


        analyzed_jobs.append(

            {

                "job": job,

                "analysis": analysis

            }

        )


    return analyzed_jobs





# =========================================================
# DISPLAY JOBS IN TERMINAL
# =========================================================

def display_jobs(analyzed_jobs):


    if not analyzed_jobs:


        print(

            "\nNo relevant jobs found."

        )

        return



    print(

        "\n================ JOB RESULTS ================\n"

    )



    for index, item in enumerate(

        analyzed_jobs,

        start=1

    ):



        job = item["job"]

        analysis = item["analysis"]



        print(

            f"{index}. {job.get('job_title','Unknown title')}"

        )


        print(

            f"Company: {job.get('employer_name','Unknown company')}"

        )


        print(

            f"Location: {job.get('job_location','Unknown location')}"

        )


        print(

            f"Source: {job.get('job_publisher','Unknown')}"

        )


        print(

            f"Relevance: {analysis.get('relevance_score',0)}/100"

        )


        print(

            f"Why: {analysis.get('reason','')}"

        )


        print(

            f"Apply: {job.get('job_apply_link','')}"

        )


        print(

            "\n---------------------------------------------\n"

        )






# =========================================================
# FORMAT JOBS FOR TELEGRAM
# =========================================================

def format_jobs_for_telegram(analyzed_jobs):


    if not analyzed_jobs:

        return "No jobs found."



    message = (

        "🔎 AI Job Finder Results\n\n"

    )


    for index, item in enumerate(

        analyzed_jobs,

        start=1

    ):


        job = item["job"]

        analysis = item["analysis"]



        message += (

            f"{index}. {job.get('job_title')}\n"

            f"🏢 Company: {job.get('employer_name')}\n"

            f"📍 Location: {job.get('job_location')}\n"

            f"🌐 Source: {job.get('job_publisher')}\n"

            f"⭐ Relevance: {analysis.get('relevance_score',0)}/100\n\n"

            f"💡 Why: {analysis.get('reason','')}\n\n"

            f"🔗 Apply: {job.get('job_apply_link')}\n"

            "\n------------------------\n\n"

        )


    return message





# =========================================================
# SEND MESSAGE TO TELEGRAM
# =========================================================

def send_telegram_message(message):


    if not TELEGRAM_BOT_TOKEN:

        print(
            "\nERROR: TELEGRAM_BOT_TOKEN is missing."
        )

        return



    if not TELEGRAM_CHAT_ID:

        print(
            "\nERROR: TELEGRAM_CHAT_ID is missing."
        )

        return



    url = (

        f"https://api.telegram.org/"

        f"bot{TELEGRAM_BOT_TOKEN}/sendMessage"

    )



    payload = {

        "chat_id": TELEGRAM_CHAT_ID,

        "text": message

    }



    try:


        response = requests.post(

            url,

            json=payload,

            timeout=10

        )

        print(response.text)

        response.raise_for_status()



        print(

            "\nTelegram message sent successfully!"

        )



    except requests.exceptions.RequestException as error:


        print(

            "\nERROR: Could not send Telegram message."

        )


        print(

            f"Details: {error}"

        )



# =========================================================
# REMOVE WEBHOOK
# =========================================================

def remove_telegram_webhook():
    """
    Remove any existing Telegram webhook.

    getUpdates polling cannot work while a webhook is active.
    """

    url = (
        f"https://api.telegram.org/"
        f"bot{TELEGRAM_BOT_TOKEN}/deleteWebhook"
    )

    try:

        response = requests.post(
            url,
            json={
                "drop_pending_updates": False
            },
            timeout=10
        )

        response.raise_for_status()

        print(
            "Telegram webhook removed."
        )

    except requests.exceptions.RequestException as error:

        print(
            "\nERROR: Could not remove Telegram webhook."
        )

        print(
            f"Details: {error}"
        )


# =========================================================
# GET TELEGRAM UPDATES
# =========================================================

def get_telegram_updates(offset=None):
    """
    Get new messages from Telegram using long polling.
    """

    url = (
        f"https://api.telegram.org/"
        f"bot{TELEGRAM_BOT_TOKEN}/getUpdates"
    )

    params = {
        "timeout": 30,
        "limit": 10,
        "allowed_updates": ["message"]
    }

    if offset is not None:
        params["offset"] = offset

    try:

        response = requests.get(
            url,
            params=params,
            timeout=40
        )

        response.raise_for_status()

        data = response.json()

        if not data.get("ok"):
            print(
                "\nTelegram API error:",
                data
            )

            return []

        return data.get(
            "result",
            []
        )

    except requests.exceptions.RequestException as error:

        print(
            "\nERROR: Could not get Telegram updates."
        )

        print(
            f"Details: {error}"
        )

        return []

# =========================================================
# PROCESS TELEGRAM MESSAGE
# =========================================================

def process_telegram_message(message_text):
    """
    Process a user's Telegram job-search request.
    """

    print(
        "\n========================================"
    )

    print(
        "New Telegram message:"
    )

    print(
        message_text
    )

    print(
        "========================================\n"
    )

    # Save message to conversation memory
    conversation_history.append(
        message_text
    )

    # ---------------------------------------------
    # PARSE USER REQUEST
    # ---------------------------------------------

    requirements = parse_user_request(
        message_text
    )

    display_search_requirements(
        requirements
    )

    # ---------------------------------------------
    # BUILD SEARCH QUERY
    # ---------------------------------------------

    search_query = build_search_query(
        requirements
    )

    print(
        f"JSearch query: {search_query}"
    )

    # ---------------------------------------------
    # SEARCH JOBS
    # ---------------------------------------------

    jobs = search_jobs(
        search_query,
        num_pages=1
    )

    if not jobs:

        return "❌ No jobs found."

    # ---------------------------------------------
    # DATE FILTER
    # ---------------------------------------------

    jobs = filter_jobs_by_date(
        jobs,
        requirements.get(
            "posted_within_days"
        )
    )

    print(
        f"After date filter: {len(jobs)} jobs"
    )

    # ---------------------------------------------
    # SOURCE FILTER
    # ---------------------------------------------

    jobs = filter_jobs_by_source(
        jobs,
        requirements.get(
            "excluded_sources",
            []
        )
    )

    print(
        f"After source filter: {len(jobs)} jobs"
    )

    if not jobs:

        return (
            "❌ No jobs matched your filters."
        )

    # ---------------------------------------------
    # AI ANALYSIS
    # ---------------------------------------------

    analyzed_jobs = analyze_jobs(
        jobs,
        search_query
    )

    # ---------------------------------------------
    # SORT BY RELEVANCE
    # ---------------------------------------------

    analyzed_jobs.sort(
        key=lambda item: item["analysis"].get(
            "relevance_score",
            0
        ),
        reverse=True
    )

    # ---------------------------------------------
    # TOP N
    # ---------------------------------------------

    quantity = requirements.get(
        "quantity",
        5
    )

    top_jobs = analyzed_jobs[
        :quantity
    ]

    # ---------------------------------------------
    # FORMAT TELEGRAM MESSAGE
    # ---------------------------------------------

    telegram_message = (
        format_jobs_for_telegram(
            top_jobs
        )
    )

    return telegram_message


# =========================================================
# MAIN PROGRAM
# =========================================================

def main():


    print(

        "\n========================================"

    )

    print(

        "        AI JOB FINDER"

    )

    print(

        "========================================\n"

    )



    while True:


        user_request = input(

            "What jobs are you looking for? "

        ).strip()



        if user_request.lower() in [

            "exit",

            "quit",

            "bye"

        ]:


            print(

                "\nGoodbye!"

            )

            break



        if not user_request:

            continue



        conversation_history.append(

            user_request

        )



        requirements = parse_user_request(

            user_request

        )



        display_search_requirements(

            requirements

        )



        search_query = build_search_query(

            requirements

        )


        print(

            f"JSearch query: {search_query}"

        )



        jobs = search_jobs(

            search_query,

            num_pages=1

        )



        if not jobs:


            print(

                "\nNo jobs returned from JSearch."

            )

            continue



        jobs = filter_jobs_by_date(

            jobs,

            requirements.get(

                "posted_within_days"

            )

        )



        print(

            f"After date filter: {len(jobs)} jobs"

        )



        jobs = filter_jobs_by_source(

            jobs,

            requirements.get(

                "excluded_sources",

                []

            )

        )



        print(

            f"After source filter: {len(jobs)} jobs"

        )



        analyzed_jobs = analyze_jobs(

            jobs,

            search_query

        )



        analyzed_jobs.sort(

            key=lambda item:

            item["analysis"].get(

                "relevance_score",

                0

            ),

            reverse=True

        )



        quantity = requirements.get(

            "quantity",

            5

        )



        top_jobs = analyzed_jobs[:quantity]



        display_jobs(

            top_jobs

        )



        telegram_message = format_jobs_for_telegram(

            top_jobs

        )


        send_telegram_message(

            telegram_message

        )

# =========================================================
# TELEGRAM BOT LOOP
# =========================================================

def telegram_bot_loop():
    """
    Continuously listen for Telegram messages.
    """

    print(
        "\n========================================"
    )

    print(
        "        AI JOB FINDER BOT"
    )

    print(
        "========================================\n"
    )

    print(
        "Telegram bot is running..."
    )

    print(
        "Waiting for messages...\n"
    )

    # Remove webhook before starting polling
    remove_telegram_webhook()

    offset = None

    while True:

        updates = get_telegram_updates(
            offset
        )

        for update in updates:

            # Confirm this update
            offset = (
                update["update_id"] + 1
            )

            # Get message
            message = update.get(
                "message"
            )

            if not message:
                continue

            # Get text
            message_text = message.get(
                "text"
            )

            if not message_text:
                continue

            # Get chat ID
            chat_id = message["chat"]["id"]

            print(
                f"Message from chat: {chat_id}"
            )

            # -----------------------------------------
            # PROCESS USER REQUEST
            # -----------------------------------------

            try:

                result = process_telegram_message(
                    message_text
                )

            except Exception as error:

                print(
                    "\nERROR while processing message:"
                )

                print(
                    error
                )

                result = (
                    "❌ Sorry, something went wrong "
                    "while searching for jobs."
                )

            # -----------------------------------------
            # SEND RESULT
            # -----------------------------------------

            send_telegram_message(
                result
            )

            print(
                "\nWaiting for next message...\n"
            )



# =========================================================
# RUN PROGRAM
# =========================================================

if __name__ == "__main__":

    telegram_bot_loop()