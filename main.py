import os
import requests
import json

from dotenv import load_dotenv
from groq import Groq


# ==========================================
# LOAD ENVIRONMENT VARIABLES
# ==========================================

load_dotenv()

RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# ==========================================
# API CONFIGURATION
# ==========================================

API_URL = "https://jsearch-mega.p.rapidapi.com/search"
API_HOST = "jsearch-mega.p.rapidapi.com"


# ==========================================
# GROQ CLIENT
# ==========================================

if GROQ_API_KEY:
    groq_client = Groq(api_key=GROQ_API_KEY)
else:
    groq_client = None


# ==========================================
# CONVERSATION MEMORY
# ==========================================

conversation_history = []


def get_recent_history(limit=6):
    """
    Return the most recent conversation messages.

    limit:
        Maximum number of messages to remember.
    """

    return conversation_history[-limit:]


# ==========================================
# PARSE USER REQUEST
# ==========================================

def parse_user_request(user_request):
    """
    Understand the user's natural-language job search request
    and convert it into structured search requirements.

    The AI also uses recent conversation history so that
    follow-up requests can refer to previous searches.
    """

    if not groq_client:
        print("\nERROR: GROQ_API_KEY was not found in .env")
        return None

    # --------------------------------------
    # GET RECENT CONVERSATION
    # --------------------------------------

    recent_history = get_recent_history()

    history_text = ""

    if recent_history:

        history_text = "\n".join(
            [
                f"{message['role']}: {message['content']}"
                for message in recent_history
            ]
        )

    # --------------------------------------
    # BUILD AI PROMPT
    # --------------------------------------

    prompt = f"""
You are a job search request parser.

Here is the recent conversation history:

{history_text}

The user's latest message is:

"{user_request}"

Your job is to understand the user's latest
job search request using the previous conversation
when necessary.

IMPORTANT:

The latest message may be a follow-up request.

If the latest message depends on the previous request,
preserve the relevant information from the previous request.

Examples:

Previous:
"Show me 5 Webflow jobs"

Latest:
"Only remote ones"

Then understand it as:

5 Webflow jobs
employment_type = remote

Another example:

Previous:
"Show me Webflow jobs"

Latest:
"Not from Upwork"

Then understand it as:

Webflow jobs
excluded_sources = ["Upwork"]

Another example:

Previous:
"Show me 5 Webflow jobs posted within 3 days"

Latest:
"Only Europe"

Then preserve:

quantity = 5
stack = Webflow
posted_within_days = 3

and add:

location = Europe

If the latest message contains a completely new
job search, use the latest request instead of
incorrectly carrying over unrelated information.

Extract the following information:

1. quantity

- Number of jobs the user wants.
- If the user does not specify a number,
  use 5.
- Must be a number.

2. stack

- The main technology, platform, skill,
  or job type the user is looking for.

Examples:

Webflow
Kajabi
n8n
React
Python
AI automation
WordPress

3. posted_within_days

- If the user asks for jobs posted within
  a certain number of days, return that number.

Examples:

"posted within 3 days" → 3
"last 24 hours" → 1
"posted today" → 1

- If no date restriction is mentioned,
  return null.

4. location

- If the user specifies a location, return it.

Examples:

"Webflow jobs in Europe" → "Europe"
"Webflow jobs from USA" → "USA"
"remote Webflow jobs" → null

- If no location is specified, return null.

5. employment_type

If the user specifies:

remote
contract
full-time
part-time

return the appropriate value.

Otherwise return null.

Important:

"remote" should be treated as an employment/location
requirement because the job needs to allow remote work.

6. excluded_sources

Extract any job platforms, companies, or sources
the user explicitly wants to exclude.

Examples:

"not from Upwork" → ["Upwork"]

"exclude Fiverr" → ["Fiverr"]

"not from Upwork or Fiverr"
→ ["Upwork", "Fiverr"]

If none are mentioned:

[]

IMPORTANT:

- Do not invent information.
- Use previous conversation only when needed
  to understand the latest request.
- Preserve previous search requirements when
  the user is clearly modifying the previous search.
- If a new search is clearly started, do not
  carry over unrelated old requirements.
- quantity must be a number.
- posted_within_days must be a number or null.
- excluded_sources must always be an array.
- Return ONLY valid JSON.

Use exactly this format:

{{
    "quantity": 5,
    "stack": "Webflow",
    "posted_within_days": 3,
    "location": null,
    "employment_type": null,
    "excluded_sources": ["Upwork"]
}}
"""

    # --------------------------------------
    # CALL GROQ
    # --------------------------------------

    try:

        response = groq_client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            response_format={
                "type": "json_object"
            },
            temperature=0
        )

        content = response.choices[0].message.content

        result = json.loads(content)

        return result

    except json.JSONDecodeError:

        print("\nERROR: Groq returned invalid JSON.")
        return None

    except Exception as error:

        print("\nERROR: Could not understand the job request.")
        print(f"Details: {error}")

        return None


# ==========================================
# FILTER JOBS BY POSTED DATE
# ==========================================

def filter_jobs_by_date(jobs, max_days):
    """
    Keep only jobs that were posted within
    the requested number of days.
    """

    if not max_days:
        return jobs

    filtered_jobs = []

    for job in jobs:

        posted_text = job.get(
            "job_posted_at",
            ""
        ).lower()

        # ----------------------------------
        # Hours
        # Example: "16 hours ago"
        # ----------------------------------

        if "hour" in posted_text:

            filtered_jobs.append(job)
            continue

        # ----------------------------------
        # Today
        # ----------------------------------

        if "today" in posted_text:

            filtered_jobs.append(job)
            continue

        # ----------------------------------
        # Days
        # Example: "2 days ago"
        # ----------------------------------

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


# ==========================================
# FILTER JOBS BY SOURCE
# ==========================================

def filter_jobs_by_source(jobs, excluded_sources):
    """
    Remove jobs from sources the user wants to exclude.
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

        # Check both publisher and employer
        should_exclude = any(
            source in publisher
            or source in employer
            for source in excluded_sources_lower
        )

        if not should_exclude:

            filtered_jobs.append(job)

    return filtered_jobs


# ==========================================
# SEARCH JOBS
# ==========================================

def search_jobs(query, num_pages=1):
    """
    Search for jobs using JSearch Mega API.
    """

    if not RAPIDAPI_KEY:
        print(
            "\nERROR: RAPIDAPI_KEY was not found in .env"
        )
        return []

    headers = {
        "x-rapidapi-key": RAPIDAPI_KEY,
        "x-rapidapi-host": API_HOST
    }

    params = {
        "query": query,
        "page": "1",
        "num_pages": str(num_pages)
    }

    try:

        response = requests.get(
            API_URL,
            headers=headers,
            params=params,
            timeout=20
        )

        response.raise_for_status()

        data = response.json()

        return data.get("data", [])

    except requests.exceptions.Timeout:

        print(
            "\nERROR: The API request timed out."
        )
        return []

    except requests.exceptions.HTTPError as error:

        print(
            "\nERROR: JSearch API returned an HTTP error."
        )

        if response.status_code == 401:

            print(
                "Your RapidAPI key may be invalid."
            )

        elif response.status_code == 403:

            print(
                "Your RapidAPI subscription may not "
                "allow this request."
            )

        elif response.status_code == 429:

            print(
                "You have reached your API request limit."
            )

        else:

            print(
                f"Status code: {response.status_code}"
            )

        print(
            f"Details: {error}"
        )

        return []

    except requests.exceptions.RequestException as error:

        print(
            "\nERROR: Could not connect to JSearch."
        )

        print(
            f"Details: {error}"
        )

        return []

    except ValueError:

        print(
            "\nERROR: API returned invalid JSON."
        )

        return []


# ==========================================
# ANALYZE ONE JOB WITH GROQ
# ==========================================

def analyze_job(job, search_query):
    """
    Analyze one job using Groq.

    Returns:
        Dictionary containing requirements,
        relevance score and reason.
    """

    if not groq_client:

        print(
            "\nERROR: GROQ_API_KEY was not found in .env"
        )

        return None

    title = job.get(
        "job_title",
        ""
    )

    description = job.get(
        "job_description",
        ""
    )

    if not description:

        return None

    prompt = f"""
You are an AI job relevance analyzer.

The user is searching for:

"{search_query}"

Analyze the following job.

JOB TITLE:
{title}

JOB DESCRIPTION:
{description}

Your tasks:

1. Extract the actual technical skills,
   tools, technologies and professional
   requirements from the job.

2. Determine how relevant this job is
   to the user's search query.

3. Give a relevance score from 0 to 100.

Scoring guidance:

90-100:
Directly matches the requested job/skill.

70-89:
Strongly related and likely relevant.

40-69:
Some relevant aspects but not primarily
the requested job.

1-39:
Very little relevance.

0:
Not relevant.

IMPORTANT:

- Do not invent requirements.
- Only use information explicitly present
  in the job description.
- Do not treat salary, location, portfolio
  requests, availability or application
  instructions as skills.
- Keep requirements concise.
- The relevance score must reflect the
  actual job content, not just keywords.
- If the search term is the name of a company
  rather than the technology/platform, do not
  assume the job is relevant to the technology.

Return ONLY valid JSON in this exact format:

{{
    "requirements": [
        "requirement 1",
        "requirement 2"
    ],
    "relevance_score": 95,
    "reason": "Short explanation of why this job is relevant."
}}
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
            response_format={
                "type": "json_object"
            },
            temperature=0
        )

        content = response.choices[0].message.content

        return json.loads(content)

    except json.JSONDecodeError:

        print(
            "\nERROR: Groq returned invalid JSON."
        )

        return None

    except Exception as error:

        print(
            "\nERROR: AI analysis failed."
        )

        print(
            f"Details: {error}"
        )

        return None


# ==========================================
# DISPLAY SEARCH REQUIREMENTS
# ==========================================

def display_search_requirements(requirements):
    """
    Display the structured requirements extracted
    from the user's natural-language request.
    """

    print("\n==============================")
    print("     SEARCH REQUIREMENTS")
    print("==============================")

    print(
        f"Quantity: "
        f"{requirements.get('quantity', 5)}"
    )

    print(
        f"Stack: "
        f"{requirements.get('stack')}"
    )

    print(
        f"Posted within days: "
        f"{requirements.get('posted_within_days')}"
    )

    print(
        f"Location: "
        f"{requirements.get('location')}"
    )

    print(
        f"Employment type: "
        f"{requirements.get('employment_type')}"
    )

    print(
        f"Excluded sources: "
        f"{requirements.get('excluded_sources', [])}"
    )


# ==========================================
# DISPLAY ANALYZED JOBS
# ==========================================

def display_jobs(analyzed_jobs):
    """
    Display analyzed jobs in a clean format.
    """

    if not analyzed_jobs:

        print(
            "\nNo analyzed jobs available."
        )

        return

    print("\n")
    print("==============================")
    print("       ANALYZED JOBS")
    print("==============================")

    for index, item in enumerate(
        analyzed_jobs,
        start=1
    ):

        job = item["job"]
        analysis = item["analysis"]

        title = job.get(
            "job_title",
            "Unknown title"
        )

        company = job.get(
            "employer_name",
            "Unknown company"
        )

        location = job.get(
            "job_location",
            "Unknown location"
        )

        publisher = job.get(
            "job_publisher",
            "Unknown"
        )

        apply_link = job.get(
            "job_apply_link",
            "No apply link"
        )

        posted_at = job.get(
            "job_posted_at",
            "Unknown"
        )

        score = analysis.get(
            "relevance_score",
            0
        )

        requirements = analysis.get(
            "requirements",
            []
        )

        reason = analysis.get(
            "reason",
            ""
        )

        print(
            "\n------------------------------"
        )

        print(
            f"{index}. {title}"
        )

        print(
            f"Company: {company}"
        )

        print(
            f"Location: {location}"
        )

        print(
            f"Posted: {posted_at}"
        )

        print(
            f"Source: {publisher}"
        )

        print(
            f"\nRelevance Score: {score}%"
        )

        print("\nRequirements:")

        for requirement in requirements:

            print(
                f"- {requirement}"
            )

        print(
            f"\nWhy: {reason}"
        )

        print(
            f"\nApply: {apply_link}"
        )


# ==========================================
# MAIN PROGRAM
# ==========================================

def main():

    print("==============================")
    print("        AI JOB FINDER")
    print("==============================")

    # --------------------------------------
    # CONTINUOUS CONVERSATION LOOP
    # --------------------------------------

    while True:

        # ----------------------------------
        # GET NATURAL LANGUAGE REQUEST
        # ----------------------------------

        user_request = input(
            "\nWhat jobs are you looking for?\n"
        )

        # ----------------------------------
        # EXIT COMMAND
        # ----------------------------------

        if user_request.lower().strip() in [
            "exit",
            "quit",
            "bye"
        ]:

            print(
                "\nGoodbye!"
            )

            break

        # ----------------------------------
        # EMPTY REQUEST
        # ----------------------------------

        if not user_request.strip():

            print(
                "\nJob request cannot be empty!"
            )

            continue

        # ----------------------------------
        # SAVE USER MESSAGE TO MEMORY
        # ----------------------------------

        conversation_history.append(
            {
                "role": "user",
                "content": user_request
            }
        )

        # ----------------------------------
        # UNDERSTAND USER REQUEST
        # ----------------------------------

        print(
            "\nUnderstanding your request..."
        )

        search_requirements = parse_user_request(
            user_request
        )

        if not search_requirements:

            continue

        # ----------------------------------
        # SHOW WHAT AI UNDERSTOOD
        # ----------------------------------

        display_search_requirements(
            search_requirements
        )

        # ----------------------------------
        # GET SEARCH STACK
        # ----------------------------------

        stack = search_requirements.get(
            "stack"
        )

        if not stack:

            print(
                "\nERROR: Could not determine "
                "what type of job to search for."
            )

            continue

        # ----------------------------------
        # SEARCH JSEARCH
        # ----------------------------------

        print(
            f"\nSearching JSearch for: {stack}"
        )

        jobs = search_jobs(
            stack
        )

        if not jobs:

            print(
                "\nNo jobs found."
            )

            continue

        print(
            f"\nFound {len(jobs)} jobs from JSearch."
        )

        # ----------------------------------
        # FILTER BY POSTED DATE
        # ----------------------------------

        posted_within_days = (
            search_requirements.get(
                "posted_within_days"
            )
        )

        if posted_within_days:

            jobs = filter_jobs_by_date(
                jobs,
                posted_within_days
            )

            print(
                f"Found {len(jobs)} jobs "
                f"posted within the last "
                f"{posted_within_days} days."
            )

            if not jobs:

                print(
                    f"\nNo jobs found within the "
                    f"last {posted_within_days} days."
                )

                continue

        # ----------------------------------
        # FILTER BY EXCLUDED SOURCES
        # ----------------------------------

        excluded_sources = (
            search_requirements.get(
                "excluded_sources",
                []
            )
        )

        if excluded_sources:

            jobs = filter_jobs_by_source(
                jobs,
                excluded_sources
            )

            print(
                f"Found {len(jobs)} jobs "
                f"after excluding: "
                f"{', '.join(excluded_sources)}."
            )

            if not jobs:

                print(
                    "\nNo jobs found after applying "
                    "source exclusions."
                )

                continue

        # ----------------------------------
        # ANALYZE JOBS WITH AI
        # ----------------------------------

        print(
            "\nAnalyzing jobs with AI..."
        )

        analyzed_jobs = []

        for index, job in enumerate(
            jobs,
            start=1
        ):

            print(
                f"Analyzing job "
                f"{index}/{len(jobs)}..."
            )

            analysis = analyze_job(
                job,
                stack
            )

            if analysis:

                analyzed_jobs.append(
                    {
                        "job": job,
                        "analysis": analysis
                    }
                )

        # ----------------------------------
        # SORT BY RELEVANCE SCORE
        # ----------------------------------

        analyzed_jobs.sort(
            key=lambda item: item["analysis"].get(
                "relevance_score",
                0
            ),
            reverse=True
        )

        # ----------------------------------
        # GET REQUESTED NUMBER OF JOBS
        # ----------------------------------

        quantity = search_requirements.get(
            "quantity",
            5
        )

        top_jobs = analyzed_jobs[:quantity]

        # ----------------------------------
        # DISPLAY TOP JOBS
        # ----------------------------------

        print(
            f"\nShowing top {len(top_jobs)} "
            f"jobs based on relevance."
        )

        display_jobs(
            top_jobs
        )


# ==========================================
# PROGRAM ENTRY POINT
# ==========================================

if __name__ == "__main__":

    main()