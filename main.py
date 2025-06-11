import os
import requests
import json
from datetime import datetime, timedelta
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate 
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools import tool
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
import gradio as gr
from dotenv import load_dotenv

# --- 0. Environment Setup for OpenRouter ---
load_dotenv() 

llm = ChatOpenAI(
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    model="deepseek/deepseek-chat-v3-0324:free",
    temperature=0.7
)

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# --- 1. Tool Definitions ---
@tool
def create_calendar_event(time: str, date: str = None, description: str = None) -> str:
    """
    Creates a calendar event.
    Parameters:
    - time (str): The time of the event (e.g., "17:00" or "5 PM").
    - date (str): The date of the event (e.g., "2025-06-11" or "tomorrow").
    - description (str): Description of the event (e.g., "Meeting with George").
    """
    if date is None and description is None and isinstance(time, str) and time.strip().startswith('{') and time.strip().endswith('}'):
        try:
            parsed_data = json.loads(time)
            if "time" in parsed_data:
                time = parsed_data["time"]
            if "date" in parsed_data:
                date = parsed_data["date"]
            if "description" in parsed_data:
                description = parsed_data["description"]
            print(f"DEBUG: Successfully parsed nested JSON in create_calendar_event. Corrected: time='{time}', date='{date}', description='{description}'")
        except json.JSONDecodeError:
            print(f"DEBUG: 'time' parameter looked like JSON but failed to parse in create_calendar_event: {time}")
            pass 
    
    if not all([time, date, description]):
        return "Error: Missing required parameters (time, date, description) for calendar event creation."

    print(f"DEBUG: Creating calendar event: {description} on {date} at {time}")
    return f"The event '{description}' has been scheduled for {date} at {time}."


@tool
def set_reminder(input_json_string: str) -> str:
    """
    Sets a reminder for a specific time and date.
    Parameters:
    - input_json_string (str): A JSON string containing 'time', 'date', and 'reminder_text'.
                               Example: '{"time": "10:00", "date": "2025-06-12", "reminder_text": "Call George"}'
    """
    try:
        data = json.loads(input_json_string)
        time = data.get("time")
        date = data.get("date")
        reminder_text = data.get("reminder_text")

        if not all([time, date, reminder_text]):
            return "Error: Missing required parameters (time, date, reminder_text) in the input JSON string."

        print(f"DEBUG: Setting reminder: {reminder_text} on {date} at {time}")
        return f"The reminder '{reminder_text}' has been set for {date} at {time}."
    except json.JSONDecodeError:
        return "Error: Invalid JSON format for reminder input."
    except Exception as e:
        return f"An unexpected error occurred: {e}"

@tool
def summarize_text(text: str) -> str:
    """
    Summarizes a given text.
    Parameters:
    - text (str): The text to be summarized.
    """
    print(f"DEBUG: Summarizing text: {text[:100]}...")
    summarized_content = llm.invoke(f"Summarize me this text: {text}").content
    return f"Summary of the text: {summarized_content}"


@tool
def rewrite_text(text: str, style: str) -> str:
    """
    Rewrites a text based on a given style/goal.
    Parameters:
    - text (str): The text to be rewritten.
    - style (str): The desired style or goal (e.g., "formal", "friendly", "for social media").
    """
    print(f"DEBUG: Rewriting text: {text[:100]}... in style: {style}")
    rewritten_content = llm.invoke(f"Rewrite the following text in a {style} style: {text}").content
    return f"Rewritten text in {style} style: {rewritten_content}"

@tool
def get_weather(city: str, day: str = "today") -> str:
    """
    Provides a weather forecast for a specific city.
    Uses the Open-Meteo.com API which is free for non-commercial use and does not require an API key.
    Note: Open-Meteo.com primarily uses coordinates. This tool will first try to find the coordinates for the city.
    Parameters:
    - city (str): The city for which the weather is requested (e.g., "Athens", "Thessaloniki").
    - day (str): (Optional) The day for which the weather is requested (e.g., "today", "tomorrow", "Sunday"). Default: "today".
    """
    original_city_param = city
    original_day_param = day

    parsed_data = None
    if isinstance(city, str) and city.strip().startswith('{') and city.strip().endswith('}'):
        try:
            parsed_data = json.loads(city)
            print(f"DEBUG: Attempting to parse 'city' as JSON: {parsed_data}")
        except json.JSONDecodeError:
            print(f"DEBUG: The 'city' parameter looked like JSON but failed to parse: {city}")
            pass

    if parsed_data and isinstance(parsed_data, dict):
        if "city" in parsed_data:
            city = parsed_data["city"]
        if "day" in parsed_data:
            day = parsed_data["day"]
        print(f"DEBUG: Successfully parsed incorrect input. Corrected city='{city}', day='{day}'")
    else:
        print(f"DEBUG: Input is already correctly formatted or cannot be parsed as JSON. city='{city}', day='{day}'")

    geocoding_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}"
    print(f"DEBUG: Calling geocoding API for city='{city}'")
    try:
        geo_response = requests.get(geocoding_url).json()
        if not geo_response or not geo_response.get("results"):
            return f"No coordinates found for city '{city}'. Please try a different name or provide a more specific location."
        
        result = geo_response["results"][0]
        latitude = result["latitude"]
        longitude = result["longitude"]
        city_name_found = result.get("name", city)

        print(f"DEBUG: Coordinates found for {city_name_found}: Lat={latitude}, Lon={longitude}")

    except requests.exceptions.RequestException as e:
        return f"A network error occurred during geocoding for city {city}: {e}"
    except Exception as e:
        return f"An unexpected error occurred during geocoding: {e}"

    forecast_url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={latitude}&longitude={longitude}"
        f"&daily=temperature_2m_max,temperature_2m_min,weathercode,precipitation_sum,precipitation_hours"
        f"&timezone=auto"
    )
    print(f"DEBUG: Calling weather forecast API for {city_name_found}")

    try:
        weather_response = requests.get(forecast_url).json()
        
        if not weather_response or "daily" not in weather_response:
            return f"Could not retrieve weather data for city {city_name_found}."

        daily_data = weather_response["daily"]
        
        today = datetime.now().date()
        
        weather_codes = {
            0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
            45: "Fog", 48: "Depositing rime fog",
            51: "Drizzle (light)", 53: "Drizzle (moderate)", 55: "Drizzle (dense intensity)",
            56: "Freezing Drizzle (light)", 57: "Freezing Drizzle (dense intensity)",
            61: "Rain (slight)", 63: "Rain (moderate)", 65: "Rain (heavy intensity)",
            66: "Freezing Rain (light)", 67: "Freezing Rain (heavy intensity)",
            71: "Snow fall (slight)", 73: "Snow fall (moderate)", 75: "Snow fall (heavy intensity)",
            77: "Snow grains",
            80: "Rain showers (slight)", 81: "Rain showers (moderate)", 82: "Rain showers (violent)",
            85: "Snow showers (slight)", 86: "Snow showers (heavy)",
            95: "Thunderstorm (slight/moderate)", 96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail"
        }

        forecast_dates = [datetime.fromisoformat(d).date() for d in daily_data["time"]]
        
        target_date = None
        
        if day.lower() == "today":
            target_date = today
        elif day.lower() == "tomorrow":
            target_date = today + timedelta(days=1)
        elif day.lower() in ["sunday", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday"]:
            day_map = {"monday":0, "tuesday":1, "wednesday":2, "thursday":3, "friday":4, "saturday":5, "sunday":6}
            target_weekday = day_map[day.lower()]
            
            next_target_date_found = False
            for i, d in enumerate(forecast_dates):
                if d.weekday() == target_weekday and d >= today:
                    target_date = d
                    next_target_date_found = True
                    break
            if not next_target_date_found:
                return f"No forecast found for the next {day} in the available data for {city_name_found}."
        else:
            return f"I can only provide forecasts for 'today', 'tomorrow', or specific days of the week. The value '{day}' is not supported."


        if target_date and target_date in forecast_dates:
            idx = forecast_dates.index(target_date)
            max_temp = daily_data["temperature_2m_max"][idx]
            min_temp = daily_data["temperature_2m_min"][idx]
            weather_code = daily_data["weathercode"][idx]
            precipitation_sum = daily_data["precipitation_sum"][idx]
            precipitation_hours = daily_data["precipitation_hours"][idx]
            
            description = weather_codes.get(weather_code, "unknown conditions")

            response_date_str = ""
            if target_date == today:
                response_date_str = "today"
            elif target_date == today + timedelta(days=1):
                response_date_str = "tomorrow"
            else:
                response_date_str = target_date.strftime("%A, %d/%m")

            return (
                f"The forecast for {city_name_found} {response_date_str} is: "
                f"Max Temp: {max_temp}°C, Min Temp: {min_temp}°C. "
                f"Conditions: {description}. "
                f"Total Precipitation: {precipitation_sum}mm ({precipitation_hours} hours)."
            )
        else:
            return f"No forecast found for {day} in the available data for {city_name_found}."

    except requests.exceptions.RequestException as e:
        return f"A network error occurred while fetching weather for {city_name_found}: {e}"
    except Exception as e:
        return f"An unexpected error occurred while processing weather data: {e}"


# --- 2. Information Agent (RAG Setup) ---
retriever_for_rag = None
vectorstore_for_rag = None

def setup_rag(documents_path: str):
    """
    Loads and processes documents for the RAG system.
    """
    global retriever_for_rag, vectorstore_for_rag
    try:
        print(f"DEBUG: Loading documents from: {documents_path}")
        loader = PyPDFLoader(documents_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        
        collection_name = os.path.basename(documents_path).replace('.', '_')
        
        vectorstore_for_rag = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            collection_name=collection_name
        )
        retriever_for_rag = vectorstore_for_rag.as_retriever()
        print(f"DEBUG: RAG setup complete for {documents_path}.")
        return "The document has been loaded and is ready for questions."
    except Exception as e:
        print(f"RAG Setup ERROR: {e}")
        return f"Failed to load the document: {e}"

@tool
def ask_document(query: str) -> str:
    """
    Retrieves and summarizes information from loaded RAG documents to answer a question.
    A document must first be loaded using a 'load file: <filename.pdf>' command.
    Parameters:
    - query (str): The document-based question.
    """
    global retriever_for_rag
    if retriever_for_rag is None:
        return "No document has been loaded yet. Please load a document first using 'load file: <filename.pdf>'."

    qa_prompt = PromptTemplate.from_template("""Answer the user's question based on the provided context.
    If you cannot find the answer in the context, state that you are unsure,
    but do not make up information.
    
    Context: {context}
    Question: {input}""")


    Youtube_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(retriever_for_rag, Youtube_chain)

    print(f"DEBUG: Executing RAG query: {query}")
    try:
        response = rag_chain.invoke({"input": query, "chat_history": []}) 
        return response["answer"]
    except Exception as e:
        print(f"RAG Query ERROR: {e}")
        return f"An error occurred while retrieving information: {e}"

# --- 3. Workflow Management Agent (Coordinator Agent) ---
coordinator_tools = [
    create_calendar_event,
    set_reminder,
    summarize_text,
    rewrite_text,
    get_weather,
    ask_document
]


coordinator_prompt = PromptTemplate.from_template("""You are the central coordinator of a personal assistant system with multiple agents.
Your role is to analyze the user's query and decide:
1. If the query can be directly answered using the available tools.
2. If the query requires delegation to a specialized agent. (Although in this implementation,
   you directly perform the functions of agents via your tools).

You have access to the following tools: {tools}
You can use the following tool names: {tool_names}

Use the following format for tasks requiring tools:

Thought: you should always think about what to do
Action: the name of the tool to use (if any)
Action Input: A JSON object string (and nothing else) that matches the tool's signature, e.g., {{"param1": "value1", "param2": "value2"}}
Observation: the result of the action
... (this can repeat N times)
Thought: I have all the information I need to answer the question.
Final Answer: the final answer to the original question

If the user asks you to "write an email" or "draft an email" or similar, do NOT use a tool. Instead, directly generate the email draft in the following **markdown format**:
**Recipient:** [email address]
**Subject:** [subject line]
**Body:**
```
[email body]
```
Make sure the markdown code block for the body is properly formatted.

History:
{chat_history}

Question: {input}
{agent_scratchpad}""")


coordinator_agent = create_react_agent(llm, coordinator_tools, coordinator_prompt)
coordinator_agent_executor = AgentExecutor(
    agent=coordinator_agent,
    tools=coordinator_tools,
    verbose=True,
    handle_parsing_errors=True
)

# --- 4. Main Application Flow (User Interaction) ---
chat_history = []

def run_personal_assistant(user_input: str):
    global chat_history, retriever_for_rag

    print(f"\nUser: {user_input}")

    if user_input.lower().startswith("load file:"):
        file_path = user_input.split("load file:", 1)[1].strip()
        print(f"Attempting to load file for RAG: {file_path}")
        rag_status = setup_rag(file_path)
        print(f"Assistant: {rag_status}")
        return rag_status
    if user_input.lower().startswith("write an email") or user_input.lower().startswith("draft an email"):
        email_prompt = f"Based on the following request, draft an email. Respond ONLY with the email in the specified markdown format, no extra thoughts or text:\n\n{user_input}\n\n**Recipient:** [email address]\n**Subject:** [subject line]\n**Body:**\n```\n[email body]\n```"
        try:
            email_response = llm.invoke(email_prompt).content
            print(f"Assistant (Email Draft): {email_response}")
            chat_history.extend([HumanMessage(content=user_input), AIMessage(content=email_response)])
            return email_response
        except Exception as e:
            print(f"An error occurred during email drafting: {e}")
            chat_history.extend([HumanMessage(content=user_input), AIMessage(content=f"Sorry, I encountered a problem drafting the email: {e}. Can you try again?")])
            return f"Sorry, I encountered a problem drafting the email: {e}. Can you try again."

    history_string = "\n".join([f"{msg.type}: {msg.content}" for msg in chat_history])

    input_with_history = {
        "input": user_input,
        "chat_history": history_string 
    }

    try:
        response = coordinator_agent_executor.invoke(input_with_history)
        ai_response = response["output"]
        print(f"Assistant: {ai_response}")
        chat_history.extend([HumanMessage(content=user_input), AIMessage(content=ai_response)])
        return ai_response
    except Exception as e:
        print(f"An error occurred: {e}")
        chat_history.extend([HumanMessage(content=user_input), AIMessage(content=f"Sorry, I encountered a problem: {e}. Can you try again?")])
        return f"Sorry, I encountered a problem: {e}. Can you try again."

# --- Gradio Frontend Setup ---
def chat_interface(message, history):
    global chat_history
    current_langchain_history = []
    for human, ai in history:
        current_langchain_history.append(HumanMessage(content=human))
        current_langchain_history.append(AIMessage(content=ai))
    
    chat_history = current_langchain_history

    ai_response = run_personal_assistant(message)
    return ai_response

if __name__ == "__main__":
    print("Starting Personal Assistant...")
    iface = gr.ChatInterface(
        chat_interface,
        title="Your Personal AI Assistant, powered by P21020 & P21219",
        description="Ask me to manage your calendar, set reminders, write emails, summarize text, get weather, or ask questions about documents!",
        examples=[
            "Schedule a meeting tomorrow at 10 AM, 'Project Sync'",
            "Set a reminder for Tuesday at 3 PM: 'Call John'",
            "Write an email to support@example.com with subject 'Issue Report' and body 'I am experiencing a login problem.'",
            "Summarize the following text: Artificial intelligence is changing the world.",
            "Tell me the weather for Athens today.",
            "load file: my_document.pdf",
            "Answer from the document: Who built the Eiffel Tower?"
        ]
    )

    iface.launch(server_name="0.0.0.0", server_port=8888)