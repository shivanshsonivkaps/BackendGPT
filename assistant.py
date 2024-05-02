from openai import OpenAI 
import shelve 
import os
import time
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)


def upload_file(path): # Convert relative path to absolute path 
 abs_path = os.path.abspath(path)
 file = client.files.create(file=open(abs_path, "rb"), purpose="assistants")
 return file




def create_assistant():
    """
    You currently cannot set the temperature for Assistant via the API.
    """
    assistant = client.beta.assistants.create(
        name="PDF BOT",
        instructions="You're a helpful PDF bot that can assist guests that can answer user query. Use your knowledge base to best respond to customer queries. If you don't know the answer, say simply that you cannot help with question and advice to contact the host directly. Be friendly and funny.",
        tools=[{"type": "code_interpreter"}],
        model="gpt-3.5-turbo",
        #  documents=[{"text": text_data}],
    )
    return assistant 

def update_assistant(dataset,asst_id):
    """
    You currently cannot set the temperature for Assistant via the API.
    """
    file = upload_file("dataset.txt")
    if(file):
        assistant = client.beta.assistants.update(
        assistant_id=asst_id,
        file_ids=[file.id],
        )
        return assistant 
    


def check_if_thread_exists(wa_id):
    with shelve.open("threads_db") as threads_shelf:
        return threads_shelf.get(wa_id, None)


def store_thread(wa_id, thread_id):
    with shelve.open("threads_db", writeback=True) as threads_shelf:
        threads_shelf[wa_id] = thread_id


def generate_response(message_body, wa_id, asst_id):
    # Check if there is already a thread_id for the wa_id
    thread_id = check_if_thread_exists(wa_id)

    # If a thread doesn't exist, create one and store it
    if thread_id is None:
        # print(f"Creating new thread  with wa_id {wa_id}")
        thread = client.beta.threads.create()
        store_thread(wa_id, thread.id)
        thread_id = thread.id

    # Otherwise, retrieve the existing thread
    else:
        # print(f"Retrieving existing thread for wa_id {wa_id}")
        thread = client.beta.threads.retrieve(thread_id)

    # Add message to thread
    message = client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=message_body,
    )

    # Run the assistant and get the new message
    new_message = run_assistant(thread , asst_id)
    # print(f"To {name}:", new_message)
    return new_message


def run_assistant(thread, asst_id):
    # Retrieve the Assistant
    assistant = client.beta.assistants.retrieve(asst_id)
    # Run the assistant
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id,
    )
    # Wait for completion
    while run.status != "completed":
        # Be nice to the API
        time.sleep(0.5)
        run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)

    # Retrieve the Messages
    messages = client.beta.threads.messages.list(thread_id=thread.id)
    new_message = messages.data[0].content[0].text.value
    # print(f"Generated message: {new_message}")
    return new_message

