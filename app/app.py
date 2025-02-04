import os
import random
import uuid
from base64 import b64encode
from datetime import datetime
from mimetypes import guess_type
from pathlib import Path
from typing import Optional

import gradio as gr
from feedback import save_feedback, scheduler
from gradio.components.chatbot import Option
from huggingface_hub import InferenceClient
from pandas import DataFrame

LANGUAGES: dict[str, str] = {
    "English": "You are a helpful assistant that speaks English.",
    "Spanish": "Tu eres un asistente útil que habla español.",
    "Hebrew": " אתה עוזר טוב ומועיל שמדבר בעברית ועונה בעברית.",
    "Dutch": "Je bent een handige assistent die Nederlands spreekt.",
    "Italian": "Tu sei un assistente utile che parla italiano.",
    "French": "Tu es un assistant utile qui parle français.",
    "German": "Du bist ein hilfreicher Assistent, der Deutsch spricht.",
    "Portuguese": "Você é um assistente útil que fala português.",
    "Russian": "Ты полезный помощник, который говорит по-русски.",
    "Chinese": "你是一个有用的助手，会说中文。",
    "Japanese": "あなたは役立つ助け役で、日本語を話します。",
    "Korean": "당신은 유용한 도우미이며 한국어를 말합니다.",
}

client = InferenceClient(
    token=os.getenv("HF_TOKEN"),
    model=(
        os.getenv("MODEL", "meta-llama/Llama-3.2-11B-Vision-Instruct")
        if not os.getenv("BASE_URL")
        else None
    ),
    base_url=os.getenv("BASE_URL"),
)


def add_user_message(history, message):
    for x in message["files"]:
        history.append({"role": "user", "content": {"path": x}})
    if message["text"] is not None:
        history.append({"role": "user", "content": message["text"]})
    return history, gr.MultimodalTextbox(value=None, interactive=False)


def format_system_message(language: str, history: list):
    if history:
        if history[0]["role"] == "system":
            history = history[1:]
    system_message = [
        {
            "role": "system",
            "content": LANGUAGES[language],
        }
    ]
    history = system_message + history
    return history


def format_history_as_messages(history: list):
    messages = []
    current_role = None
    current_message_content = []

    for entry in history:
        content = entry["content"]

        if entry["role"] != current_role:
            if current_role is not None:
                messages.append(
                    {"role": current_role, "content": current_message_content}
                )
            current_role = entry["role"]
            current_message_content = []

        if isinstance(content, tuple):  # Handle file paths
            for temp_path in content:
                if space_host := os.getenv("SPACE_HOST"):
                    url = f"https://{space_host}/gradio_api/file%3D{temp_path}"
                else:
                    url = _convert_path_to_data_uri(temp_path)
                current_message_content.append(
                    {"type": "image_url", "image_url": {"url": url}}
                )
        elif isinstance(content, str):  # Handle text
            current_message_content.append({"type": "text", "text": content})

    if current_role is not None:
        messages.append({"role": current_role, "content": current_message_content})

    return messages


def _convert_path_to_data_uri(path) -> str:
    mime_type, _ = guess_type(path)
    with open(path, "rb") as image_file:
        data = image_file.read()
        data_uri = f"data:{mime_type};base64," + b64encode(data).decode("utf-8")
    return data_uri


def _is_file_safe(path) -> bool:
    try:
        return Path(path).is_file()
    except Exception:
        return ""


def _process_content(content) -> str | list[str]:
    if isinstance(content, str) and _is_file_safe(content):
        return _convert_path_to_data_uri(content)
    elif isinstance(content, list) or isinstance(content, tuple):
        return _convert_path_to_data_uri(content[0])
    return content


def _process_rating(rating) -> int:
    if isinstance(rating, str):
        return 0
    elif isinstance(rating, int):
        return rating
    else:
        raise ValueError(f"Invalid rating: {rating}")


def add_fake_like_data(
    history: list, session_id: str, language: str, liked: bool = False
) -> None:
    data = {
        "index": len(history) - 1,
        "value": history[-1],
        "liked": liked,
    }
    _, dataframe = wrangle_like_data(
        gr.LikeData(target=None, data=data), history.copy()
    )
    submit_conversation(dataframe, session_id, language)


def respond_system_message(
    history: list, temperature: Optional[float] = None, seed: Optional[int] = None
) -> list:  # -> list:
    """Respond to the user message with a system message

    Return the history with the new message"""
    messages = format_history_as_messages(history)
    response = client.chat.completions.create(
        messages=messages,
        max_tokens=2000,
        stream=False,
        seed=seed,
        temperature=temperature,
    )
    content = response.choices[0].message.content
    message = gr.ChatMessage(role="assistant", content=content)
    history.append(message)
    return history


def update_dataframe(dataframe: DataFrame, history: list) -> DataFrame:
    """Update the dataframe with the new message"""
    data = {
        "index": 9999,
        "value": None,
        "liked": False,
    }
    _, dataframe = wrangle_like_data(
        gr.LikeData(target=None, data=data), history.copy()
    )
    return dataframe


def wrangle_like_data(x: gr.LikeData, history) -> DataFrame:
    """Wrangle conversations and liked data into a DataFrame"""

    if isinstance(x.index, int):
        liked_index = x.index
    else:
        liked_index = x.index[0]

    output_data = []
    for idx, message in enumerate(history):
        if isinstance(message, gr.ChatMessage):
            message = message.__dict__
        if idx == liked_index:
            message["metadata"] = {"title": "liked" if x.liked else "disliked"}
        if not isinstance(message["metadata"], dict):
            message["metadata"] = message["metadata"].__dict__
        rating = message["metadata"].get("title")
        if rating == "liked":
            message["rating"] = 1
        elif rating == "disliked":
            message["rating"] = -1
        else:
            message["rating"] = 0

        message["chosen"] = ""
        message["rejected"] = ""
        if message["options"]:
            for option in message["options"]:
                if not isinstance(option, dict):
                    option = option.__dict__
                message[option["label"]] = option["value"]
        else:
            if message["rating"] == 1:
                message["chosen"] = message["content"]
            elif message["rating"] == -1:
                message["rejected"] = message["content"]

        output_data.append(
            dict(
                [(k, v) for k, v in message.items() if k not in ["metadata", "options"]]
            )
        )

    return history, DataFrame(data=output_data)


def wrangle_edit_data(
    x: gr.EditData, history: list, dataframe: DataFrame, session_id: str, language: str
) -> list:
    """Edit the conversation and add negative feedback if assistant message is edited, otherwise regenerate the message

    Return the history with the new message"""
    if isinstance(x.index, int):
        index = x.index
    else:
        index = x.index[0]

    original_message = gr.ChatMessage(
        role="assistant", content=dataframe.iloc[index]["content"]
    ).__dict__

    if history[index]["role"] == "user":
        # Add feedback on original and corrected message
        add_fake_like_data(history[: index + 2], session_id, language, liked=True)
        add_fake_like_data(
            history[: index + 1] + [original_message], session_id, language
        )
        history = respond_system_message(
            history[: index + 1],
            temperature=random.randint(1, 100) / 100,
            seed=random.randint(0, 1000000),
        )
        return history
    else:
        # Add feedback on original and corrected message
        add_fake_like_data(history[: index + 1], session_id, language, liked=True)
        add_fake_like_data(history[:index] + [original_message], session_id, language)
        history = history[: index + 1]
        # add chosen and rejected options
        history[-1]["options"] = [
            Option(label="chosen", value=x.value),
            Option(label="rejected", value=original_message["content"]),
        ]
        return history


def wrangle_retry_data(
    x: gr.RetryData, history: list, dataframe: DataFrame, session_id: str, language: str
) -> list:
    """Respond to the user message with a system message and add negative feedback on the original message

    Return the history with the new message"""
    add_fake_like_data(history, session_id, language)

    # Return the history without a new message
    history = respond_system_message(
        history[:-1],
        temperature=random.randint(1, 100) / 100,
        seed=random.randint(0, 1000000),
    )
    return history, update_dataframe(dataframe, history)


def submit_conversation(dataframe, session_id, language):
    """ "Submit the conversation to dataset repo"""
    if dataframe.empty or len(dataframe) < 2:
        gr.Info("No feedback to submit.")
        return (gr.Dataframe(value=None, interactive=False), [])

    dataframe["content"] = dataframe["content"].apply(_process_content)
    dataframe["rating"] = dataframe["rating"].apply(_process_rating)
    conversation = dataframe.to_dict(orient="records")
    conversation_data = {
        "conversation": conversation,
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "conversation_id": str(uuid.uuid4()),
        "language": language,
    }
    save_feedback(input_object=conversation_data)
    gr.Info("Submitted your feedback!")
    return (gr.Dataframe(value=None, interactive=False), [])


css = """
.options.svelte-pcaovb {
    display: none !important;
}
.option.svelte-pcaovb {
    display: none !important;
}
"""

with gr.Blocks(css=css) as demo:
    ##############################
    # Chatbot
    ##############################
    gr.Markdown("""
    # ♾️ FeeL - a real-time Feedback Loop for LMs
    """)

    with gr.Accordion("Explanation") as explanation:
        gr.Markdown(f"""
        FeeL is a collaboration between Hugging Face and MIT. It is a community-driven project to provide a real-time feedback loop for VLMs, where your feedback is continuously used to train the model. The [dataset](https://huggingface.co/datasets/{scheduler.repo_id}) and [code](https://github.com/huggingface/feel) are public.

        Start by selecting your language, chat with the model with text and images and provide feedback in different ways.

        - ✏️ Edit a message
        - 👍/👎 Like or dislike a message
        - 🔄 Regenerate a message

        Some feedback is automatically submitted allowing you to continue chatting, but you can also submit and reset the conversation by clicking "💾 Submit conversation" (under the chat) or trash the conversation by clicking "🗑️" (upper right corner).
        """)
        language = gr.Dropdown(
            choices=list(LANGUAGES.keys()), label="Language", interactive=True
        )

    session_id = gr.Textbox(
        interactive=False,
        value=str(uuid.uuid4()),
        visible=False,
    )

    chatbot = gr.Chatbot(
        elem_id="chatbot",
        editable="all",
        bubble_full_width=False,
        value=[
            {
                "role": "system",
                "content": LANGUAGES[language.value],
            }
        ],
        type="messages",
        feedback_options=["Like", "Dislike"],
    )

    chat_input = gr.MultimodalTextbox(
        interactive=True,
        file_count="multiple",
        placeholder="Enter message or upload file...",
        show_label=False,
        submit_btn=True,
    )

    dataframe = gr.Dataframe(wrap=True, label="Collected feedback")

    submit_btn = gr.Button(
        value="💾 Submit conversation",
    )

    ##############################
    # Deal with feedback
    ##############################

    language.change(
        fn=format_system_message,
        inputs=[language, chatbot],
        outputs=[chatbot],
    )

    chat_input.submit(
        fn=add_user_message,
        inputs=[chatbot, chat_input],
        outputs=[chatbot, chat_input],
    ).then(respond_system_message, chatbot, chatbot, api_name="bot_response").then(
        lambda: gr.Textbox(interactive=True), None, [chat_input]
    ).then(update_dataframe, inputs=[dataframe, chatbot], outputs=[dataframe])

    chatbot.like(
        fn=wrangle_like_data,
        inputs=[chatbot],
        outputs=[chatbot, dataframe],
        like_user_message=False,
    )

    chatbot.retry(
        fn=wrangle_retry_data,
        inputs=[chatbot, dataframe, session_id, language],
        outputs=[chatbot, dataframe],
    )

    chatbot.edit(
        fn=wrangle_edit_data,
        inputs=[chatbot, dataframe, session_id, language],
        outputs=[chatbot],
    ).then(update_dataframe, inputs=[dataframe, chatbot], outputs=[dataframe])

    submit_btn.click(
        fn=submit_conversation,
        inputs=[dataframe, session_id, language],
        outputs=[dataframe, chatbot],
    )
    demo.load(
        lambda: str(uuid.uuid4()),
        inputs=[],
        outputs=[session_id],
    )

demo.launch()

# /private/var/folders/9t/msy700h16jz3q35qvg4z1ln40000gn/T/gradio/a5013b9763ad9f2192254540fee226539fbcd1382cbc2317b916aef469bb01b9/Screenshot 2025-01-13 at 08.02.26.png
