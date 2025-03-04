from agents import *
from utils import *
import streamlit as st
import tempfile
import time

def get_response(image_path, query, extraction=None):
    
    # If no extraction exists (or image changed), generate it.
    if extraction is None:
        extraction = extract_agent.run(query, image_path)
    
    # Run the multi-agent router to get the analysis or answer to the query.
    output = agent_router.run("Here is extracted information about the chart: " + extraction + "\n\n" + query)
    answer = output["execution"]['response']


    # Run the prompt recommendation agent to get recommended prompts.
    chart_recommendation = parse_json_from_string(recommend_agent.run(f"""This is the query: {query}

Here is the answer question: {answer}. Give me some short recommended prompts based on this information.
"""))

    
    return answer, chart_recommendation["recommended_prompts"], extraction


if __name__ == "__main__":
    # Enable wide mode
    st.set_page_config(layout="wide")

    st.title("Chatbot Image Analysis and Recommendation App")

    # Initialize session state variables.
    if "image_path" not in st.session_state:
        st.session_state.image_path = None
    if "extraction" not in st.session_state:
        st.session_state.extraction = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "new_image_file" not in st.session_state:
        st.session_state.new_image_file = None

    # Display previous chat history
    for chat in st.session_state.chat_history:
        with st.chat_message(chat["sender"]):
            st.markdown(chat["message"])

    with st.sidebar:
        st.header("Upload Chart Image")
        new_image_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"], key="uploader")
        # When a new image is uploaded, save it to a temporary file.
        # If the file differs from the previous one, update image_path and reset extraction.
        if new_image_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                tmp.write(new_image_file.read())
                new_path = tmp.name
            if st.session_state["new_image_file"] != new_image_file:
                st.session_state.image_path = new_path
                st.session_state.extraction = None  # Reset extraction when image changes.
                st.session_state.chat_history = []  # Reset chat history when image changes.
                st.session_state["new_image_file"] = new_image_file

            # Display the uploaded image.
            image = PIL.Image.open(new_path)
            st.image(image, caption="Uploaded Chart", use_column_width=True)


    # Get user query from chat input or recommendation button
    user_query = st.chat_input("Message chatbot")
    if user_query:
        
        if not st.session_state.image_path:
            st.warning("Please upload a chart image first.")
            st.stop()

        st.session_state.chat_history.append({
            "sender": "human",
            "message": user_query
        })

        with st.chat_message("human"):
            st.markdown(user_query)

        with st.spinner("Thinking..."):
            start_time = time.time()
            
            # Get analysis and recommendations
            answer, recommendations, extraction = get_response(
                st.session_state.image_path, user_query, st.session_state.extraction
            )
            end_time = time.time()
            
            if st.session_state.extraction is None:
                st.session_state.extraction = extraction

            # Format the AI response
            response_text = f"{answer}"
            response_text += f"\n\n_Response time: {end_time-start_time:.2f} seconds_\n\n"

            # Append AI response to chat history
            st.session_state.chat_history.append({
                "sender": "assistant",
                "message": response_text
            })

            # Display AI response
            with st.chat_message("assistant"):
                st.markdown(response_text)

                # Display recommended prompts
                st.markdown(f"**Recommended**\n")
                for idx, rec in enumerate(recommendations):
                    st.write(f"{idx+1}. {rec}")

