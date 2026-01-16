# OpenTCM

OpenTCM is a web application designed for intelligent question answering in Traditional Chinese Medicine (TCM). Built upon a knowledge graph and a Large Language Model (LLM, using Kimi as an example), OpenTCM combines structured TCM knowledge with the understanding and generation capabilities of LLMs. This enables the system to provide users with well-sourced, comprehensive, and easy-to-understand TCM information. The project supports streaming responses to enhance user interaction.

Our paper, *"OpenTCM: A GraphRAG-Empowered LLM-based System for Traditional Chinese Medicine Knowledge Retrieval and Diagnosis"*, was accepted by BIGCOM25 and received the Best Paper Award 🏆.

> **Note**: Due to privacy and copyright reasons, the dataset is not included. You can download it yourself.

---

## Technical Architecture

The project follows a front-end/back-end architecture:

### Backend (Python & Flask)
- **Flask**: Serves as the web framework, providing API endpoints.
- **Core Logic**:
  - `TCMKnowledgeGraph`: Handles the construction, management, and querying of the knowledge graph.
  - `GraphRAG`: Implements the GraphRAG pipeline, including interaction with the Kimi API, prompt construction, knowledge retrieval, and answer synthesis.
  - `TCMGraphRAGApp`: Encapsulates the application logic for API calls.
- **Dependencies**:
  - `requests`: Communicates with external LLM APIs (e.g., Moonshot Kimi).
  - `python-dotenv`: Manages environment variables (e.g., API keys).
  - `Flask-CORS`: Handles Cross-Origin Resource Sharing.

### Frontend (HTML, CSS, Vanilla JavaScript)
- Built with pure HTML/CSS/JS for the user interface.
- Uses the `Workspace` API for asynchronous communication with the backend.
- Implements streaming reception and display of responses from the backend.

---

## Technology Stack

- **Backend**: Python 3.x, Flask, Pandas, NetworkX, Requests
- **Frontend**: HTML5, CSS3, JavaScript (ES6+)
- **LLM API**: Moonshot (Kimi) API (can be replaced with other compatible APIs)
- **Environment Management**: python-dotenv

---

## Project Structure

```
├── /data
│   ├── TCMKG.py
│   └── data.csv    # Add your dataset here
├── app.py
├── GraphRAG.py
├── .env
├── requirements.txt
├── /templates
│   ├── welcome.html
│   └── chat.html
├── /static
│   ├── /css
│   │   └── style.css
│   └── /images
│       └── opentcm_logo.png
└── README.md
```

---

## How to Use

1. **Set Up the Environment**:
   - Create a Conda environment (Python 3.10 or above is recommended).
2. **Prepare the Dataset**:
   - Use `TCMKG.py` to create your dataset.
3. **Run the Application**:
   - Execute `app.py`.
4. **Access the Web Interface**:
   - Open your web browser and navigate to:
     - `http://127.0.0.1:8000/`
     - or `http://localhost:8000/`
   - You will see the welcome page. Click the **Start Chat** button to go to the chat interface.
   - Type your TCM-related question into the input box and click **Send** or press **Enter**.
   - The system will process your query and progressively display the answer on the interface via streaming output.

---

Enjoy exploring the world of Traditional Chinese Medicine with OpenTCM! 🌿
