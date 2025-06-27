# ArXiv Recommender Interface (Dynamic RAG System)

This project implements a dynamic Retrieval-Augmented Generation (RAG) system integrated with user feedback for recommending relevant academic papers from ArXiv.

## Features

* **Semantic Search**: Uses semantic similarity to retrieve relevant documents from ArXiv.
* **Feedback Integration**: Collects user ratings to dynamically adjust recommendations and query expansions.
* **Interactive UI**: Provides an intuitive, responsive interface for searching and rating recommendations.
* **AI-Powered Recommendations**: Generates insightful recommendations and analyses based on user queries and retrieved papers.

## Tech Stack

### Frontend

* HTML, CSS, JavaScript
* Responsive UI design with dynamic content loading

### Backend

* Python (Flask)
* SQLite for feedback storage
* SentenceTransformers, ChromaDB, LlamaIndex, HuggingFace embeddings
* OpenAI API for language modeling (GPT-4)

## Project Structure

```
project-root/
├── app.py                     # Flask API endpoints
├── rag_system.py              # Handles semantic search and recommendation logic
├── feedback_system.py         # Manages feedback collection and learning
├── index.html                 # Frontend UI
├── feedback.db                # SQLite database for feedback (generated at runtime)
├── query_expansions.pkl       # Learned query expansions (generated at runtime)
├── chroma_db/                 # Vector database for papers (generated at runtime)
└── requirements.txt           # Python dependencies
```

## Setup Instructions

### Prerequisites

* Python 3.8+
* OpenAI API key

### Installation

1. **Clone the repository**:

```sh
git clone <repository-url>
cd project-root
```

2. **Install dependencies**:

```sh
pip install -r requirements.txt
```

3. **Set up environment variables**:

Create a `.env` file in the project root and add your OpenAI API key:

```
OPENAI_API_KEY=your_openai_api_key
```

### Running the Application

Start the Flask backend:

```sh
python app.py
```

The application will be available at:

```
http://localhost:5000
```

## Usage

* **Search**: Enter a query in the search box and specify the number of results.
* **Rate Results**: Click stars to rate papers and provide feedback.
* **Quick Queries**: Use quick-action buttons for common topics.

## API Endpoints

* `POST /api/search`: Retrieve relevant papers based on query
* `POST /api/feedback`: Submit user feedback for specific results
* `GET /api/status`: Get system initialization status
* `GET /api/analytics`: Get feedback analytics and insights

## Feedback and Analytics

Feedback submitted by users dynamically influences future recommendations through query expansion and result ranking.

Feedback analytics include:

* Popular and struggling queries
* Most relevant papers based on user interactions
* Trends over time

## Development and Debugging

* Detailed logging provided in `app.log`
* SQLite database (`feedback.db`) accessible for direct inspection and analysis

## License

This project is open-source and available under the MIT License.
