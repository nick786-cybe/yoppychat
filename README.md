# YoppyChat AI 🎙️

YoppyChat is a powerful web application that transforms any YouTube channel into a smart, conversational AI assistant. It allows fans and viewers to chat directly with an AI persona of their favorite creator, trained on the channel's entire video library.

This project is built with a modern Python backend using Flask and a dynamic, vanilla JavaScript frontend. It leverages a sophisticated RAG (Retrieval-Augmented Generation) pipeline with hybrid search to provide accurate, context-aware answers that capture the creator's unique voice and style.



## ✨ Key Features

- **Automated Channel Processing:** Simply provide a YouTube channel URL, and a background task will fetch video transcripts, generate embeddings, and build the AI's knowledge base.
- **Persona-Driven Chat:** The AI doesn't just give factual answers; it adopts the personality, tone, and style of the YouTuber it represents, thanks to a robust persona prompt.
- **Accurate Sourcing:** Every answer is backed by sources, linking directly to the YouTube video and timestamp where the information was found.
- **Hybrid Search:** Combines the precision of keyword search with the contextual power of semantic vector search to find the most relevant information for any question.
- **Dynamic UI:** A responsive, single-page-style chat interface with real-time streaming answers, typing animations, and a clean user experience.
- **Creator-Focused Analytics:** Automatically generates a channel summary and a list of popular topics discussed in the videos, providing valuable insights for creators.
- **SEO & Social Sharing:** Optimized with rich meta tags (Open Graph) for beautiful and informative link previews on social media.
- **Scalable Background Tasks:** Uses a Redis and Huey-based queue to handle long-running processes like video transcription and embedding without blocking the user interface.

---

## ⚙️ How It Works

1.  **Add a Channel:** A user provides a YouTube channel URL.
2.  **Background Processing:** A `Huey` background task is initiated. It uses the YouTube Data API to fetch video details and transcripts.
3.  **Knowledge Base Creation:** The transcripts are chunked, and embeddings are created using a configurable AI model (e.g., OpenAI, Groq). These embeddings are stored in a Supabase Postgres database with `pgvector`.
4.  **Topic & Summary Generation:** A powerful LLM (e.g., Llama 3) analyzes the content to generate a channel summary and a list of popular topics.
5.  **Chat Interaction:** When a user asks a question, the app performs a hybrid search:
    * **Keyword Search:** Scans video titles for direct matches.
    * **Semantic Search:** Queries the vector database to find the most contextually relevant video chunks.
6.  **Answer Generation:** The combined, re-ranked search results are injected into a detailed persona prompt, and an LLM generates a conversational, in-character response that is streamed back to the user in real-time.

---

## 🚀 Technology Stack

- **Backend:** Flask, Gunicorn
- **Database:** Supabase (PostgreSQL with pgvector)
- **Authentication:** Supabase Auth
- **Background Tasks:** Huey, Redis
- **AI / LLMs:** OpenAI, Groq (configurable via `.env`)
- **Frontend:** Vanilla JavaScript, HTML5, CSS3
- **Deployment:** Nginx, Google Cloud VM (or any Linux server)

---

## 🛠️ Setup and Installation

### Prerequisites

- Python 3.10+
- A Supabase project with a database
- A Redis instance
- API keys for YouTube, OpenAI, and/or Groq

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd talktoyoutuber-v9
