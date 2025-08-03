import os
import json
import time
import logging
import numpy as np
import requests
from functools import lru_cache
from typing import Iterator
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder

from . import prompts
from .supabase_client import get_supabase_client, get_supabase_admin_client

# Load environment variables from .env file
load_dotenv()
cross_encoder = None

def _get_api_key(provider: str) -> str:
    """Gets the appropriate API key from environment variables."""
    key_map = {
        'openai': 'OPENAI_API_KEY',
        'gemini': 'GEMINI_API_KEY',
        'groq': 'GROQ_API_KEY'
    }
    env_var = key_map.get(provider)
    return os.environ.get(env_var) if env_var else None

# --- Provider-Specific Embedding Functions ---
def _create_openai_embedding(texts: list, model, api_key):
    try:
        import openai
        client = openai.OpenAI(api_key=api_key)
        response = client.embeddings.create(input=texts, model=model)
        return [np.array(data.embedding).astype('float32') for data in response.data]
    except Exception as e:
        logging.error(f"Failed to create OpenAI batch embedding: {e}", exc_info=True)
        return None

def _create_groq_embedding(texts: list, model, api_key):
    logging.warning("Groq embedding function is not implemented.")
    return [None] * len(texts)

def _create_gemini_embedding(texts: list, model, api_key):
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model_name = f"models/{model}" if not model.startswith('models/') else model
        result = genai.embed_content(model=model_name, content=texts, task_type="retrieval_document")
        return [np.array(embedding).astype('float32') for embedding in result['embedding']]
    except Exception as e:
        logging.error(f"Failed to create Gemini batch embedding: {e}", exc_info=True)
        return None

def _create_ollama_embedding(texts: list, model, ollama_url):
    embeddings = []
    for text in texts:
        try:
            response = requests.post(f"{ollama_url}/api/embeddings", json={"model": model, "prompt": text}, timeout=30)
            if response.status_code == 200 and 'embedding' in response.json():
                embeddings.append(np.array(response.json()['embedding']).astype('float32'))
            else:
                logging.error(f"Failed to get Ollama embedding for a chunk. Status: {response.status_code}")
                embeddings.append(None)
        except Exception as e:
            logging.error(f"Exception during Ollama embedding for a chunk: {e}", exc_info=True)
            embeddings.append(None)
    return [emb for emb in embeddings if emb is not None]

EMBEDDING_PROVIDER_MAP = {
    'openai': _create_openai_embedding,
    'gemini': _create_gemini_embedding,
    'ollama': _create_ollama_embedding,
    'groq': _create_groq_embedding
}

def get_routed_context(question, channel_data, user_id, access_token):
    """
    Intelligently routes a user's question to the correct retrieval method.
    """
    question_lower = question.lower()
    
    if 'latest video' in question_lower or 'newest video' in question_lower or 'most recent' in question_lower:
        print("Query routed to: get_latest_video")
        if channel_data and channel_data.get('videos'):
            latest_video_id = channel_data['videos'][0]['video_id']
            admin_supabase = get_supabase_admin_client()
            response = admin_supabase.table('embeddings').select('*').eq('video_id', latest_video_id).execute()
            if response.data:
                print(f"Metadata search successful. Found {len(response.data)} chunks for the latest video.")
                return [row['metadata'] for row in response.data]
        
        print("Metadata search for latest video failed or returned no chunks. Falling back to semantic search.")

    print("Query routed to: semantic_search")
    video_ids = {v['video_id'] for v in channel_data.get('videos', [])} if channel_data else None
    return search_and_rerank_chunks(question, user_id, access_token, video_ids)


# --- Provider-Specific LLM STREAMING Functions ---
def _get_openai_answer_stream(prompt, model, api_key, **kwargs):
    try:
        import openai
        base_url = kwargs.get('base_url')
        temperature = kwargs.get('temperature', 0.7)
        client = openai.OpenAI(api_key=api_key, base_url=base_url)
        response_stream = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=temperature,
            stream=True
        )
        for chunk in response_stream:
            content = chunk.choices[0].delta.content
            if content:
                yield content
    except Exception as e:
        logging.error(f"Failed to get OpenAI stream: {e}", exc_info=True)
        yield "Error: Could not get a response from the provider."

def _get_groq_answer_stream(prompt, model, api_key, **kwargs):
    try:
        headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}
        data = {'model': model, 'messages': [{"role": "user", "content": prompt}], 'max_tokens': 1024, 'temperature': 0.7, 'stream': True}
        with requests.post('https://api.groq.com/openai/v1/chat/completions', headers=headers, json=data, stream=True) as response:
            for chunk in response.iter_lines():
                if chunk and chunk.startswith(b'data: '):
                    chunk_data = chunk.decode('utf-8')[6:].strip()
                    if chunk_data != '[DONE]':
                        try:
                            json_data = json.loads(chunk_data)
                            content = json_data['choices'][0]['delta'].get('content')
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            continue
    except Exception as e:
        logging.error(f"Failed to get Groq stream: {e}", exc_info=True)
        yield "Error: Could not get a response from the provider."

def _get_gemini_answer_stream(prompt, model, api_key, **kwargs):
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        gemini_model = genai.GenerativeModel(model)
        response_stream = gemini_model.generate_content(prompt, generation_config=genai.types.GenerationConfig(max_output_tokens=1024, temperature=0.7), stream=True)
        for chunk in response_stream:
            if chunk.text:
                yield chunk.text
    except Exception as e:
        logging.error(f"Failed to get Gemini stream: {e}", exc_info=True)
        yield "Error: Could not get a response from the provider."

def _get_ollama_answer_stream(prompt, model, ollama_url, **kwargs):
    try:
        response = requests.post(f"{ollama_url}/api/chat", json={"model": model, "messages": [{"role": "user", "content": prompt}], "stream": True}, stream=True)
        for chunk in response.iter_lines():
            if chunk:
                json_data = json.loads(chunk)
                content = json_data['message'].get('content')
                if content:
                    yield content
    except Exception as e:
        logging.error(f"Failed to get Ollama stream: {e}", exc_info=True)
        yield "Error: Could not get a response from the provider."

LLM_STREAM_PROVIDER_MAP = {
    'openai': _get_openai_answer_stream,
    'groq': _get_groq_answer_stream,
    'gemini': _get_gemini_answer_stream,
    'ollama': _get_ollama_answer_stream
}

def rerank_with_cross_encoder(query: str, chunks: list):
    """
    Re-ranks chunks using a Cross-Encoder model, lazy-loaded on first call.
    """
    global cross_encoder
    if cross_encoder is None:
        try:
            print("Loading Cross-Encoder model for the first time...")
            cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            print("Cross-Encoder model loaded successfully.")
        except Exception as e:
            logging.warning(f"Could not load Cross-Encoder model: {e}. Re-ranking will be disabled.")
            cross_encoder = 'failed_to_load'
    if cross_encoder == 'failed_to_load' or not chunks:
        return chunks
    
    start_time = time.perf_counter()
    print(f"Re-ranking {len(chunks)} chunks for query: '{query[:50]}...'")
    pairs = [[query, chunk['chunk_text']] for chunk in chunks]
    scores = cross_encoder.predict(pairs)
    for chunk, score in zip(chunks, scores):
        chunk['relevance_score'] = float(score)
    sorted_chunks = sorted(chunks, key=lambda x: x.get('relevance_score', 0), reverse=True)
    end_time = time.perf_counter()
    print(f"[TIME_LOG] Re-ranking with Cross-Encoder took {end_time - start_time:.4f} seconds.")
    return sorted_chunks

def search_and_rerank_chunks(query: str, user_id: str, access_token: str, video_ids: set = None):
    total_start_time = time.perf_counter()
    
    def create_query_embedding(query_text):
        provider = os.environ.get('EMBED_PROVIDER', 'openai')
        model = os.environ.get('EMBED_MODEL')
        api_key = _get_api_key(provider)
        ollama_url = os.environ.get('OLLAMA_URL')
        embedding_function = EMBEDDING_PROVIDER_MAP.get(provider)
        if not embedding_function:
            logging.error(f"Unsupported embedding provider: {provider}")
            return None
        embeddings = None
        if provider == 'ollama':
            embeddings = embedding_function([query_text], model, ollama_url=ollama_url)
        else:
            if not api_key:
                logging.error(f"API key for {provider} not found in environment variables.")
                return None
            embeddings = embedding_function([query_text], model, api_key=api_key)
        return embeddings[0] if embeddings else None

    try:
        embedding_start_time = time.perf_counter()
        query_embedding = create_query_embedding(query)
        embedding_end_time = time.perf_counter()
        print(f"[TIME_LOG] Query embedding creation took {embedding_end_time - embedding_start_time:.4f} seconds.")
        if query_embedding is None:
            logging.error("Failed to create query embedding.")
            return []

        supabase = get_supabase_client(access_token=access_token) if access_token else get_supabase_admin_client()
        match_params = {
            'query_embedding': query_embedding.tolist(),
            'match_threshold': 0.4,
            'match_count': 50,
            'p_video_ids': list(video_ids) if video_ids else None
        }
        
        print(f"Calling 'match_embeddings' with params:")
        print(f"  - user_id: {user_id}")
        print(f"  - video_ids: {'All' if not video_ids else list(video_ids)}")
        print(f"  - match_threshold: {match_params['match_threshold']}")
        
        rpc_start_time = time.perf_counter()
        response = supabase.rpc('match_embeddings', match_params).execute()
        rpc_end_time = time.perf_counter()
        print(f"[TIME_LOG] Supabase 'match_embeddings' RPC call took {rpc_end_time - rpc_start_time:.4f} seconds.")
        
        if not response.data:
            logging.warning("Supabase RPC call returned no data.")
            return []
        print(f"SUCCESS: Received {len(response.data)} results from Supabase.")

        initial_results = []
        for row in response.data:
            chunk_data = row['metadata']
            chunk_data['similarity_score'] = row['similarity']
            initial_results.append(chunk_data)

        CHUNKS_TO_RERANK = 15 
        print(f"Passing the top {CHUNKS_TO_RERANK} results to the re-ranker.")
        if os.environ.get('ENABLE_RERANKING', 'true').lower() == 'true':
            reranked_results = rerank_with_cross_encoder(query, initial_results[:CHUNKS_TO_RERANK])
        else:
            print("Re-ranking is disabled via environment variable. Skipping.")
            reranked_results = initial_results
        
        filtering_start_time = time.perf_counter()
        top_k = int(os.environ.get('TOP_K', 5))
        final_results = []
        video_counts = {}
        for chunk in reranked_results:
            video_id = chunk['video_id']
            if video_counts.get(video_id, 0) < 2:
                final_results.append(chunk)
                video_counts[video_id] = video_counts.get(video_id, 0) + 1
            if len(final_results) >= top_k:
                break
        filtering_end_time = time.perf_counter()
        print(f"[TIME_LOG] Final result diversification/filtering took {filtering_end_time - filtering_start_time:.4f} seconds.")
        print(f"Selected {len(final_results)} diverse, highly relevant chunks for the context.")
        
        total_end_time = time.perf_counter()
        print(f"[TIME_LOG] Total search_and_rerank_chunks took {total_end_time - total_start_time:.4f} seconds.")
        return final_results
    
    except Exception as e:
        if hasattr(e, 'message') and 'JWT expired' in e.message:
            logging.warning("Caught expired JWT error. Notifying frontend.")
            return "JWT_EXPIRED"
        logging.error(f"Error in search_and_rerank_chunks: {e}", exc_info=True)
        return []

def answer_question_stream(question_for_prompt: str, question_for_search: str, channel_data: dict = None, video_ids: set = None, user_id: str = None, access_token: str = None) -> Iterator[str]:
    """
    Finds relevant context from documents and streams an answer to the user's question.
    """
    from tasks import post_answer_processing_task
    
    total_request_start_time = time.perf_counter()

    # --- Configuration Logging ---
    llm_provider = os.environ.get('LLM_PROVIDER', 'groq')
    llm_model = os.environ.get('MODEL_NAME', 'Not Set')
    embed_provider = os.environ.get('EMBED_PROVIDER', 'openai')
    embed_model = os.environ.get('EMBED_MODEL', 'Not Set')
    api_key = _get_api_key(llm_provider)
    masked_api_key = f"{api_key[:5]}...{api_key[-4:]}" if api_key else "Not Set"
    base_url = os.environ.get('OPENAI_API_BASE_URL', 'Default')

    print("--- Answering Question with the following configuration ---")
    print(f"  LLM Provider:         {llm_provider}")
    print(f"  LLM Model:            {llm_model}")
    print(f"  Embedding Provider:   {embed_provider}")
    print(f"  Embedding Model:      {embed_model}")
    print(f"  API Key Used:         {masked_api_key}")
    if llm_provider == 'openai' and base_url != 'Default':
        print(f"  OpenAI Base URL:      {base_url}")
    print("---------------------------------------------------------")

    # --- Separate chat history from the original question ---
    chat_history_for_prompt = ""
    original_question = question_for_prompt 
    history_marker = "Now, answer this new question, considering the history as context:\n"
    if history_marker in question_for_prompt:
        parts = question_for_prompt.split(history_marker)
        history_section = parts[0]
        original_question = parts[1]
        chat_history_for_prompt = history_section.replace("Given the following conversation history:\n", "").replace("--- End History ---\n\n", "")
    print(f"Answering question for user {user_id}: '{original_question[:100]}...'")
    
    if not user_id:
        yield "data: {\"error\": \"User not identified. Please log in.\"}\n\n"
        return

    # --- Use the dedicated `question_for_search` to find relevant documents ---
    relevant_chunks = get_routed_context(question_for_search, channel_data, user_id, access_token)

    if relevant_chunks == "JWT_EXPIRED":
        yield 'data: {"error": "JWT_EXPIRED"}\n\n'
        return
    
    if not relevant_chunks:
        yield "data: {\"answer\": \"I couldn't find any relevant information in the documents to answer your question.\"}\n\n"
        yield "data: [DONE]\n\n"
        return

    # --- The rest of the function for processing and streaming the answer ---
    sources_dict = {}
    for chunk in relevant_chunks:
        if chunk['video_url'] not in sources_dict:
            sources_dict[chunk['video_url']] = {'title': chunk['video_title'], 'url': chunk['video_url']}
    formatted_sources = sorted(list(sources_dict.values()), key=lambda s: s['title'])
    yield f"data: {json.dumps({'sources': formatted_sources})}\n\n"

    context_parts = [f"From video '{chunk['video_title']}' (uploaded on {chunk.get('upload_date', 'N/A')}): {chunk['chunk_text']}" for chunk in relevant_chunks]
    context = '\n\n'.join(context_parts)
    
    if channel_data:
        creator_name = channel_data.get('creator_name', channel_data.get('channel_name', 'the creator'))
        prompt = prompts.PERSONA_PROMPT.format(
            creator_name=creator_name, 
            channel_name=channel_data['channel_name'], 
            context=context, 
            question=original_question,
            chat_history=chat_history_for_prompt or "This is the first message in the conversation."
        )
    else:
        prompt = prompts.NEUTRAL_ASSISTANT_PROMPT.format(context=context, question=original_question)

    # --- This block now contains all necessary variables ---
    llm_provider = os.environ.get('LLM_PROVIDER', 'groq')
    model = os.environ.get('MODEL_NAME')
    api_key = _get_api_key(llm_provider)
    ollama_url = os.environ.get('OLLAMA_URL')
    openai_base_url = os.environ.get('OPENAI_API_BASE_URL') # <-- FIX: Definition added
    temperature = float(os.environ.get('LLM_TEMPERATURE', 0.7))
    
    stream_function = LLM_STREAM_PROVIDER_MAP.get(llm_provider)
    if not stream_function:
        yield "data: {\"answer\": \"Error: The selected LLM provider does not support streaming.\"}\n\n"
        yield "data: [DONE]\n\n"
        return

    full_answer = ""
    llm_stream_start_time = time.perf_counter()
    first_token_time_logged = False
    
    stream_kwargs = {
        'api_key': api_key,
        'ollama_url': ollama_url,
        'base_url': openai_base_url,
        'temperature': temperature
    }

    for chunk in stream_function(prompt, model, **stream_kwargs):
        if not first_token_time_logged:
            first_token_end_time = time.perf_counter()
            print(f"[TIME_LOG] LLM time to first token: {first_token_end_time - llm_stream_start_time:.4f} seconds.")
            first_token_time_logged = True
            
        full_answer += chunk
        yield f"data: {json.dumps({'answer': chunk})}\n\n"

    llm_stream_end_time = time.perf_counter()
    if not first_token_time_logged and not full_answer:
        print("[TIME_LOG] LLM stream produced no output.")
    else:
        print(f"[TIME_LOG] Full LLM stream generation took {llm_stream_end_time - llm_stream_start_time:.4f} seconds.")
    
    yield "data: [DONE]\n\n"
    
    if full_answer and "Error:" not in full_answer:
        post_answer_processing_task(
            user_id=user_id,
            channel_name=channel_data['channel_name'] if channel_data else 'general',
            question=original_question,
            answer=full_answer,
            sources=formatted_sources
        )
    
    total_request_end_time = time.perf_counter()
    print(f"[TIME_LOG] Total answer_question_stream request (end-to-end) took {total_request_end_time - total_request_start_time:.4f} seconds.")

def _get_openai_answer_non_stream(prompt, model, api_key, **kwargs):
    """Gets a single, non-streamed response from an OpenAI-compatible API."""
    try:
        import openai
        base_url = kwargs.get('base_url')
        temperature = kwargs.get('temperature', 0.2) # Lower temp for factual extraction
        client = openai.OpenAI(api_key=api_key, base_url=base_url)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=temperature,
            stream=False
        )
        content = response.choices[0].message.content
        return content
    except Exception as e:
        logging.error(f"Failed to get OpenAI non-stream response: {e}", exc_info=True)
        return ""
    
def extract_topics_from_text(text_sample: str) -> list:
    """Uses an LLM to extract a list of topics from a sample of text."""
    # This task is best suited for a powerful model, so we'll default to OpenAI provider
    llm_provider = os.environ.get('LLM_PROVIDER', 'openai')
    model = os.environ.get('MODEL_NAME')
    api_key = _get_api_key(llm_provider)
    base_url = os.environ.get('OPENAI_API_BASE_URL')

    if not all([llm_provider, model, api_key]):
        logging.warning("LLM provider not fully configured. Skipping topic extraction.")
        return []

    prompt = prompts.TOPIC_EXTRACTION_PROMPT.format(context=text_sample)

    try:
        # Using a non-streaming call is simpler for this kind of extraction task
        topic_string = _get_openai_answer_non_stream(prompt, model, api_key, base_url=base_url)

        if topic_string:
            # Clean up the string and split it into a list
            topics = [topic.strip() for topic in topic_string.split(',') if topic.strip()]
            logging.info(f"Extracted topics: {topics}")
            return topics
        return []
    except Exception as e:
        logging.error(f"Error during topic extraction: {e}", exc_info=True)
        return []
    
def generate_channel_summary(text_sample: str) -> str:
    """Uses an LLM to generate a short summary for the channel."""
    llm_provider = os.environ.get('LLM_PROVIDER', 'openai')
    model = os.environ.get('MODEL_NAME')
    api_key = _get_api_key(llm_provider)
    base_url = os.environ.get('OPENAI_API_BASE_URL')

    if not all([llm_provider, model, api_key]):
        logging.warning("LLM provider not fully configured. Skipping summary generation.")
        return ""

    prompt = prompts.CHANNEL_SUMMARY_PROMPT.format(context=text_sample)

    try:
        # We can increase the temperature slightly for more creative summaries
        summary = _get_openai_answer_non_stream(
            prompt, model, api_key, base_url=base_url, temperature=0.5
        )
        # Basic cleanup to remove leading/trailing whitespace
        return summary.strip() if summary else ""
    except Exception as e:
        logging.error(f"Error during summary generation: {e}", exc_info=True)
        return ""