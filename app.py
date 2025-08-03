import logging
from functools import wraps
from utils.youtube_utils import is_youtube_video_url, clean_youtube_url
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session, Response
import os
import json
import secrets
from datetime import datetime, timedelta, timezone
from tasks import huey, process_channel_task, sync_channel_task, process_telegram_update_task,delete_channel_task
from utils.qa_utils import answer_question_stream,search_and_rerank_chunks
from utils.supabase_client import get_supabase_client, get_supabase_admin_client,refresh_supabase_session
from utils.history_utils import get_chat_history, save_chat_history
from utils.telegram_utils import set_webhook, get_bot_token_and_url
from utils.config_utils import load_config
from utils.subscription_utils import get_user_subscription_status,subscription_required
from utils import prompts
import time
import redis
from postgrest.exceptions import APIError
from markupsafe import Markup
import markdown
from huey.exceptions import TaskException
from postgrest.exceptions import APIError
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)


load_dotenv()

app = Flask(__name__)

@app.context_processor
def inject_subscription_data():
    if 'user' in session:
        user_id = session['user']['id']
        subscription_data = get_user_subscription_status(user_id)
        return dict(subscription=subscription_data, user=session.get('user'))
    return dict(subscription=None, user=None)
@app.template_filter('markdown')
def markdown_filter(text):
    return Markup(markdown.markdown(text))
app.secret_key = os.environ.get('SECRET_KEY', 'a_default_dev_secret_key')

try:
    redis_client = redis.from_url(os.environ.get('REDIS_URL'))
except Exception:
    redis_client = None

def get_user_channels():
    """
    Gets all channels linked to the current user through the user_channels table.
    This is the corrected query for the new database structure.
    """
    if 'user' not in session:
        return {}

    user_id = session['user']['id']
    access_token = session.get('access_token')
    supabase = get_supabase_client(access_token)
    
    if not supabase:
        return {}

    try:
        # This is the new, correct query. It joins 'user_channels' with 'channels'.
        # The (*) tells Supabase to fetch all columns from the linked 'channels' table.
        response = supabase.table('user_channels').select('channels(*)').eq('user_id', user_id).execute()
        
        if response.data:
            # The data is nested, so we need to extract it properly.
            linked_channels = [item['channels'] for item in response.data if item.get('channels')]
            # Convert the list into the dictionary format your templates expect.
            user_channels = {item['channel_name']: item for item in linked_channels}
            return user_channels
        
    except APIError as e:
        # This will gracefully handle expired tokens and other API issues.
        logging.error(f"Could not fetch user channels due to APIError: {e.message}")
        if 'JWT expired' in e.message:
            session.clear() # Clear session on expired token
        
    except Exception as e:
        logging.error(f"An unexpected error occurred in get_user_channels: {e}")

    return {} # Return empty dictionary on failure

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            logging.warning("Unauthorized access: User not in session.")
            return jsonify({'status': 'error', 'message': 'Authentication required. Please log in again.'}), 401

        try:
            # This will run the code inside your route (e.g., the channel function)
            return f(*args, **kwargs)
        except APIError as e:
            # This block now specifically catches the expired token error
            if 'JWT' in e.message and 'expired' in e.message:
                logging.warning("Caught expired JWT. Clearing session and sending 401.")
                session.clear()  # Clear the stale server-side session
                return jsonify({
                    'status': 'error',
                    'message': 'Your session has expired. The page will now reload.',
                    'action': 'logout' # This is a useful hint for your frontend JavaScript
                }), 401
            else:
                # If it's a different API error, let the main error handler catch it
                raise e

    return decorated_function

@app.route('/auth/callback')
def auth_callback():
    """
    This page is the redirect target from Google. It contains the Supabase JS
    that handles the session from the URL hash. After the JS syncs the session,
    it redirects back to the main channel page.
    """
    # --- FIX: Pass the environment variables to the callback template ---
    return render_template(
        'callback.html',
        SUPABASE_URL=('https://glmtdjegibqaojifyxzf.supabase.co'),
        SUPABASE_ANON_KEY=('eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImdsbXRkamVnaWJxYW9qaWZ5eHpmIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTA2NjA2MDUsImV4cCI6MjA2NjIzNjYwNX0.AFqnq49ZBp-jiJ1GEHr4QDNoL0QGw3dPYFRu_2YvNVA')
    )

@app.route('/auth/set-cookie', methods=['POST'])
def set_auth_cookie():
    """
    This endpoint receives an access token from the client-side Supabase instance.
    It verifies the token, gets the user details, and sets the Flask session.
    """
    try:
        data = request.get_json()
        token = data.get('token')
        refresh_token = data.get('refresh_token') # <-- ADD THIS LINE

        if not token:
            return jsonify({'status': 'error', 'message': 'Token is missing.'}), 400

        supabase = get_supabase_client(token)
        user_response = supabase.auth.get_user(token)
        
        user = user_response.user
        if not user:
            return jsonify({'status': 'error', 'message': 'Invalid token.'}), 401

        session['user'] = user.dict()
        session['access_token'] = token
        session['refresh_token'] = refresh_token # <-- ADD THIS LINE
        
        logging.info(f"Successfully set session for user: {user.email}")
        return jsonify({'status': 'success', 'message': 'Session set successfully.'})

    except Exception as e:
        logging.error(f"Error in set-cookie endpoint: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': 'An internal error occurred.'}), 500

@app.route('/stream_answer', methods=['POST'])
@login_required
@subscription_required('query')
def stream_answer():
    user_id = session['user']['id']
    question = request.form.get('question', '').strip()
    channel_name = request.form.get('channel_name')
    access_token = session.get('access_token')

    # --- (Your existing code for history and chat limits is fine) ---
    MAX_CHAT_MESSAGES = 20
    current_channel_name_for_history = channel_name or 'general'
    history = get_chat_history(user_id, current_channel_name_for_history, access_token=access_token)

    if len(history) >= MAX_CHAT_MESSAGES:
        def limit_exceeded_stream():
            error_data = {
                "error": "QUERY_LIMIT_REACHED",
                "message": f"You have reached the chat limit of {MAX_CHAT_MESSAGES} messages. Please use the 'Clear Chat' button to start a new conversation."
            }
            yield f"data: {json.dumps(error_data)}\n\n"
            yield "data: [DONE]\n\n"
        return Response(limit_exceeded_stream(), mimetype='text/event-stream')
    
    chat_history_for_prompt = ""
    for qa in history[-5:]: 
        chat_history_for_prompt += f"Human: {qa['question']}\nAI: {qa['answer']}\n\n"

    final_question_with_history = question
    if chat_history_for_prompt:
        final_question_with_history = (
            "Given the following conversation history:\n"
            f"{chat_history_for_prompt}"
            "--- End History ---\n\n"
            "Now, answer this new question, considering the history as context:\n"
            f"{question}"
        )

    search_query = question
    if history:
        last_q = history[-1]['question']
        last_a = history[-1]['answer']
        search_query = f"In response to the question '{last_q}' and the answer '{last_a}', the user is now asking: {question}"
        
    # --- START: MODIFIED BLOCK WITH LOGGING ---
    channel_data = None
    video_ids = None

    if channel_name:
        logging.info(f"Attempting to fetch data for channel: '{channel_name}' for user: {user_id}")

        # Use the reliable get_user_channels() function to get all user channels
        all_user_channels = get_user_channels() 

        # Find the specific channel from the results in Python
        channel_data = all_user_channels.get(channel_name)

        if channel_data:
            logging.info("Successfully found channel data.")
            video_ids = {v['video_id'] for v in channel_data.get('videos', [])}
            logging.info(f"Found {len(video_ids)} video_ids for the channel.")
        else:
            logging.warning(f"Could not find data for channel '{channel_name}' in user's linked channels.")

        # Final check before calling the AI
        if video_ids:
            logging.info("Proceeding to answer question WITH channel-specific context.") # <-- ADD LOGGING
        else:
            logging.warning("Proceeding to answer question WITHOUT channel-specific context. Answers will be generic.") # <-- ADD LOGGING
    # --- END: MODIFIED BLOCK ---

    # Your existing call to the answer_question_stream function (no changes needed)
    stream = answer_question_stream(
        question_for_prompt=final_question_with_history,
        question_for_search=search_query,
        channel_data=channel_data,
        video_ids=video_ids,
        user_id=user_id,
        access_token=access_token
    )
    
    return Response(stream, mimetype='text/event-stream')

# --- MODIFICATION 1: Fix the root route ---
# Removed the @login_required decorator and simplified the function.
# Now, any user visiting the root of your site will be immediately
# redirected to the '/channel' page, which is the desired landing page.
@app.route('/')
def index():
    return redirect(url_for('channel'))


@app.route('/landing')
def landing():
    return render_template('landing.html')

@app.route('/pro')
@login_required
def pro_page():
    user_id = session['user']['id']
    subscription_data = get_user_subscription_status(user_id)
    
    logging.debug(f"Subscription data for pro_page: {subscription_data}")

    if not subscription_data:
        flash('Could not retrieve subscription information. Please try again.', 'error')
        return redirect(url_for('channel'))

    return render_template('pro.html', subscription=subscription_data, saved_channels=get_user_channels())


@app.route('/ask', methods=['GET'])
@login_required
def ask():
    user_id = session['user']['id']
    access_token = session.get('access_token')
    history = get_chat_history(user_id, 'general', access_token=access_token)
    return render_template('ask.html',
                           history=history,
                           saved_channels=get_user_channels())


@app.route('/ask/channel/<path:channel_name>') # Using <path:> is more robust for channel names
@login_required
def ask_channel(channel_name):
    """
    Displays the chat interface for a specific channel.
    This is the complete, corrected version that fixes the query and TypeError.
    """
    user = session.get('user')
    user_id = user['id']
    access_token = session.get('access_token')
    
    # --- FIX #1: Fetch ALL user channels first ---
    # This is a more reliable way to find the specific channel and also gets the
    # data needed for the sidebar in a single database call.
    all_user_channels = get_user_channels()
    
    current_channel = all_user_channels.get(channel_name)
    
    if not current_channel:
        flash(f"You do not have access to the channel '{channel_name}' or it does not exist.", "error")
        return redirect(url_for('channel'))

    # --- FIX #2: Add the missing 'access_token' to the function call ---
    history = get_chat_history(user_id, channel_name, access_token)
    
    # Pass the variables to the template
    return render_template(
        'ask.html',
        user=user,
        history=history,
        channel_name=channel_name, # Pass the name
        current_channel=current_channel, # Pass the specific channel's data
        saved_channels=all_user_channels, # Pass the full list for the sidebar
        SUPABASE_URL=os.environ.get('SUPABASE_URL'),
        SUPABASE_ANON_KEY=os.environ.get('SUPABASE_ANON_KEY')
    )


@app.route('/delete_channel/<int:channel_id>', methods=['POST'])
@login_required
def delete_channel_route(channel_id):
    """
    Handles the initial request to delete a channel.
    It verifies ownership and then queues a background task to do the actual deletion.
    """
    user_id = session['user']['id']
    supabase_admin = get_supabase_admin_client()

    try:
        # --- FIX: Check for ownership in the 'user_channels' junction table ---
        ownership_check = supabase_admin.table('user_channels') \
            .select('channel_id') \
            .eq('user_id', user_id) \
            .eq('channel_id', channel_id) \
            .limit(1).single().execute()

        # If the above line doesn't error, the user has permission.
        logging.info(f"User {user_id} initiated deletion for channel {channel_id}. Queuing background task.")
        delete_channel_task(channel_id, user_id)
        
        return jsonify({
            'status': 'success',
            'message': 'Channel deletion has been started in the background.'
        })

    except APIError as e:
        # This block will now correctly catch the case where no row is found
        if 'PGRST116' in e.message: # This is the code for 'no rows returned'
            logging.warning(f"User {user_id} failed to delete channel {channel_id}: No permission.")
            return jsonify({'status': 'error', 'message': 'Channel not found or you do not have permission.'}), 404
        
        # --- FIX: Use logging.error for proper exception info ---
        logging.error(f"API Error on channel deletion for {channel_id}: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': 'A database error occurred.'}), 500
    except Exception as e:
        # --- FIX: Use logging.error for proper exception info ---
        logging.error(f"Error initiating deletion for channel {channel_id}: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': 'An error occurred while starting the deletion process.'}), 500


# --- MODIFICATION 2: Update the channel() function ---
# This is the same function from the Canvas, but with the fix for the Jinja2 error.
@app.route('/channel', methods=['GET', 'POST'])
def channel():
    """
    Handles both displaying the channel page (GET) and processing
    a new channel submission (POST). This version uses the correct
    Huey .schedule() method.
    """
    try:
        if request.method == 'POST':
            if 'user' not in session:
                return jsonify({'status': 'error', 'message': 'Authentication required.'}), 401

            user_id = session['user']['id']
            channel_url = request.form.get('channel_url', '').strip()

            if not channel_url:
                return jsonify({'status': 'error', 'message': 'Channel URL is required'}), 400

            # It's good practice to clean the URL to have a consistent format
            cleaned_channel_url = clean_youtube_url(channel_url)
            supabase = get_supabase_client(session.get('access_token'))

            # Check if the channel already exists in our master 'channels' table
            existing_channel_res = supabase.table('channels').select('id, status').eq('channel_url', cleaned_channel_url).limit(1).execute()

            if existing_channel_res.data:
                # Channel already exists, just link it to the user
                channel_id = existing_channel_res.data[0]['id']
                link_res = supabase.table('user_channels').select('channel_id').eq('user_id', user_id).eq('channel_id', channel_id).limit(1).execute()

                if not link_res.data:
                    supabase.table('user_channels').insert({'user_id': user_id, 'channel_id': channel_id}).execute()

                return jsonify({'status': 'success', 'message': 'Channel already exists and has been added.'})
            else:
                # --- THIS IS THE CORRECTED LINE ---
                # Channel is new, schedule it for background processing
                task = process_channel_task.schedule(
                    args=(cleaned_channel_url, user_id),
                    delay=1 # Starts the task almost immediately in the background
                )
                return jsonify({'status': 'processing', 'task_id': task.id})

        # This block handles loading the page (GET request)
        return render_template(
            'channel.html',
            saved_channels=get_user_channels(),
            user=session.get('user'),
            SUPABASE_URL=os.environ.get('SUPABASE_URL'),
            SUPABASE_ANON_KEY=os.environ.get('SUPABASE_ANON_KEY')
        )

    except APIError as e:
        if 'JWT' in e.message and 'expired' in e.message:
            session.clear()
            return jsonify({'status': 'error', 'message': 'Your session has expired.'}), 401
        logging.error(f"API Error in /channel POST: {e}")
        return jsonify({'status': 'error', 'message': str(e.message)}), 500
    except Exception as e:
        logging.error(f"An unexpected error occurred in /channel: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': 'An internal server error occurred.'}), 500


@app.route('/task_result/<task_id>')
@login_required
def task_result(task_id):
    # First, try to get progress from Redis
    if redis_client:
        progress_data = redis_client.get(f"task_progress:{task_id}")
        if progress_data:
            return jsonify(json.loads(progress_data))

    # If not in Redis, check the final Huey result store as a fallback
    try:
        result = huey.result(task_id, preserve=True)
        if result is None:
            # If the task is not yet complete, and we have no Redis data,
            # it might be starting. Return a default "running" state.
            return jsonify({'status': 'processing', 'progress': 5, 'message': 'Task is starting...'})
        else:
            # If it's complete, Huey has the final result string
            return jsonify({'status': 'complete', 'progress': 100, 'message': result})
    except TaskException as e:
        logging.error(f"Task {task_id} failed: {e}")
        return jsonify({'status': 'failed', 'progress': 0, 'message': str(e)})

@app.route('/terms')
def terms():
    """Renders the terms and conditions page."""
    return render_template('terms.html',saved_channels=get_user_channels())

@app.route('/privacy')
def privacy():
    """Renders the terms and conditions page."""
    return render_template('privacy.html',saved_channels=get_user_channels())
# The login and signup routes can be removed or kept for fallback.
# For now, we'll keep them but they won't be the primary flow.
# @app.route('/signup', methods=['GET', 'POST'])
# def signup():
#     if request.method == 'POST':
#         email = request.form.get('email')
#         password = request.form.get('password')
#         supabase = get_supabase_client()
#         try:
#             auth_response = supabase.auth.sign_up({"email": email, "password": password})
#             # To support the new flow, we can log the user in directly
#             # For now, we'll keep the old flash message flow
#             flash('Sign up successful! Please check your email to verify your account.', 'success')
#             return redirect(url_for('login'))
#         except Exception as e:
#             logging.error(f"Signup failed for {email}: {e}", exc_info=True)
#             flash("An error occurred during sign up. The email may already be in use.", 'error')
#             return redirect(url_for('signup'))
#     return render_template('signup.html', saved_channels=get_user_channels())


# @app.route('/login', methods=['GET', 'POST'])
# def login():
#     if request.method == 'POST':
#         email = request.form.get('email')
#         password = request.form.get('password')
#         supabase = get_supabase_client()
#         try:
#             auth_response = supabase.auth.sign_in_with_password({"email": email, "password": password})
#             session['user'] = auth_response.user.dict()
#             session['access_token'] = auth_response.session.access_token
#             flash('Successfully logged in!', 'success')
#             return redirect(url_for('index'))
#         except Exception as e:
#             logging.error(f"Login failed for user {email}: {e}", exc_info=True)
#             flash("An error occurred during login. Please check your credentials and try again.", 'error')
#             return redirect(url_for('login'))
#     return render_template('login.html', saved_channels=get_user_channels())


@app.route('/logout')
def logout():
    """
    Clears the server-side session and then redirects to the channel page.
    The client-side Supabase session will be cleared by the browser on redirect.
    """
    session.clear()
    flash('You have been successfully logged out.', 'success')
    return redirect(url_for('channel'))

@app.route('/clear_chat', methods=['POST'])
@login_required
def clear_chat():
    channel_name = request.form.get('channel_name') or 'general'
    user_id = session['user']['id']
    
    try:
        supabase = get_supabase_client(session.get('access_token'))
        supabase.table('chat_history').delete().eq('user_id', user_id).eq('channel_name', channel_name).execute()
        return jsonify({'status': 'success', 'message': f'Chat history cleared for {channel_name}'})
    except Exception as e:
        logging.error(f"Error clearing chat history: {e}")
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/refresh_channel/<int:channel_id>', methods=['POST'])
@login_required
def refresh_channel_route(channel_id):
    user_id = session['user']['id']
    access_token = session.get('access_token')
    
    try:
        # --- PROACTIVE FIX: Check for ownership in the 'user_channels' junction table ---
        supabase = get_supabase_client(access_token)
        ownership_check = supabase.table('user_channels') \
            .select('channel_id') \
            .eq('user_id', user_id) \
            .eq('channel_id', channel_id) \
            .limit(1).single().execute()
        
        # If the above line doesn't error, the user has permission.
        task = sync_channel_task(channel_id)
        return jsonify({'status': 'success', 'message': 'Channel refresh has been queued.', 'task_id': task.id})

    except APIError:
        # If single() finds no rows, it raises an APIError.
        return jsonify({'status': 'error', 'message': 'Channel not found or you do not have permission.'}), 404
    except Exception as e:
        logging.error(f"Error initiating refresh for channel {channel_id}: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': 'An error occurred while starting the refresh.'}), 500



# --- START: TELEGRAM BOT ROUTES ---

@app.route('/telegram/connect', methods=['GET', 'POST'])
# The @login_required decorator has been REMOVED from here
def connect_telegram():
    # If the user is not logged into the website, just show them the login prompt page.
    if 'user' not in session:
        return render_template('connect_telegram.html', user=None, saved_channels={})

    # If the user IS logged in, proceed with all of your original logic.
    user_id = session['user']['id']
    supabase_admin = get_supabase_admin_client()

    existing_response = supabase_admin.table('telegram_connections').select('*').eq('app_user_id', user_id).limit(1).execute()

    if existing_response.data and existing_response.data[0]['is_active']:
        existing_data = existing_response.data[0]
        return render_template('connect_telegram.html',
                               connection_status='connected',
                               telegram_username=existing_data.get('telegram_username', 'N/A'),
                               saved_channels=get_user_channels(),
                               user=session.get('user')) # Pass the user object

    if request.method == 'POST':
        connection_code = secrets.token_hex(8)
        
        data_to_store = {
            'app_user_id': user_id,
            'telegram_chat_id': 0, 
            'connection_code': connection_code,
            'created_at': datetime.now(timezone.utc).isoformat(),
            'is_active': False
        }

        # Use an upsert for cleaner logic
        supabase_admin.table('telegram_connections').upsert(data_to_store, on_conflict='app_user_id').execute()

        token, _ = get_bot_token_and_url()
        bot_username = token.split(':')[0] if token else "YourBot"

        return render_template('connect_telegram.html',
                               connection_status='code_generated',
                               connection_code=connection_code,
                               bot_username=bot_username,
                               saved_channels=get_user_channels(),
                               user=session.get('user')) # Pass the user object

    # This is the default state for a logged-in user who hasn't connected yet
    return render_template('connect_telegram.html',
                           connection_status='not_connected',
                           saved_channels=get_user_channels(),
                           user=session.get('user'))


@app.route('/channel/<int:channel_id>/disconnect_group', methods=['POST'])
@login_required
def disconnect_group(channel_id):
    """
    Handles disconnecting a Telegram group and now redirects back to the chat page.
    """
    user_id = session['user']['id']
    supabase = get_supabase_client(session.get('access_token'))
    supabase_admin = get_supabase_admin_client()

    # Security Check: Verify the user is linked to this channel.
    link_check = supabase.table('user_channels').select('channels(channel_name)') \
        .eq('user_id', user_id).eq('channel_id', channel_id).single().execute()

    if not (link_check.data and link_check.data.get('channels')):
        flash("You do not have permission to modify this channel's connection.", "error")
        return redirect(url_for('channel'))
    
    # Get the channel name from our security check for the redirect
    channel_name = link_check.data['channels']['channel_name']

    # If the security check passes, proceed with deletion.
    try:
        supabase_admin.table('group_connections').delete().eq('linked_channel_id', channel_id).execute()
        flash("Telegram group successfully disconnected.", "success")
    except APIError as e:
        flash(f"An error occurred while disconnecting: {e.message}", "error")
    
    # --- THIS IS THE FIX ---
    # Redirect to the 'ask_channel' page using the channel name we fetched.
    return redirect(url_for('ask_channel', channel_name=channel_name))


@app.route('/channel/<int:channel_id>/connect_group')
@login_required
def connect_group(channel_id):
    # --- THIS IS THE FIX ---
    # Get a fresh database connection every time the function is called.
    supabase_admin = get_supabase_admin_client()
    supabase = get_supabase_client(session.get('access_token'))
    # --- END FIX ---

    user_id = session['user']['id']

    # Security Check: Verify the user is linked to this channel.
    link_check = supabase.table('user_channels').select('channel_id').eq('user_id', user_id).eq('channel_id', channel_id).limit(1).execute()
    if not link_check.data:
        flash("You do not have permission to access this channel.", "error")
        return redirect(url_for('channel'))

    # Fetch the channel's name for display
    channel_resp = supabase_admin.table('channels').select('id, channel_name').eq('id', channel_id).single().execute()
    if not channel_resp.data:
        flash("Channel not found.", "error")
        return redirect(url_for('channel'))

    # This query safely checks for an existing connection without crashing.
    response = supabase_admin.table('group_connections').select('*').eq('linked_channel_id', channel_id).eq('is_active', True).limit(1).execute()
    
    # Check if the data list is not empty
    if response.data:
        # If already connected, render the page in 'connected' mode
        return render_template('connect_group.html',
                               connection_status='connected',
                               channel=channel_resp.data,
                               group_details=response.data[0],
                               saved_channels=get_user_channels())

    # If not connected, proceed with generating a new code
    connection_code = secrets.token_hex(10)
    supabase_admin.table('group_connections').upsert({
        'owner_user_id': user_id,
        'linked_channel_id': channel_id,
        'connection_code': connection_code,
        'is_active': False
    }, on_conflict='linked_channel_id').execute()

    token, _ = get_bot_token_and_url()
    bot_username = token.split(':')[0] if token else "YourBot"

    return render_template('connect_group.html',
                           connection_status='code_generated',
                           channel=channel_resp.data,
                           connection_code=connection_code,
                           bot_username=bot_username,
                           saved_channels=get_user_channels())



@app.route('/telegram/webhook/<webhook_secret>', methods=['POST'])
def telegram_webhook(webhook_secret):
    config = load_config()
    token = config.get("telegram_bot_token")
    if not token:
        logging.error("Webhook received but TELEGRAM_BOT_TOKEN is not configured.")
        return "Configuration error", 500

    expected_secret = token.split(':')[-1][:10]
    header_secret = request.headers.get('X-Telegram-Bot-Api-Secret-Token')

    if not (secrets.compare_digest(webhook_secret, expected_secret) and header_secret and secrets.compare_digest(header_secret, expected_secret)):
        logging.warning("Unauthorized webhook access attempt.")
        return "Unauthorized", 403

    update = request.get_json()
    
    process_telegram_update_task(update)

    return jsonify({'status': 'ok'})

# In app.py

@app.template_filter('format_subscribers')
def format_subscribers_filter(value):
    """Formats a number into a human-readable string like 1.5K or 1.2M."""
    try:
        num = int(value)
        if num < 1000:
            return str(num)
        if num < 1000000:
            # Format to one decimal place, but remove .0 if it exists
            k_value = f"{num / 1000:.1f}"
            return k_value.replace('.0', '') + "K"
        # Format to one decimal place, but remove .0 if it exists
        m_value = f"{num / 1000000:.1f}"
        return m_value.replace('.0', '') + "M"
    except (ValueError, TypeError):
        # If the value isn't a number, return an empty string or the original value
        return ""
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(module)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    app.run(debug=False, host='0.0.0.0', port=5000)