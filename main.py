import argparse
import pandas as pd
import json
import os
import logging
from openai import OpenAI
from dotenv import load_dotenv
import ulid
from typing import Optional
import time

load_dotenv()

# Configuration: read from environment with sensible defaults
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
if not OPENROUTER_API_KEY:
    raise RuntimeError('OPENROUTER_API_KEY not set in environment')

MODEL_NAME = os.getenv('MODEL_NAME', 'meta-llama/llama-3-70b-instruct')

client = OpenAI(
    base_url=os.getenv('OPENROUTER_BASE_URL', 'https://openrouter.ai/api/v1'),
    api_key=OPENROUTER_API_KEY,
)

logging.basicConfig(level=os.getenv('LOG_LEVEL', 'INFO'))
logger = logging.getLogger('mirrorlife')

# Initialize Langfuse client and decorator if keys present
langfuse_client: Optional[object] = None
try:
    if os.getenv('LANGFUSE_PUBLIC_KEY') and os.getenv('LANGFUSE_SECRET_KEY'):
        from langfuse import Langfuse, observe
        langfuse_client = Langfuse(
            public_key=os.getenv('LANGFUSE_PUBLIC_KEY'),
            secret_key=os.getenv('LANGFUSE_SECRET_KEY'),
            host=os.getenv('LANGFUSE_HOST', 'https://challenges.reply.com/langfuse'),
        )
        logger.info('✓ Langfuse initialized')
except Exception as e:
    logger.warning('Langfuse not available: %s', e)


def parse_model_response(response) -> bool:
    """
    Parse the model response and return True if activation is requested.
    We expect the model to emit JSON like: {"activate": true, "confidence": 0.87}
    Falls back to legacy substring check if JSON cannot be parsed.
    """
    raw = ''
    try:
        # Common shapes: response.choices[0].message.content or response.choices[0].text
        raw = getattr(response.choices[0].message, 'content', None) or getattr(response.choices[0], 'text', None) or str(response)
    except Exception:
        raw = str(response)

    # Try to extract JSON object from the response
    json_text = None
    if '{' in raw and '}' in raw:
        start = raw.find('{')
        end = raw.rfind('}') + 1
        json_text = raw[start:end]

    parsed = None
    if json_text:
        try:
            parsed = json.loads(json_text)
        except Exception:
            parsed = None

    if isinstance(parsed, dict):
        activate = parsed.get('activate')
        if isinstance(activate, bool):
            return activate
        if isinstance(activate, (int, float)):
            return bool(activate)
        if isinstance(activate, str):
            return activate.strip().lower() in ('1', 'true', 'yes')

    # Fallback: legacy simple check
    return '1' in raw[:5]


def _make_traced_llm_call_function():
    """Create a Langfuse-traced LLM call function if Langfuse is available."""
    if langfuse_client is None:
        # No tracing - return a simple wrapper
        def call_model(session_id: str, citizen_id: str, messages: list) -> dict:
            llm_timeout = float(os.getenv('LLM_TIMEOUT_SECS', '30'))
            return client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                timeout=llm_timeout,
            )
        return call_model
    
    # With tracing - use @observe() decorator
    try:
        from langfuse import observe
        
        @observe()
        def call_model_traced(session_id: str, citizen_id: str, messages: list) -> dict:
            """Traced LLM call function that reports to Langfuse."""
            # Update the trace with session_id so all calls are grouped
            try:
                langfuse_client.update_current_trace(session_id=session_id)
            except Exception as e:
                logger.debug('Failed to update trace: %s', e)
            
            llm_timeout = float(os.getenv('LLM_TIMEOUT_SECS', '30'))
            return client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                timeout=llm_timeout,
            )
        
        return call_model_traced
    except Exception as e:
        logger.warning('Failed to create traced LLM function: %s, falling back to untraced', e)
        def call_model(session_id: str, citizen_id: str, messages: list) -> dict:
            llm_timeout = float(os.getenv('LLM_TIMEOUT_SECS', '30'))
            return client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                timeout=llm_timeout,
            )
        return call_model


# Create the traced LLM call function
call_model = _make_traced_llm_call_function()



def solve_mirror_life(dry_run: bool = False, limit: int | None = None):
    try:
        # Generate a session id for tracing and grouping
        session_id = f"{os.getenv('TEAM_NAME', 'tutorial')}-{ulid.new().str}"
        logger.info('Session id: %s', session_id)

        logger.info('Loading data files')
        status_df = pd.read_csv('status.csv')

        with open('users.json', 'r') as f:
            users_data = json.load(f)
            users_df = pd.DataFrame(users_data)

        with open('locations.json', 'r') as f:
            locations_data = json.load(f)
            locations_df = pd.DataFrame(locations_data)

        recommended_citizens = []
        processed = 0

        for citizen_id, history in status_df.groupby('CitizenID'):
            if limit is not None and processed >= limit:
                break
            processed += 1

            # Support different column names across files: prefer 'CitizenID' but fall back to 'user_id'
            if 'CitizenID' in users_df.columns:
                user_profile = users_df[users_df['CitizenID'] == citizen_id].to_dict('records')
            elif 'user_id' in users_df.columns:
                user_profile = users_df[users_df['user_id'] == citizen_id].to_dict('records')
            else:
                user_profile = []

            if 'BioTag' in locations_df.columns:
                spatial_history = locations_df[locations_df['BioTag'] == citizen_id].tail(5).to_dict('records')
            elif 'user_id' in locations_df.columns:
                spatial_history = locations_df[locations_df['user_id'] == citizen_id].tail(5).to_dict('records')
            else:
                spatial_history = []

            if not user_profile:
                logger.warning('No user profile found for %s', citizen_id)
            health_signals = history.sort_values('Timestamp').tail(10).to_dict('records')

            logger.debug('Invoking model for citizen %s', citizen_id)
            system_message = {
                'role': 'system',
                'content': (
                    'You are The Eye, a cooperative multi-agent system for MirrorLife. '
                    'Analyze demographic data, GPS history, and health signals to identify '
                    'subtle deviations from optimal well-being trajectories. '
                    'Respond with a JSON object: {"activate": true|false, "confidence": 0.0-1.0}. '
                    'Do not include additional commentary outside the JSON.'
                )
            }
            user_message = {
                'role': 'user',
                'content': json.dumps({'profile': user_profile, 'health': health_signals, 'locations': spatial_history})
            }
            messages = [system_message, user_message]

            # Retry/backoff configuration
            max_retries = int(os.getenv('LLM_MAX_RETRIES', '3'))
            base_backoff = float(os.getenv('LLM_BACKOFF_SECS', '1.0'))
            response = None
            last_exc = None
            for attempt in range(1, max_retries + 1):
                try:
                    response = call_model(session_id, citizen_id, messages)
                    last_exc = None
                    break
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    last_exc = e
                    backoff = base_backoff * (2 ** (attempt - 1))
                    logger.warning('LLM call failed (attempt %d/%d): %s — backing off %.1fs', attempt, max_retries, e, backoff)
                    time.sleep(backoff)

            if response is None and last_exc is not None:
                logger.error('LLM call failed after %d attempts: %s', max_retries, last_exc)
                activate = False
            else:
                try:
                    activate = parse_model_response(response)
                except Exception as e:
                    logger.warning('Failed to parse model response for %s: %s', citizen_id, e)
                    activate = False

            if activate:
                recommended_citizens.append(citizen_id)

        if dry_run:
            logger.info('Dry run: would recommend %d citizens', len(recommended_citizens))
            for cid in recommended_citizens[:20]:
                logger.info('Recommended: %s', cid)
        else:
            with open('submission.txt', 'w') as f:
                for cid in recommended_citizens:
                    f.write(f"{cid}\n")
            logger.info('Analysis complete. Submission file generated.')
        
        # Flush Langfuse traces if client initialized
        if langfuse_client is not None:
            try:
                langfuse_client.flush()
                logger.info('Langfuse traces flushed')
            except Exception as e:
                logger.warning('Langfuse flush failed: %s', e)
    except KeyboardInterrupt:
        logger.warning('Interrupted by user')
        raise
    except Exception as e:
        logger.exception('Operational Error')



def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dry-run', action='store_true', help='Do not write submission file')
    p.add_argument('--limit', type=int, default=None, help='Limit number of citizens to process')
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    solve_mirror_life(dry_run=args.dry_run, limit=args.limit)