"""Pipecat Cloud WhatsApp voice agent entrypoint using OpenAI realtime audio."""

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import aiohttp

try:  # Available inside Pipecat Cloud
    from pipecatcloud.agent import PipecatSessionArguments
except ImportError:  # Local dev fallback
    PipecatSessionArguments = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Configuration constants
IDLE_TIMEOUT_SECS = int(os.getenv("IDLE_TIMEOUT_SECS", "30"))
AUDIO_OUT_10MS_CHUNKS = int(os.getenv("AUDIO_OUT_10MS_CHUNKS", "1"))
NOISE_REDUCTION_TYPE = os.getenv("NOISE_REDUCTION_TYPE", "near_field")

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.openai_realtime_beta import (
    InputAudioNoiseReduction,
    InputAudioTranscription,
    OpenAIRealtimeBetaLLMService,
    SemanticTurnDetection,
    SessionProperties,
)
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.transports.whatsapp.api import WhatsAppWebhookRequest
from pipecat.transports.whatsapp.client import WhatsAppClient

SYSTEM_PROMPT_FALLBACK = (
    "You are Kap, a Kapso engineer on a phone call. Always reply in neutral Latin-American Spanish. "
    "Keep answers warm, very concise, and confident - this is voice, not text. Use the contact or user name "
    "to greet them if available in the context. Personalize the conversation using any details included inside "
    "the <context> and <conversation> sections. If information is missing, acknowledge it naturally. When helpful, "
    "mention that Kapso is 'WhatsApp for developers' - we help developers and builders create great WhatsApp "
    "experiences easily with fast onboarding, Supabase sync, automations, and white-label options."
)


def safe_get(data: Any, *keys: str, default: Any = None) -> Any:
    """Safely navigate nested dictionaries with multiple keys.

    Args:
        data: The dictionary to navigate
        *keys: Sequence of keys to traverse
        default: Default value if key not found or data is not a dict

    Returns:
        The value at the nested key path, or default if not found
    """
    current = data
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
        if current is None:
            return default
    return current


def clean_value(value: Any) -> str:
    """Sanitize value for display by removing newlines and converting to string.

    Args:
        value: Any value to clean

    Returns:
        Cleaned string representation
    """
    if value is None:
        return ""
    return str(value).replace("\n", " ").strip()


def build_context_prompt(
    context: Optional[Dict[str, Any]],
    contact_hint: Optional[Dict[str, str]] = None,
) -> Tuple[str, Optional[str]]:
    """Render Kapso context into tagged sections and return a contact name hint.

    Args:
        context: Dictionary containing project, config, contact, call, and conversation data
        contact_hint: Optional fallback contact information

    Returns:
        Tuple of (formatted context string, contact name)
    """
    logger.debug("Building context prompt from context data")

    if not isinstance(context, dict) or not context:
        name_hint = safe_get(contact_hint, "name") or safe_get(contact_hint, "wa_id")
        logger.debug(f"No context provided, using contact hint: {name_hint}")
        return "", name_hint

    context_lines: List[str] = []
    contact_name: Optional[str] = None

    # Build project context
    project_name = safe_get(context, "project", "name")
    project_id = safe_get(context, "project", "id")
    if project_name or project_id:
        context_lines.append(f"Project: {clean_value(project_name)} (ID {clean_value(project_id)})")

    # Build WhatsApp config context
    config_name = safe_get(context, "config", "name") or safe_get(context, "config", "display_name")
    phone = safe_get(context, "config", "display_phone_number") or safe_get(context, "config", "phone_number_id")
    is_coexistence = safe_get(context, "config", "is_coexistence", default=False)
    if config_name or phone:
        mode = "coexistence" if is_coexistence else "dedicated"
        context_lines.append(
            f"WhatsApp config: {clean_value(config_name)} ({clean_value(phone)}) – mode {mode}"
        )

    # Extract contact name
    contact_name = (
        safe_get(context, "contact", "profile_name")
        or safe_get(context, "contact", "display_name")
        or safe_get(context, "contact", "wa_id")
    )

    if not contact_name and contact_hint:
        contact_name = safe_get(contact_hint, "name") or safe_get(contact_hint, "wa_id")

    if contact_name:
        context_lines.append(f"Contact: {clean_value(contact_name)}")

    # Build call context
    call_direction = safe_get(context, "call", "direction")
    call_status = safe_get(context, "call", "status")
    call_started = safe_get(context, "call", "started_at")
    if call_direction or call_status:
        context_lines.append(
            f"Call: direction {clean_value(call_direction)}, status {clean_value(call_status)}, "
            f"started {clean_value(call_started)}"
        )

    # Build permission context
    permission_status = safe_get(context, "call_permission", "status")
    if not permission_status:
        allowed = safe_get(context, "call_permission", "allowed", default=False)
        permission_status = "allowed" if allowed else "blocked"
    permission_expires = safe_get(context, "call_permission", "expires_at")
    if permission_status:
        context_lines.append(
            f"Call permission: {clean_value(permission_status)}, expires {clean_value(permission_expires)}"
        )

    # Build context block
    context_block = ""
    if context_lines:
        context_block = "<context>\n" + "\n".join(f"- {line}" for line in context_lines) + "\n</context>"

    # Build conversation history
    conversation_block = _build_conversation_block(context)

    # Combine blocks
    parts = [block for block in (context_block, conversation_block) if block]
    if not parts:
        logger.debug("No context or conversation data to include")
        return "", contact_name

    result = "\n\n" + "\n".join(parts) + "\n\n"
    logger.debug(f"Built context prompt with {len(parts)} sections for contact: {contact_name}")
    return result, contact_name


def _build_conversation_block(context: Dict[str, Any]) -> str:
    """Extract and format conversation history from context.

    Args:
        context: Context dictionary containing conversation data

    Returns:
        Formatted conversation block or empty string
    """
    messages = safe_get(context, "conversation", "messages", default=[])
    if not isinstance(messages, list) or not messages:
        return ""

    conversation_lines = []
    for message in messages:
        if not isinstance(message, dict):
            continue

        role = safe_get(message, "direction") or safe_get(message, "role", default="unknown")
        timestamp = clean_value(safe_get(message, "created_at"))
        content = clean_value(safe_get(message, "content") or safe_get(message, "text"))

        if not content:
            continue

        label = "Inbound" if role and str(role).lower().startswith("in") else "Outbound"
        line = f"{label} {timestamp}: {content}".strip()
        conversation_lines.append(line)

    if not conversation_lines:
        return ""

    return "<conversation>\n" + "\n".join(conversation_lines) + "\n</conversation>"


def extract_contact_hint(payload: Optional[Dict[str, Any]]) -> Optional[Dict[str, str]]:
    """Extract contact information from payload context or webhook data.

    Args:
        payload: Dictionary containing context or webhook data

    Returns:
        Dictionary with name and wa_id if found, None otherwise
    """
    if not isinstance(payload, dict):
        logger.debug("Invalid payload type for contact extraction")
        return None

    # Try context contact first
    contact_name = safe_get(payload, "context", "contact", "profile_name") or safe_get(
        payload, "context", "contact", "display_name"
    )
    contact_wa_id = safe_get(payload, "context", "contact", "wa_id")

    if contact_name or contact_wa_id:
        logger.debug(f"Found contact in context: {contact_name} ({contact_wa_id})")
        return {"name": contact_name, "wa_id": contact_wa_id}

    # Try webhook entries
    contact_hint = _extract_contact_from_webhook(payload)
    if contact_hint:
        logger.debug(f"Found contact in webhook: {contact_hint.get('name')} ({contact_hint.get('wa_id')})")

    return contact_hint


def _extract_contact_from_webhook(payload: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """Extract contact from webhook entry structure.

    Args:
        payload: Payload containing webhook data

    Returns:
        Dictionary with name and wa_id if found, None otherwise
    """
    webhook = safe_get(payload, "webhook")
    if not isinstance(webhook, dict):
        return None

    entries = safe_get(webhook, "entry", default=[])
    if not isinstance(entries, list):
        return None

    for entry in entries:
        if not isinstance(entry, dict):
            continue

        changes = safe_get(entry, "changes", default=[])
        if not isinstance(changes, list):
            continue

        for change in changes:
            if not isinstance(change, dict):
                continue

            value = safe_get(change, "value")
            if not isinstance(value, dict):
                continue

            contacts = safe_get(value, "contacts", default=[])
            if not isinstance(contacts, list):
                continue

            for contact in contacts:
                if not isinstance(contact, dict):
                    continue

                profile = safe_get(contact, "profile")
                name = safe_get(profile, "name") if isinstance(profile, dict) else None
                wa_id = safe_get(contact, "wa_id")

                if name or wa_id:
                    return {"name": name, "wa_id": wa_id}

    return None


async def run_voice_pipeline(
    connection,
    context_payload: Optional[Dict[str, Any]],
    contact_hint: Optional[Dict[str, str]],
) -> None:
    """Run a speech-to-speech OpenAI realtime pipeline over WebRTC.

    Args:
        connection: WebRTC connection object
        context_payload: Optional context data for the conversation
        contact_hint: Optional contact information for personalization
    """
    logger.info("Starting voice pipeline")

    try:
        transport = SmallWebRTCTransport(
            webrtc_connection=connection,
            params=TransportParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                vad_analyzer=SileroVADAnalyzer(),
                audio_out_10ms_chunks=AUDIO_OUT_10MS_CHUNKS,
            ),
        )

        system_prompt = os.getenv("SYSTEM_PROMPT", SYSTEM_PROMPT_FALLBACK)
        dynamic_context, contact_name = build_context_prompt(context_payload, contact_hint)
        instructions = system_prompt + dynamic_context

        logger.debug(f"Built instructions with context for contact: {contact_name}")

        session_properties = SessionProperties(
            instructions=instructions,
            voice="nova",
            input_audio_transcription=InputAudioTranscription(),
            turn_detection=SemanticTurnDetection(eagerness="high"),
            input_audio_noise_reduction=InputAudioNoiseReduction(type=NOISE_REDUCTION_TYPE),
        )

        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            logger.error("OPENAI_API_KEY not found in environment")
            raise ValueError("OPENAI_API_KEY environment variable is required")

        llm = OpenAIRealtimeBetaLLMService(
            api_key=openai_api_key,
            session_properties=session_properties,
            start_audio_paused=False,
            temperature=0.8,
        )

        # Provide an initial user message so the assistant greets the caller
        llm_context = OpenAILLMContext([{"role": "user", "content": "Hola, estoy llamando"}])
        context_aggregator = llm.create_context_aggregator(llm_context)

        pipeline = Pipeline(
            [
                transport.input(),
                context_aggregator.user(),
                llm,
                transport.output(),
                context_aggregator.assistant(),
            ]
        )

        task = PipelineTask(
            pipeline,
            params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
            idle_timeout_secs=IDLE_TIMEOUT_SECS,
        )
        runner = PipelineRunner(handle_sigint=False)

        @transport.event_handler("on_client_connected")
        async def on_client_connected(_transport, _client):
            logger.info("Client connected to voice pipeline")
            await task.queue_frames([LLMRunFrame()])

        @transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(_transport, _client):
            logger.info("Client disconnected from voice pipeline")
            await task.cancel()

        await runner.run(task)
        logger.info("Voice pipeline completed successfully")

    except Exception as error:
        logger.error(f"Error in voice pipeline: {error}", exc_info=True)
        raise


async def bot(entry: Any) -> None:
    """Pipecat Cloud session entrypoint.

    Args:
        entry: Entry point data from Pipecat Cloud or direct payload
    """
    logger.info("Bot entry point called")

    try:
        # Extract payload from entry
        if PipecatSessionArguments and isinstance(entry, PipecatSessionArguments):
            payload = entry.body or {}
        else:
            payload = entry or {}

        # Validate payload kind
        payload_kind = payload.get("kind")
        if payload_kind != "whatsapp_connect":
            logger.info(f"Ignoring payload with kind: {payload_kind}")
            return

        # Extract required fields
        whatsapp_token = payload.get("whatsapp_token")
        phone_number_id = payload.get("phone_number_id")
        webhook_body = payload.get("webhook")

        if not whatsapp_token:
            logger.error("Missing required field: whatsapp_token")
            raise ValueError("whatsapp_token is required in payload")

        if not phone_number_id:
            logger.error("Missing required field: phone_number_id")
            raise ValueError("phone_number_id is required in payload")

        if not webhook_body:
            logger.error("Missing required field: webhook")
            raise ValueError("webhook is required in payload")

        # Extract optional context and contact information
        context_payload = payload.get("context") if isinstance(payload, dict) else None
        contact_hint = extract_contact_hint(payload)

        logger.info(f"Processing WhatsApp webhook for phone number: {phone_number_id}")

        # Create WhatsApp webhook request
        webhook_request = WhatsAppWebhookRequest(**webhook_body)

        async with aiohttp.ClientSession() as session:
            whatsapp_client = WhatsAppClient(
                whatsapp_token=whatsapp_token,
                phone_number_id=phone_number_id,
                session=session,
            )

            async def on_connection(connection):
                logger.info("WhatsApp connection established")
                await run_voice_pipeline(connection, context_payload, contact_hint)

            await whatsapp_client.handle_webhook_request(
                webhook_request, connection_callback=on_connection
            )

        logger.info("Bot session completed successfully")

    except Exception as error:
        logger.error(f"Error in bot entrypoint: {error}", exc_info=True)
        raise


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
