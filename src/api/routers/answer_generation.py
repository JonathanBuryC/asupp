from fastapi import APIRouter, Response, status, HTTPException
from fastapi.concurrency import run_in_threadpool
import asyncio
import logging
from logging.handlers import RotatingFileHandler
import os
import time
from typing import Any, Dict, Optional, Set

from src.api.base_models.base_model import LLMInput  # (conservé si référencé ailleurs)
from src.api.utils.portail_iag import PortailIAG
from src.api.constants.api_launch import (
    API_ROUTER_PREFIX,
)  # (conservé si utilisé pour l'inclusion ailleurs)
from src.api.constants.openai import (
    DEFAULT_MODEL,
    MODEL_GEMINI,
    MODEL_MISTRAL,
    MODEL_MISTRAL_SMALL,
    MISTRAL_MODELS,
    MISTRAL_TIMEOUT_SECONDS,  # <-- timeout par défaut (secondes)
    build_prompt_for_model,
    build_mistral_jsonify_prompt,
    REFERENCE_KEY,
    JSON_SCHEMA_MISTRAL,
)
from src.api.utils.query import (
    try_parse_json_strict,
    try_parse_json_relaxed,
    normalize_llm_output,
    normalize_llm_json_output,
    repair_gemini_json,
    remap_unknown_references,
    count_tokens,
    collect_streaming_content,
    parse_and_normalize_json,
)

router_LLM = APIRouter(tags=["answer_generation"])
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)





# Seuil informatif sur la taille du prompt (pure télémétrie)
_PROMPT_WARN_LEN = 120_000
_PROMPT_HARD_WARN_LEN = 160_000


def _resolve_model(payload: Dict[str, Any]) -> str:
    """
    Résout le modèle à utiliser avec la précédence :
    1) variable d'environnement MODEL (non vide)
    2) payload["model"]
    3) DEFAULT_MODEL
    """
    env_model = os.getenv("MODEL")
    return (
        env_model.strip() if env_model and env_model.strip() else payload.get("model")
    ) or DEFAULT_MODEL


def _get_mistral_timeout_seconds() -> int:
    """
    Résout le timeout Mistral (secondes) avec la précédence :
    1) variable d'environnement MISTRAL_TIMEOUT (int >= 1)
    2) constante MISTRAL_TIMEOUT_SECONDS
    """
    raw = os.getenv("MISTRAL_TIMEOUT")
    if raw is not None:
        try:
            value = int(raw)
            if value >= 1:
                return value
            logger.warning(
                f"MISTRAL_TIMEOUT défini mais invalide (value={raw!r}); utilisation de la valeur par défaut {MISTRAL_TIMEOUT_SECONDS}s"
            )
        except ValueError:
            logger.warning(
                f"MISTRAL_TIMEOUT n'est pas un entier (value={raw!r}); utilisation de la valeur par défaut {MISTRAL_TIMEOUT_SECONDS}s"
            )
    return MISTRAL_TIMEOUT_SECONDS


def _run_generation_logic(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Logique synchrone et bloquante de génération de réponse, invoquée dans un threadpool.

    Étapes :
      1) Résolution du modèle et construction du prompt
      2) Appel Portail IAG (Gemini ou Mistral)
      3) Parsing/normalisation JSON
      4) Fallback "jsonify" (Mistral pass2) si nécessaire
      5) Nettoyage des références et retour
    """
    try:
        query: Optional[str] = payload.get("query")
        chunks: Optional[list] = payload.get("chunks")

        # Résolution du modèle (env > payload > défaut)
        model: str = _resolve_model(payload)

        # Mode (historique) avec défaut "fast"
        mode: str = payload.get("mode", "fast")

        context: Optional[dict] = payload.get(
            "context"
        )  # {"original_query": "...", "initial_answer": [...]}
        deepen_selected: Optional[list] = payload.get(
            "deepen_selected"
        )  # [{"subtheme": "...", "content": "..."}, ...]

        if not query or not isinstance(chunks, list):
            raise HTTPException(
                status_code=400, detail="query ou chunks manquants/invalides"
            )

        # Approfondissement uniquement pour Mistral (mode 'fast') ou Gemini
        if deepen_selected:
            allow_mistral_fast = model in MISTRAL_MODELS and mode == "fast"
            allow_gemini = model == MODEL_GEMINI
            if not (allow_mistral_fast or allow_gemini):
                raise HTTPException(
                    status_code=400,
                    detail="Approfondissement disponible uniquement pour Mistral (mode 'fast') ou Gemini.",
                )

        # 1) Prompt & response_format (selon modèle)
        cfg = build_prompt_for_model(
            model=model,
            query=query,
            chunks=chunks,
            mode=mode,
            context=context,
            deepen_selected=deepen_selected,
        )
        prompt: str = cfg["prompt"]
        response_format: Optional[dict] = cfg["response_format"]

        # Télémétrie taille de prompt (logs)
        if len(prompt) >= _PROMPT_HARD_WARN_LEN:
            logger.warning(
                f"[answer_generation] Prompt très volumineux: len={len(prompt)} (>= {_PROMPT_HARD_WARN_LEN})"
            )
        elif len(prompt) >= _PROMPT_WARN_LEN:
            logger.info(
                f"[answer_generation] Prompt volumineux: len={len(prompt)} (>= {_PROMPT_WARN_LEN})"
            )

        iag_portail = PortailIAG()
        logger.info(
            f"[answer_generation] model={model} prompt_len={len(prompt)}  deepen={bool(deepen_selected)}"
        )

        # 2) Appel principal Portail IAG (mesure globale)
        t_portail_start = time.perf_counter()

        if model == MODEL_GEMINI and response_format is not None:
            logger.debug("[LLM][Gemini] Appel Gemini démarré…")
            completion = iag_portail.query_gemini(model, prompt, query, response_format)
            # Selon le client, choices[0].message.content doit exister
            content = (completion.choices[0].message.content or "").strip()
            logger.debug(
                f"[LLM][Gemini] Réponse brute len={len(content)} preview={content[:200]!r}"
            )
        else:
            logger.debug("[LLM][Mistral] Appel Mistral (streaming) démarré…")
            completion, correlation_id = iag_portail.query_mistral(model, prompt, query)
            logger.error(f"[MISTRAL] Correlation ID: {correlation_id}")

            full_content = ""
            chunk_count = 0
            for event in completion:
                chunk_count += 1
                delta = event.choices[0].delta
                if delta and delta.content:
                    full_content += delta.content

            content = full_content.strip()

            logger.info(
                "\n===== PASS1 LLM OUTPUT =====\n"
                f"model={model}\n"
                f"prompt_len={len(prompt)}\n"
                f"chunk_count={chunk_count}\n"
                "----- RAW CONTENT START -----\n"
                f"{content}\n"
                "----- RAW CONTENT END -----\n"
            )

            logger.info(
                f"[LLM][Mistral] Fin streaming : chunks={chunk_count}, total_len={len(content)}, preview={content[:800]!r}"
            )

        if not content:
            logger.error(
                "[LLM] Aucune réponse reçue du LLM (content vide après appel Portail IAG)"
            )
            raise HTTPException(status_code=502, detail="Réponse vide du LLM (pass1)")

        # 3) Parsing & normalisation
        parsed = (
            try_parse_json_strict(content)
            or try_parse_json_relaxed(content)
            or repair_gemini_json(content)
        )

        if parsed is None and model == MODEL_GEMINI:
            # Réparation spécifique Gemini si JSON bancal malgré response_format
            parsed = repair_gemini_json(content)

        parsed = normalize_llm_output(parsed, REFERENCE_KEY)

        # 4) Fallback Mistral jsonify pass (pass2) si nécessaire
        if (parsed is None) and model in MISTRAL_MODELS:
            logger.warning(
                f"Pass1 non JSON-compliant. Lancement du pass2 (Mistral jsonify). Longuer du contenu initial {len(content)}"
            )
            jsonify_prompt = build_mistral_jsonify_prompt(content, REFERENCE_KEY)
            content2 = iag_portail.query_mistral_json_structured(
                model, jsonify_prompt, JSON_SCHEMA_MISTRAL
            )

            content2 = content2.strip()

            logger.info(
                "\n===== PASS2 JSONIFY OUTPUT =====\n"
                f"model={MODEL_MISTRAL_SMALL}\n"
                f"source_model={model}\n"
                "----- RAW CONTENT START -----\n"
                f"{content2}\n"
                "----- RAW CONTENT END -----\n"
            )

            # parsed2 = try_parse_json_strict(content2) or try_parse_json_relaxed(content2)
            parsed2 = (
                try_parse_json_strict(content2)
                or try_parse_json_relaxed(content2)
                or repair_gemini_json(content2)
            )
            logger.info(
                f"[LLM][Mistral] Fin streaming : total_len={len(content2)}, preview={content2[:800]!r}"
            )
            parsed = normalize_llm_output(parsed2, REFERENCE_KEY)

        if parsed is None:
            raise HTTPException(
                status_code=502,
                detail="Le modèle n'a pas produit un JSON valide malgré la coercition.",
            )

        # 5) Validation/correction des références
        valid_ids: Set[Any] = {
            c.get("chunk_id")
            for c in chunks
            if isinstance(c, dict) and c.get("chunk_id")
        }
        parsed, ref_report = remap_unknown_references(
            parsed,
            valid_ids=valid_ids,
            reference_key=REFERENCE_KEY,  # "references"
            enable_fuzzy=True,
            threshold=0.8,
        )
        if ref_report.get("removed") or ref_report.get("remapped"):
            logger.info(f"[answer_generation] references cleanup: {ref_report}")

        # Mesure finale (couvre pass1 + éventuel pass2)
        t_portail_end = time.perf_counter()
        duree_portail_ms = int((t_portail_end - t_portail_start) * 1000)

        return {
            "answer": parsed,
            "timings": {"portail_ms": duree_portail_ms},
            "num_tokens_entry": count_tokens(query),
            "num_tokens_output": count_tokens(str(parsed)),
            "correlation_id": correlation_id or None
        }

    except HTTPException as he:
        # Laisser remonter les HTTPException pour gestion par l'endpoint
        raise he

    except Exception as e:
        logger.exception("Failed to run generation")
        # Wrapper en HTTPException pour homogénéité des erreurs
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@router_LLM.post("/answer_generation")
async def generate_answer(payload: Dict[str, Any], response: Response):
    """
    Endpoint asynchrone d'orchestration :
      - Résout le modèle effectif
      - Applique un timeout uniquement pour Mistral
      - Exécute la logique bloquante dans un threadpool
    """
    try:
        # Même règle de précédence (env > payload > défaut)
        effective_model = _resolve_model(payload)

        # Timeout uniquement pour Mistral (env > constante)
        mistral_timeout = (
            _get_mistral_timeout_seconds()
            if effective_model in MISTRAL_MODELS
            else None
        )

        if mistral_timeout:
            # Enforce timeout pour les workloads Mistral
            result = await asyncio.wait_for(
                run_in_threadpool(_run_generation_logic, payload),
                timeout=mistral_timeout,
            )
        else:
            result = await run_in_threadpool(_run_generation_logic, payload)

        response.status_code = status.HTTP_200_OK
        return result

    except asyncio.TimeoutError:
        # Standard pour upstream timeouts : 504
        response.status_code = status.HTTP_504_GATEWAY_TIMEOUT
        # mistral_timeout ne peut pas être None si on arrive ici
        minutes = (mistral_timeout or 0) / 60.0
        return {
            "error": (
                f"⏱️ **Délai dépassé**\n\n"
                f"Le Portail IAG met plus de temps que prévu pour générer la réponse "
                f"(timeout de {minutes:.1f} minutes atteint).\n\n"
                "Merci de réessayer ou d'utiliser le mode *Rapide* si ce n'est pas déjà le cas."
            )
        }

    except HTTPException as he:
        response.status_code = he.status_code
        return {"error": he.detail}

    except Exception as e:
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return {"error": str(e)}
