from typing import Any, TypeVar

from openai import BadRequestError


T = TypeVar("T")


class PlatformGuardrailError(RuntimeError):
    """Raised when the model provider blocks a request before inference."""

    def __init__(
        self,
        *,
        provider: str,
        code: str,
        message: str,
    ) -> None:
        super().__init__(message)

        self.provider = provider
        self.code = code
        self.message = message


def _extract_error_body(
    exception: BadRequestError,
) -> dict[str, Any]:
    body = getattr(
        exception,
        "body",
        None,
    )

    if isinstance(body, dict):
        return body

    response = getattr(
        exception,
        "response",
        None,
    )

    if response is not None:
        try:
            response_body = response.json()

            if isinstance(response_body, dict):
                return response_body
        except Exception:
            pass

    return {}


def _extract_error_details(
    exception: BadRequestError,
) -> tuple[str | None, str]:
    body = _extract_error_body(exception)

    # Depending on the SDK/version, the error properties may be
    # located directly in body or nested under "error".
    nested_error = body.get("error")

    if isinstance(nested_error, dict):
        error_data = nested_error
    else:
        error_data = body

    code = error_data.get("code")
    message = error_data.get("message")

    if not isinstance(code, str):
        code = None

    if not isinstance(message, str):
        message = str(exception)

    return code, message


def invoke_model_safely(
    runnable: Any,
    model_input: Any,
    *,
    provider: str = "Azure AI Foundry",
) -> T:
    """
    Invoke a LangChain runnable and translate provider-level
    content filtering into a domain-specific exception.
    """
    try:
        return runnable.invoke(
            model_input
        )

    except BadRequestError as exception:
        code, message = _extract_error_details(
            exception
        )

        if code in {
            "content_filter",
            "ContentFiltered",
        }:
            raise PlatformGuardrailError(
                provider=provider,
                code=code,
                message=message,
            ) from exception

        raise