"""OIDC/OAuth2 authentication middleware for the AI Capsule Platform.

Supports two modes:
1. **Production**: Validates JWT Bearer tokens from the OIDC provider (e.g., Okta, Auth0, Cognito).
2. **Development**: Uses the X-User header (no verification) when AUTH_DISABLED=true.

Configuration via environment variables:
- AUTH_DISABLED: Set to "true" to disable auth (development only)
- OIDC_ISSUER_URL: OIDC issuer URL (e.g., https://auth.company.com)
- OIDC_AUDIENCE: Expected JWT audience
- OIDC_JWKS_URI: JWKS endpoint (auto-derived from issuer if not set)
"""

import logging
import os

from fastapi import HTTPException, Request, status
from fastapi.security import HTTPBearer
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

# Auth configuration
AUTH_DISABLED = os.getenv("AUTH_DISABLED", "false").lower() == "true"
OIDC_ISSUER_URL = os.getenv("OIDC_ISSUER_URL", "")
OIDC_AUDIENCE = os.getenv("OIDC_AUDIENCE", "ai-capsule-platform")
OIDC_JWKS_URI = os.getenv("OIDC_JWKS_URI", "")

# Public paths that skip authentication
PUBLIC_PATHS = {"/", "/health", "/docs", "/openapi.json", "/redoc"}

security = HTTPBearer(auto_error=False)


def _get_jwks_uri() -> str:
    """Derive JWKS URI from OIDC issuer if not explicitly set."""
    if OIDC_JWKS_URI:
        return OIDC_JWKS_URI
    if OIDC_ISSUER_URL:
        return f"{OIDC_ISSUER_URL.rstrip('/')}/.well-known/jwks.json"
    return ""


async def _validate_token(token: str) -> dict:
    """
    Validate a JWT Bearer token against the OIDC provider.

    Returns the decoded token claims if valid.
    Raises HTTPException if invalid.
    """
    try:
        import httpx
        import jwt

        jwks_uri = _get_jwks_uri()
        if not jwks_uri:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="OIDC not configured — set OIDC_ISSUER_URL",
            )

        # Fetch JWKS (in production, cache this with TTL)
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(jwks_uri)
            resp.raise_for_status()
            jwks = resp.json()

        # Decode and validate the token
        header = jwt.get_unverified_header(token)
        key = None
        for k in jwks.get("keys", []):
            if k.get("kid") == header.get("kid"):
                key = jwt.algorithms.RSAAlgorithm.from_jwk(k)
                break

        if key is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token: key not found in JWKS",
            )

        claims = jwt.decode(
            token,
            key,
            algorithms=["RS256"],
            audience=OIDC_AUDIENCE,
            issuer=OIDC_ISSUER_URL,
        )
        return claims

    except ImportError as e:
        logger.error(f"Authentication backend misconfigured: missing JWT/HTTPX dependencies ({e})")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication backend misconfigured: missing JWT/HTTPX dependencies",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token validation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {str(e)}",
        )


class AuthMiddleware(BaseHTTPMiddleware):
    """
    Authentication middleware.

    In development (AUTH_DISABLED=true): trusts X-User header.
    In production: validates JWT Bearer token from Authorization header.
    """

    async def dispatch(self, request: Request, call_next):
        # Skip auth for public paths
        if request.url.path in PUBLIC_PATHS:
            request.state.user = "anonymous"
            request.state.user_email = ""
            request.state.user_groups = []
            return await call_next(request)

        if AUTH_DISABLED:
            # Development mode: trust X-User header
            request.state.user = request.headers.get("X-User", "dev-user")
            request.state.user_email = f"{request.state.user}@local"
            request.state.user_groups = []
            return await call_next(request)

        # Production mode: validate Bearer token
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing or invalid Authorization header",
                headers={"WWW-Authenticate": "Bearer"},
            )

        token = auth_header[7:]  # Strip "Bearer "
        claims = await _validate_token(token)

        request.state.user = claims.get("sub", "unknown")
        request.state.user_email = claims.get("email", "")
        request.state.user_groups = claims.get("groups", [])

        return await call_next(request)
