"""Role-Based Access Control (RBAC) — Control plane and tool permission scoping.

Defines role hierarchy: ADMIN > OPERATOR > USER.
Provides FastAPI dependencies for endpoint authorization and tool permission validation.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from enum import Enum, StrEnum

from fastapi import HTTPException, Request

logger = logging.getLogger(__name__)


class Role(StrEnum):
    ADMIN = "admin"
    OPERATOR = "operator"
    USER = "user"


ROLE_HIERARCHY: dict[Role, int] = {
    Role.ADMIN: 3,
    Role.OPERATOR: 2,
    Role.USER: 1,
}


class RBACManager:
    """Manage roles and endpoint/tool permissions."""

    @staticmethod
    def get_user_role(request: Request) -> Role:
        """Extract user role from request headers or state."""
        role_header = request.headers.get("X-User-Role", "").lower()
        if role_header in {r.value for r in Role}:
            return Role(role_header)

        # Admin route default check
        if request.url.path.startswith("/admin"):
            api_key = request.headers.get("Authorization", "")
            if "admin" in api_key.lower():
                return Role.ADMIN

        return Role.USER

    @staticmethod
    def has_permission(user_role: Role, required_role: Role) -> bool:
        """Check if user_role satisfies required_role in the hierarchy."""
        return ROLE_HIERARCHY.get(user_role, 0) >= ROLE_HIERARCHY.get(required_role, 0)


def require_role(required_role: Role) -> Callable:
    """FastAPI dependency enforcing minimum role requirements.

    Usage:
        @router.post("/admin/reload", dependencies=[Depends(require_role(Role.ADMIN))])
    """

    async def _role_checker(request: Request) -> Role:
        user_role = RBACManager.get_user_role(request)
        if not RBACManager.has_permission(user_role, required_role):
            logger.warning(
                f"RBAC Denied: User with role '{user_role.value}' attempted to access "
                f"endpoint '{request.url.path}' requiring '{required_role.value}'."
            )
            raise HTTPException(
                status_code=403,
                detail=f"Permission denied: Requires '{required_role.value}' role.",
            )
        return user_role

    return _role_checker
