from fastapi import HTTPException, Depends
from fastapi.security import APIKeyHeader
import config.settings as settings

api_key_header = APIKeyHeader(name="Authorization")

async def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key != "Bearer " + settings.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return api_key
