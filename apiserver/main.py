from contextlib import asynccontextmanager

from fastapi import FastAPI

from models.load_model import get_model
from retrieve.retrieve_documents import retrieve_documents

from apiserver.exception import resister_exception_handler
from apiserver.slack import APP_TOKEN, register_slack_app, get_slack
from apiserver.model import build_response, ChatRequest
from apiserver.service import request_chat
from apiserver.model import ChatCompletionRequest, build_chat_prompt

from slack_bolt.adapter.socket_mode.aiohttp import AsyncSocketModeHandler

slack_app = get_slack()
slack_socket_handler = AsyncSocketModeHandler(slack_app, APP_TOKEN)

@asynccontextmanager
async def lifespan(app: FastAPI):
    get_model("llama")
    get_model("koelectra")
    register_slack_app(slack_app)

    await slack_socket_handler.connect_async()
    try:
        yield
    finally:
        await slack_socket_handler.close_async()


app = FastAPI(lifespan=lifespan)
resister_exception_handler(app)

# @app.on_event("startup")
# async def startup():
#     get_model("llama")

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/rag")
async def chat(request: ChatRequest):
    result = request_chat(request.message, request.enable_rag, user_id=request.user_id, history_limit=request.history_limit)
    return result

@app.post("/context")
async def rag(request: ChatRequest):
    ret = retrieve_documents(request.message, retrieve_topk=20, rerank_topk=5)
    return {"ret": ret}

# @app.post("/v1/chat/completions")
# async def chat_v1_completions(request: ChatCompletionRequest):
#     return chat_completions(request)

@app.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if request.model == "dobby":
        result = request_chat(request.messages[0].content, True, user_id=request.user_id, history_limit=request.history_limit)
    elif request.model == "llama3.1":
        messages = build_chat_prompt(request)
        result = request_chat(messages, False, False, user_id=request.user_id, history_limit=request.history_limit)
        if request.response_format is not None:
            result = "{" + result
    else:
        raise ValueError(f"Invalid model name: {request.model}")

    response = build_response(request, result)
    return response


@app.post("/chat/test")
async def chat_test(request: ChatCompletionRequest):
    if request.model == "dobby":
        result = request_chat(request.messages[0].content, True, return_judge=True, user_id=request.user_id, history_limit=request.history_limit)
    # elif request.model == "llama3.1":
    #     messages = build_chat_prompt(request)
    #     result = request_chat(messages, False, False)
    #     if request.response_format is not None:
    #         result = "{" + result
    else:
        raise ValueError(f"Invalid model name: {request.model}")

    # response = build_response(request, result, return_judge=True)
    return result


@app.get("/v1/models")
async def models():
    return {"models": ["dobby", "llama3.1"]}


# @app.middleware("http")
# async def log_request_body(request: Request, call_next):
#     body = await request.body()
    
#     print(body)
#     response = await call_next(request)
#     return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

"""
{"model":"dobby","stream":false,"response_format":{"type":"json_schema","json_schema":{"name":"extract","strict":true,"schema":{"$schema":"https://json-schema.org/draft/2020-12/schema","type":"object","properties":{"score":{"type":"string","title":"extract"},"reasoning":{"type":"string","title":"extract"}},"required":["score","reasoning"],"additionalProperties":false,"title":"extract"}}},"messages":[{"role":"user","content":"Extract a score (1-5) and reasoning from this text: \\"This is a test. It worked perfectly because it matched all passing criteria.\\""}]}
"""