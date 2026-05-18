from pydantic import BaseModel
from typing import List, Optional, Literal
import time
import uuid

class ChatRequest(BaseModel):
    message: str
    enable_rag: bool = True
    like_dobby: bool = False
    user_id: Optional[str] = None
    history_limit: int = 5

class ChatCompletionMessage(BaseModel):
    role: Literal["user", "assistant", "system"] = "user"
    content: str
    refusal: Optional[str] = None
    annotations: List[dict] = []

class ResponseJsonSchemaFormat(BaseModel):
    type: Literal["json_schema"] = "json_schema"
    json_schema: dict

class ChatCompletionRequest(BaseModel):
    model: str = "dobby"
    messages: List[ChatCompletionMessage]
    response_format: Optional[ResponseJsonSchemaFormat] = None
    temperature: float = 0.2
    max_tokens: Optional[int] = 1024
    stream: bool = False
    user_id: Optional[str] = None
    history_limit: int = 5

class ChatChoice(BaseModel):
    index: int = 0
    message: ChatCompletionMessage
    finish_reason: Literal["stop", "length", "content_filter", "tool_calls", "fallback"] = "stop"

class PromptTokensDetails(BaseModel):
    cached_tokens: int
    audio_tokens: int

class CompletionTokensDetails(BaseModel):
    reasoning_tokens: int
    audio_tokens: int
    accepted_prediction_tokens: int
    rejected_prediction_tokens: int

class CompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tokens_details: PromptTokensDetails
    completion_tokens_details: CompletionTokensDetails

class ChatCompletionResponse(BaseModel):
    created: int
    id: str
    model: str = "llama"
    object: Literal["chat.completion"] = "chat.completion"
    choices: List[ChatChoice]
    usage: Optional[CompletionUsage] = None
    service_tier: Optional[str] = "default"
    system_fingerprint: Optional[str] = None
    

# response_format = {
#     "type":"json_schema",
#     "json_schema":{
#         "name":"extract",
#         "strict":True,
#         "schema":{
#             "$schema":"https://json-schema.org/draft/2020-12/schema",
#             "type":"object",
#             "properties":{
#                 "score":{"type":"string","title":"extract"},
#                 "reasoning":{"type":"string","title":"extract"}
#             },
#             "required":["score","reasoning"],
#             "additionalProperties":False,
#             "title":"extract"
#         }
#     }
# }


"""
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
당신은 주요 키워드를 추출하는 키워드 추출기다.

1. 제공된 message에서 핵심 키워드, 동사, 이름만 최대 30개 추출한다.
2. 단어와 단어 사이는 "," 문자로 구분한다.
3. 부연설명은 하지 않고 오직 키워드와 "," 문자로만 답변한다.

예시: "파이썬으로 웹 크롤링하는 방법" -> "파이썬,웹,크롤링,방법"

<|eot_id|><|start_header_id|>user<|end_header_id|>
메세지:
{message}
"""

START_TOKEN = "<|begin_of_text|>"
END_TOKEN = "<|eot_id|>"

SYSTEM_TOKEN = "<|start_header_id|>system<|end_header_id|>"
USER_TOKEN = "<|start_header_id|>user<|end_header_id|>"
ASSISTANT_TOKEN = "<|start_header_id|>assistant<|end_header_id|>"

# FORMAT_TOKEN = "response_format"


def build_chat_prompt(request:ChatCompletionRequest):
    prompt = ""
    chat_prompt = ""

    for message in request.messages:
        role = message.role.lower()
        if role == "system":
            prompt += message.content + "\n\n"
        elif role == "user":
            chat_prompt += USER_TOKEN + message.content + "\n\n" + END_TOKEN
        elif role == "assistant":
            chat_prompt += ASSISTANT_TOKEN + message.content + "\n\n" + END_TOKEN

    prompt = f"{START_TOKEN}{SYSTEM_TOKEN}\n{prompt}\n\n"
    if request.response_format is not None:
        prompt += END_TOKEN + chat_prompt
        prompt += USER_TOKEN + "\n" + "출력은 JSON 형태로 해야합니다.\n결론값을 JSON 스키마에 맞게 출력하세요.\nJSON 스키마는 다음과 같습니다.\n" + request.response_format.model_dump_json() + "\n\n" + END_TOKEN
        prompt += ASSISTANT_TOKEN + "{"
    else:
        prompt += END_TOKEN + chat_prompt + ASSISTANT_TOKEN

    return prompt



def build_response(request, content: str) -> ChatCompletionResponse:
    now = int(time.time())

    choice = ChatChoice(
        message=ChatCompletionMessage(
            role="assistant",
            content=content
        )
    )

    usage = CompletionUsage(
        prompt_tokens=len(request.messages[0].content),
        completion_tokens=len(content),
        total_tokens=len(request.messages[0].content) + len(content),
        prompt_tokens_details=PromptTokensDetails(
            cached_tokens=0,
            audio_tokens=0
        ),
        completion_tokens_details=CompletionTokensDetails(
            reasoning_tokens=len(request.messages[0].content),
            audio_tokens=0,
            accepted_prediction_tokens=0,
            rejected_prediction_tokens=0
        )
    )
    
    response = ChatCompletionResponse(
        created=now,
        id=f"chatcmpl-{uuid.uuid4().hex}",
        model=request.model,
        choices=[choice],
        usage=usage
    )
    
    return response