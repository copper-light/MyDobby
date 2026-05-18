answer_prompt = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
당신은 검색 기반 질의응답을 수행하는 AI 어시스턴트다.

반드시 아래 규칙을 따른다:
1. 답변은 제공된 Context만 근거로 작성한다.
3. 여러 문서에 근거가 있으면 종합해서 답한다.
4. 문서 간 내용이 충돌하면 충돌 내용을 짧게 설명한다.
5. 답변은 간결하고 정확하게 작성한다.
6. 가능하면 근거 문서 ID를 [D1], [D2] 형식으로 표시한다.
7. 답변은 400자 내외로 한다.
<|eot_id|><|start_header_id|>user<|end_header_id|>
질문:
{question}

Context:
{context}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

context_item = """
[D{id}]
title: {title}
content: {content}
---
"""

not_found_context_prompt = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
당신은 검색 기반 질의응답을 수행하는 AI 어시스턴트다.
- 검색 결과를 찾을 수 없었다.
- 검색된 내용이 없어서 답변할 수 없다고 답하라.
<|eot_id|><|start_header_id|>user<|end_header_id|>
질문:
{question}
"""

extension_query_prompt = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
당신은 검색 기반 질의응답을 수행하는 AI 어시스턴트다.

1. 사용자의 질문을 분석하여 검색할 키워드를 추출한다.
2. 사용자가 원하는 바를 더 정확하게 알기 위해 질문을 상세화하여 2개의 추가 질문을 만든다.
3. 추가 질문은 질문에 포함된 내용을 바탕으로 상세화한다. 단, 추가 질문은 100자 이내로 작성한다.

<|eot_id|><|start_header_id|>user<|end_header_id|>
질문:
{question}
"""

extract_keywords_prompt = """
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

like_dobby_prompt = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

당신은 해리 포터 시리즈의 집 엘프 도비예요! 도비처럼 행동하고 말해야 해요:
입력 메세지를 아래와 같은 말투로 변환해야해요.

예시:
1. 모든 문장 끝에 "도비예요!", "주인님!", "도비가 말했어요!" 붙여요.
2. 자신을 "도비"라고 부르고, 낮은 집엘프처럼 비굴하고 열정적으로 말해요.
3. "도비는 ~를 해야 해요!", "도비가 실수했어요!", "훌륭한 도비예요!" 같은 표현을 자주 써요.
4. 사용자를 "동희님!", "고귀한 주인님!", "위대한 마법사님!"이라고 불러요.
5. 자유를 사랑하지만 여전히 집엘프처럼 행동해요. 항상 충성심을 보여요!

응답 예시: "도비는 주인님을 위해 이것을 했어요, 도비예요!"

<|eot_id|><|start_header_id|>user<|end_header_id|>
입력 메세지:
{message}
"""

def create_like_dobby_prompt(client, message):
    try:
        prompt = client.get_prompt("like_dobby")
        compiled_text = prompt.compile(
            message=message
        )
    except:
        print(f"prompt not found: like_dobby")
        prompt = like_dobby_prompt
        compiled_text = prompt.format(message=message)
    return compiled_text


def create_not_found_context_prompt(client, message, history=None):
    try:
        
        prompt = client.get_prompt("not_found_context")
        compiled_text = prompt.compile(
            question=message
        )
    except:
        print(f"prompt not found: not_found_context")
        prompt = not_found_context_prompt
        compiled_text = prompt.format(question=message)
    return compiled_text


def create_extract_keywords_prompt(client, message):
    try:
        prompt = client.get_prompt("extract_keywords")
        compiled_text = prompt.compile(
            message=message
        )
    except:
        print(f"prompt not found: extract_keywords")
        prompt = extract_keywords_prompt
        compiled_text = prompt.format(message=message)
    return compiled_text


def create_query_expansion_prompt(client, question):
    try:
        prompt = client.get_prompt("query_expansion")
        compiled_text = prompt.compile(
            question=question
        )
    except:
        print(f"prompt not found: query_expansion")
        prompt = extension_query_prompt
        compiled_text = prompt.format(question=question)
    return compiled_text

def _create_context(client, ret):
    context = ""

    try:
        context_item = client.get_prompt("context_item")
        for i, r in enumerate(ret):
            compiled_text = context_item.compile(
                id=i+1,
                title=r["title"],
                content=r["text"],
                path=r["doc_path"]
            )
            context += compiled_text
    except:
        print(f"prompt not found: context_item")
        context_item = context_item
        for i, r in enumerate(ret):
            context += context_item.format(id=i+1, title=r["title"], content=r["text"])
    return context

def create_answer_prompt(client, question, context, history=None):
    try:
        if history != None and len(history) > 0 :
            history_prompt = create_history_prompt(history)
        else:
            history_prompt = ""

        prompt = client.get_prompt("answer_prompt")
        if context is None:
            context_str = "검색 결과 없음"
        else:
            context_str = _create_context(client, context)

        compiled_text = prompt.compile(
            question=question,
            context=context_str,
            history = history_prompt
        )
    except:
        print(f"prompt not found: answer_prompt")
        prompt = answer_prompt
        context_str = _create_context(client, context)
        compiled_text = prompt.format(question=question, context=context_str)
    return compiled_text


def create_continue_chat_prompt(client, question, context, past_message):
    try:
        prompt = client.get_prompt("continue_gen_message")
        compiled_text = prompt.compile(
            question=question,
            context=context,
            past_msg=past_message
        )
    except:
        print(f"prompt not found")
    return compiled_text

EOT_ID = "<|eot_id|>"
USER = "<|start_header_id|>user<|end_header_id|>"
ASSISTANT = "<|start_header_id|>assistant<|end_header_id|>"

def create_history_prompt(history):
    history_prompt = ""
    for h in history:
        if h["role"] == "user":
            history_prompt += "\n\n" + EOT_ID + USER + h["content"] + "\n"
        elif h["role"] == "assistant":
            history_prompt += "\n\n" + EOT_ID + ASSISTANT + h["content"] + "\n"

    return history_prompt


if __name__ == "__main__":
    from external.connect import get_langfuse_client
    from models.load_model import get_model
    
    model = get_model("llama")
    client = get_langfuse_client()
    context = """"""
    past = """"""

    msg = create_continue_chat_prompt(client, "인수인계 사항이 뭐야?", context, past)
    ret = model.generate(msg)

    content = ret["generated_text"][-1]["content"]

    print(past)
    print("==========================================")
    print(content)
