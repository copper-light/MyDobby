import json
from external.connect import get_langfuse_client, get_mongo_db
from apiserver.pompt import create_answer_prompt, create_query_expansion_prompt, create_like_dobby_prompt, create_not_found_context_prompt
from retrieve.retrieve_documents import retrieve_documents
from models.load_model import get_model
from session.session import get_or_create_session, append_message, get_recent_messages
# from retrieve.embedding import extract_keyword

def request_chat(message: str, enable_rag: bool = True, return_judge: bool = False, user_id = None, history_limit: int = 5):
    model = get_model("llama")
    client = get_langfuse_client()
    db = get_mongo_db()

    # 세션 관리: user_id가 있으면 대화 이력 로드
    history = []
    if user_id:
        get_or_create_session(db, user_id)
        history = get_recent_messages(db, user_id, n=history_limit)

    try:
        prompt_obj = client.get_prompt("answer_prompt")
        with client.start_as_current_observation(as_type="generation", name="request", prompt=prompt_obj) as trace:
            trace.update(input={"request": message, "enable_rag": enable_rag})
            
            if enable_rag:
                with trace.start_as_current_observation(as_type="generation", name="query_expansion") as query_expansion_span:
                    try_count = 0
                    while True:
                        try:
                            query_expansion_prompt = create_query_expansion_prompt(client, message)
                            query_expansion_span.update(input={"query_expansion_prompt": query_expansion_prompt})
                            content, input_token_len, output_token_len = model.generate(query_expansion_prompt, top_p=0.8, temperature=0.1)
                            new_querys = json.loads("{"+ content)
                            new_querys["questions"].insert(0, message)
                            # new_querys = {
                            #     "questions": [message],
                            # }

                            keywords = []
                            ex_model = get_model("keyword_extractor")
                            for q in new_querys["questions"]:
                                keywords += ex_model.generate(q)
                            new_querys["keywords"] = " ".join(list(set(keywords)))
                            break
                        except Exception as e:
                            try_count += 1
                            print(f"query_expansion try count: {try_count} : {e}, \n{content}")
                            if try_count == 3:
                                raise Exception("다시 시도해 주세요.")

                    query_expansion_span.update(output=new_querys)

                    with trace.start_as_current_observation(as_type="retrieval", name="retrieve") as retrieve_span:
                        retrieve_span.update(input=new_querys)
                        context = retrieve_documents(new_querys, retrieve_topk=30, rerank_topk=5, trace = retrieve_span)
                        retrieve_span.update(output=context)
                        
                        history = None
                        if len(context) > 0:
                            prompt = create_answer_prompt(client, message, context, history)
                        else:
                            prompt = create_not_found_context_prompt(client, message, history)
            else:
                prompt = message
                
            with trace.start_as_current_observation(as_type="generation", name="generate") as generate_span:
                generate_span.update(input=prompt)
                content, input_token_len, output_token_len = model.generate(prompt, top_p=0.8, temperature=0.2)

                generate_span.update(output={"answer": content})
                generate_span.update(usage_details={
                    "input": input_token_len,
                    "output": output_token_len,
                    "total": input_token_len + output_token_len,
            })

            if enable_rag:
                judge = {
                    "query": message,
                    "expansion_query": new_querys,
                    "answer": content,
                    "context": context,
                    "prompt": prompt,
                    "history": history
                }

                trace.update(output=judge, metadata ={"rag": True,})
                if return_judge:
                    return judge
            else:
                trace.update(output={"query": message, "answer": content}, metadata ={"rag": False,})
                


        # 세션에 대화 저장
        if user_id:
            append_message(db, user_id, message, content, "llama")

        client.flush()
        return content

    except Exception as e:
        print(e)
        return "[내부 에러] 다시 시도해 주세요."