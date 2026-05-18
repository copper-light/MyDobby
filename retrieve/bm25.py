from rank_bm25 import BM25Okapi

documents = [
    "강아지가 좋아하는 음식은 사료다",
    "고양이도 음식을 좋아한다",
    "강아지 사료는 영양가가 높다",
    "음식은 생명의 근원이다"
]

tokenized_docs = [doc.split() for doc in documents]
bm25 = BM25Okapi(tokenized_docs)

query = "강아지 음식"
tokenized_query = query.split()

scores = bm25.get_scores(tokenized_query)
top_docs = bm25.get_top_n(tokenized_query, documents, n=3)

print(scores)
print(top_docs)
