import pandas as pd
import random
import qdrant_client
from qdrant_client.http import models
from typing import List, Dict, Any
from collections import defaultdict
import argparse
from datetime import datetime
from external.connect import get_qdrant_client

def extract_all_points_from_qdrant(
    client: qdrant_client.QdrantClient,
    collection_name: str,
    limit_per_doc: int = 3
) -> pd.DataFrame:
    """
    Qdrant 컬렉션에서 모든 points를 scroll로 추출하고,
    doc_id별로 1~limit_per_doc 샘플링해서 DataFrame 반환.
    
    필드 가정: payload에 doc_id, parent_id, chunk_id, title, text, create_at
    """
    # Scroll로 모든 points 가져오기 (offset 기반)
    all_points = []
    offset = None
    while True:
        points, next_offset = client.scroll(
            collection_name=collection_name,
            limit=100,  # batch size
            with_payload=True,
            with_vectors=False,
            offset=offset
        )
    
        points_batch = points
        if not points_batch:
            break
            
        all_points.extend(points_batch)
        offset = next_offset
        if offset is None:
            break
        print(f"가져온 points: {len(all_points)}")
    
    print(f"총 points 수: {len(all_points)}")
    
    # Payload를 DataFrame으로 변환
    data = []
    for point in all_points:
        payload = point.payload or {}
        data.append({
            'point_id': point.id,
            'doc_id': payload.get('doc_id'),
            'parent_id': payload.get('parent_id'),
            'chunk_id': payload.get('chunk_id'),
            'title': payload.get('title', ''),
            'text': payload.get('text', ''),
            'create_at': payload.get('create_at', ''),
            'text_len': len(payload.get('text', '').strip()) if payload.get('text') else 0
        })
    
    df = pd.DataFrame(data)
    
    # 빈 text나 doc_id 없는 것 필터
    df = df[(df['text'].str.strip() != '') & df['doc_id'].notna()]
    print(f"유효 청크 수: {len(df)}")
    
    # doc_id별 그룹화하고 샘플링
    grouped = df.groupby('doc_id')
    sampled_rows = []
    
    for doc_id, group in grouped:
        # text_len 기준 정렬 후 긴 것 우선 (품질 좋을 가능성)
        sorted_group = group.sort_values('text_len', ascending=False)
        n_sample = min(limit_per_doc, len(sorted_group))
        samples = sorted_group.head(n_sample).copy()
        sampled_rows.extend(samples.to_dict('records'))
    
    sampled_df = pd.DataFrame(sampled_rows)
    print(f"샘플링 결과: {len(sampled_df)}개 (문서 {len(grouped)}개 커버)")
    
    return sampled_df[['doc_id', 'parent_id', 'chunk_id', 'title', 'text', 'create_at']]

def main():
    parser = argparse.ArgumentParser(description="Qdrant에서 문서당 1~3개 청크 샘플링 CSV 생성")
    parser.add_argument('--collection', default="child_chunks", help='컬렉션 이름')
    parser.add_argument('--limit-per-doc', type=int, default=3, help='문서당 최대 샘플 수 (1~3)')
    parser.add_argument('--output', default='sampled_chunks.csv', help='출력 CSV 파일')
    
    args = parser.parse_args()
    
    # Qdrant 클라이언트 연결
    client = get_qdrant_client()
    
    print(f"컬렉션 '{args.collection}'에서 샘플링 시작...")
    print(f"문서당 최대 {args.limit_per_doc}개 청크 추출")
    
    # 데이터 추출 및 샘플링
    sampled_df = extract_all_points_from_qdrant(
        client, args.collection, args.limit_per_doc
    )
    
    # CSV 저장 (UTF-8)
    sampled_df.to_csv(args.output, index=False, encoding='utf-8-sig')
    
    print(f"\n[OK] 샘플링 완료!")
    print(f"총 {len(sampled_df)}개 청크 저장: {args.output}")
    print("\n컬럼 확인:")
    print(sampled_df.head())
    print(f"\n문서 커버리지: {sampled_df['doc_id'].nunique()}개 문서")

if __name__ == "__main__":
    main()