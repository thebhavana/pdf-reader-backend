import os
import traceback
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.conf import settings
from .serializers import UploadSerializer, QuerySerializer
from .utils import extract_pages_text, chunk_text, get_embedding, upsert_vectors, search_index, ask_llm

class UploadPDF(APIView):
    def post(self, request, *args, **kwargs):
        try:
            serializer = UploadSerializer(data=request.data)
            serializer.is_valid(raise_exception=True)
            f = serializer.validated_data['file']
            
            # Save file
            out_dir = settings.MEDIA_ROOT
            os.makedirs(out_dir, exist_ok=True)
            path = os.path.join(out_dir, f.name)
            
            with open(path, 'wb') as dest:
                for chunk in f.chunks():
                    dest.write(chunk)

            # Extract pages text
            pages = extract_pages_text(path)
            if not pages:
                return Response({'error': 'No text extracted from PDF.'}, status=status.HTTP_400_BAD_REQUEST)

            vectors = []
            metadatas = []
            for p in pages:
                page = p['page']
                text = p['text'] or ''
                cks = chunk_text(text)
                for i, ck in enumerate(cks):
                    emb = get_embedding(ck)
                    vectors.append(emb)
                    metadatas.append({'page': page, 'text': ck, 'file_name': f.name})

            # Upsert vectors to FAISS
            index_path = settings.FAISS_INDEX_PATH
            upsert_vectors(vectors, metadatas, index_path)

            return Response({'status': 'ok', 'file_path': path})
        
        except Exception as e:
            print("Exception during PDF upload:", str(e))
            print(traceback.format_exc())
            return Response({'error': 'Failed to process PDF upload.', 'details': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class QueryView(APIView):
    def post(self, request, *args, **kwargs):
        try:
            serializer = QuerySerializer(data=request.data)
            serializer.is_valid(raise_exception=True)
            q = serializer.validated_data['question']
            file_path = serializer.validated_data.get('file_path', None)
            
            # Embed query
            qvec = get_embedding(q)
            index_path = settings.FAISS_INDEX_PATH
            results = search_index(qvec, index_path, top_k=5)
            
            contexts = []
            seen_pages = set()
            for r in results:
                if r['file_name'] and file_path and r['file_name'] != os.path.basename(file_path):
                    continue
                if r['page'] not in seen_pages:
                    contexts.append({'page': r['page'], 'text': r['text']})
                    seen_pages.add(r['page'])

            answer, pages = ask_llm(q, contexts)
            return Response({'answer': answer, 'pages': pages})
        
        except Exception as e:
            print("Exception during query processing:", str(e))
            print(traceback.format_exc())
            return Response({'error': 'Failed to process query.', 'details': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
