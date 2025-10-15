import os
import json
from pathlib import Path
import PyPDF2
import google.generativeai as genai

class RAGEngine:
    def __init__(self, model_name="gemini-2.0-flash"):
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        genai.configure(api_key=api_key)
        
        self.model = genai.GenerativeModel(f"models/{model_name}")
        self.model_name = model_name
        self.documents = {}
        self.embeddings = {}
        self.db_path = 'knowledge_base.json'
        self.load_from_disk()
    
    def ingest_document(self, file_path, filename):
        doc_id = f"doc_{len(self.documents)}_{int(Path(file_path).stat().st_mtime)}"
        
        try:
            if file_path.endswith('.pdf'):
                text = self._extract_pdf(file_path)
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            
            self.documents[doc_id] = {
                'filename': filename,
                'content': text,
                'size': len(text),
                'chunks': self._chunk_text(text)
            }
            
            self.save_to_disk()
            
            return doc_id
        except Exception as e:
            raise Exception(f"Failed to ingest {filename}: {str(e)}")
    
    def _extract_pdf(self, pdf_path):
        text = []
        try:
            with open(pdf_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text.append(page.extract_text())
            return '\n'.join(text)
        except Exception as e:
            raise Exception(f"PDF extraction failed: {str(e)}")
    
    def _chunk_text(self, text, chunk_size=500, overlap=50):
        chunks = []
        words = text.split()
        chunk_words = []
        
        for word in words:
            chunk_words.append(word)
            if len(' '.join(chunk_words)) >= chunk_size:
                chunks.append(' '.join(chunk_words))
                chunk_words = chunk_words[-int(overlap/5):]
        
        if chunk_words:
            chunks.append(' '.join(chunk_words))
        
        return chunks
    
    def query(self, user_query):
        relevant_chunks = self.search(user_query, top_k=5)
        
        if not relevant_chunks:
            return {
                'answer': "No relevant documents found in the knowledge base.",
                'sources': [],
                'confidence': 0.0
            }
        
        # Prepare context
        context = "\n\n".join([f"Source: {c['source']}\n{c['text']}" for c in relevant_chunks])
        
        # Generate answer using Claude
        prompt = f"""You are a helpful assistant that answers questions based on provided documents.

Context from knowledge base:
{context}

User Question: {user_query}

Please provide a concise, accurate answer based only on the provided context. If the answer cannot be found in the context, say so clearly."""
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=1024,
                    temperature=0.7
                )
            )
            
            answer = response.text
            
            # Calculate confidence based on context relevance
            confidence = min(1.0, len(relevant_chunks) / 5)
            
            return {
                'answer': answer,
                'sources': [c['source'] for c in relevant_chunks],
                'confidence': confidence
            }
        except Exception as e:
            raise Exception(f"LLM query failed: {str(e)}")
    
    def search(self, query_term, top_k=5):
        results = []
        
        query_terms = query_term.lower().split()
        
        for doc_id, doc in self.documents.items():
            for i, chunk in enumerate(doc['chunks']):
                chunk_lower = chunk.lower()
                match_count = sum(1 for term in query_terms if term in chunk_lower)
                
                if match_count > 0:
                    relevance_score = match_count / len(query_terms)
                    results.append({
                        'text': chunk,
                        'source': f"{doc['filename']} (chunk {i+1})",
                        'doc_id': doc_id,
                        'score': relevance_score
                    })
        
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]
    
    def delete_document(self, doc_id):
        if doc_id in self.documents:
            del self.documents[doc_id]
            self.save_to_disk()
        else:
            raise Exception(f"Document {doc_id} not found")
    
    def get_documents_metadata(self):
        return [
            {
                'doc_id': doc_id,
                'filename': doc['filename'],
                'size': doc['size'],
                'chunks': len(doc['chunks'])
            }
            for doc_id, doc in self.documents.items()
        ]
    
    def save_to_disk(self):
        data = {doc_id: {
            'filename': doc['filename'],
            'content': doc['content'],
            'size': doc['size']
        } for doc_id, doc in self.documents.items()}
        
        with open(self.db_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_from_disk(self):
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'r') as f:
                    data = json.load(f)
                    for doc_id, doc_data in data.items():
                        self.documents[doc_id] = {
                            'filename': doc_data['filename'],
                            'content': doc_data['content'],
                            'size': doc_data['size'],
                            'chunks': self._chunk_text(doc_data['content'])
                        }
            except Exception as e:
                print(f"Error loading from disk: {str(e)}")
