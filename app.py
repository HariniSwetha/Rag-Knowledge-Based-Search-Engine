from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import os
import json
from datetime import datetime
from dotenv import load_dotenv
from rag_engine import RAGEngine

load_dotenv()

app = Flask(__name__, template_folder='.', static_folder='.')
CORS(app)

rag_engine = RAGEngine()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

@app.route('/api/ingest', methods=['POST'])
def ingest_documents():
    if 'files' not in request.files:
        return jsonify({'error': 'No files provided'}), 400
    
    files = request.files.getlist('files')
    if not files:
        return jsonify({'error': 'No files selected'}), 400
    
    results = []
    for file in files:
        if file.filename == '':
            continue
        
        try:
            temp_path = os.path.join('temp', file.filename)
            os.makedirs('temp', exist_ok=True)
            file.save(temp_path)
            
            doc_id = rag_engine.ingest_document(temp_path, file.filename)
            results.append({
                'filename': file.filename,
                'doc_id': doc_id,
                'status': 'success'
            })
            
            os.remove(temp_path)
        except Exception as e:
            results.append({
                'filename': file.filename,
                'error': str(e),
                'status': 'failed'
            })
    
    return jsonify({
        'message': 'Document ingestion completed',
        'results': results,
        'total_documents': len(rag_engine.documents)
    })

@app.route('/api/query', methods=['POST'])
def query():
    data = request.json
    if not data or 'query' not in data:
        return jsonify({'error': 'No query provided'}), 400
    
    user_query = data['query'].strip()
    if not user_query:
        return jsonify({'error': 'Query cannot be empty'}), 400
    
    try:
        result = rag_engine.query(user_query)
        return jsonify({
            'query': user_query,
            'answer': result['answer'],
            'sources': result['sources'],
            'confidence': result['confidence'],
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/documents', methods=['GET'])
def list_documents():
    docs = rag_engine.get_documents_metadata()
    return jsonify({
        'total': len(docs),
        'documents': docs
    })

@app.route('/api/documents/<doc_id>', methods=['DELETE'])
def delete_document(doc_id):
    try:
        rag_engine.delete_document(doc_id)
        return jsonify({'message': f'Document {doc_id} deleted successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/search', methods=['GET'])
def search():
    query_term = request.args.get('q', '').strip()
    top_k = request.args.get('top_k', 5, type=int)
    
    if not query_term:
        return jsonify({'error': 'Search term required'}), 400
    
    try:
        results = rag_engine.search(query_term, top_k=top_k)
        return jsonify({
            'query': query_term,
            'results': results,
            'count': len(results)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)