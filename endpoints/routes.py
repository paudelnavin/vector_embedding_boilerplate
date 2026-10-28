python
from flask import Blueprint, jsonify, request
import urllib.request
import json

api_bp = Blueprint("api", __name__)

@api_bp.route("/api/vectorize", methods=["POST"])
def get_vector_embeddings():
    try:
        # 1. Catch the incoming payload from your local FastAPI server
        body = request.get_json() or {}
        text_input = body.get("text", "")
        
        if not text_input:
            return jsonify({"error": "Missing 'text' key in JSON request body"}), 400
            
        # 2. Free, zero-dependency streaming model provider pipeline
        hf_url = "https://huggingface.co"
        req = urllib.request.Request(
            hf_url, 
            data=json.dumps({"inputs": text_input}).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        
        # 3. Stream the raw vector arrays back to your database layer
        with urllib.request.urlopen(req) as response:
            vector_array = json.loads(response.read().decode())
            return jsonify({"vector": vector_array}), 200
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500
