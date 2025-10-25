import os
import time
from typing import Any, Dict, List

from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
import requests

app = Flask(__name__)

load_dotenv()
api_key = os.environ.get('API_KEY')
base_url = os.environ.get('BASE_URL')

# Endpoint for the external chat/vision agent API. If needed this can be
# overridden via env; otherwise we default to using base_url + /qa/chat
REKA_VIDEO_QA_ENDPOINT = os.environ.get(
    'REKA_VIDEO_QA_ENDPOINT', 
    f"{base_url.rstrip('/')}/qa/chat"
)

# Simple in-memory cache for videos to avoid hitting the API on every request.
_VIDEO_CACHE: Dict[str, Any] = {
    "timestamp": 0.0,
    "ttl": 60.0,
    "results": []
}


def fetch_videos() -> List[Dict[str, Any]]:
    """
    Fetch the list of videos from Reka Vision API, with basic caching.

    The API is expected to respond with a JSON structure containing a
    "results" key that holds a list of video objects. Each video includes
    metadata with fields like "title" and "thumbnail".

    Returns:
        List[Dict[str, Any]]: List of video dictionaries from the API.
    """
    now = time.time()
    is_stale = (now - _VIDEO_CACHE["timestamp"]) > _VIDEO_CACHE["ttl"]

    if not base_url:
        # Without BASE_URL we can't call the API; return empty.
        return []

    url = f"{base_url.rstrip('/')}/videos/get"
    headers = {}
    if api_key:
        headers["X-Api-Key"] = api_key

    try:
        response = requests.post(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        results = data.get("results", [])
        _VIDEO_CACHE.update({
            "timestamp": now,
            "results": results
        })
        return results
    except Exception as e:
        # On failure, keep old cache if available; otherwise empty list.
        if _VIDEO_CACHE["results"]:
            return _VIDEO_CACHE["results"]
        return []


def call_reka_vision_qa(video_id: str) -> Dict[str, Any]:
    """Call the Reka Video QA API for a given video.

    The request format follows the user's provided specification. We issue a
    POST request with the video_id and a static user prompt asking to gently
    roast the person in the video.

    Environment Variables:
        REKA_VIDEO_QA_ENDPOINT: Optional override for the API endpoint.
            If not set, defaults to {base_url}/qa/chat
        api_key or API_KEY: API key placed in the X-Api-Key header.

    Parameters:
        video_id (str): The UUID of the video to query.

    Returns:
        Dict[str, Any]: Parsed JSON response (may include keys like
        chat_response, system_message, error, status, etc.). On total failure
        returns a dict with an 'error' key.
    """
    headers = {}
    if api_key:
        headers['X-Api-Key'] = api_key

    payload = {
        "video_id": video_id,
        "messages": [
            {
                "role": "user",
                "content": "Write a funny and gently roast about the person, or the voice in this video. Reply in a markdown format."
            }
        ]
    }

    try:
        resp = requests.post(
            REKA_VIDEO_QA_ENDPOINT,
            headers=headers,
            json=payload,
            timeout=30
        )
        # Even on non-2xx we attempt to parse JSON for richer error context.
        data: Dict[str, Any]
        try:
            data = resp.json()
        except Exception:
            data = {"error": f"Non-JSON response (status {resp.status_code})"}

        if not resp.ok and 'error' not in data:
            data['error'] = f"HTTP {resp.status_code} calling chat endpoint"
        return data
    except requests.Timeout:
        return {"error": "Request to chat API timed out"}
    except Exception as e:  # broad catch to avoid propagating unexpected errors
        return {"error": f"Chat API call failed: {e}"}


def simple_markdown_to_html(md: str) -> str:
    """
    Convert Markdown text to HTML using the Python-Markdown library.

    This function uses the 'markdown' package for robust Markdown parsing and HTML output.
    Any HTML in the source is safely handled by the library to mitigate injection risks.

    Parameters:
        md (str): Markdown input string.

    Returns:
        str: HTML output.
    """
    if not md:
        return ""
    import markdown
    # Use 'extra' and 'sane_lists' extensions for better Markdown support
    return markdown.markdown(md, extensions=['extra', 'sane_lists'])


def analyze_video_content(video_id: str) -> Dict[str, Any]:
    """
    Analyze video for educational content, topics, and key concepts.
    
    This function uses Reka Vision API to analyze educational videos and extract:
    - Main subject/topic
    - Key concepts and strategies
    - Difficulty level
    - Learning objectives
    - Important timestamps
    
    Parameters:
        video_id (str): The UUID of the video to analyze
        
    Returns:
        Dict[str, Any]: Structured analysis data with educational insights
    """
    headers = {}
    if api_key:
        headers['X-Api-Key'] = api_key

    analysis_prompt = """
    Analyze this educational video and provide a comprehensive educational analysis. 
    
    Please identify and return the following information in JSON format:
    
    1. **Subject Area**: What is the main academic subject? (e.g., Mathematics, Science, History, Language Arts, etc.)
    2. **Topic**: What specific topic is being taught? (e.g., "Quadratic Equations", "Photosynthesis", "World War II")
    3. **Difficulty Level**: Rate the content difficulty (beginner, intermediate, advanced)
    4. **Key Concepts**: List 3-5 main concepts or strategies being taught
    5. **Learning Objectives**: What should students learn from this video?
    6. **Key Moments**: Identify important timestamps and what concepts are taught at each moment
    7. **Educational Value**: What makes this video educationally valuable?
    8. **Prerequisites**: What prior knowledge might students need?
    
    Return the response as a valid JSON object with these exact field names:
    {
        "subject": "string",
        "topic": "string", 
        "difficulty": "string",
        "key_concepts": ["concept1", "concept2", "concept3"],
        "learning_objectives": ["objective1", "objective2"],
        "key_moments": [
            {"timestamp": 120, "concept": "factoring", "description": "Shows how to factor quadratic equations"}
        ],
        "educational_value": "string",
        "prerequisites": ["prerequisite1", "prerequisite2"]
    }
    
    Focus on identifying educational content that could be used to create meaningful quiz questions.
    """

    payload = {
        "video_id": video_id,
        "messages": [
            {
                "role": "user",
                "content": analysis_prompt
            }
        ]
    }

    try:
        resp = requests.post(
            REKA_VIDEO_QA_ENDPOINT,
            headers=headers,
            json=payload,
            timeout=60  # Longer timeout for analysis
        )
        
        data: Dict[str, Any]
        try:
            data = resp.json()
        except Exception:
            data = {"error": f"Non-JSON response (status {resp.status_code})"}

        if not resp.ok and 'error' not in data:
            data['error'] = f"HTTP {resp.status_code} calling analysis endpoint"
        return data
    except requests.Timeout:
        return {"error": "Video analysis timed out - video may be too long"}
    except Exception as e:
        return {"error": f"Video analysis failed: {e}"}


def generate_quiz_questions(video_id: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate quiz questions based on video analysis.
    
    This function creates educational quiz questions using the video analysis
    to ensure questions are relevant to the content and appropriate difficulty.
    
    Parameters:
        video_id (str): The UUID of the video
        analysis (Dict[str, Any]): The analysis results from analyze_video_content
        
    Returns:
        Dict[str, Any]: Generated quiz questions and answers
    """
    headers = {}
    if api_key:
        headers['X-Api-Key'] = api_key

    # Build a comprehensive quiz generation prompt
    quiz_prompt = f"""
    Based on this {analysis.get('subject', 'educational')} video about {analysis.get('topic', 'the topic')}, 
    create a comprehensive educational quiz.
    
    Video Analysis Summary:
    - Subject: {analysis.get('subject', 'Not specified')}
    - Topic: {analysis.get('topic', 'Not specified')}
    - Difficulty: {analysis.get('difficulty', 'Not specified')}
    - Key Concepts: {', '.join(analysis.get('key_concepts', []))}
    - Learning Objectives: {', '.join(analysis.get('learning_objectives', []))}
    
    Create a quiz with the following structure:
    
    1. **Multiple Choice Questions (3 questions)**: Test understanding of key concepts
    2. **Short Answer Questions (2 questions)**: Test deeper comprehension  
    3. **Problem-Solving Question (1 question)**: Test application of concepts
    
    For each question, provide:
    - question_text: The question itself
    - question_type: "multiple_choice", "short_answer", or "problem_solving"
    - options: Array of choices (for multiple choice only)
    - correct_answer: The correct answer
    - explanation: Why this answer is correct
    - difficulty_points: 1-5 scale
    - concept_tested: Which key concept this tests
    
    Return the response as a valid JSON object with this exact structure:
    {{
        "quiz_title": "Quiz based on [topic]",
        "quiz_description": "Test your understanding of [topic] concepts",
        "total_questions": 6,
        "estimated_time": "10-15 minutes",
        "questions": [
            {{
                "question_id": 1,
                "question_text": "What is the main concept discussed?",
                "question_type": "multiple_choice",
                "options": ["Option A", "Option B", "Option C", "Option D"],
                "correct_answer": "Option B",
                "explanation": "This is correct because...",
                "difficulty_points": 3,
                "concept_tested": "main_concept"
            }}
        ]
    }}
    
    Make sure questions are:
    - Directly related to the video content
    - Appropriate for the difficulty level
    - Test actual understanding, not just memorization
    - Include clear explanations for learning
    """

    payload = {
        "video_id": video_id,
        "messages": [
            {
                "role": "user",
                "content": quiz_prompt
            }
        ]
    }

    try:
        resp = requests.post(
            REKA_VIDEO_QA_ENDPOINT,
            headers=headers,
            json=payload,
            timeout=90  # Longer timeout for quiz generation
        )
        
        data: Dict[str, Any]
        try:
            data = resp.json()
        except Exception:
            data = {"error": f"Non-JSON response (status {resp.status_code})"}

        if not resp.ok and 'error' not in data:
            data['error'] = f"HTTP {resp.status_code} calling quiz generation endpoint"
        return data
    except requests.Timeout:
        return {"error": "Quiz generation timed out - video may be too complex"}
    except Exception as e:
        return {"error": f"Quiz generation failed: {e}"}


@app.route('/')
def home() -> str:
    """
    Render the home page with welcome text.

    Returns:
        str: Rendered HTML template for the home page.
    """
    return render_template('index.html')


@app.route('/form')
def form_page() -> str:
    """
    Render the form page with dynamic video selection grid.

    Returns:
        str: Rendered HTML template for the form page.
    """
    videos = fetch_videos()

    # Transform videos to a simplified structure for the template.
    template_videos = []
    for v in videos:
        meta = v.get("metadata", {})
        template_videos.append({
            "id": v.get("video_id"),
            "name": meta.get("title") or meta.get("video_name") or "Untitled",
            # fallback
            "thumbnail": meta.get("thumbnail") or "/static/images/image1.jpg",
            "url": v.get("url") or meta.get("url") or "",
        })

    return render_template('form.html', videos=template_videos)


@app.route('/api/upload_video', methods=['POST'])
def upload_video() -> Dict[str, Any]:
    """
    Upload a new video to the Reka Vision API.

    Expects JSON body: { "video_name": "string", "video_url": "string" }

    Returns:
        Dict[str, Any]: JSON response with fields:
            success (bool)
            video_id (str) when successful
            error (str) when not successful
    """
    data = request.get_json() or {}
    video_name = data.get('video_name', '').strip()
    video_url = data.get('video_url', '').strip()

    if not video_name or not video_url:
        return jsonify({"error": "Both video_name and video_url are required"}), 400

    if not api_key:
        return jsonify({"error": "API key not configured"}), 500

    # Call Reka API to upload video
    try:
        response = requests.post(
            f"{base_url.rstrip('/')}/videos/upload",
            headers={
                "X-Api-Key": api_key
            },
            data={
                'video_name': video_name,
                'index': 'true',
                'video_url': video_url
            },
            timeout=30
        )
        
        # Try to parse the response
        try:
            response_data = response.json()
        except Exception:
            response_data = {}

        if response.ok:
            # Invalidate cache to force refresh
            _VIDEO_CACHE["timestamp"] = 0.0
            
            video_id = response_data.get('video_id', 'unknown')
            return jsonify({
                "success": True, 
                "video_id": video_id,
                "message": "Video uploaded successfully"
            })
        else:
            error_msg = response_data.get('error') or response_data.get('message') or f"HTTP {response.status_code}"
            return jsonify({"success": False, "error": f"Upload failed: {error_msg}"}), response.status_code

    except requests.Timeout:
        return jsonify({"success": False, "error": "Request timed out"}), 504
    except Exception as e:
        return jsonify({"success": False, "error": f"Upload failed: {str(e)}"}), 500


@app.route('/api/analyze', methods=['POST'])
def analyze_video() -> Dict[str, Any]:
    """
    Analyze a video for educational content and topics.
    
    Expects JSON body: { "video_id": "uuid" }
    
    Returns:
        Dict[str, Any]: JSON response with analysis data or error
    """
    data = request.get_json() or {}
    video_id = data.get('video_id')

    if not video_id:
        return jsonify({"error": "No video ID provided"}), 400

    # Call the analysis function
    analysis_data = analyze_video_content(video_id)

    # Check if analysis was successful
    if 'error' in analysis_data:
        return jsonify({"success": False, "error": analysis_data['error']}), 500

    # Try to parse the chat response as JSON
    chat_response = analysis_data.get('chat_response')
    if chat_response:
        try:
            import json
            # Parse the JSON response
            parsed_analysis = json.loads(chat_response)
            
            # Validate that we have the expected structure
            if isinstance(parsed_analysis, dict) and 'subject' in parsed_analysis:
                return jsonify({
                    "success": True, 
                    "analysis": parsed_analysis,
                    "message": "Video analysis completed successfully"
                })
            else:
                # If structure is wrong, return the raw response
                return jsonify({
                    "success": True,
                    "analysis": {"raw_response": chat_response},
                    "message": "Analysis completed but structure may be unexpected"
                })
        except (json.JSONDecodeError, ValueError) as e:
            # If JSON parsing fails, return the raw response
            return jsonify({
                "success": True,
                "analysis": {"raw_response": chat_response},
                "message": "Analysis completed but response format may be unexpected"
            })
    
    # No chat response available
    system_msg = analysis_data.get('system_message')
    api_error = analysis_data.get('error')
    fallback = system_msg or api_error or "No analysis data received"
    
    return jsonify({"success": False, "error": fallback}), 500


@app.route('/api/generate_quiz', methods=['POST'])
def generate_quiz() -> Dict[str, Any]:
    """
    Generate a quiz based on video analysis.
    
    Expects JSON body: { "video_id": "uuid", "analysis": {...} }
    
    Returns:
        Dict[str, Any]: JSON response with quiz data or error
    """
    data = request.get_json() or {}
    video_id = data.get('video_id')
    analysis = data.get('analysis', {})

    if not video_id:
        return jsonify({"error": "No video ID provided"}), 400

    if not analysis:
        return jsonify({"error": "No analysis data provided. Please analyze the video first."}), 400

    # Call the quiz generation function
    quiz_data = generate_quiz_questions(video_id, analysis)

    # Check if quiz generation was successful
    if 'error' in quiz_data:
        return jsonify({"success": False, "error": quiz_data['error']}), 500

    # Try to parse the chat response as JSON
    chat_response = quiz_data.get('chat_response')
    if chat_response:
        try:
            import json
            # Parse the JSON response
            parsed_quiz = json.loads(chat_response)
            
            # Validate that we have the expected structure
            if isinstance(parsed_quiz, dict) and 'questions' in parsed_quiz:
                return jsonify({
                    "success": True, 
                    "quiz": parsed_quiz,
                    "message": "Quiz generated successfully"
                })
            else:
                # If structure is wrong, return the raw response
                return jsonify({
                    "success": True,
                    "quiz": {"raw_response": chat_response},
                    "message": "Quiz generated but structure may be unexpected"
                })
        except (json.JSONDecodeError, ValueError) as e:
            # If JSON parsing fails, return the raw response
            return jsonify({
                "success": True,
                "quiz": {"raw_response": chat_response},
                "message": "Quiz generated but response format may be unexpected"
            })
    
    # No chat response available
    system_msg = quiz_data.get('system_message')
    api_error = quiz_data.get('error')
    fallback = system_msg or api_error or "No quiz data received"
    
    return jsonify({"success": False, "error": fallback}), 500





if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
