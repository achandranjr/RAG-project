from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import logging
import time
import threading
import os
import pandas as pd
from datetime import datetime
import sqlite3
import uuid

# Import your existing RAG system
from rag_system import ArxivRAGSystem

# Import the new feedback system
from feedback_system import FeedbackSystem, integrate_feedback_with_rag

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global variables
rag_system = None
feedback_system = None
startup_status = {
    'rag_initialized': False,
    'data_loaded': False,
    'index_built': False,
    'feedback_initialized': False,
    'ready': False,
    'errors': [],
    'start_time': time.time(),
    'total_papers': 0,
    'has_data': False
}

def auto_initialize_system():
    """Automatically initialize the entire RAG system with feedback on startup"""
    global rag_system, startup_status, feedback_system
    
    startup_start = time.time()
    logger.info("🚀 Starting automatic system initialization...")
    
    try:
        # Step 1: Initialize RAG system
        logger.info("📦 Step 1: Initializing RAG system...")
        rag_system = ArxivRAGSystem(llm_option="openai")
        startup_status['rag_initialized'] = True
        logger.info("✅ RAG system initialized")
        
        # Step 1.5: Initialize feedback system
        logger.info("🧠 Step 1.5: Initializing feedback system...")
        feedback_system = FeedbackSystem(rag_system=rag_system)
        startup_status['feedback_initialized'] = True
        logger.info("✅ Feedback system initialized")
        
        # Step 2: Check for existing data and index
        logger.info("🔍 Step 2: Checking for existing data...")
        
        try:
            # Try to load existing index and data
            rag_system.load_existing_index()
            startup_status['total_papers'] = len(rag_system.papers_df)
            startup_status['data_loaded'] = True
            startup_status['has_data'] = True
            startup_status['index_built'] = True
            logger.info(f"✅ Loaded existing index with {startup_status['total_papers']} papers")
        except FileNotFoundError:
            logger.info("📥 No existing data found, need to load fresh data...")
            logger.info("🚨 Please load ArXiv data using the /api/load_data endpoint")
            logger.info("💡 Example: POST to /api/load_data with JSON body:")
            logger.info('   {"data_file": "path/to/arxiv-metadata.json", "sample_size": 1000}')
            
            # Mark as initialized but without data
            startup_status['data_loaded'] = False
            startup_status['has_data'] = False
            startup_status['index_built'] = False
        
        # Step 3: Integrate feedback system with RAG (if we have an index)
        if startup_status['index_built']:
            logger.info("🔗 Step 3: Integrating feedback system with RAG...")
            integrate_feedback_with_rag(rag_system, feedback_system)
            logger.info("✅ Feedback integration complete")
        
        # Mark system as ready (even without data, for manual loading)
        startup_status['ready'] = True
        startup_time = time.time() - startup_start
        
        if startup_status['has_data']:
            logger.info(f"🎉 System initialization completed successfully in {startup_time:.2f} seconds!")
            logger.info(f"📊 Total papers: {startup_status['total_papers']}")
            logger.info(f"🧠 Feedback learning: Active")
            logger.info(f"🚀 System ready for searches!")
        else:
            logger.info(f"⚡ System partially initialized in {startup_time:.2f} seconds")
            logger.info("📋 Ready for manual data loading via API")
            logger.info("🔧 Use /api/load_data endpoint to load ArXiv data")
        
    except Exception as e:
        error_msg = f"❌ System initialization failed: {str(e)}"
        logger.error(error_msg)
        startup_status['errors'].append(error_msg)
        startup_status['ready'] = False

# Start initialization in background thread
initialization_thread = threading.Thread(target=auto_initialize_system, daemon=True)
initialization_thread.start()

# Routes
@app.route('/')
def index():
    """Serve the main interface"""
    return send_from_directory('.', 'index.html')

@app.route('/api/status')
def get_status():
    """Get system initialization status"""
    current_time = time.time()
    elapsed_time = current_time - startup_status['start_time']
    
    status = {
        **startup_status,
        'elapsed_time': elapsed_time,
        'timestamp': datetime.now().isoformat()
    }
    
    return jsonify(status)

@app.route('/api/search', methods=['POST'])
def search_papers():
    """Search for papers using RAG system with feedback learning"""
    try:
        if not startup_status['ready']:
            return jsonify({
                'error': 'System not ready yet', 
                'status': startup_status,
                'message': 'Please wait for system initialization to complete'
            }), 503
        
        data = request.json
        query = data.get('query', '').strip()
        top_k = int(data.get('top_k', 5))
        session_id = data.get('session_id', str(uuid.uuid4()))
        use_feedback = data.get('use_feedback', True)
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        logger.info(f"🔍 Searching for: '{query}' (top_k={top_k}, feedback={use_feedback})")
        
        # Measure search time
        start_time = time.time()
        
        # Enhanced search with feedback (if available)
        if feedback_system and hasattr(rag_system, 'feedback_system'):
            paper_results = rag_system.search_papers(query, top_k=top_k, use_feedback=use_feedback)
            # Note: Enhanced query is already applied in the integrated search
            enhanced_query = feedback_system.get_enhanced_search_query(query) if use_feedback else query
        else:
            # Fallback to regular search
            paper_results = rag_system.search_papers(query, top_k=top_k)
            enhanced_query = query
        
        # Get AI recommendations
        recommendations = rag_system.get_recommendations(query, top_k=top_k)
        
        search_time = time.time() - start_time
        
        response = {
            'query': query,
            'enhanced_query': enhanced_query,
            'recommendations': recommendations,
            'papers': paper_results,
            'search_time': search_time,
            'num_results': len(paper_results),
            'session_id': session_id,
            'timestamp': datetime.now().isoformat(),
            'feedback_enabled': feedback_system is not None
        }
        
        logger.info(f"✅ Search completed in {search_time:.2f}s, found {len(paper_results)} papers")
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"❌ Search error: {e}")
        return jsonify({'error': f'Search failed: {str(e)}'}), 500

@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
    """Submit user feedback with enhanced processing"""
    try:
        if not feedback_system:
            return jsonify({'error': 'Feedback system not available'}), 503
        
        data = request.json
        
        # Validate required fields
        required_fields = ['query', 'feedback_type']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Create enhanced feedback record
        feedback_record = {
            'query': data['query'],
            'feedback_type': data['feedback_type'],  # 'paper', 'overall', 'search_quality'
            'timestamp': datetime.now().isoformat(),
            'session_id': data.get('session_id'),
            'rating': data.get('rating'),
            'paper_id': data.get('paper_id'),
            'paper_rank': data.get('paper_rank'),
            'search_results_count': data.get('search_results_count'),
            'response_time': data.get('response_time'),
            'user_comments': data.get('user_comments'),
            'context_data': data.get('context_data', {})
        }
        
        # Record feedback
        success = feedback_system.record_feedback(feedback_record)
        
        if success:
            logger.info(f"📝 Feedback recorded: {data['feedback_type']} for query '{data['query']}'")
            
            # Get suggestions if this was negative feedback
            suggestions = []
            if data.get('rating') in ['negative', 'not_helpful', 'irrelevant']:
                suggestions = feedback_system._generate_query_suggestions(data['query'])
            
            return jsonify({
                'status': 'success',
                'message': 'Feedback received and processed',
                'suggestions': suggestions,
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({'error': 'Failed to process feedback'}), 500
        
    except Exception as e:
        logger.error(f"❌ Feedback error: {e}")
        return jsonify({'error': f'Failed to submit feedback: {str(e)}'}), 500

@app.route('/api/analytics', methods=['GET'])
def get_analytics():
    """Get comprehensive feedback analytics"""
    try:
        if not feedback_system:
            return jsonify({'error': 'Feedback system not available'}), 503
        
        analytics = feedback_system.get_feedback_analytics()
        
        # Add system-specific analytics
        analytics['system_info'] = {
            'total_papers': len(rag_system.papers_df) if rag_system and hasattr(rag_system, 'papers_df') else 0,
            'feedback_system_active': True,
            'query_expansions_learned': len(feedback_system.query_expansions),
            'database_path': feedback_system.db_path
        }
        
        return jsonify(analytics)
        
    except Exception as e:
        logger.error(f"❌ Analytics error: {e}")
        return jsonify({'error': f'Failed to get analytics: {str(e)}'}), 500

@app.route('/api/suggest', methods=['POST'])
def get_query_suggestions():
    """Get query suggestions based on feedback history"""
    try:
        if not feedback_system:
            return jsonify({'error': 'Feedback system not available'}), 503
        
        data = request.json
        query = data.get('query', '').strip()
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        # Get enhanced query
        enhanced_query = feedback_system.get_enhanced_search_query(query)
        
        # Get suggestions for improvement
        suggestions = feedback_system._generate_query_suggestions(query)
        
        # Check query performance history
        conn = sqlite3.connect(feedback_system.db_path)
        cursor = conn.cursor()
        
        normalized_query = feedback_system._normalize_query(query)
        cursor.execute(
            'SELECT performance_score, total_searches FROM query_performance WHERE normalized_query = ?',
            (normalized_query,)
        )
        performance_data = cursor.fetchone()
        conn.close()
        
        response = {
            'original_query': query,
            'enhanced_query': enhanced_query,
            'suggestions': suggestions,
            'performance_history': {
                'score': performance_data[0] if performance_data else None,
                'search_count': performance_data[1] if performance_data else 0
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"❌ Query suggestion error: {e}")
        return jsonify({'error': f'Failed to get suggestions: {str(e)}'}), 500

@app.route('/api/feedback/batch', methods=['POST'])
def submit_batch_feedback():
    """Submit feedback for multiple papers at once"""
    try:
        if not feedback_system:
            return jsonify({'error': 'Feedback system not available'}), 503
        
        data = request.json
        feedback_items = data.get('feedback_items', [])
        session_id = data.get('session_id')
        query = data.get('query')
        
        if not feedback_items:
            return jsonify({'error': 'No feedback items provided'}), 400
        
        results = []
        
        for item in feedback_items:
            feedback_record = {
                'query': query,
                'feedback_type': 'paper',
                'timestamp': datetime.now().isoformat(),
                'session_id': session_id,
                'rating': item.get('rating'),
                'paper_id': item.get('paper_id'),
                'paper_rank': item.get('paper_rank'),
                'search_results_count': len(feedback_items),
                'user_comments': item.get('comments'),
                'context_data': item.get('context_data', {})
            }
            
            success = feedback_system.record_feedback(feedback_record)
            results.append({
                'paper_id': item.get('paper_id'),
                'success': success
            })
        
        successful_count = sum(1 for r in results if r['success'])
        
        logger.info(f"📝 Batch feedback: {successful_count}/{len(feedback_items)} items processed for query '{query}'")
        
        return jsonify({
            'status': 'success',
            'message': f'Processed {successful_count}/{len(feedback_items)} feedback items',
            'results': results,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Batch feedback error: {e}")
        return jsonify({'error': f'Failed to submit batch feedback: {str(e)}'}), 500

@app.route('/api/insights', methods=['GET'])
def get_learning_insights():
    """Get insights about what the system has learned"""
    try:
        if not feedback_system:
            return jsonify({'error': 'Feedback system not available'}), 503
        
        conn = sqlite3.connect(feedback_system.db_path)
        
        insights = {}
        
        # Top performing queries
        df_queries = pd.read_sql_query('''
            SELECT query, performance_score, total_searches, positive_feedback, negative_feedback
            FROM query_performance 
            ORDER BY performance_score DESC 
            LIMIT 10
        ''', conn)
        
        insights['top_queries'] = df_queries.to_dict('records') if not df_queries.empty else []
        
        # Most popular papers
        df_papers = pd.read_sql_query('''
            SELECT paper_id, title, view_count, relevance_score, positive_feedback
            FROM paper_popularity 
            ORDER BY relevance_score DESC 
            LIMIT 10
        ''', conn)
        
        insights['popular_papers'] = df_papers.to_dict('records') if not df_papers.empty else []
        
        # Recent learning trends
        df_recent = pd.read_sql_query('''
            SELECT DATE(timestamp) as date, 
                   COUNT(*) as feedback_count,
                   AVG(CASE WHEN rating IN ('positive', 'helpful', 'relevant') THEN 1.0 ELSE 0.0 END) as satisfaction_rate
            FROM feedback 
            WHERE timestamp >= datetime('now', '-30 days')
            GROUP BY DATE(timestamp)
            ORDER BY date DESC
        ''', conn)
        
        insights['recent_trends'] = df_recent.to_dict('records') if not df_recent.empty else []
        
        # Query expansions learned
        insights['learned_expansions'] = {
            'total_queries': len(feedback_system.query_expansions),
            'sample_expansions': list(feedback_system.query_expansions.keys())[:5]
        }
        
        # Failed queries that need attention
        df_failed = pd.read_sql_query('''
            SELECT query, COUNT(*) as failure_count, MAX(timestamp) as last_failure
            FROM failed_queries 
            WHERE timestamp >= datetime('now', '-7 days')
            GROUP BY query
            ORDER BY failure_count DESC
            LIMIT 5
        ''', conn)
        
        insights['queries_needing_attention'] = df_failed.to_dict('records') if not df_failed.empty else []
        
        conn.close()
        
        return jsonify(insights)
        
    except Exception as e:
        logger.error(f"❌ Insights error: {e}")
        return jsonify({'error': f'Failed to get insights: {str(e)}'}), 500

@app.route('/api/feedback/export', methods=['GET'])
def export_feedback():
    """Export feedback data for analysis"""
    try:
        if not feedback_system:
            return jsonify({'error': 'Feedback system not available'}), 503
        
        format_type = request.args.get('format', 'json')
        days = int(request.args.get('days', 30))
        
        conn = sqlite3.connect(feedback_system.db_path)
        
        # Get feedback data from last N days
        df = pd.read_sql_query('''
            SELECT * FROM feedback 
            WHERE timestamp >= datetime('now', '-{} days')
            ORDER BY timestamp DESC
        '''.format(days), conn)
        
        conn.close()
        
        if format_type == 'csv':
            from flask import Response
            from io import StringIO
            output = StringIO()
            df.to_csv(output, index=False)
            
            return Response(
                output.getvalue(),
                mimetype='text/csv',
                headers={'Content-Disposition': f'attachment; filename=feedback_export_{days}days.csv'}
            )
        else:
            # JSON format
            return jsonify({
                'data': df.to_dict('records'),
                'metadata': {
                    'export_date': datetime.now().isoformat(),
                    'days_included': days,
                    'total_records': len(df)
                }
            })
        
    except Exception as e:
        logger.error(f"❌ Export error: {e}")
        return jsonify({'error': f'Failed to export feedback: {str(e)}'}), 500

@app.route('/api/feedback/reset', methods=['POST'])
def reset_feedback_learning():
    """Reset feedback learning data (development only)"""
    try:
        if not feedback_system:
            return jsonify({'error': 'Feedback system not available'}), 503
        
        # Only allow in development mode
        if not app.debug:
            return jsonify({'error': 'Reset only allowed in debug mode'}), 403
        
        # Clear query expansions
        feedback_system.query_expansions = {}
        feedback_system.save_query_expansions()
        
        # Clear database tables (optional - uncomment if needed)
        # conn = sqlite3.connect(feedback_system.db_path)
        # cursor = conn.cursor()
        # cursor.execute('DELETE FROM feedback')
        # cursor.execute('DELETE FROM query_performance')
        # cursor.execute('DELETE FROM paper_popularity')
        # cursor.execute('DELETE FROM failed_queries')
        # conn.commit()
        # conn.close()
        
        logger.info("🔄 Feedback learning data reset")
        
        return jsonify({
            'status': 'success',
            'message': 'Feedback learning data reset',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Reset error: {e}")
        return jsonify({'error': f'Failed to reset learning data: {str(e)}'}), 500

# Legacy endpoints for compatibility
@app.route('/api/load_data', methods=['POST'])
def load_data():
    """Load ArXiv data from JSON file"""
    try:
        if not rag_system:
            return jsonify({'error': 'RAG system not initialized'}), 503
        
        data = request.json or {}
        data_file = data.get('data_file', 'arxiv-metadata-oai-snapshot.json')
        sample_size = data.get('sample_size', 1000)
        
        logger.info(f"🔄 Manual data loading requested: {data_file}, sample_size: {sample_size}")
        
        # Check if file exists
        if not os.path.exists(data_file):
            return jsonify({
                'error': f'Data file not found: {data_file}',
                'suggestion': 'Please provide the correct path to your ArXiv JSON data file'
            }), 400
        
        # Load new data
        start_time = time.time()
        papers_df = rag_system.load_arxiv_data(data_file, sample_size=sample_size)
        load_time = time.time() - start_time
        
        # Build index
        logger.info("🔨 Building search index...")
        index_build_start = time.time()
        rag_system.build_index()
        index_time = time.time() - index_build_start
        
        # Update status
        startup_status['total_papers'] = len(papers_df)
        startup_status['data_loaded'] = True
        startup_status['has_data'] = True
        startup_status['index_built'] = True
        
        # Integrate feedback system
        if feedback_system:
            logger.info("🔗 Integrating feedback system...")
            integrate_feedback_with_rag(rag_system, feedback_system)
        
        total_time = load_time + index_time
        logger.info(f"✅ Data loaded and indexed: {len(papers_df)} papers in {total_time:.2f}s")
        
        return jsonify({
            'status': 'success',
            'message': f'Loaded and indexed {len(papers_df)} papers',
            'total_papers': len(papers_df),
            'load_time': load_time,
            'index_time': index_time,
            'total_time': total_time,
            'data_file': data_file,
            'sample_size': sample_size
        })
        
    except Exception as e:
        logger.error(f"❌ Data loading error: {e}")
        return jsonify({'error': f'Failed to load data: {str(e)}'}), 500

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'system_ready': startup_status['ready'],
        'components': {
            'rag_system': rag_system is not None,
            'feedback_system': feedback_system is not None,
            'has_data': startup_status['has_data']
        }
    })

@app.route('/debug')
def debug():
    """Simple debug endpoint to test Flask is working"""
    return """
    <h1>🚀 Flask is working!</h1>
    <p>If you see this, Flask is running correctly.</p>
    <h2>System Status:</h2>
    <ul>
        <li>RAG System: {}</li>
        <li>Feedback System: {}</li>
        <li>Has Data: {}</li>
        <li>Ready: {}</li>
    </ul>
    <h2>Quick Links:</h2>
    <ul>
        <li><a href="/test">Test Page</a></li>
        <li><a href="/api/status">API Status</a></li>
        <li><a href="/api/health">Health Check</a></li>
    </ul>
    """.format(
        "✅" if rag_system else "❌",
        "✅" if feedback_system else "❌", 
        "✅" if startup_status['has_data'] else "❌",
        "✅" if startup_status['ready'] else "❌"
    )

@app.route('/test')
def test_page():
    """Serve the test page"""
    return send_from_directory('.', 'test.html')
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    logger.info("🚀 Starting ArXiv RAG Flask application with AI feedback learning...")
    logger.info("🌐 Frontend will be available at: http://localhost:5000")
    logger.info("📡 API endpoints available at: http://localhost:5000/api/")
    logger.info("🧠 AI learning system: Active")
    
    # Run in debug mode for development
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True,
        use_reloader=False  # Disable reloader to prevent double initialization
    )