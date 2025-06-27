import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import sqlite3
import logging
from collections import defaultdict, Counter
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import re

logger = logging.getLogger(__name__)

class FeedbackSystem:
    """
    Comprehensive feedback system for the ArXiv RAG
    Handles collection, storage, analysis, and learning from user feedback
    """
    
    def __init__(self, db_path="feedback.db", rag_system=None):
        self.db_path = db_path
        self.rag_system = rag_system
        self.init_database()
        
        # Learning parameters
        self.feedback_weights = {
            'positive': 1.0,
            'negative': -0.5,
            'relevant': 1.0,
            'irrelevant': -0.8,
            'helpful': 0.8,
            'not_helpful': -0.6
        }
        
        # Query expansion storage
        self.query_expansions = {}
        self.load_query_expansions()
        
    def init_database(self):
        """Initialize SQLite database for feedback storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Feedback table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                session_id TEXT,
                query TEXT NOT NULL,
                feedback_type TEXT NOT NULL,
                rating TEXT,
                paper_id TEXT,
                paper_rank INTEGER,
                search_results_count INTEGER,
                response_time REAL,
                user_comments TEXT,
                context_data TEXT
            )
        ''')
        
        # Query performance tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS query_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                normalized_query TEXT,
                avg_rating REAL,
                total_searches INTEGER,
                positive_feedback INTEGER,
                negative_feedback INTEGER,
                last_updated TEXT,
                performance_score REAL
            )
        ''')
        
        # Paper popularity tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS paper_popularity (
                paper_id TEXT PRIMARY KEY,
                title TEXT,
                view_count INTEGER DEFAULT 0,
                positive_feedback INTEGER DEFAULT 0,
                negative_feedback INTEGER DEFAULT 0,
                relevance_score REAL DEFAULT 0.0,
                last_accessed TEXT
            )
        ''')
        
        # Failed queries (no good results)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS failed_queries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                timestamp TEXT,
                reason TEXT,
                suggestions TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Feedback database initialized")
    
    def record_feedback(self, feedback_data: Dict[str, Any]) -> bool:
        """
        Record user feedback in the database
        
        Args:
            feedback_data: Dictionary containing feedback information
        
        Returns:
            bool: Success status
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Insert feedback
            cursor.execute('''
                INSERT INTO feedback (
                    timestamp, session_id, query, feedback_type, rating,
                    paper_id, paper_rank, search_results_count, response_time,
                    user_comments, context_data
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                feedback_data.get('timestamp', datetime.now().isoformat()),
                feedback_data.get('session_id'),
                feedback_data.get('query'),
                feedback_data.get('feedback_type'),
                feedback_data.get('rating'),
                feedback_data.get('paper_id'),
                feedback_data.get('paper_rank'),
                feedback_data.get('search_results_count'),
                feedback_data.get('response_time'),
                feedback_data.get('user_comments'),
                json.dumps(feedback_data.get('context_data', {}))
            ))
            
            conn.commit()
            
            # Update derived tables
            self._update_query_performance(feedback_data)
            self._update_paper_popularity(feedback_data)
            
            conn.close()
            
            # Trigger learning process
            self._process_feedback_for_learning(feedback_data)
            
            logger.info(f"Recorded feedback: {feedback_data.get('feedback_type')} for query '{feedback_data.get('query')}'")
            return True
            
        except Exception as e:
            logger.error(f"Failed to record feedback: {e}")
            return False
    
    def _update_query_performance(self, feedback_data: Dict[str, Any]):
        """Update query performance metrics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = feedback_data.get('query')
        normalized_query = self._normalize_query(query)
        rating = feedback_data.get('rating')
        
        # Get existing performance data
        cursor.execute(
            'SELECT * FROM query_performance WHERE normalized_query = ?',
            (normalized_query,)
        )
        existing = cursor.fetchone()
        
        if existing:
            # Update existing record
            total_searches = existing[4] + 1
            positive_feedback = existing[5] + (1 if rating in ['positive', 'helpful', 'relevant'] else 0)
            negative_feedback = existing[6] + (1 if rating in ['negative', 'not_helpful', 'irrelevant'] else 0)
            
            # Calculate performance score
            performance_score = self._calculate_performance_score(positive_feedback, negative_feedback, total_searches)
            
            cursor.execute('''
                UPDATE query_performance 
                SET total_searches = ?, positive_feedback = ?, negative_feedback = ?,
                    last_updated = ?, performance_score = ?
                WHERE normalized_query = ?
            ''', (total_searches, positive_feedback, negative_feedback,
                  datetime.now().isoformat(), performance_score, normalized_query))
        else:
            # Create new record
            positive_feedback = 1 if rating in ['positive', 'helpful', 'relevant'] else 0
            negative_feedback = 1 if rating in ['negative', 'not_helpful', 'irrelevant'] else 0
            performance_score = self._calculate_performance_score(positive_feedback, negative_feedback, 1)
            
            cursor.execute('''
                INSERT INTO query_performance 
                (query, normalized_query, total_searches, positive_feedback, 
                 negative_feedback, last_updated, performance_score)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (query, normalized_query, 1, positive_feedback, negative_feedback,
                  datetime.now().isoformat(), performance_score))
        
        conn.commit()
        conn.close()
    
    def _update_paper_popularity(self, feedback_data: Dict[str, Any]):
        """Update paper popularity metrics"""
        paper_id = feedback_data.get('paper_id')
        if not paper_id:
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        rating = feedback_data.get('rating')
        
        # Get existing paper data
        cursor.execute('SELECT * FROM paper_popularity WHERE paper_id = ?', (paper_id,))
        existing = cursor.fetchone()
        
        if existing:
            # Update existing record
            view_count = existing[2] + 1
            positive_feedback = existing[3] + (1 if rating in ['positive', 'helpful', 'relevant'] else 0)
            negative_feedback = existing[4] + (1 if rating in ['negative', 'not_helpful', 'irrelevant'] else 0)
            
            relevance_score = self._calculate_relevance_score(positive_feedback, negative_feedback, view_count)
            
            cursor.execute('''
                UPDATE paper_popularity 
                SET view_count = ?, positive_feedback = ?, negative_feedback = ?,
                    relevance_score = ?, last_accessed = ?
                WHERE paper_id = ?
            ''', (view_count, positive_feedback, negative_feedback, relevance_score,
                  datetime.now().isoformat(), paper_id))
        else:
            # Create new record
            positive_feedback = 1 if rating in ['positive', 'helpful', 'relevant'] else 0
            negative_feedback = 1 if rating in ['negative', 'not_helpful', 'irrelevant'] else 0
            relevance_score = self._calculate_relevance_score(positive_feedback, negative_feedback, 1)
            
            # Try to get paper title from RAG system
            title = "Unknown"
            if self.rag_system and hasattr(self.rag_system, 'papers_df'):
                try:
                    paper_row = self.rag_system.papers_df[
                        self.rag_system.papers_df['arxiv_id'] == paper_id
                    ]
                    if not paper_row.empty:
                        title = paper_row.iloc[0]['title']
                except:
                    pass
            
            cursor.execute('''
                INSERT INTO paper_popularity 
                (paper_id, title, view_count, positive_feedback, negative_feedback,
                 relevance_score, last_accessed)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (paper_id, title, 1, positive_feedback, negative_feedback,
                  relevance_score, datetime.now().isoformat()))
        
        conn.commit()
        conn.close()
    
    def _process_feedback_for_learning(self, feedback_data: Dict[str, Any]):
        """Process feedback to improve future searches"""
        query = feedback_data.get('query')
        rating = feedback_data.get('rating')
        paper_id = feedback_data.get('paper_id')
        
        # Learn query expansions from positive feedback
        if rating in ['positive', 'helpful', 'relevant'] and paper_id:
            self._learn_query_expansion(query, paper_id)
        
        # Identify failed queries
        if rating in ['negative', 'not_helpful', 'irrelevant']:
            self._handle_failed_query(query, feedback_data)
    
    def _learn_query_expansion(self, query: str, relevant_paper_id: str):
        """Learn to expand queries based on relevant papers"""
        if not self.rag_system or not hasattr(self.rag_system, 'papers_df'):
            return
        
        try:
            # Get the relevant paper
            paper_row = self.rag_system.papers_df[
                self.rag_system.papers_df['arxiv_id'] == relevant_paper_id
            ]
            
            if paper_row.empty:
                return
            
            paper = paper_row.iloc[0]
            
            # Extract keywords from title and abstract
            text = f"{paper['title']} {paper['abstract']}"
            keywords = self._extract_keywords(text)
            
            # Store query expansion
            normalized_query = self._normalize_query(query)
            if normalized_query not in self.query_expansions:
                self.query_expansions[normalized_query] = {
                    'keywords': Counter(),
                    'categories': Counter(),
                    'authors': Counter()
                }
            
            # Update expansions
            self.query_expansions[normalized_query]['keywords'].update(keywords)
            self.query_expansions[normalized_query]['categories'][paper['primary_category']] += 1
            
            # Extract author keywords
            if paper['authors'] and paper['authors'] != 'Unknown':
                authors = [author.strip() for author in paper['authors'].split(',')]
                self.query_expansions[normalized_query]['authors'].update(authors[:2])  # Top 2 authors
            
            # Save expansions
            self.save_query_expansions()
            
        except Exception as e:
            logger.error(f"Failed to learn query expansion: {e}")
    
    def _handle_failed_query(self, query: str, feedback_data: Dict[str, Any]):
        """Handle queries that received negative feedback"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if this is a recurring failed query
        cursor.execute(
            'SELECT COUNT(*) FROM failed_queries WHERE query = ? AND timestamp > ?',
            (query, (datetime.now() - timedelta(days=7)).isoformat())
        )
        recent_failures = cursor.fetchone()[0]
        
        if recent_failures >= 2:  # Multiple failures in a week
            # Generate suggestions for query improvement
            suggestions = self._generate_query_suggestions(query)
            
            cursor.execute('''
                INSERT INTO failed_queries (query, timestamp, reason, suggestions)
                VALUES (?, ?, ?, ?)
            ''', (query, datetime.now().isoformat(), 
                  "Multiple negative feedback instances", 
                  json.dumps(suggestions)))
        
        conn.commit()
        conn.close()
    
    def get_enhanced_search_query(self, original_query: str) -> str:
        """
        Enhance a search query based on learned feedback
        
        Args:
            original_query: The user's original query
            
        Returns:
            Enhanced query string
        """
        normalized_query = self._normalize_query(original_query)
        
        # Check if we have learned expansions for this query
        if normalized_query in self.query_expansions:
            expansions = self.query_expansions[normalized_query]
            
            # Add top keywords
            top_keywords = [word for word, count in expansions['keywords'].most_common(3)]
            
            # Create enhanced query
            enhanced_parts = [original_query]
            if top_keywords:
                enhanced_parts.extend(top_keywords)
            
            enhanced_query = ' '.join(enhanced_parts)
            logger.info(f"Enhanced query: '{original_query}' -> '{enhanced_query}'")
            return enhanced_query
        
        return original_query
    
    def get_boosted_results(self, search_results: List[Dict], query: str) -> List[Dict]:
        """
        Boost search results based on feedback history
        
        Args:
            search_results: Original search results
            query: The search query
            
        Returns:
            Reranked results with feedback-based boosting
        """
        if not search_results:
            return search_results
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        boosted_results = []
        
        for result in search_results:
            paper_id = result.get('arxiv_id')
            
            # Get popularity data
            cursor.execute(
                'SELECT relevance_score FROM paper_popularity WHERE paper_id = ?',
                (paper_id,)
            )
            popularity_data = cursor.fetchone()
            
            relevance_boost = 0.0
            if popularity_data:
                relevance_boost = popularity_data[0] * 0.1  # 10% boost based on relevance score
            
            # Apply boost to score
            original_score = result.get('score', 1.0)
            boosted_score = original_score + relevance_boost
            
            result['original_score'] = original_score
            result['relevance_boost'] = relevance_boost
            result['score'] = boosted_score
            
            boosted_results.append(result)
        
        # Re-sort by boosted score
        boosted_results.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        conn.close()
        return boosted_results
    
    def get_feedback_analytics(self) -> Dict[str, Any]:
        """Get comprehensive feedback analytics"""
        conn = sqlite3.connect(self.db_path)
        
        analytics = {}
        
        # Overall feedback stats
        df_feedback = pd.read_sql_query('SELECT * FROM feedback', conn)
        
        if not df_feedback.empty:
            analytics['total_feedback'] = len(df_feedback)
            analytics['feedback_by_type'] = df_feedback['feedback_type'].value_counts().to_dict()
            analytics['rating_distribution'] = df_feedback['rating'].value_counts().to_dict()
            
            # Query performance insights
            df_queries = pd.read_sql_query('SELECT * FROM query_performance', conn)
            if not df_queries.empty:
                analytics['top_performing_queries'] = df_queries.nlargest(5, 'performance_score')[
                    ['query', 'performance_score', 'total_searches']
                ].to_dict('records')
                
                analytics['struggling_queries'] = df_queries.nsmallest(5, 'performance_score')[
                    ['query', 'performance_score', 'total_searches']
                ].to_dict('records')
            
            # Paper popularity insights
            df_papers = pd.read_sql_query('SELECT * FROM paper_popularity', conn)
            if not df_papers.empty:
                analytics['most_popular_papers'] = df_papers.nlargest(5, 'relevance_score')[
                    ['paper_id', 'title', 'relevance_score', 'view_count']
                ].to_dict('records')
            
            # Recent trends
            df_recent = df_feedback[df_feedback['timestamp'] >= (datetime.now() - timedelta(days=7)).isoformat()]
            analytics['recent_feedback_count'] = len(df_recent)
            analytics['recent_satisfaction'] = len(df_recent[df_recent['rating'].isin(['positive', 'helpful', 'relevant'])]) / len(df_recent) if len(df_recent) > 0 else 0
        
        conn.close()
        return analytics
    
    def _normalize_query(self, query: str) -> str:
        """Normalize query for comparison and storage"""
        return re.sub(r'[^\w\s]', '', query.lower().strip())
    
    def _extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """Extract keywords from text using TF-IDF"""
        try:
            # Simple keyword extraction using TF-IDF
            vectorizer = TfidfVectorizer(
                max_features=max_keywords,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            tfidf_matrix = vectorizer.fit_transform([text])
            feature_names = vectorizer.get_feature_names_out()
            scores = tfidf_matrix.toarray()[0]
            
            # Get top keywords
            keyword_scores = list(zip(feature_names, scores))
            keyword_scores.sort(key=lambda x: x[1], reverse=True)
            
            return [keyword for keyword, score in keyword_scores if score > 0]
        except:
            # Fallback to simple word extraction
            words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
            return list(set(words))[:max_keywords]
    
    def _calculate_performance_score(self, positive: int, negative: int, total: int) -> float:
        """Calculate performance score for queries"""
        if total == 0:
            return 0.0
        
        # Weighted score considering positive/negative feedback and recency
        positive_rate = positive / total
        negative_rate = negative / total
        
        # Score ranges from -1 to 1
        score = positive_rate - (negative_rate * 0.5)
        return max(-1.0, min(1.0, score))
    
    def _calculate_relevance_score(self, positive: int, negative: int, views: int) -> float:
        """Calculate relevance score for papers"""
        if views == 0:
            return 0.0
        
        # Consider both feedback ratio and popularity
        feedback_score = (positive - negative) / views
        popularity_bonus = min(0.1, views / 100)  # Small bonus for popular papers
        
        return max(0.0, feedback_score + popularity_bonus)
    
    def _generate_query_suggestions(self, failed_query: str) -> List[str]:
        """Generate suggestions for improving failed queries"""
        suggestions = []
        
        # Basic suggestions
        suggestions.append("Try using more specific technical terms")
        suggestions.append("Include author names if you know them")
        suggestions.append("Add year or time period")
        
        # Look for similar successful queries
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT DISTINCT query FROM query_performance 
            WHERE performance_score > 0.5 
            ORDER BY performance_score DESC LIMIT 5
        ''')
        
        successful_queries = [row[0] for row in cursor.fetchall()]
        
        # Simple similarity matching
        failed_words = set(failed_query.lower().split())
        for successful_query in successful_queries:
            successful_words = set(successful_query.lower().split())
            if len(failed_words.intersection(successful_words)) >= 1:
                suggestions.append(f"Try: '{successful_query}'")
        
        conn.close()
        return suggestions[:5]  # Limit to 5 suggestions
    
    def save_query_expansions(self):
        """Save query expansions to disk"""
        try:
            with open('query_expansions.pkl', 'wb') as f:
                pickle.dump(self.query_expansions, f)
        except Exception as e:
            logger.error(f"Failed to save query expansions: {e}")
    
    def load_query_expansions(self):
        """Load query expansions from disk"""
        try:
            if os.path.exists('query_expansions.pkl'):
                with open('query_expansions.pkl', 'rb') as f:
                    self.query_expansions = pickle.load(f)
                logger.info(f"Loaded {len(self.query_expansions)} query expansions")
        except Exception as e:
            logger.error(f"Failed to load query expansions: {e}")
            self.query_expansions = {}

# Integration functions for the RAG system
def integrate_feedback_with_rag(rag_system, feedback_system):
    """
    Integrate feedback system with RAG system
    """
    
    # Monkey patch the search method to include feedback learning
    original_search = rag_system.search_papers
    
    def enhanced_search(query, top_k=5, use_feedback=True):
        if use_feedback:
            # Enhance query based on feedback
            enhanced_query = feedback_system.get_enhanced_search_query(query)
            
            # Perform search with enhanced query
            results = original_search(enhanced_query, top_k)
            
            # Boost results based on feedback history
            results = feedback_system.get_boosted_results(results, query)
            
            return results
        else:
            return original_search(query, top_k)
    
    # Replace the method
    rag_system.search_papers = enhanced_search
    rag_system.feedback_system = feedback_system
    
    logger.info("Integrated feedback system with RAG")

# Example usage
if __name__ == "__main__":
    # Initialize feedback system
    feedback_system = FeedbackSystem()
    
    # Example feedback recording
    feedback_data = {
        'query': 'transformer neural networks',
        'feedback_type': 'paper',
        'rating': 'positive',
        'paper_id': '1706.03762',
        'paper_rank': 1,
        'search_results_count': 5,
        'response_time': 0.8,
        'session_id': 'user123_session1'
    }
    
    feedback_system.record_feedback(feedback_data)
    
    # Get analytics
    analytics = feedback_system.get_feedback_analytics()
    print("Feedback Analytics:", analytics)