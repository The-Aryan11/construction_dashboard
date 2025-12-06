# data_processor.py
"""
Advanced NLP Data Processor for Construction Inspection Reports
Handles keyword extraction, topic modeling, and sentiment analysis
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.cluster import KMeans
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import json
from datetime import datetime, timedelta
import random

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('punkt_tab')

class ConstructionInspectionAnalyzer:
    """
    Comprehensive analyzer for construction inspection reports
    """
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Add construction-specific stop words
        self.custom_stop_words = {
            'also', 'would', 'could', 'shall', 'must', 'may', 
            'project', 'site', 'inspection', 'report', 'date',
            'inspector', 'noted', 'observed', 'checked', 'verified'
        }
        self.stop_words.update(self.custom_stop_words)
        
        # Construction domain keywords
        self.domain_keywords = {
            'safety': ['safety', 'hazard', 'protection', 'ppe', 'warning', 'risk', 'danger', 'secure'],
            'structural': ['concrete', 'steel', 'foundation', 'beam', 'column', 'slab', 'reinforcement', 'rebar'],
            'electrical': ['electrical', 'wiring', 'conduit', 'panel', 'circuit', 'grounding', 'voltage'],
            'plumbing': ['plumbing', 'pipe', 'drainage', 'water', 'sewage', 'valve', 'fitting'],
            'hvac': ['hvac', 'ventilation', 'duct', 'cooling', 'heating', 'air', 'conditioning'],
            'compliance': ['code', 'compliance', 'regulation', 'standard', 'permit', 'approved', 'violation'],
            'quality': ['quality', 'defect', 'crack', 'damage', 'repair', 'rework', 'finish'],
            'progress': ['progress', 'schedule', 'delay', 'milestone', 'completion', 'phase', 'timeline']
        }
        
        self.vectorizer = None
        self.lda_model = None
        self.nmf_model = None
        
    def generate_sample_data(self, n_reports=500):
        """
        Generate realistic sample inspection reports
        """
        
        report_templates = [
            # Safety-related
            "Safety inspection revealed {issue}. Workers {compliance} wearing proper PPE. {additional}",
            "On-site safety audit completed. {finding}. Recommended {action} for hazard mitigation.",
            "Emergency exit pathways {status}. Fire extinguishers {fire_status}. Safety signage {sign_status}.",
            
            # Structural
            "Foundation inspection shows {foundation_status}. Concrete curing {curing_status}. {rebar_note}",
            "Structural steel installation {steel_status}. Welding quality {weld_quality}. Beam connections {connection_status}.",
            "Column alignment verified. {alignment_note}. Load-bearing capacity {capacity_status}.",
            
            # Electrical
            "Electrical rough-in inspection: {electrical_finding}. Conduit installation {conduit_status}.",
            "Panel installation {panel_status}. Grounding system {ground_status}. Wire gauge {wire_status}.",
            
            # Plumbing
            "Plumbing inspection: {plumbing_finding}. Pressure test results {pressure_status}.",
            "Drainage system {drain_status}. Vent stack installation {vent_status}. {leak_note}",
            
            # Quality
            "Quality control inspection identified {quality_issue}. Finish work {finish_status}.",
            "Material quality {material_status}. Workmanship {workmanship_status}. {recommendation}",
            
            # Progress
            "Construction progress at {progress}% completion. {schedule_status}. Next milestone: {milestone}.",
            "Phase {phase} inspection completed. Timeline {timeline_status}. {delay_note}"
        ]
        
        issues = ["minor safety violations", "adequate safety measures", "critical safety concerns", 
                  "excellent safety compliance", "moderate safety issues requiring attention"]
        compliance = ["were", "were not", "partially", "consistently", "intermittently"]
        additional = ["Immediate corrective action required.", "Continue monitoring.", 
                     "Excellent performance noted.", "Training recommended.", "Follow-up inspection scheduled."]
        
        findings = ["No major deficiencies found", "Several items require attention", 
                   "Critical issues identified", "All items meet standards", "Minor corrections needed"]
        actions = ["immediate repair", "additional training", "equipment replacement", 
                  "procedure review", "enhanced monitoring"]
        
        statuses = ["clear and accessible", "partially blocked", "properly maintained", 
                   "requires attention", "meets all requirements"]
        
        foundation_statuses = ["proper compaction achieved", "minor settling observed", 
                              "excellent bearing capacity", "requires additional testing", "meets specifications"]
        curing_statuses = ["proceeding normally", "requires extended time", "completed successfully", 
                         "temperature monitored", "humidity controlled"]
        
        phases = ["1", "2", "3", "4", "5"]
        milestones = ["foundation completion", "structural framing", "MEP rough-in", 
                     "exterior envelope", "interior finishes", "final inspection"]
        
        project_types = ["Commercial Building", "Residential Complex", "Industrial Facility", 
                        "Infrastructure", "Renovation", "Healthcare Facility", "Educational Building"]
        
        locations = ["Downtown District", "North Industrial Zone", "Suburban Development Area",
                    "Waterfront Project Site", "Highway Corridor", "Mixed-Use Zone", "Tech Park"]
        
        inspectors = ["John Smith", "Maria Garcia", "Robert Johnson", "Sarah Chen", 
                     "Michael Brown", "Emily Davis", "David Wilson", "Lisa Anderson"]
        
        reports = []
        base_date = datetime.now() - timedelta(days=365)
        
        for i in range(n_reports):
            template = random.choice(report_templates)
            
            # Fill in template variables
            report_text = template.format(
                issue=random.choice(issues),
                compliance=random.choice(compliance),
                additional=random.choice(additional),
                finding=random.choice(findings),
                action=random.choice(actions),
                status=random.choice(statuses),
                fire_status=random.choice(["inspected and valid", "requires replacement", "properly placed"]),
                sign_status=random.choice(["adequate", "needs updating", "meets standards"]),
                foundation_status=random.choice(foundation_statuses),
                curing_status=random.choice(curing_statuses),
                rebar_note=random.choice(["Rebar placement correct.", "Spacing verified.", "Cover depth adequate."]),
                steel_status=random.choice(["on schedule", "ahead of schedule", "slightly delayed"]),
                weld_quality=random.choice(["excellent", "acceptable", "requires re-inspection"]),
                connection_status=random.choice(["properly torqued", "needs verification", "meets specs"]),
                alignment_note=random.choice(["Plumb within tolerance.", "Minor adjustment needed.", "Perfect alignment."]),
                capacity_status=random.choice(["confirmed", "pending verification", "exceeds requirements"]),
                electrical_finding=random.choice(["All circuits properly labeled", "Minor code violations found", "Excellent installation"]),
                conduit_status=random.choice(["properly secured", "needs additional supports", "meets code"]),
                panel_status=random.choice(["complete", "in progress", "requires correction"]),
                ground_status=random.choice(["properly installed", "needs testing", "verified"]),
                wire_status=random.choice(["appropriate", "needs verification", "correct per plans"]),
                plumbing_finding=random.choice(["System properly installed", "Minor leaks detected", "Excellent workmanship"]),
                pressure_status=random.choice(["passed", "requires retest", "exceeds requirements"]),
                drain_status=random.choice(["properly sloped", "needs adjustment", "correctly installed"]),
                vent_status=random.choice(["complete", "in progress", "verified"]),
                leak_note=random.choice(["No leaks detected.", "Minor repairs needed.", "All joints sealed."]),
                quality_issue=random.choice(["minor surface defects", "excellent finish quality", "areas requiring touch-up"]),
                finish_status=random.choice(["acceptable", "excellent", "needs improvement"]),
                material_status=random.choice(["approved", "pending verification", "meets specifications"]),
                workmanship_status=random.choice(["excellent", "good", "acceptable", "needs improvement"]),
                recommendation=random.choice(["Continue current practices.", "Additional QC recommended.", "Ready for next phase."]),
                progress=random.randint(10, 100),
                schedule_status=random.choice(["On schedule", "Ahead of schedule", "Behind schedule", "Critical delay"]),
                milestone=random.choice(milestones),
                phase=random.choice(phases),
                timeline_status=random.choice(["on track", "extended", "compressed", "revised"]),
                delay_note=random.choice(["Weather delays noted.", "Material delivery issues.", "On track for completion.", "Labor shortage impact."])
            )
            
            # Add severity score
            severity_keywords = {
                'critical': 5, 'danger': 5, 'violation': 4, 'hazard': 4,
                'requires': 3, 'attention': 3, 'delayed': 3,
                'minor': 2, 'good': 1, 'excellent': 0, 'meets': 0
            }
            
            severity = 2  # Default moderate
            for word, score in severity_keywords.items():
                if word in report_text.lower():
                    severity = score
                    break
            
            report = {
                'id': f"INS-{2024}-{str(i+1).zfill(5)}",
                'date': (base_date + timedelta(days=random.randint(0, 365))).strftime('%Y-%m-%d'),
                'project_type': random.choice(project_types),
                'location': random.choice(locations),
                'inspector': random.choice(inspectors),
                'report_text': report_text,
                'severity': severity,
                'status': random.choice(['Approved', 'Pending', 'Requires Follow-up', 'Critical']),
                'category': random.choice(['Safety', 'Structural', 'Electrical', 'Plumbing', 'HVAC', 'Quality', 'Progress']),
                'compliance_score': random.randint(60, 100),
                'issues_count': random.randint(0, 10),
                'resolution_time': random.randint(1, 30)
            }
            reports.append(report)
        
        return pd.DataFrame(reports)
    
    def preprocess_text(self, text):
        """
        Clean and preprocess text for NLP analysis
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [
            self.lemmatizer.lemmatize(token) 
            for token in tokens 
            if token not in self.stop_words and len(token) > 2
        ]
        
        return ' '.join(tokens)
    
    def extract_keywords(self, df, text_column='report_text', top_n=30):
        """
        Extract keywords using TF-IDF
        """
        # Preprocess texts
        processed_texts = df[text_column].apply(self.preprocess_text)
        
        # TF-IDF Vectorization
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            max_df=0.85,
            min_df=2
        )
        
        tfidf_matrix = self.vectorizer.fit_transform(processed_texts)
        
        # Get feature names and their scores
        feature_names = self.vectorizer.get_feature_names_out()
        tfidf_scores = np.array(tfidf_matrix.mean(axis=0)).flatten()
        
        # Create keyword dataframe
        keywords_df = pd.DataFrame({
            'keyword': feature_names,
            'tfidf_score': tfidf_scores
        }).sort_values('tfidf_score', ascending=False).head(top_n)
        
        return keywords_df, tfidf_matrix
    
    def perform_topic_modeling(self, df, text_column='report_text', n_topics=8):
        """
        Perform LDA topic modeling
        """
        # Preprocess texts
        processed_texts = df[text_column].apply(self.preprocess_text)
        
        # Count Vectorization for LDA
        count_vectorizer = CountVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            max_df=0.85,
            min_df=2
        )
        
        count_matrix = count_vectorizer.fit_transform(processed_texts)
        feature_names = count_vectorizer.get_feature_names_out()
        
        # LDA Model
        self.lda_model = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            max_iter=20,
            learning_method='online'
        )
        
        lda_output = self.lda_model.fit_transform(count_matrix)
        
        # Extract topics
        topics = []
        topic_names = [
            "Safety & Compliance", "Structural Work", "Electrical Systems",
            "Plumbing & Water", "Quality Control", "Progress Tracking",
            "Material & Resources", "Environmental & HVAC"
        ]
        
        for idx, topic in enumerate(self.lda_model.components_):
            top_words_idx = topic.argsort()[-10:][::-1]
            top_words = [feature_names[i] for i in top_words_idx]
            top_weights = [topic[i] for i in top_words_idx]
            
            topics.append({
                'topic_id': idx,
                'topic_name': topic_names[idx] if idx < len(topic_names) else f"Topic {idx+1}",
                'words': top_words,
                'weights': top_weights
            })
        
        # Assign dominant topic to each document
        df['dominant_topic'] = lda_output.argmax(axis=1)
        df['topic_name'] = df['dominant_topic'].apply(
            lambda x: topic_names[x] if x < len(topic_names) else f"Topic {x+1}"
        )
        df['topic_confidence'] = lda_output.max(axis=1)
        
        return topics, lda_output, df
    
    def calculate_statistics(self, df):
        """
        Calculate comprehensive statistics
        """
        stats = {
            'total_reports': len(df),
            'avg_compliance': df['compliance_score'].mean(),
            'total_issues': df['issues_count'].sum(),
            'avg_resolution_time': df['resolution_time'].mean(),
            'critical_count': len(df[df['status'] == 'Critical']),
            'pending_count': len(df[df['status'] == 'Pending']),
            'approved_count': len(df[df['status'] == 'Approved']),
            'category_distribution': df['category'].value_counts().to_dict(),
            'project_type_distribution': df['project_type'].value_counts().to_dict(),
            'location_distribution': df['location'].value_counts().to_dict(),
            'monthly_trend': df.groupby(pd.to_datetime(df['date']).dt.to_period('M')).size().to_dict()
        }
        return stats

# Initialize analyzer
analyzer = ConstructionInspectionAnalyzer()