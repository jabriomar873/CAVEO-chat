# Enhanced Retrieval Functions for Better Accuracy and Precision

from config import QUERY_ENHANCEMENT, VALIDATION_CONFIG, PROMPT_TEMPLATES

def enhance_query(query):
    """Enhance user query to improve retrieval accuracy"""
    
    enhanced_query = query.lower()
    
    # Add related terms for better matching
    if any(keyword in enhanced_query for keyword in QUERY_ENHANCEMENT["phase_keywords"]):
        enhanced_query += " phase étape processus développement phase00 phase01 phase02 phase03"
    
    if any(keyword in enhanced_query for keyword in QUERY_ENHANCEMENT["process_keywords"]):
        enhanced_query += " phase procédure workflow méthodologie"
    
    if any(keyword in enhanced_query for keyword in QUERY_ENHANCEMENT["completeness_keywords"]):
        enhanced_query += " nombre total liste complète toutes phases étapes phase00 phase01 phase02 phase03 étude chiffrage conception développement réalisation production"
    
    # Special enhancement for "how many phases" type questions
    if any(phrase in enhanced_query for phrase in ["how many", "combien", "number of"]):
        enhanced_query += " phase00 phase01 phase02 phase03 étude chiffrage conception réalisation production toutes phases complète liste"
    
    return enhanced_query

def rerank_retrieved_docs(docs, query):
    """Re-rank retrieved documents based on query relevance"""
    
    query_lower = query.lower()
    
    # Keywords that indicate completeness requests
    completeness_indicators = ["combien", "toutes", "liste", "how many", "all", "complete"]
    
    scored_docs = []
    for doc in docs:
        score = 0
        content = doc.page_content.lower()
        
        # Boost score for documents containing phase information
        if "phase" in content:
            score += 10
        
        # Special boost for Phase 00 which is often missed
        if any(indicator in content for indicator in ["phase 00", "phase 0", "étude et chiffrage", "study and estimation"]):
            score += 20  # High priority for Phase 00
        
        # Boost score for specific phases
        phase_mentions = 0
        for i in range(0, 6):  # Check phases 0-5
            if f"phase {i:02d}" in content or f"phase {i}" in content:
                phase_mentions += 1
                score += 8
        
        # Boost score for numbered lists or sequences
        if any(str(i) in content for i in range(0, 10)):
            score += 5
        
        # Boost score for completeness if query asks for it
        if any(indicator in query_lower for indicator in completeness_indicators):
            if "phase" in content and phase_mentions >= 2:
                score += 15
            # Extra boost if document contains Phase 00
            if any(p00 in content for p00 in ["phase 00", "phase 0", "étude et chiffrage"]):
                score += 25
        
        # Boost score for document structure indicators
        if any(marker in content for marker in [":", "•", "-", "1.", "2.", "3."]):
            score += 3
        
        # Boost for comprehensive content
        if len(content.split()) > 100:  # Longer documents likely more comprehensive
            score += 2
        
        scored_docs.append((score, doc))
    
    # Sort by score (descending) and return documents
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    return [doc for score, doc in scored_docs]

def create_enhanced_context(retrieved_docs, query):
    """Create enhanced context from retrieved documents"""
    
    query_lower = query.lower()
    
    # Check if query is asking for phases/steps
    is_phase_query = any(keyword in query_lower for keyword in 
                        ["phase", "étape", "combien", "how many", "liste", "all"])
    
    if is_phase_query:
        # For phase queries, prioritize documents with phase information
        phase_docs = []
        other_docs = []
        
        for doc in retrieved_docs:
            if "phase" in doc.page_content.lower():
                phase_docs.append(doc)
            else:
                other_docs.append(doc)
        
        # Combine with phase docs first
        ordered_docs = phase_docs + other_docs
    else:
        ordered_docs = retrieved_docs
    
    # Create enhanced context string
    context_parts = []
    for i, doc in enumerate(ordered_docs[:8]):  # Limit to top 8 docs
        context_parts.append(f"[DOCUMENT {i+1}]\n{doc.page_content}\n")
    
    return "\n".join(context_parts)

def validate_response_completeness(response, query):
    """Validate if response is complete and accurate for phase-related queries"""
    
    query_lower = query.lower()
    response_lower = response.lower()
    
    # Check for phase completeness
    if any(keyword in query_lower for keyword in ["combien", "how many", "toutes", "all", "phases"]):
        if "phase" in query_lower or "étape" in query_lower:
            # Look for all possible phase formats
            phases_found = set()
            
            # Check for Phase 00, Phase 0, Phase 01, etc.
            phase_patterns = [
                r'phase\s*0*0',   # Phase 00, Phase 0
                r'phase\s*0*1',   # Phase 01, Phase 1  
                r'phase\s*0*2',   # Phase 02, Phase 2
                r'phase\s*0*3',   # Phase 03, Phase 3
                r'phase\s*0*4',   # Phase 04, Phase 4
                r'phase\s*0*5'    # Phase 05, Phase 5
            ]
            
            import re
            for pattern in phase_patterns:
                matches = re.findall(pattern, response_lower)
                for match in matches:
                    # Extract the phase number
                    phase_num = re.search(r'\d+', match)
                    if phase_num:
                        phases_found.add(int(phase_num.group()))
            
            # Special check for Phase 00 keywords
            if not any(p == 0 for p in phases_found):
                phase_00_indicators = [
                    "étude et chiffrage", "study and estimation", 
                    "feasibility", "faisabilité", "estimation"
                ]
                
                if any(indicator in response_lower for indicator in phase_00_indicators):
                    phases_found.add(0)
            
            # Check for hallucination warning - phases mentioned but not in typical project documents
            suspicious_phases = [p for p in phases_found if p > 3]
            if suspicious_phases and any(phrase in response_lower for phrase in 
                ["non mentionnée", "non trouvée", "pas dans les documents", "documents fournis"]):
                return False, f"⚠️ ERREUR: La réponse mentionne des phases inexistantes (Phase {', '.join(map(str, suspicious_phases))}). Basez-vous uniquement sur ce qui existe dans les documents."
            
            # Check if response seems incomplete (missing important phases)
            if len(phases_found) < 3:
                return False, f"La réponse semble incomplète. Phases détectées: {sorted(phases_found)}. Vérifiez si toutes les phases sont mentionnées."
            
            # Specific check for Phase 00
            if 0 not in phases_found:
                return False, "⚠️ ATTENTION: La Phase 00 (Étude et chiffrage) semble manquer dans la réponse."
            
            # Check for counting accuracy
            total_claimed = None
            count_patterns = [
                r'il y a (\d+) phases',
                r'(\d+) phases principales',
                r'(\d+) phases dans',
                r'total[^0-9]*(\d+)[^0-9]*phases'
            ]
            
            for pattern in count_patterns:
                match = re.search(pattern, response_lower)
                if match:
                    total_claimed = int(match.group(1))
                    break
            
            if total_claimed and total_claimed != len(phases_found):
                return False, f"⚠️ INCOHÉRENCE: Le nombre annoncé ({total_claimed}) ne correspond pas aux phases listées ({len(phases_found)}). Vérifiez le décompte."
    
    return True, ""
