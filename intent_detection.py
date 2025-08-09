# Greeting and Intent Detection Functions

import re

def detect_intent(user_input):
    """Detect user intent to provide appropriate responses"""
    
    user_input_lower = user_input.lower().strip()
    
    # Simple greeting patterns
    greeting_patterns = [
        r'^(hello|hi|hey|bonjour|salut)$',
        r'^(how are you|comment allez-vous)$',
        r'^(good morning|good afternoon|good evening|bonsoir|bonne journée)$',
        r'^(merci|thank you|thanks)$',
        r'^(au revoir|goodbye|bye)$'
    ]
    
    # Simple chat patterns (not greetings but basic conversation)
    simple_chat_patterns = [
        r'^(ça va|comment ça va|how are you doing)$',
        r'^(quoi de neuf|what\'s up|quoi de nouveau)$',
        r'^(comment vas-tu|how do you do)$'
    ]
    
    # Document analysis patterns
    document_patterns = [
        r'\b(phase|étape|processus|projet|développement)\b',
        r'\b(combien|how many|liste|list)\b',
        r'\b(que|what|pourquoi|why|comment|how)\b.*\b(phase|document|processus)\b',
        r'\b(décris|describe|explique|explain)\b'
    ]
    
    # Check for greetings
    for pattern in greeting_patterns:
        if re.search(pattern, user_input_lower):
            return "greeting"
    
    # Check for simple chat
    for pattern in simple_chat_patterns:
        if re.search(pattern, user_input_lower):
            return "simple_chat"
    
    # Check for document analysis requests
    for pattern in document_patterns:
        if re.search(pattern, user_input_lower):
            return "document_analysis"
    
    # If input is very short and doesn't contain document keywords
    if len(user_input.split()) <= 3 and not any(keyword in user_input_lower for keyword in 
        ['phase', 'processus', 'projet', 'document', 'étape', 'développement']):
        return "simple_chat"
    
    # Default to document analysis for longer queries
    return "document_analysis"

def generate_greeting_response():
    """Generate appropriate greeting responses"""
    
    import random
    
    greeting_responses = [
        "Bonjour ! Je suis votre assistant pour l'analyse de documents PDF. Comment puis-je vous aider aujourd'hui ?",
        "Salut ! Je suis là pour répondre à vos questions sur les documents que vous avez uploadés. Que souhaitez-vous savoir ?",
        "Bonjour ! Je peux vous aider à analyser et comprendre vos documents PDF. Posez-moi une question !",
        "Hello ! Je suis prêt à analyser vos documents. Que voulez-vous savoir ?",
        "Bonjour ! Je vais bien, merci ! Je suis votre assistant d'analyse documentaire. Comment puis-je vous aider ?"
    ]
    
    return random.choice(greeting_responses)

def generate_simple_chat_response(user_input):
    """Generate simple chat responses for basic interactions"""
    
    user_lower = user_input.lower()
    
    if any(word in user_lower for word in ['merci', 'thank', 'thanks']):
        return "De rien ! N'hésitez pas si vous avez d'autres questions sur vos documents."
    
    elif any(word in user_lower for word in ['bye', 'au revoir', 'goodbye']):
        return "Au revoir ! J'espère avoir pu vous aider avec l'analyse de vos documents."
    
    elif any(word in user_lower for word in ['ça va', 'how are you', 'comment allez']):
        return "Je vais très bien, merci ! Je suis prêt à analyser vos documents PDF. Avez-vous des questions ?"
    
    else:
        return "Je suis votre assistant d'analyse de documents PDF. Posez-moi une question sur vos documents uploadés !"

def should_use_enhanced_retrieval(intent, user_input):
    """Determine if enhanced retrieval should be used"""
    
    if intent in ["greeting", "simple_chat"]:
        return False
    
    # Use enhanced retrieval for document analysis
    if intent == "document_analysis":
        return True
    
    # Check for specific keywords that indicate need for document search
    document_keywords = ['phase', 'processus', 'projet', 'étape', 'développement', 'document', 'combien', 'liste']
    return any(keyword in user_input.lower() for keyword in document_keywords)
