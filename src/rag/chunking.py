from typing import List, Dict, Any
import re

def split_text(text: str, max_length: int = 500) -> List[str]:
    """
    Découpe un texte en morceaux de max_length caractères 
    en essayant de ne pas couper les phrases au milieu.
    """
    if not text:
        return []
    
    # Nettoyage basique
    text = text.strip()
    
    # Si le texte est déjà court, on le renvoie tel quel
    if len(text) <= max_length:
        return [text]
    
    chunks = []
    current_chunk = ""
    
    # On découpe par phrases (approximatif : point, point d'exclamation, point d'interrogation + espace)
    # Le regex split garde les délimiteurs grâce aux parenthèses
    sentences = re.split(r'([.!?]+\s+)', text)
    
    # On recombine les phrases et leurs délimiteurs
    # Ex: ["Bonjour", ". ", "Ça va", "? "] -> ["Bonjour. ", "Ça va? "]
    combined_sentences = []
    for i in range(0, len(sentences)-1, 2):
        combined_sentences.append(sentences[i] + sentences[i+1])
    if len(sentences) % 2 != 0:
        combined_sentences.append(sentences[-1])
        
    for sentence in combined_sentences:
        # Si la phrase seule est plus grande que max_length (cas rare), on force la découpe
        if len(sentence) > max_length:
            # Si on a déjà du contenu dans current_chunk, on le sauve d'abord
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            
            # On découpe la longue phrase brutalement ou par mots si on voulait être plus fin
            # Ici on accepte que ce chunk soit la phrase entière (ou on pourrait la recouper)
            # Pour faire simple : on l'ajoute comme un chunk unique même si > max_length
            # Alternative : couper par mots.
            chunks.append(sentence.strip())
            
        elif len(current_chunk) + len(sentence) <= max_length:
            current_chunk += sentence
        else:
            # Le chunk est plein, on l'enregistre
            chunks.append(current_chunk.strip())
            current_chunk = sentence
            
    # Ajouter le dernier morceau s'il reste quelque chose
    if current_chunk:
        chunks.append(current_chunk.strip())
        
    return chunks

def process_ticket(ticket: Dict[str, Any], max_length=500) -> List[Dict[str, Any]]:
    """
    Applique le chunking sur un ticket et attache les métadonnées à chaque chunk.
    """
    full_text = f"{ticket.get('titre', '')}. {ticket.get('description', '')}"
    text_chunks = split_text(full_text, max_length)
    
    chunked_docs = []
    for i, chunk in enumerate(text_chunks):
        doc = {
            "page_content": chunk,
            "metadata": {
                "ticket_id": ticket.get("id"),
                "priorite": ticket.get("priorite"),
                "categorie": ticket.get("categorie"),
                "chunk_index": i,
                "source": "ticket_history"
            }
        }
        chunked_docs.append(doc)
    return chunked_docs

def main():
    print("✂️ Test du découpage (Chunking)...\n")
    
    # Ticket de test (long)
    ticket_test = {
        "id": "TICKET-2024-001",
        "priorite": "Haute",
        "categorie": "Réseau",
        "titre": "Problème connexion VPN instable",
        "description": (
            "Bonjour, depuis ce matin j'ai des déconnexions intempestives du VPN. "
            "Cela m'empêche d'accéder aux serveurs de fichiers. J'ai essayé de redémarrer ma box et mon PC mais rien n'y fait. "
            "Le message d'erreur est 'Connection reset by peer'. " * 5  # On répète pour faire du volume
            + "Merci de votre aide rapidement car je dois finir un rapport pour la direction avant 17h. "
            "Cordialement, Jean Dupont."
        )
    }
    
    print(f"Ticket original (longueur: {len(ticket_test['description']) + len(ticket_test['titre'])} caractères)")
    
    resultats = process_ticket(ticket_test, max_length=200) # Max 200 pour forcer le découpage sur ce test
    
    print(f"Nombre de chunks générés : {len(resultats)}\n")
    
    for i, doc in enumerate(resultats):
        print(f"--- Chunk {i+1} ---")
        print(f"Contenu ({len(doc['page_content'])} chars): {doc['page_content']}")
        print(f"Métadonnées: {doc['metadata']}")
        print()

if __name__ == "__main__":
    main()
