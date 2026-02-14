from langchain_community.embeddings import SentenceTransformerEmbeddings

def main():
    print("⏳ Chargement du modèle d'embedding...")
    # Code pour charger le modèle
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    print("✅ Modèle chargé. Test de vectorisation...")
    # Testez : print(embeddings.embed_query(bonjour))
    vector = embeddings.embed_query("bonjour")
    
    print(f"Vecteur généré (taille: {len(vector)}) :")
    print(vector[:5], "...") # On affiche juste le début pour vérifier que ce sont des chiffres

if __name__ == "__main__":
    main()
