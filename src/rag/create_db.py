import os
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

# On importe notre fonction de chunking précédemment créée
from chunking import split_text

def main():
    print("⏳ Chargement du modèle d'embedding...")
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # Simulation de données : quelques documents "propres"
    raw_documents = [
        "Pour réinitialiser le mot de passe wifi, il faut aller sur 192.168.1.1 et entrer admin/admin.",
        "Si l'imprimante ne répond pas, vérifiez qu'elle est bien connectée au réseau et qu'il y a du papier.",
        "La procédure de demande de congés se fait via le portail RH rubrique 'Mes absences'.",
        "En cas de panne réseau générale, contacter le 0800 123 456.",
        "Le VPN nécessite l'installation du client Cisco AnyConnect et un certificat valide."
    ]

    print(f"📝 Préparation des données: {len(raw_documents)} documents sources.")
    
    all_chunks = []
    # On découpe (même si ici ils sont courts, c'est pour l'exemple)
    for doc in raw_documents:
        chunks = split_text(doc)
        all_chunks.extend(chunks)

    print(f"✂️  Nombre total de chunks à ingérer: {len(all_chunks)}")

    persist_directory = "./chroma_db"
    
    print(f"💾 Création/Mise à jour de la base Chroma dans '{persist_directory}'...")
    
    # Création de la base Vectorielle
    # Note: Chroma gère la persistance automatiquement dans les versions récentes
    db = Chroma.from_texts(
        texts=all_chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )

    print("✅ Base de données vectorielle créée avec succès!")

    # Test de recherche
    query = "problème wifi"
    print(f"\n🔍 Test de recherche pour: '{query}'")
    
    results = db.similarity_search(query, k=2)
    
    for i, res in enumerate(results):
        print(f"\n--- Résultat {i+1} ---")
        print(f"Contenu: {res.page_content}")
        # Note: metadata est vide ici car on a utilisé from_texts simple, 
        # mais on pourrait passer des metadatas avec from_texts(texts, metadatas=...)

if __name__ == "__main__":
    main()
