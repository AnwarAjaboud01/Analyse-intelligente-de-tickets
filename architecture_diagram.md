# Architecture Diagram: Intelligent Ticket Analysis

This diagram represents the internal architecture of the provided solution, focusing on the local machine learning pipelines (Classification & Regression) and excluding external API dependencies.

```mermaid
graph TD
    %% Styling
    classDef client fill:#f9f,stroke:#333,stroke-width:2px,color:black;
    classDef server fill:#bbf,stroke:#333,stroke-width:2px,color:black;
    classDef ml fill:#bfb,stroke:#333,stroke-width:2px,color:black;
    classDef data fill:#ff9,stroke:#333,stroke-width:2px,color:black;

    %% Client Layer
    subgraph "Présentation (Frontend)"
        UI[💻 Interface Web<br/>(HTML5 / CSS3 / JS)]:::client
    end

    %% Application Layer
    subgraph "Serveur d'Application (Backend)"
        API[🚀 Flask API<br/>(app.py)]:::server
        Router[📡 Routeur<br/>(/api/tickets)]:::server
    end

    %% ML Layer
    subgraph "Moteur d'Intelligence Artificielle (src/ml)"
        Preprocessing[⚙️ Prétraitement<br/>(TF-IDF Vectorization)]:::ml
        
        subgraph "Pipelines de Prédiction"
            C1[🏷️ Classification: Urgence<br/>(Logistic Regression)]:::ml
            C2[📂 Classification: Catégorie<br/>(SGD Classifier)]:::ml
            C3[📝 Classification: Type<br/>(SGD Classifier)]:::ml
            R1[⏱️ Régression: Temps<br/>(Ridge Regression)]:::ml
        end
        
        Aggregator[🧩 Agrégateur de Résultats]:::ml
    end

    %% Data Layer
    subgraph "Persistance de Données"
        DB[(💾 Base de Données CSV/JSON)]:::data
        Models[(📦 Modèles Sérialisés .pkl)]:::data
    end

    %% Flow/Connections
    UI -- "1. Envoi Ticket (JSON)" --> Router
    Router -- "2. Traitement Requête" --> API
    API -- "3. Appel Pipeline" --> Preprocessing
    
    Preprocessing -- "Features" --> C1
    Preprocessing -- "Features" --> C2
    Preprocessing -- "Features" --> C3
    Preprocessing -- "Features" --> R1
    
    Models -. "Chargement" .-> C1
    Models -. "Chargement" .-> C2
    Models -. "Chargement" .-> C3
    Models -. "Chargement" .-> R1
    
    C1 & C2 & C3 & R1 --> Aggregator
    Aggregator -- "4. Prédictions Structurées" --> API
    API -- "5. Sauvegarde" --> DB
    API -- "6. Réponse (JSON)" --> UI
```
