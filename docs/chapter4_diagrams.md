# Chapter 4: System Design Diagrams

## 4.1 Design Method
This system uses **Object-Oriented Design** with UML diagrams following the Agile/Hybrid methodology.

---

## 4.2 UML Diagrams

### 4.2.1 Use Case Diagram

```mermaid
graph TB
    subgraph "Sabah Informal Housing Detection System"
        UC1["Analyze Selected Area"]
        UC2["View Detection Results"]
        UC3["Navigate Map"]
        UC4["Switch Base Layers"]
        UC5["View Confidence Score"]
    end
    
    User((User))
    System((System))
    Supabase[(Supabase DB)]
    Sentinel[(Sentinel API)]
    
    User --> UC1
    User --> UC2
    User --> UC3
    User --> UC4
    User --> UC5
    
    UC1 --> System
    System --> Supabase
    System --> Sentinel
```

### 4.2.2 Activity Diagram - Detect Informal Housing

```mermaid
flowchart TD
    A([Start]) --> B[User clicks on map]
    B --> C[System calculates 512x512 bounding box]
    C --> D{Local tiles available?}
    D -->|Yes| E[Load local tiles]
    D -->|No| F[Fetch live tiles from Sentinel]
    E --> G[Preprocess image]
    F --> G
    G --> H[Run Original GRAM model]
    G --> I[Run Extended GRAM model]
    H --> J[Combine predictions with ensemble weights]
    I --> J
    J --> K[Apply threshold and morphological cleanup]
    K --> L[Generate detection overlay]
    L --> M[Calculate confidence score]
    M --> N[Display results on map]
    N --> O([End])
```

### 4.2.3 Activity Diagram - View Results

```mermaid
flowchart TD
    A([Start]) --> B[Detection overlay displayed]
    B --> C[User views red overlay on map]
    C --> D[System shows confidence percentage]
    D --> E[System shows detection coverage]
    E --> F{User wants to analyze another area?}
    F -->|Yes| G[Click new location]
    G --> B
    F -->|No| H([End])
```

### 4.2.4 Sequence Diagram - Detection Request

```mermaid
sequenceDiagram
    actor User
    participant Dashboard
    participant Server
    participant Supabase
    participant Sentinel
    participant GRAM_Model
    
    User->>Dashboard: Click on map location
    Dashboard->>Dashboard: Calculate 512x512 bbox
    Dashboard->>Server: POST /snapshot (bbox)
    
    Server->>Supabase: Query tile_index for bbox
    Supabase-->>Server: Return matching tiles
    
    alt Local tiles exist
        Server->>Server: Load local PNG
    else No local tiles
        Server->>Sentinel: Fetch live tiles (z14)
        Sentinel-->>Server: Return satellite imagery
    end
    
    Server->>Server: Preprocess (resize, normalize)
    Server->>GRAM_Model: Run inference (Original + Extended)
    GRAM_Model-->>Server: Probability maps
    Server->>Server: Ensemble + Threshold + Morphology
    Server->>Server: Generate overlay PNG
    Server-->>Dashboard: Return overlay + confidence
    Dashboard->>User: Display detection overlay
```

### 4.2.5 Use Case Specifications

#### Use Case 1: Analyze Selected Area

| Field | Description |
|-------|-------------|
| **Use Case ID** | UC-01 |
| **Use Case Name** | Analyze Selected Area |
| **Actor** | User |
| **Preconditions** | Dashboard is loaded, map is visible |
| **Main Flow** | 1. User selects marker tool<br>2. User clicks on map location<br>3. System calculates 512x512 analysis area<br>4. System fetches/loads satellite imagery<br>5. System runs GRAM ensemble model<br>6. System displays detection overlay |
| **Postconditions** | Detection overlay visible, confidence score displayed |
| **Alternative Flow** | If no local tiles, system fetches live tiles from Sentinel |

#### Use Case 2: View Detection Results

| Field | Description |
|-------|-------------|
| **Use Case ID** | UC-02 |
| **Use Case Name** | View Detection Results |
| **Actor** | User |
| **Preconditions** | Analysis has been performed |
| **Main Flow** | 1. User views red overlay on map<br>2. User reads confidence score in sidebar<br>3. User reads detection coverage percentage |
| **Postconditions** | User understands detection results |

#### Use Case 3: Navigate Map

| Field | Description |
|-------|-------------|
| **Use Case ID** | UC-03 |
| **Use Case Name** | Navigate Map |
| **Actor** | User |
| **Preconditions** | Dashboard is loaded |
| **Main Flow** | 1. User can pan map by dragging<br>2. User can zoom with scroll wheel<br>3. User can switch between Sentinel and Google layers |
| **Postconditions** | Map shows desired location |

---

## 4.3 Database Design (ERD)

Since the system uses **Supabase (PostgreSQL with PostGIS)**, the database schema is represented using an Entity Relationship Diagram (ERD):

```mermaid
erDiagram
    TILE_INDEX {
        int id PK
        string png_path
        string mask_path
        string tif_path
        geometry geometry
    }
    
    DISTRICTS {
        int id PK
        string name
        string type
        geometry geometry
    }
    
    DETECTION_RESULT {
        int id PK
        int tile_id FK
        float confidence
        float coverage
        geometry bbox
        timestamp created_at
    }
    
    DISTRICTS ||--o{ TILE_INDEX : "contains"
    TILE_INDEX ||--o{ DETECTION_RESULT : "generates"
```

### Table Descriptions

| Table | Description |
|-------|-------------|
| **TILE_INDEX** | Stores satellite tile metadata and spatial boundaries |
| **DISTRICTS** | Stores Sabah district boundaries and names |
| **DETECTION_RESULT** | Stores inference results with confidence scores |

---

## 4.4 AI Inference Design (Class Diagram)

The machine learning inference pipeline is represented using a Class Diagram:

```mermaid
classDiagram
    class GRAMModel {
        -string checkpoint_path
        -float weight
        -device device
        +load_checkpoint() void
        +predict(image) ProbabilityMap
        +to_device(device) void
    }
    
    class EnsemblePredictor {
        -GRAMModel original_model
        -GRAMModel extended_model
        -float weight_original
        -float weight_extended
        -float threshold
        +predict(image) DetectionResult
        +combine_predictions(prob1, prob2) ProbabilityMap
        +apply_threshold(prob_map) BinaryMask
        +postprocess(mask) CleanedMask
    }
    
    class DetectionResult {
        +float confidence
        +float coverage
        +ndarray overlay
        +geometry bbox
        +to_png() bytes
        +to_geojson() dict
    }
    
    class ImagePreprocessor {
        +int target_size
        +transform transform
        +preprocess(image) Tensor
        +normalize(tensor) Tensor
    }
    
    EnsemblePredictor "1" *-- "2" GRAMModel : contains
    EnsemblePredictor --> DetectionResult : produces
    EnsemblePredictor --> ImagePreprocessor : uses
```

### Class Descriptions

| Class | Responsibility |
|-------|----------------|
| **GRAMModel** | Loads and runs GRAM neural network inference |
| **EnsemblePredictor** | Combines two models with weighted averaging |
| **DetectionResult** | Holds detection output with confidence metrics |
| **ImagePreprocessor** | Resizes and normalizes input images |

---

## 4.5 Interface Design

### 4.5.1 Main Dashboard Interface

![Main Dashboard View](C:/Users/junyi/.gemini/antigravity/brain/3595c388-33c2-4caa-92a0-143c363cfcfe/dashboard_main_view_1766930517099.png)

**Description**: The main interface shows an interactive Leaflet map with Sentinel-2 satellite imagery. The sidebar contains:
- Application title and description
- How to use instructions
- Analysis result panel
- Map legend

### 4.5.2 Detection Result View

![Detection Result Overlay](C:/Users/junyi/.gemini/antigravity/brain/3595c388-33c2-4caa-92a0-143c363cfcfe/detection_result_overlay_1766930599125.png)

**Description**: After clicking on the map, the system displays:
- Blue bounding box (512×512 analysis area)
- Red overlay (possible informal housing detections)
- Green/yellow areas (high confidence detections)
- Confidence score and coverage percentage in sidebar

### User Interaction

| User Type | Interface | Actions |
|-----------|-----------|---------|
| General User | Dashboard | Click map to analyze, view results, navigate |
| Researcher | Dashboard | Compare different areas, evaluate model performance |
| Urban Planner | Dashboard | Identify informal settlements for planning purposes |
