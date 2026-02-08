graph TD
        %% Styles
        classDef model fill:#e1f5fe,stroke:#01579b,stroke-width:2px;
        classDef data fill:#fff3e0,stroke:#e65100,stroke-width:2px;
        classDef process fill:#f3e5f5,stroke:#4a148c,stroke-width:2px,stroke-dasharray: 5 5;
        classDef infra fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px;
        classDef decision fill:#fff9c4,stroke:#fbc02d,stroke-width:2px,shape:diamond;

        subgraph Data_Generation ["Phase 1: Synthetic Data Generation (Knowledge Distillation)"]
            direction TB
            MIPD[("MIPD Dataset<br/>(Text + Labels)")]:::data
            Qwen[("Teacher Model<br/>Qwen-2.5-7B-Instruct")]:::model
            Script_Gen["Hard-Constraint Generator<br/>(Python Script)"]:::process
            
            MIPD --> Script_Gen
            Qwen --> Script_Gen
            Script_Gen -->|Generates Reasoning| CoT_Data[("Synthetic Dataset<br/>(Text + Reasoning + Labels)")]:::data
        end

        subgraph Training ["Phase 2: Supervised Fine-Tuning (SFT)"]
            direction TB
            Bielik_Base[("Student Model<br/>Bielik-4.5B-Instruct")]:::model
            Unsloth["Unsloth Trainer<br/>(QLoRA, 4-bit, RoPE Scaling, 16k ctx)"]:::process
            Adapter[("LoRA Adapter")]:::model
            
            CoT_Data --> Unsloth
            Bielik_Base --> Unsloth
            Unsloth --> Adapter
        end

        subgraph Inference ["Phase 3: Local Inference Architecture"]
            direction TB
            Bielik_GGUF[("Base Model<br/>Q8_0.gguf")]:::model
            LoRA_GGUF[("Adapter<br/>F32.gguf")]:::model
            Ollama["Ollama Server<br/>(Runtime Adapter Loading)"]:::infra
            React["React Frontend<br/>(Streaming UI)"]:::infra
            
            Bielik_GGUF -.-> Ollama
            LoRA_GGUF -.-> Ollama
            Ollama <-->|"/api/chat (SSE)"| React
        end

        subgraph MLOps ["Phase 4: MLOps Loop (Human-in-the-Loop)"]
            direction TB
            Engineer["Engineer (Admin Mode)"]:::infra
            Upload["Upload Retraining Dataset<br/>(.jsonl)"]:::process
            FastAPI["Backend Orchestrator<br/>(FastAPI)"]:::process
            Retrain["Trigger Retraining"]:::process
            New_Adapter[("New Adapter")]:::model
            Benchmark["Auto-Benchmark"]:::process
            Result_View["View Results:<br/>Baseline F1 vs New F1"]:::infra
            Decision{"Approve New<br/>Adapter?"}:::decision
            
            Engineer --> Upload
            Upload --> FastAPI
            FastAPI --> Retrain
            Retrain --> New_Adapter
            New_Adapter --> Benchmark
            Benchmark --> Result_View
            Result_View --> Decision
            Decision -->|Yes| FastAPI
            FastAPI -.->|Hot-Swap Model| Ollama
            Decision -->|No| Discard["Discard Adapter"]:::process
        end

        %% Cross-graph connections
        Adapter -.->|Convert to GGUF| LoRA_GGUF