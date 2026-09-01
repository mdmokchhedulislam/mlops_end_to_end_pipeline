                    ┌──────────────┐
                    │     Git      │
                    └──────┬───────┘
                           │
                           ▼
                 ┌──────────────────┐
                 │        CI         │
                 ├──────────────────┤
                 │ Ruff / Lint      │
                 │ Unit Test        │
                 │ Data Validation  │
                 │ Security Scan    │
                 │ Docker Build     │
                 │ Image Scan       │
                 │ Docker Push      │
                 └────────┬─────────┘
                          │
                          ▼
                  Docker Registry
                          │
                   CI DONE ✅
                          │
                          ▼
              ┌─────────────────────┐
              │   Argo Workflow     │
              │        (CD/ML)      │
              └──────────┬──────────┘
                         │
             ┌───────────┴───────────┐
             ▼                       ▼
      Docker Registry              DVC
             │                       │
             ▼                       ▼
       Training Image             Dataset
             │                       │
             └───────────┬───────────┘
                         ▼
                    Train Model
                         │
                         ▼
                    Evaluate
                         │
                         ▼
                   Quality Gate
                    /        \
                 FAIL          PASS
                  │              │
                  ▼              ▼
                STOP       MLflow Registry
                                  │
                                  ▼
                            Set Alias
                           `champion`
                                  │
                                  ▼
                              Deploy
                                  │
                                  ▼
                             Kubernetes
                                  │
                                  ▼
                              FastAPI
                                  │
                                  ▼
                        MLflow champion model