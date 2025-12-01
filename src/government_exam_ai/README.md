# Government Exam AI/ML Model

## Project Overview
Custom AI/ML model trained for 150+ government exams across 15 categories including UPSC, SSC, Banking, Railways, Defence, and State Government exams.

## Exam Categories
1. **UPSC/Civil Services** - IAS, IPS, IFS, IFoS, CDS, NDA
2. **SSC Exams** - CGL, CHSL, MTS, GD, CPO, JE, Stenographer
3. **Banking** - IBPS PO/Clerk, SBI PO/Clerk, RBI Grade B
4. **Railways** - RRB NTPC, Group D, JE, ALP
5. **Defence** - NDA, CDS, AFCAT, Army/Navy/Airforce Agniveer
6. **State Government** - UPPSC, MPPSC, BPSC, RPSC, MPSC, etc.
7. **Teaching** - CTET, UPTET, DSSSB, KVS
8. **Insurance** - LIC AAO, NIACL, GIC
9. **PSU** - GATE-based PSU, Coal India, SAIL
10. **Metro/Transport** - DMRC, NMRC, BMRC
11. **Judiciary** - Judicial Services, AIBE
12. **Research/Academic** - UGC NET, CSIR NET
13. **Health/Medical** - AIIMS NORCET, ESIC Nursing
14. **Clerk/Assistant** - Various assistant positions
15. **Specialized** - FCI, EPFO, India Post GDS

## Architecture Components
- **Data Collection Pipeline** - Automated data gathering from exam sources
- **Content Processing** - PDF parsing, OCR, text extraction
- **Question Classification** - Subject tagging, difficulty assessment
- **Answer Evaluation** - NLP-based scoring and feedback
- **Mock Test Generation** - Adaptive test creation
- **Performance Analytics** - Progress tracking and insights
- **Personalization Engine** - Customized learning paths

## Tech Stack
- **ML Framework**: PyTorch, Transformers, Scikit-learn
- **NLP**: spaCy, NLTK, Hugging Face Transformers
- **Data Processing**: Pandas, NumPy, OpenCV (for OCR)
- **Database**: MongoDB for unstructured data, PostgreSQL for structured
- **Deployment**: FastAPI, Docker, Kubernetes
- **Frontend**: React.js with TypeScript

## Project Structure
```
├── data_ingestion/          # Data collection and preprocessing
├── ml_models/              # Core ML models and algorithms
├── evaluation/             # Answer evaluation and scoring
├── test_generation/        # Mock test generation engine
├── analytics/              # Performance analytics and insights
├── api/                    # FastAPI backend
├── frontend/               # React frontend
├── config/                 # Configuration files
├── tests/                  # Unit and integration tests
└── docs/                   # Documentation
```

## Getting Started
1. Set up data collection from exam sources
2. Preprocess and clean exam data
3. Train question classification models
4. Build answer evaluation system
5. Create mock test generation pipeline
6. Deploy and test with real exam data