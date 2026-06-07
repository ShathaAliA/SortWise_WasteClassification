# 🌱 SortWise: AI-Powered Waste Classification & Eco-Disposal Portal

SortWise is an end-to-end, locally deployed artificial intelligence application designed to automate household waste classification and support sustainable recycling habits. Utilizing deep computer vision architectures and specialized engineering practices, the system processes consumer waste imagery, computes predictive categorical boundaries, displays multi-class probabilistic analytics, and delivers real-time eco-disposal instructions.

---

## 🚀 Key Features

- **Deep Learning Core:** Powered by a customized MobileNetV2 convolutional neural network fine-tuned on household waste metrics.
- **Explainability Metrics:** Displays the primary classification alongside the top two alternative model probabilities to provide system classification transparency.
- **Eco Disposal & Recycling Guidance Engine:** Automatically maps the predicted waste item to an explicit, interactive recycling instruction block.
- **Production-Grade Logging & Monitoring:** Maintains an active local diagnostic pipeline (`sortwise_app.log`) capturing model loads, operational latency benchmarks, and application lifecycle events.
- **State-Based Application Caching:** Injects memory optimization protocols (`@st.cache_resource`) to streamline heavy framework processing and lower execution runtime latency down to a local target.
- **Automated Continuous Integration (CI):** Formulated with an active GitHub Actions workflow to run code linting and environment matrix validation dynamically upon code delivery.

---

## 📁 Repository Structure

The project directory is structured cleanly to decouple model development phases from core software engineering deployment routines:

```text
SortWise_WasteClassification/
│
├── .github/
│   └── workflows/
│       └── test_ci.yml          # GitHub Actions Automated CI configuration
│
├── Notebooks/                  # Model design, training, and exploration pipeline
│   ├── Data_Pre_processing.ipynb
│   ├── Evaluation_and_Visualization.ipynb
│   ├── Libraries&Load_the_data.ipynb
│   ├── Model_Training1.ipynb
│   └── Real_time_inference.ipynb
│
├── .gitignore                  # Strict filtration rules to bypass binary/log uploads
├── app.py                      # Main Streamlit web application & local production pipeline
└── requirements.txt            # Explicit third-party system dependencies
