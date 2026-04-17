

\# Colorectal 28-Day Pathway Discrete Event Simulator


## Disclaimer

This model is intended for exploratory analysis and scenario testing only.
All parameters are illustrative and do not represent any specific NHS Trust,
service, or patient cohort.

##



A daily-step discrete event simulation (DES) of the colorectal 28-day diagnostic pathway, with an interactive Streamlit interface.



The model supports:

\- Multiple referral routes (OP F2F, OP Tel, IDA, Sulis)

\- Resource-specific capacity and working days

\- Retests, histology, and patient communication delays

\- Warm-up periods and multiple replications

\- Patient-level and step-level KPIs (including 28-day compliance)

\- Bottleneck identification

\- Resource utilisation summary



\## Project structure



```text

.

├── app.py                  # Streamlit user interface

├── engine/

│   └── colorectal\_sim.py   # Core simulation engine

├── requirements.txt

└── README.md



