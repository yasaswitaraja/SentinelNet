# SentinelNet Project
This project is for AI-powered Network Intrusion Detection System. SentinelNet is a project that analyzes network intrusion datasets to help detect cyber attacks. It’s designed to explore, visualize, and categorize network traffic into different types of attacks, giving a clear picture of which attacks are more frequent and how balanced the dataset is.

The project focuses on two widely used datasets:

NSL-KDD – A classic network intrusion detection dataset with labels for attacks like DoS (Denial of Service), Probe, R2L (Remote to Local), U2R (User to Root), and Normal traffic.

CICIDS2017 – A modern network dataset with real traffic captured over an afternoon, including multiple attack types like DDoS, PortScan, and more.

Project Structure

SentinelNet/
├── data/                         # Folder for raw datasets
│   ├── NSL-KDD/
│   │   └─ KDDTrain+.txt
│   └── CICIDS-2017/
│       └─ Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
├── script/                       # Python scripts for analysis
│   └─ main.py                     # Loads datasets, processes data, generates plots
├── outputs/                      # Generated outputs
│   ├─ kdd_attack_types.png
│   ├─ kdd_attack_categories.png
│   └─ cicids_top10_attack_types.png
├── notebooks/                     # Optional notebooks for exploration
│   └─ load_and_explore.ipynb
├── requirements.txt               # Required Python packages
└── README.md                      # Project overview and instructions


Run the main script:
'''python scripts/main.py'''
