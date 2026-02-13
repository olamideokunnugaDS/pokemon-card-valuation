#!/bin/bash

# Root level directories
mkdir -p src/{vision_module,market_module,fusion_module,data_pipeline,evaluation,utils}
mkdir -p configs/{vision,market,fusion,data}
mkdir -p data/{raw,processed,embeddings,external}
mkdir -p models/{vision,market,fusion,checkpoints}
mkdir -p notebooks/{exploratory,analysis,visualization}
mkdir -p scripts/{data_collection,preprocessing,training,evaluation}
mkdir -p tests/{unit,integration}
mkdir -p results/{figures,tables,metrics,reports}
mkdir -p docs/{architecture,methodology,api}
mkdir -p logs

# Create __init__.py files for Python packages
find src -type d -exec touch {}/__init__.py \;
find tests -type d -exec touch {}/__init__.py \;

echo "Directory structure created successfully!"
