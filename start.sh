#!/bin/bash
cd cropstress_back/cropstress_back-main
uvicorn pipeline:app --host 0.0.0.0 --port 8000