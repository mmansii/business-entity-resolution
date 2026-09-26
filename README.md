# Business Entity Resolution

AI-powered business entity resolution solution for the Amazon ML Challenge.

## Project Overview

This project solves the Business Entity Resolution Challenge.

The task is to identify which records from Source 2 and Source 3 refer to the same real-world business as each Source 1 reference entity.

The data contains noisy and inconsistent business names, addresses, and country information.

## Challenge Structure

The challenge contains three independent data sources:

- **Source 1** — deduplicated reference businesses
- **Source 2** — business records to be matched
- **Source 3** — business records to be matched

For every Source 1 entity, the system finds all matching Source 2 and Source 3 records.

A Source 1 entity may have:

- zero matches
- one match
- multiple matches

## Input Data

The files are tab-separated (`.tsv`) files.

Columns:

- `entity_id`
- `business_name`
- `business_address`
- `country`

Example:

```python
import pandas as pd

df = pd.read_csv(
    "dataset/train/train_source1.tsv",
    sep="\t"
)