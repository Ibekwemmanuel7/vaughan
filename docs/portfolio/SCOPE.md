# Vaughan scope and portfolio boundary

Updated 17 October 2026.

Portfolio hub: [Lillie US](https://github.com/Ibekwemmanuel7/Lillie_US)

Portfolio map: [Product responsibilities](https://github.com/Ibekwemmanuel7/Lillie_US/blob/main/docs/PORTFOLIO.md)

## Domain and status

**Domain:** Hurricanes, flood and coastal impacts

**Status:** Existing hurricane research code; flood and coastal-impact extensions are planned.

## Current baseline

The implemented research focus is hurricane structure retrieval and related evaluation. Existing package, experiments and scientific documentation are retained.

## Planned scope

Observed flood extent, event-related coastal inundation and shoreline disturbance, exposed-asset screening, access and inspection priorities, and recovery evidence. None of these additions is claimed to be implemented by this scope change.

## Ownership and interfaces

Vaughan owns event-based flood and coastal impacts, including non-hurricane floods as a planned extension. Velo owns routine water quantity, quality screening and watershed context. Vepris and Vantage consume event products for forestry and farming decisions.

## Delivery gate

Define the buyer, decision, last useful delivery time, reference data, baseline and acceptance criteria before implementation. Outputs must carry observation and processing dates, provenance, quality, uncertainty and unavailable-data states. Customer-provided records and assessments require explicit access control. No new operational or scientific performance claim follows from adding this roadmap.

The repository owns its reusable domain code, methods, tests and domain documentation. Lillie US owns cross-product integration, shared customer workflows, portfolio configuration and product-release coordination. Do not make this repository depend on Lillie US merely to create a backlink.
