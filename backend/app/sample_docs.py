"""Sample financial documents and profiles database seeder."""

import logging
from .rag import RAGPipeline

logger = logging.getLogger(__name__)

# Structured profile metrics to populate company_metrics table on startup
SAMPLE_COMPANIES = [
    {
        "ticker": "AAPL",
        "company_name": "Apple Inc.",
        "sector": "Technology",
        "competitors": ["MSFT", "GOOG", "TSLA"],
        "revenue": 395000.0,
        "net_income": 97000.0,
        "operating_income": 115000.0,
        "total_assets": 340000.0,
        "total_liabilities": 200000.0,
        "cash": 45000.0
    },
    {
        "ticker": "MSFT",
        "company_name": "Microsoft Corporation",
        "sector": "Technology",
        "competitors": ["AAPL", "GOOG", "AMZN"],
        "revenue": 245000.0,
        "net_income": 88000.0,
        "operating_income": 102000.0,
        "total_assets": 470000.0,
        "total_liabilities": 220000.0,
        "cash": 75000.0
    },
    {
        "ticker": "NVDA",
        "company_name": "NVIDIA Corporation",
        "sector": "Technology",
        "competitors": ["AMD", "INTC", "MSFT"],
        "revenue": 96000.0,
        "net_income": 53000.0,
        "operating_income": 60000.0,
        "total_assets": 120000.0,
        "total_liabilities": 35000.0,
        "cash": 32000.0
    },
    {
        "ticker": "TSLA",
        "company_name": "Tesla Inc.",
        "sector": "Automotive",
        "competitors": ["F", "GM", "BYD"],
        "revenue": 97000.0,
        "net_income": 13000.0,
        "operating_income": 11000.0,
        "total_assets": 110000.0,
        "total_liabilities": 45000.0,
        "cash": 28000.0
    }
]

# Unstructured filing files to ingest into SQLite vector store
SAMPLE_DOCUMENTS = {
    "aapl_10k_fy25.md": """---
ticker: AAPL
fiscal_year: 2025
document_type: 10-K
---
# Apple Inc. Form 10-K Summary (FY25)

## Business Overview
Apple Inc. continues to design, manufacture, and market smartphones, personal computers, tablets, wearables, and accessories. The company's strategic roadmap in FY25 focused heavily on 'Apple Intelligence' – a deep integration of generative artificial intelligence features across iOS, iPadOS, and macOS.

## Financial Performance
In FY25, total revenue expanded to $395,000 million (up from $385,000 million in FY24), powered by early upgrade cycles for Apple Intelligence compatible devices. Operating income was $115,000 million, representing strong operational efficiency. Net income reached $97,000 million.

## Risk Factors
1. **Supply Chain Constraints**: Apple relies on custom silicon and advanced components. Short-term constraints in display technology and camera modules represent key risks.
2. **Regulatory Pressures**: Digital markets laws in Europe and App Store fee antitrust lawsuits present ongoing headwinds to services margins.
3. **Intellectual Property**: Ongoing patent battles regarding biosensors and connectivity chips require continuous R&D outlays.
""",

    "aapl_transcript_fy25.md": """---
ticker: AAPL
fiscal_year: 2025
document_type: transcript
---
# Apple Inc. Q4 FY25 Earnings Call Transcript

## Executives Present
- Tim Cook, CEO
- Luca Maestri, CFO

## Tim Cook's Comments
"We are thrilled with the response to Apple Intelligence. Customers are loving the localized natural language processing, writing tools, and Siri upgrades. Our personal devices are more capable than ever. Looking forward, our silicon pipeline is stronger than ever."

## Luca Maestri on Margins
"Gross margin was 46.2%, sitting at the top end of our guidance. Capital expenditures for the year reached $12,500 million, primarily supporting data center partnerships for Apple Intelligence model hosting. Our cash balance stands at $45,000 million, allowing us to remain highly active in our capital return program."

## Q&A Highlights
- *Analyst*: Tim, what are the primary limitations to Apple Intelligence deployment?
- *Tim Cook*: "Right now, it is supply constraints for camera sensors and display drivers, and getting localization out. We are working hard to expand geographic coverage by early 2026."
""",

    "msft_10k_fy25.md": """---
ticker: MSFT
fiscal_year: 2025
document_type: 10-K
---
# Microsoft Corporation Form 10-K Summary (FY25)

## Business Overview
Microsoft Corporation is a global leader in software, cloud computing (Azure), enterprise services, and AI. The main operational focus in FY25 remains Copilot integration across Office suite products and heavy infrastructure expansion for AI training.

## Financial Performance
Microsoft reported total revenue of $245,000 million in FY25. Operating income reached $102,000 million, and net income was $88,000 million. Azure cloud services grew 28% year-over-year, driving cloud segment dominance.

## Risk Factors
1. **AI Capital Spending**: Massively expanding GPU data centers leads to high capital expenditures, which could impact free cash flow if cloud monetization slows down.
2. **Cybersecurity Moats**: High-profile security vulnerabilities require additional security investments and represent operational risks.
3. **Competition**: Google Cloud, Amazon Web Services, and emerging specialized GPU clouds represent key competitive threats.
""",

    "msft_transcript_fy25.md": """---
ticker: MSFT
fiscal_year: 2025
document_type: transcript
---
# Microsoft Corp Q4 FY25 Earnings Call Transcript

## Executives Present
- Satya Nadella, CEO
- Amy Hood, CFO

## Satya Nadella's Comments
"This year, Microsoft Copilot has transitioned from an experimental product to an enterprise necessity. Over 65% of Fortune 500 companies are utilizing Copilot. Azure OpenAI API utilization is up 150%, and we are seeing excellent returns on our platform investments."

## Amy Hood on Capex
"We invested $22,000 million in capital expenditures this year, which is a significant increase. This spending is directly aligned with cloud and AI capacity constraints. Our overall cash position stands at $75,000 million, giving us unmatched balance sheet strength."
""",

    "nvda_10k_fy25.md": """---
ticker: NVDA
fiscal_year: 2025
document_type: 10-K
---
# NVIDIA Corporation Form 10-K Summary (FY25)

## Business Overview
NVIDIA is the pioneer of GPU-accelerated computing. The company designs chips, systems, and software (CUDA platform) that power generative AI workloads.

## Financial Performance
NVIDIA experienced explosive growth in FY25, with revenue climbing to $96,000 million. Operating income reached $60,000 million and net income was $53,000 million, reflecting an industry-leading operating margin.

## Risk Factors
1. **Supply Chain Concentration**: NVIDIA relies on TSMC for chip fabrication. Any geopolitical disruption in Taiwan would severely halt operations.
2. **Blackwell Product Transition**: Initial production issues with Blackwell packaging require careful quality control and yield-rate monitoring.
3. **Hyperscaler Insourcing**: Major customers (Apple, Microsoft, Google) are developing custom internal TPUs/ASICs to reduce dependence.
""",

    "nvda_transcript_fy25.md": """---
ticker: NVDA
fiscal_year: 2025
document_type: transcript
---
# NVIDIA Corp Q4 FY25 Earnings Call Transcript

## Executives Present
- Jensen Huang, CEO
- Colette Kress, CFO

## Jensen Huang's Comments
"The generative AI wave is in its infancy. Blackwell demand is off the charts. We are producing Blackwell at scale, and customer allocations are locked in. The CUDA ecosystem remains our strongest competitive advantage."

## Colette Kress on Margins
"Gross margin came in at 75.1%, driven by datacenter chip demand. Cash and cash equivalents reached $32,000 million. We expect Blackwell volumes to ramp significantly throughout FY26, easing current allocation constraints."
""",

    "tsla_10k_fy25.md": """---
ticker: TSLA
fiscal_year: 2025
document_type: 10-K
---
# Tesla Inc. Form 10-K Summary (FY25)

## Business Overview
Tesla Inc. designs and manufactures electric vehicles, battery energy storage systems, solar products, and autonomous driving technology.

## Financial Performance
Tesla reported total revenue of $97,000 million in FY25. Net income was $13,000 million, and operating income was $11,000 million. Automotive gross margins declined slightly due to pricing pressure, but energy storage expanded 120% YoY.

## Risk Factors
1. **Global EV Competition**: Severe pricing competition in China and Europe from local EV makers impacts average selling prices.
2. **Autonomous Driving Regulation**: Regulatory reviews of Full Self-Driving (FSD) represent significant uncertainty.
3. **Factory Ramps**: Scaling gigafactories in Texas and Berlin requires capital outlays.
""",

    "tsla_transcript_fy25.md": """---
ticker: TSLA
fiscal_year: 2025
document_type: transcript
---
# Tesla Inc. Q4 FY25 Earnings Call Transcript

## Executives Present
- Elon Musk, CEO
- Vaibhav Taneja, CFO

## Elon Musk's Comments
"Tesla is not just a car company; we are an AI and robotics company. Our energy storage deployment reached record gigawatt-hours. FSD version 12 is performing exceptionally well, and we are working with global regulators to deploy robotaxis."

## Vaibhav Taneja on Cash
"Operating margins were 11.3%. Capital expenditures reached $8,500 million. Our cash and equivalents remain strong at $28,000 million. We are optimizing manufacturing costs to offset global pricing pressure."
"""
}


def populate_sample_data(pipeline: RAGPipeline) -> None:
    """Seed SQLite database with structured metrics and unstructured SEC documents."""
    logger.info("Starting database seeding...")
    
    # 1. Seed structured company metrics
    for comp in SAMPLE_COMPANIES:
        pipeline.store.upsert_company_metrics(
            ticker=comp["ticker"],
            company_name=comp["company_name"],
            sector=comp["sector"],
            competitors=comp["competitors"],
            revenue=comp["revenue"],
            net_income=comp["net_income"],
            operating_income=comp["operating_income"],
            total_assets=comp["total_assets"],
            total_liabilities=comp["total_liabilities"],
            cash=comp["cash"]
        )
    logger.info("Structured company registry seeded.")

    # 2. Seed unstructured document vector store
    for name, content in SAMPLE_DOCUMENTS.items():
        pipeline.ingest(name, content)
        
    logger.info("Unstructured vector store filings seeded successfully.")
