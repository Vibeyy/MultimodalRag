"""
Example: Processing Large PDFs Cost-Effectively
"""

from pathlib import Path
from multimodal_rag.ingestion.pdf_processor import PdfProcessor

# Example 1: Default settings (balanced)
print("=" * 60)
print("Example 1: Default Settings (Recommended)")
print("=" * 60)

processor = PdfProcessor(
    use_vision=True,
    vision_fallback_threshold=100,
    max_pages=100
)

pdf_path = Path("large_book.pdf")

# Always estimate cost first!
estimate = processor.estimate_cost(pdf_path)
print(f"\n📊 Cost Estimate:")
print(f"  Total pages: {estimate['total_pages']}")
print(f"  Pages to process: {estimate['pages_to_process']}")
print(f"  Estimated Vision pages: {estimate['estimated_vision_pages']}")
print(f"  Estimated traditional pages: {estimate['estimated_traditional_pages']}")
print(f"  💰 Estimated cost: ${estimate['estimated_cost_usd']}")

# Decide whether to proceed
if estimate['estimated_cost_usd'] < 2.00:
    print("\n✅ Cost acceptable, processing...")
    pages_data = processor.extract_text(pdf_path)
    print(f"✅ Processed {len(pages_data)} pages successfully!")
else:
    print("\n⚠️ Cost too high! Consider:")
    print("  - Reducing max_pages")
    print("  - Disabling Vision (use_vision=False)")
    print("  - Processing specific page range")

print()

# Example 2: Budget mode (minimize costs)
print("=" * 60)
print("Example 2: Budget Mode (Free/Minimal Cost)")
print("=" * 60)

budget_processor = PdfProcessor(
    use_vision=False,  # No Vision API = free!
    max_pages=50       # Only first 50 pages
)

estimate2 = budget_processor.estimate_cost(pdf_path)
print(f"\n📊 Cost Estimate:")
print(f"  Pages to process: {estimate2['pages_to_process']}")
print(f"  💰 Estimated cost: ${estimate2['estimated_cost_usd']} (embeddings only)")

print()

# Example 3: High quality mode (for scanned PDFs)
print("=" * 60)
print("Example 3: High Quality Mode (Scanned Documents)")
print("=" * 60)

quality_processor = PdfProcessor(
    use_vision=True,
    vision_fallback_threshold=500,  # Use Vision more aggressively
    max_pages=50  # Limit pages since Vision is expensive
)

estimate3 = quality_processor.estimate_cost(pdf_path)
print(f"\n📊 Cost Estimate:")
print(f"  Pages to process: {estimate3['pages_to_process']}")
print(f"  Estimated Vision pages: {estimate3['estimated_vision_pages']}")
print(f"  💰 Estimated cost: ${estimate3['estimated_cost_usd']}")

print()

# Example 4: Process specific page range
print("=" * 60)
print("Example 4: Process Specific Chapters/Pages")
print("=" * 60)

# Process pages 10-30 only (e.g., Chapter 2)
processor = PdfProcessor()
pages_data = processor.extract_text(pdf_path, page_range=(10, 30))
print(f"✅ Processed pages 10-30: {len(pages_data)} pages extracted")

print()

# Example 5: Environment-based configuration
print("=" * 60)
print("Example 5: Use Environment Variables")
print("=" * 60)

import os

# Set via environment (.env file)
processor = PdfProcessor(
    max_pages=int(os.getenv("PDF_MAX_PAGES", 100)),
    use_vision=os.getenv("PDF_USE_VISION", "true").lower() == "true",
    vision_fallback_threshold=int(os.getenv("PDF_VISION_THRESHOLD", 100))
)

print(f"Settings from environment:")
print(f"  max_pages: {processor._max_pages}")
print(f"  use_vision: {processor._use_vision}")
print(f"  vision_threshold: {processor._vision_threshold}")

print()
print("=" * 60)
print("💡 Key Takeaways:")
print("=" * 60)
print("✅ Always call estimate_cost() before processing")
print("✅ Set max_pages to avoid huge costs (default: 100)")
print("✅ Use use_vision=False for text-only PDFs (free!)")
print("✅ Adjust vision_threshold based on your needs")
print("✅ Process specific page ranges for chapters")
print("=" * 60)
