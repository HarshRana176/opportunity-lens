import pymupdf


def extract_text_from_pdf(pdf_path: str) -> str:

    doc = pymupdf.open(pdf_path)

    try:
        pages_text = []

        for page in doc:
            pages_text.append(
                page.get_text("text")
            )

        return "\n".join(pages_text)

    finally:
        doc.close()
