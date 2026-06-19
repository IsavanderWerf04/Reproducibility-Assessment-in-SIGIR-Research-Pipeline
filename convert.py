import os
import pymupdf4llm


def convert_pdf_to_markdown(pdf_path, output_root):
    filename = os.path.splitext(os.path.basename(pdf_path))[0]

    safe_filename = filename.replace("(", "-").replace(")", "-")

    output_path = os.path.join(output_root, f"{filename}.md")

    image_folder = os.path.join(output_root, f"{safe_filename}_images")
    os.makedirs(image_folder, exist_ok=True)

    # Extra folder pymupdf4llm tries to use internally
    os.makedirs(f"{safe_filename}_images", exist_ok=True)

    md_text = pymupdf4llm.to_markdown(
        pdf_path,
        write_images=True,
        image_path=f"{safe_filename}_images",
        image_format="png",
        dpi=200,
        page_chunks=False,
        table_strategy="lines_strict",
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md_text)

    print(f"Created markdown: {output_path}")
    print(f"Created images: {image_folder}")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base_dir)

    pdf_paths = [
        os.path.join(base_dir, "3477495.3532051.pdf"),
        os.path.join(base_dir, "3527546.3527552.pdf"),
    ]

    for pdf_path in pdf_paths:
        convert_pdf_to_markdown(pdf_path, base_dir)