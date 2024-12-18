import os
import re
from dataclasses import dataclass, field
from operator import itemgetter
from typing import List, Optional, Tuple
from urllib.parse import urlparse

import fitz
import requests
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import AzureOpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from src.templates import discuss_prompt_template


class ArxivDownloadError(Exception):
    pass


class PDFParsingError(Exception):
    pass


@dataclass
class Section:
    title: str
    content: str
    level: int = 1
    subsections: List["Section"] = field(default_factory=list)

    def to_text(self) -> str:
        """Convert section and subsections to formatted text."""
        text = f"{'#' * self.level} {self.title}\n{self.content}\n"
        for subsection in self.subsections:
            text += subsection.to_text()
        return text


def extract_paper_id(url: str) -> str:
    parsed_url = urlparse(url)
    if "pdf" in url:
        return parsed_url.path.split("/")[-1].replace(".pdf", "")
    return parsed_url.path.split("/")[-1]


def download_arxiv_pdf(url: str, base_dir: str = "downloads") -> Tuple[str, str]:
    """Download a PDF from an arXiv URL.

    Args:
        url (str): The arXiv URL. Can be either the abstract page or direct PDF link.
        base_dir: Base directory for saving papers

    Returns:
        - Path to the downloaded PDF
        - Paper ID

    Example:
        >>> download_arxiv_pdf("https://arxiv.org/abs/2103.00020")
        'downloads/2103.00020/paper.pdf'
    """
    try:
        paper_id = extract_paper_id(url)
        paper_dir = os.path.join(base_dir, paper_id)
        os.makedirs(paper_dir, exist_ok=True)

        if "pdf" not in url:
            pdf_url = f"https://arxiv.org/pdf/{paper_id}.pdf"
        else:
            pdf_url = url

        output_path = os.path.join(paper_dir, "paper.pdf")

        response = requests.get(pdf_url, stream=True)
        response.raise_for_status()

        if "application/pdf" not in response.headers.get("content-type", ""):
            raise ValueError("The URL did not return a PDF file")

        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        return output_path, paper_id

    except Exception as e:
        raise ArxivDownloadError(f"Failed to download PDF: {str(e)}")


def parse_arxiv_pdf(pdf_path: str) -> str:
    try:
        parsed_path = os.path.join(os.path.dirname(pdf_path), "parsed.txt")

        doc = fitz.open(pdf_path)

        sections: List[Section] = []
        current_section: Optional[Section] = None
        current_level = 1
        collecting = True

        section_pattern = re.compile(
            r"^(?:(\d+\.)+\s*)?(?:Introduction|Background|Method(?:ology)?|"
            r"Results|Discussion|Conclusion|References|Abstract)",
            re.IGNORECASE,
        )

        for page in doc:
            if not collecting:
                break

            blocks = page.get_text("blocks")
            blocks.sort(key=lambda b: (b[1], b[0]))

            for block in blocks:
                text = block[4].strip()

                if not text:
                    continue

                if re.search(r"^\s*(?:\d+\.)?\s*Conclusion", text, re.IGNORECASE):
                    if current_section:
                        sections.append(current_section)
                    current_section = Section(
                        title="Conclusion",
                        content=text[text.lower().index("conclusion") :],
                        level=1,
                    )
                    sections.append(current_section)
                    collecting = False
                    break

                if section_pattern.match(text):
                    level = len(text.split(".")[0].strip().split("."))

                    new_section = Section(title=text, content="", level=level)

                    if current_section:
                        if level > current_section.level:
                            current_section.subsections.append(new_section)
                        else:
                            sections.append(current_section)
                            current_section = new_section
                    else:
                        current_section = new_section
                else:
                    if current_section:
                        current_section.content += f"\n{text}"

        processed_text = ""
        for section in sections:
            processed_text += section.to_text()

        processed_text = re.sub(r"-\s*\n\s*", "", processed_text)
        processed_text = re.sub(r"\s+", " ", processed_text)
        processed_text = re.sub(r"\[[\d,\s-]+\]", "", processed_text)

        if not processed_text.strip():
            raise PDFParsingError("No text could be extracted from the PDF")

        with open(parsed_path, "w", encoding="utf-8") as f:
            f.write(processed_text)

        doc.close()
        return parsed_path

    except Exception as e:
        raise PDFParsingError(f"Failed to parse PDF: {str(e)}")


def process_arxiv_paper(url: str, base_dir: str = "downloads") -> Tuple[str, str, str]:
    pdf_path, paper_id = download_arxiv_pdf(url, base_dir)
    parsed_path = parse_arxiv_pdf(pdf_path)
    return pdf_path, parsed_path, paper_id


def parse_script_plan(ai_message: AIMessage) -> list:
    sections = []
    current_section = []

    lines = ai_message.content.strip().splitlines()
    lines = lines[1:]

    header_pattern = re.compile(r"^#+\s")
    bullet_pattern = re.compile(r"^- ")

    for line in lines:
        if header_pattern.match(line):
            if current_section:
                sections.append(" ".join(current_section))
                current_section = []
            current_section.append(line.strip())
        elif bullet_pattern.match(line):
            current_section.append(line.strip())

    if current_section:
        sections.append(" ".join(current_section))

    return sections


def get_paper_head(pdf_path: str) -> str:
    pdf_reader = PdfReader(pdf_path)

    extracted_text = []
    collecting = True

    for page in pdf_reader.pages:
        text = page.extract_text()
        if text and collecting:
            # Stop collecting once "Introduction" is found
            if "Introduction" in text:
                introduction_index = text.index("Introduction")
                extracted_text.append(text[:introduction_index])
                break
            else:
                extracted_text.append(text)

    return "\n".join(extracted_text)


def initialize_vectorstore(txt_file):
    loader = TextLoader(txt_file, encoding="UTF-8")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    return Chroma.from_documents(documents=splits, embedding=AzureOpenAIEmbeddings())


def get_discussion_chain(vectorstore: Chroma, llm):
    """Create discussion chain using existing vectorstore."""
    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 6}
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    return (
        {
            "additional_context": itemgetter("section_plan") | retriever | format_docs,
            "section_plan": itemgetter("section_plan"),
            "previous_dialogue": itemgetter("previous_dialogue"),
        }
        | discuss_prompt_template
        | llm
        | StrOutputParser()
    )


def initialize_discussion_chain(txt_file, llm):
    # Load, chunk and index the contents of the blog.
    loader = TextLoader(txt_file, encoding="UTF-8")
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(
        documents=splits, embedding=AzureOpenAIEmbeddings()
    )

    # Retrieve and generate using the relevant snippets of the blog.
    retriever = vectorstore.as_retriever()

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    discuss_rag_chain = (
        {
            "additional_context": itemgetter("section_plan") | retriever | format_docs,
            "section_plan": itemgetter("section_plan"),
            "previous_dialogue": itemgetter("previous_dialogue"),
        }
        | discuss_prompt_template
        | llm
        | StrOutputParser()
    )
    return discuss_rag_chain
