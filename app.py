"""
Streamlit application for evaluating startup project proposals from a venture‑capital
perspective.  Users can upload a proposal document (PDF, DOCX or plain text) and
receive an AI‑generated analysis based on common investment criteria.

The app uses OpenAI’s chat API (e.g. gpt‑5‑mini‑2025‑08‑07) to produce a detailed
assessment.  You will need a valid API key with access to the requested model.

The evaluation criteria are derived from the FNR JUMP Programme 2025 guidelines
for reviewers.  Each proposal is assessed on:

  * **Novelty** – Is the underlying technology or concept truly novel or
    disruptive?  Does the proposal provide sufficient data to demonstrate
    scientific soundness?
  * **Market & customer understanding** – For commercial projects, does the
    team understand the target market and customers?  Are there credible
    strategies for commercialisation?
  * **Social need & methodology** – For social impact projects, has the
    proposal clearly identified the social need?  Is the proposed methodology
    appropriate for addressing that need and delivering value to society?
  * **Impact magnitude** – What is the expected scale of the impact?  How big is
    the potential market or societal effect?
  * **Intellectual property & validation** – What is the state of the IP
    protection and prior art?  Does the project include validation data or
    prototypes that de‑risk the technology?
  * **Team competences** – Does the team have the right mix of technical and
    business expertise to execute?  If an Entrepreneur in Residence (EiR) is
    involved, are they suitably qualified and available?  How strong is the
    broader advisory or mentor network?

The tool produces an analysis but does not make funding decisions.  Final
decisions rest with human investors; the AI serves purely as an analytical aid.
"""

import io
import os
from typing import Optional

import streamlit as st

# Attempt to import libraries for document parsing.  pypdf handles PDF files,
# python‑docx handles DOCX files.  Both are optional: if they are not
# available, the app falls back to treating uploads as plain text.
try:
    from pypdf import PdfReader  # type: ignore
except Exception:
    PdfReader = None  # type: ignore

try:
    from docx import Document  # type: ignore
except Exception:
    Document = None  # type: ignore

try:
    import openai  # type: ignore
except Exception:
    openai = None  # type: ignore


def extract_text(file_buffer: io.BytesIO, mime_type: str) -> str:
    """Extract text from an uploaded file based on its MIME type.

    Parameters
    ----------
    file_buffer: io.BytesIO
        The raw file data as a bytes buffer.
    mime_type: str
        The MIME type of the uploaded file.

    Returns
    -------
    str
        The extracted textual content.
    """
    # Reset the buffer’s pointer to the start of the file
    file_buffer.seek(0)

    # PDF extraction
    if mime_type == "application/pdf" and PdfReader is not None:
        try:
            reader = PdfReader(file_buffer)
            pages_text = []
            for page in reader.pages:
                try:
                    pages_text.append(page.extract_text() or "")
                except Exception:
                    # If extraction fails for a page, skip it gracefully
                    continue
            return "\n".join(pages_text).strip()
        except Exception:
            pass  # Fall back to treating as plain text

    # DOCX extraction
    if mime_type in [
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/msword",
    ] and Document is not None:
        try:
            document = Document(file_buffer)
            paragraphs = [p.text for p in document.paragraphs]
            return "\n".join(paragraphs).strip()
        except Exception:
            pass  # Fall back to treating as plain text

    # Fallback: treat as plain text
    try:
        text_bytes = file_buffer.read()
        return text_bytes.decode("utf-8", errors="ignore")
    except Exception:
        return ""


def build_prompt(document_text: str) -> str:
    """Construct the prompt to be sent to the LLM.

    The prompt instructs the model to evaluate the proposal against
    predefined criteria derived from the JUMP Programme guidelines.  The
    model is told explicitly not to rely on sensitive personal data.
    """
    criteria_description = (
        "You are an experienced venture‑capital analyst.  Based on the FNR JUMP "
        "Programme review guidelines, evaluate the following project proposal.  "
        "Assess it against these criteria:\n"
        "1. Novelty – is the technology or concept novel/disruptive, and is there "
        "enough evidence to show scientific robustness?\n"
        "2. Market/customer understanding and commercialisation strengths (for "
        "commercial projects) – how well does the team understand the market and "
        "target customers, and what are the plans for bringing the product to market?\n"
        "3. Social needs and methodology (for social impact projects) – has the "
        "proposal identified a clear societal need and provided a suitable methodology "
        "for addressing it?\n"
        "4. Impact magnitude – what is the expected scale of impact?\n"
        "5. Intellectual property and validation – what is the status of IP and "
        "technical validation?\n"
        "6. Team competences and expertise – assess the team’s experience and the "
        "fit of any entrepreneur in residence (EiR) if applicable.\n\n"
        "Provide a structured analysis highlighting strengths, weaknesses, risks and "
        "opportunities for each criterion.  Offer recommendations for improvement "
        "but **do not** make a final funding decision.  Do **not** use sensitive "
        "personal attributes (e.g. race, gender, health) in your reasoning.  "
        "Instead focus on objective, project‑related factors.  At the end, "
        "summarise the overall investment outlook."
    )

    return f"{criteria_description}\n\nHere is the proposal:\n\n{document_text.strip()}"


def evaluate_proposal(document_text: str, api_key: str) -> Optional[str]:
    """Send the proposal and evaluation instructions to the OpenAI API.

    Parameters
    ----------
    document_text: str
        The text of the uploaded proposal.
    api_key: str
        The user's OpenAI API key.  Must be valid and have access to the
        requested model.

    Returns
    -------
    Optional[str]
        The AI's response text, or None if an error occurred.
    """
    if openai is None:
        st.error(
            "The openai package is not available.  Please ensure it is installed."
        )
        return None

    if not api_key:
        st.error("Please provide a valid OpenAI API key.")
        return None

    openai.api_key = api_key
    prompt = build_prompt(document_text)

    try:
        response = openai.ChatCompletion.create(
            model="gpt-5-mini-2025-08-07",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1024,
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        st.error(f"Error communicating with OpenAI API: {exc}")
        return None


def main() -> None:
    """Run the Streamlit app."""
    st.set_page_config(page_title="Startup Proposal Evaluator", page_icon="📊")
    st.title("Startup Proposal Evaluator")
    st.markdown(
        "Upload a startup project proposal to receive a structured VC‑style analysis "
        "based on the FNR JUMP review criteria.  Your API key is required to call "
        "OpenAI’s chat models.  The tool does not make funding decisions; it "
        "provides analysis to aid your judgement."
    )

    api_key_input = st.text_input(
        "OpenAI API Key",
        type="password",
        value=os.getenv("OPENAI_API_KEY", ""),
        help="Your key will only be used locally to query the OpenAI API.",
    )

    uploaded_file = st.file_uploader(
        "Upload proposal (PDF, DOCX, or TXT)",
        type=["pdf", "docx", "txt"],
    )

    if uploaded_file is not None:
        mime_type = uploaded_file.type
        file_buffer = io.BytesIO(uploaded_file.read())
        proposal_text = extract_text(file_buffer, mime_type)
        if not proposal_text:
            st.error(
                "Could not extract text from the uploaded file.  Please upload a "
                "document in PDF, DOCX, or plain text format."
            )
        else:
            st.write(
                f"Extracted {len(proposal_text):,} characters of text from the uploaded proposal."
            )

            if st.button("Evaluate Proposal"):
                with st.spinner("Evaluating... this may take a minute"):
                    result = evaluate_proposal(proposal_text, api_key_input)
                    if result:
                        st.subheader("AI‑Generated Analysis")
                        st.markdown(result)


if __name__ == "__main__":
    main()