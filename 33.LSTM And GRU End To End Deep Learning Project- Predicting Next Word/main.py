import html
import logging
import pickle
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


LOGGER = logging.getLogger(__name__)
PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "next_word.h5"
TOKENIZER_PATH = PROJECT_DIR / "tokenizere.pickle"
DEFAULT_PROMPT = "To be or not to"
EXAMPLE_PROMPTS = (
    "To be or not to",
    "Something is rotten",
    "The rest is silence",
)
HISTORY_LIMIT = 8


st.set_page_config(
    page_title="Next Word Prediction | NLP Lab",
    page_icon=":material/neurology:",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_resource(show_spinner=False)
def load_assets():
    """Load the existing model and tokenizer once per Streamlit process."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Required model artifact not found: {MODEL_PATH.name}")
    if not TOKENIZER_PATH.is_file():
        raise FileNotFoundError(
            f"Required tokenizer artifact not found: {TOKENIZER_PATH.name}"
        )

    model = load_model(MODEL_PATH, compile=False)
    with TOKENIZER_PATH.open("rb") as handle:
        tokenizer = pickle.load(handle)

    input_shape = model.input_shape
    if isinstance(input_shape, list) or len(input_shape) != 2:
        raise ValueError("The loaded model must accept a single sequence input.")
    input_length = input_shape[1]
    if not isinstance(input_length, int) or input_length <= 0:
        raise ValueError("The loaded model does not define a fixed input length.")

    return model, tokenizer, input_length


def prepare_input(tokenizer, text, input_length):
    """Match the model's original tokenizer and pre-padding inference pipeline."""
    token_ids = tokenizer.texts_to_sequences([text])[0]
    model_token_ids = token_ids[-input_length:]
    padded_sequence = pad_sequences(
        [model_token_ids],
        maxlen=input_length,
        padding="pre",
        truncating="pre",
    )
    return token_ids, model_token_ids, padded_sequence


def predict_next_word(model, tokenizer, text, input_length):
    token_ids, model_token_ids, padded_sequence = prepare_input(
        tokenizer, text, input_length
    )
    if not model_token_ids:
        return None, token_ids, model_token_ids, padded_sequence

    scores = model.predict(padded_sequence, verbose=0)
    predicted_index = int(np.argmax(scores, axis=1)[0])
    predicted_word = tokenizer.index_word.get(predicted_index)
    return predicted_word, token_ids, model_token_ids, padded_sequence


def clear_current_input():
    st.session_state["input_text"] = ""
    st.session_state["prediction"] = None


def choose_prompt(prompt):
    st.session_state["input_text"] = prompt
    st.session_state["prediction"] = None


def reuse_history_input(text):
    st.session_state["input_text"] = text
    st.session_state["prediction"] = None


def clear_history():
    st.session_state["prediction_history"] = []


def describe_layer(layer, layer_number):
    config = layer.get_config()
    layer_type = layer.__class__.__name__

    if layer_type == "Embedding":
        details = (
            f"{config.get('output_dim', '?')} dimensions · "
            f"{config.get('input_dim', '?')} embedding rows"
        )
        title = "Embedding"
    elif layer_type == "LSTM":
        details = f"{config.get('units', '?')} units"
        title = f"LSTM layer {layer_number}"
    elif layer_type == "Dropout":
        details = f"{config.get('rate', '?')} dropout rate"
        title = "Dropout"
    elif layer_type == "Dense":
        details = (
            f"{config.get('units', '?')} outputs · "
            f"{config.get('activation', 'linear')} activation"
        )
        title = "Dense output"
    else:
        title = layer_type
        details = "Layer in the loaded model"

    return title, details, layer_type


def render_styles():
    st.markdown(
        """
        <style>
        :root {
            --ink: #f2f4ff;
            --muted: #a3abc2;
            --violet: #a694ff;
            --cyan: #73d9ed;
            --line: rgba(167, 180, 221, 0.16);
            --panel: rgba(16, 22, 39, 0.76);
        }

        html, body, [class*="css"] {
            font-family: "Inter", "Segoe UI", sans-serif;
        }

        .stApp {
            color: var(--ink);
            background:
                radial-gradient(ellipse at 80% 0%, rgba(75, 68, 157, 0.22), transparent 34rem),
                radial-gradient(ellipse at 0% 32%, rgba(35, 112, 138, 0.12), transparent 34rem),
                linear-gradient(145deg, #080b14 0%, #0b1020 52%, #090d17 100%);
        }

        .main .block-container {
            max-width: 1220px;
            padding-top: 2.2rem;
            padding-bottom: 4rem;
        }

        header[data-testid="stHeader"] {
            background: rgba(8, 11, 20, 0.35);
        }

        .hero {
            position: relative;
            overflow: hidden;
            isolation: isolate;
            padding: clamp(2rem, 6vw, 4.7rem);
            border: 1px solid rgba(173, 184, 231, 0.18);
            border-radius: 28px;
            background:
                radial-gradient(ellipse at 82% 45%, rgba(117, 93, 225, 0.2), transparent 29rem),
                linear-gradient(115deg, rgba(20, 27, 48, 0.95), rgba(13, 19, 35, 0.86));
            box-shadow: 0 28px 90px rgba(0, 0, 0, 0.28), inset 0 1px rgba(255, 255, 255, 0.035);
        }

        .hero::before {
            position: absolute;
            z-index: -1;
            inset: 0;
            content: "";
            opacity: 0.58;
            background:
                radial-gradient(circle at 76% 27%, rgba(137, 124, 255, 0.96) 0 2px, transparent 3px),
                radial-gradient(circle at 90% 42%, rgba(115, 217, 237, 0.9) 0 2px, transparent 3px),
                radial-gradient(circle at 81% 70%, rgba(137, 124, 255, 0.78) 0 2px, transparent 3px),
                radial-gradient(circle at 68% 56%, rgba(115, 217, 237, 0.8) 0 2px, transparent 3px),
                linear-gradient(28deg, transparent 74%, rgba(125, 145, 229, 0.18) 74.15%, transparent 74.35%),
                linear-gradient(153deg, transparent 71%, rgba(125, 145, 229, 0.15) 71.15%, transparent 71.35%),
                linear-gradient(103deg, transparent 83%, rgba(125, 145, 229, 0.12) 83.15%, transparent 83.35%);
            mask-image: linear-gradient(90deg, transparent 42%, #000 100%);
            animation: network-breathe 8s ease-in-out infinite alternate;
        }

        .hero::after {
            position: absolute;
            z-index: -1;
            top: -9rem;
            right: 8%;
            width: 24rem;
            height: 24rem;
            border: 1px solid rgba(161, 147, 255, 0.1);
            border-radius: 50%;
            box-shadow: 0 0 0 3rem rgba(161, 147, 255, 0.025), 0 0 0 6rem rgba(161, 147, 255, 0.018);
            content: "";
        }

        @keyframes network-breathe {
            from { opacity: 0.36; transform: translateY(0); }
            to { opacity: 0.7; transform: translateY(-5px); }
        }

        .hero-eyebrow, .section-kicker, .micro-label {
            color: #aeb6d0;
            font-size: 0.69rem;
            font-weight: 700;
            letter-spacing: 0.17em;
            text-transform: uppercase;
        }

        .hero h1 {
            max-width: 720px;
            margin: 1.05rem 0 0.8rem;
            color: #f7f7ff;
            font-family: "Inter", "Segoe UI", sans-serif;
            font-size: clamp(2.55rem, 6.2vw, 5rem);
            font-weight: 800;
            letter-spacing: -0.065em;
            line-height: 1.04;
        }

        .hero h1 span {
            color: #b5a7ff;
            background: linear-gradient(100deg, #c0b4ff, #83dcec 92%);
            background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .hero-copy {
            max-width: 600px;
            margin: 0;
            color: #aeb7cf;
            font-size: clamp(1rem, 1.8vw, 1.12rem);
            line-height: 1.7;
        }

        .hero-badges {
            display: flex;
            flex-wrap: wrap;
            gap: 0.7rem;
            margin-top: 2rem;
        }

        .hero-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.55rem;
            padding: 0.55rem 0.82rem;
            border: 1px solid rgba(169, 181, 222, 0.18);
            border-radius: 99px;
            background: rgba(8, 13, 26, 0.55);
            color: #d8def2;
            font-size: 0.77rem;
        }

        .status-dot {
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background: #63dfb0;
            box-shadow: 0 0 12px rgba(99, 223, 176, 0.72);
        }

        .section-kicker {
            margin: 2.5rem 0 0.4rem;
            color: #9c8dff;
        }

        .section-title {
            margin: 0 0 0.25rem;
            color: #f3f5ff;
            font-family: "Inter", "Segoe UI", sans-serif;
            font-size: clamp(1.5rem, 3vw, 2rem);
            font-weight: 750;
            letter-spacing: -0.035em;
        }

        .section-copy {
            margin: 0 0 1.2rem;
            color: var(--muted);
            font-size: 0.96rem;
        }

        .st-key-prediction-studio, .st-key-result-card, .st-key-pipeline-panel,
        .st-key-architecture-panel, .st-key-history-panel {
            padding: clamp(1.05rem, 3vw, 1.8rem);
            border: 1px solid var(--line);
            border-radius: 22px;
            background: var(--panel);
            box-shadow: 0 16px 46px rgba(0, 0, 0, 0.16), inset 0 1px rgba(255, 255, 255, 0.025);
        }

        .st-key-prediction-studio {
            background:
                radial-gradient(ellipse at 98% 0%, rgba(104, 85, 206, 0.11), transparent 28rem),
                var(--panel);
        }

        .stTextArea textarea {
            min-height: 150px;
            border: 1px solid rgba(159, 173, 221, 0.2);
            border-radius: 15px;
            background: rgba(7, 12, 24, 0.72);
            color: #f0f3ff;
            font-size: 1.05rem;
            line-height: 1.65;
            transition: border-color 180ms ease, box-shadow 180ms ease;
        }

        .stTextArea textarea:focus {
            border-color: rgba(166, 148, 255, 0.72);
            box-shadow: 0 0 0 3px rgba(142, 119, 255, 0.13);
        }

        div[data-testid="stButton"] button {
            min-height: 2.8rem;
            border: 1px solid rgba(162, 174, 218, 0.18);
            border-radius: 12px;
            background: rgba(25, 33, 54, 0.88);
            color: #e8ebf8;
            transition: transform 160ms ease, border-color 160ms ease, box-shadow 160ms ease, background 160ms ease;
        }

        div[data-testid="stButton"] button:hover {
            transform: translateY(-1px);
            border-color: rgba(164, 148, 255, 0.58);
            background: rgba(39, 46, 75, 0.96);
            box-shadow: 0 8px 25px rgba(87, 74, 174, 0.18);
        }

        div[data-testid="stButton"] button[kind="primary"] {
            border-color: rgba(172, 157, 255, 0.65);
            background: linear-gradient(105deg, #715be1, #635bd0 55%, #347f9c);
            color: white;
            font-weight: 700;
            box-shadow: 0 8px 27px rgba(102, 83, 211, 0.24);
        }

        div[data-testid="stButton"] button[kind="primary"]:hover {
            box-shadow: 0 10px 34px rgba(119, 100, 238, 0.38);
        }

        .example-label {
            margin: 0.3rem 0 0.65rem;
            color: #9ea8c2;
            font-size: 0.76rem;
            font-weight: 600;
            letter-spacing: 0.04em;
        }

        .st-key-result-card {
            margin-top: 1.15rem;
            border-color: rgba(151, 134, 255, 0.32);
            background:
                radial-gradient(ellipse at 95% 0%, rgba(115, 217, 237, 0.09), transparent 25rem),
                linear-gradient(130deg, rgba(25, 28, 56, 0.91), rgba(16, 23, 40, 0.91));
        }

        .result-word {
            margin: 0.15rem 0 0.55rem;
            color: #c2b7ff;
            font-family: "Inter", "Segoe UI", sans-serif;
            font-size: clamp(2.3rem, 5vw, 3.6rem);
            font-weight: 800;
            letter-spacing: -0.055em;
            line-height: 1.1;
            overflow-wrap: anywhere;
        }

        .result-sentence {
            margin-top: 0.5rem;
            padding: 0.95rem 1.1rem;
            border: 1px solid rgba(162, 176, 220, 0.13);
            border-radius: 12px;
            background: rgba(5, 10, 20, 0.46);
            color: #d6dcef;
            line-height: 1.7;
            overflow-wrap: anywhere;
        }

        .flow-node, .architecture-node {
            position: relative;
            height: 100%;
            min-height: 105px;
            padding: 0.95rem 0.85rem;
            border: 1px solid rgba(159, 173, 221, 0.15);
            border-radius: 15px;
            background: linear-gradient(150deg, rgba(28, 36, 59, 0.76), rgba(14, 20, 36, 0.72));
        }

        .st-key-pipeline-flow .stColumn:not(:last-child) .flow-node::after,
        .st-key-architecture-flow .stColumn:not(:last-child) .architecture-node::after {
            position: absolute;
            top: 43%;
            right: -0.55rem;
            z-index: 1;
            color: #9a8cff;
            content: "→";
            font-size: 1rem;
        }

        .flow-index {
            color: #9585ff;
            font-size: 0.66rem;
            font-weight: 700;
            letter-spacing: 0.12em;
        }

        .flow-title, .architecture-title {
            margin: 0.35rem 0 0.22rem;
            color: #eff2ff;
            font-size: 0.84rem;
            font-weight: 700;
        }

        .flow-detail, .architecture-detail {
            color: #a5aec6;
            font-size: 0.72rem;
            line-height: 1.45;
            overflow-wrap: anywhere;
        }

        .st-key-pipeline-flow .stHorizontalBlock,
        .st-key-architecture-flow .stHorizontalBlock {
            flex-wrap: wrap;
            gap: 0.55rem;
        }

        .st-key-pipeline-flow .stColumn {
            min-width: 108px !important;
            flex: 1 1 108px !important;
        }

        .st-key-architecture-flow .stColumn {
            min-width: 150px !important;
            flex: 1 1 150px !important;
        }

        .st-key-pipeline-flow .stColumn,
        .st-key-architecture-flow .stColumn {
            margin-bottom: 0.55rem;
        }

        @media (max-width: 1000px) {
            .st-key-pipeline-flow .stColumn {
                min-width: calc(25% - 0.5rem) !important;
                flex: 1 1 calc(25% - 0.5rem) !important;
            }

            .st-key-architecture-flow .stColumn {
                min-width: calc(33.333% - 0.5rem) !important;
                flex: 1 1 calc(33.333% - 0.5rem) !important;
            }

            .st-key-pipeline-flow .flow-node::after,
            .st-key-architecture-flow .architecture-node::after {
                display: none;
            }
        }

        .flow-title {
            font-size: 0.74rem;
            overflow-wrap: normal;
            word-break: keep-all;
        }

        .st-key-pipeline-panel pre, .st-key-pipeline-panel code {
            color: #c7d4f3;
        }

        .history-entry {
            margin: 0.75rem 0;
            padding: 1rem 1.1rem;
            border: 1px solid rgba(159, 173, 221, 0.13);
            border-radius: 14px;
            background: rgba(8, 13, 25, 0.46);
        }

        @media (max-width: 640px) {
            .main .block-container {
                padding: 1rem 1rem 2.5rem;
            }

            .hero {
                border-radius: 21px;
            }

            .hero::before {
                mask-image: linear-gradient(90deg, transparent 20%, #000 100%);
            }

            .st-key-pipeline-flow .flow-node::after,
            .st-key-architecture-flow .architecture-node::after {
                display: none;
            }

            .st-key-pipeline-flow .stColumn {
                min-width: calc(50% - 0.5rem) !important;
                flex: 1 1 calc(50% - 0.5rem) !important;
            }

            .st-key-architecture-flow .stColumn {
                min-width: 100% !important;
                flex: 1 1 100% !important;
            }

            .st-key-prediction-studio, .st-key-result-card, .st-key-pipeline-panel,
            .st-key-architecture-panel, .st-key-history-panel {
                border-radius: 17px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


try:
    model, tokenizer, input_length = load_assets()
except Exception:
    LOGGER.exception("Unable to load the next-word prediction assets.")
    st.error(
        "The prediction model could not be loaded. Confirm that "
        "`next_word.h5` and `tokenizere.pickle` are present beside `main.py`, "
        "then check the server logs for details."
    )
    st.stop()


st.session_state.setdefault("input_text", DEFAULT_PROMPT)
st.session_state.setdefault("prediction", None)
st.session_state.setdefault("prediction_history", [])

render_styles()

st.markdown(
    """
    <section class="hero">
        <div class="hero-eyebrow">NLP LAB / LANGUAGE INTELLIGENCE</div>
        <h1>Predict What<br><span>Comes Next.</span></h1>
        <p class="hero-copy">
            Explore how recurrent neural networks use the context in a sequence
            to predict its next word, one token at a time.
        </p>
        <div class="hero-badges">
            <span class="hero-badge"><i class="status-dot"></i> Trained LSTM model loaded</span>
            <span class="hero-badge">LSTM-based language prediction</span>
        </div>
    </section>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="section-kicker">01 / INTERACTIVE</div>', unsafe_allow_html=True)
st.markdown('<h2 class="section-title">Prediction Studio</h2>', unsafe_allow_html=True)
st.markdown(
    '<p class="section-copy">Give the model a little context and let it complete the next step.</p>',
    unsafe_allow_html=True,
)

with st.container(key="prediction-studio"):
    st.text_area(
        "Your text sequence",
        key="input_text",
        height=155,
        placeholder=DEFAULT_PROMPT,
        help="The tokenizer recognizes words from the model's training vocabulary.",
    )
    utility_left, utility_right = st.columns([5, 1])
    with utility_left:
        st.caption(f"{len(st.session_state.input_text)} characters")
    with utility_right:
        st.button(
            "Clear input",
            key="clear-input",
            on_click=clear_current_input,
            width="stretch",
        )

    st.markdown('<div class="example-label">TRY AN EXAMPLE</div>', unsafe_allow_html=True)
    prompt_columns = st.columns(len(EXAMPLE_PROMPTS))
    for index, (column, prompt) in enumerate(zip(prompt_columns, EXAMPLE_PROMPTS)):
        with column:
            st.button(
                prompt,
                key=f"example-{index}",
                on_click=choose_prompt,
                args=(prompt,),
                width="stretch",
            )

    run_prediction = st.button(
        "Predict Next Word",
        key="predict-next-word",
        type="primary",
        icon=":material/arrow_forward:",
        width="stretch",
    )

    if run_prediction:
        st.session_state["prediction"] = None
        submitted_text = st.session_state.input_text
        if not submitted_text.strip():
            st.warning("Enter a sentence or sequence before asking for a prediction.")
        else:
            try:
                started_at = time.perf_counter()
                with st.spinner("Reading the sequence through the LSTM…"):
                    predicted_word, token_ids, model_token_ids, padded_sequence = (
                        predict_next_word(
                            model, tokenizer, submitted_text, input_length
                        )
                    )
                if not model_token_ids:
                    st.warning(
                        "No words in that input were recognized by this tokenizer. "
                        "Try one of the examples or use words from the model's vocabulary."
                    )
                elif predicted_word is None:
                    st.warning(
                        "The model selected an output that has no matching word in "
                        "the saved tokenizer."
                    )
                else:
                    latency_ms = (time.perf_counter() - started_at) * 1000
                    completed_sentence = (
                        f"{submitted_text.rstrip()} {predicted_word}".strip()
                    )
                    st.session_state["prediction"] = {
                        "input": submitted_text,
                        "word": predicted_word,
                        "completed": completed_sentence,
                        "latency_ms": latency_ms,
                        "token_ids": token_ids,
                        "model_token_ids": model_token_ids,
                        "padded_sequence": padded_sequence[0].tolist(),
                    }
                    st.session_state.prediction_history.insert(
                        0,
                        {
                            "input": submitted_text,
                            "word": predicted_word,
                            "completed": completed_sentence,
                            "timestamp": datetime.now().strftime("%H:%M:%S"),
                        },
                    )
                    del st.session_state.prediction_history[HISTORY_LIMIT:]
            except Exception:
                LOGGER.exception("Next-word inference failed.")
                st.error(
                    "The model could not process this sequence. Please try again; "
                    "the server logs contain the technical details."
                )


prediction = st.session_state.prediction
if prediction:
    st.markdown('<div class="section-kicker">02 / MODEL OUTPUT</div>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">Your prediction</h2>', unsafe_allow_html=True)
    with st.container(key="result-card"):
        st.caption(f'Original input · {prediction["input"]}')
        st.markdown(
            f'<div class="micro-label">NEXT WORD</div>'
            f'<div class="result-word">{html.escape(prediction["word"])}</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="micro-label">COMPLETED SEQUENCE</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="result-sentence">{html.escape(prediction["completed"])}</div>',
            unsafe_allow_html=True,
        )
        result_meta_left, result_meta_right = st.columns([1, 5])
        with result_meta_left:
            st.metric("Inference", f'{prediction["latency_ms"]:.1f} ms')
        with result_meta_right:
            st.button(
                "Start another prediction",
                key="reset-result",
                on_click=clear_current_input,
            )


st.markdown('<div class="section-kicker">03 / INSIDE THE MODEL</div>', unsafe_allow_html=True)
st.markdown('<h2 class="section-title">From text to prediction</h2>', unsafe_allow_html=True)
st.markdown(
    '<p class="section-copy">The sequence below reflects the loaded model and tokenizer, not training metrics.</p>',
    unsafe_allow_html=True,
)

with st.container(key="pipeline-panel"):
    metadata_columns = st.columns(4)
    metadata = (
        ("Model input", str(model.input_shape)),
        ("Tokenizer vocabulary", f"{len(tokenizer.word_index):,} words"),
        ("Output classes", f"{model.output_shape[-1]:,}"),
        ("Input positions", str(input_length)),
    )
    for column, (label, value) in zip(metadata_columns, metadata):
        with column:
            st.metric(label, value)

    layer_type_counts = {}
    layers = []
    for layer in model.layers:
        layer_type = layer.__class__.__name__
        layer_type_counts[layer_type] = layer_type_counts.get(layer_type, 0) + 1
        layers.append(describe_layer(layer, layer_type_counts[layer_type]))
    embedding_details = next(
        (details for title, details, _ in layers if title == "Embedding"),
        "Embedding dimensions are not available",
    )
    lstm_units = [
        str(layer.get_config().get("units", "?"))
        for layer in model.layers
        if layer.__class__.__name__ == "LSTM"
    ]
    dense_activation = next(
        (
            layer.get_config().get("activation", "linear")
            for layer in model.layers
            if layer.__class__.__name__ == "Dense"
        ),
        "unknown",
    )
    pipeline_steps = (
        ("Text input", "sentence"),
        ("Tokenization", "word IDs"),
        ("Pre-padding", f"{input_length} positions"),
        ("Embedding", embedding_details.split(" · ")[0]),
        ("LSTM layers", " + ".join(lstm_units) + " units" if lstm_units else "model layers"),
        (
            "Softmax output" if dense_activation == "softmax" else "Output layer",
            f"{model.output_shape[-1]:,} classes",
        ),
        (
            "Predicted word",
            prediction["word"] if prediction else "argmax index",
        ),
    )

    st.markdown('<div class="micro-label">ACTUAL INFERENCE PIPELINE</div>', unsafe_allow_html=True)
    with st.container(key="pipeline-flow"):
        flow_columns = st.columns(len(pipeline_steps))
        for index, (column, (title, detail)) in enumerate(
            zip(flow_columns, pipeline_steps), start=1
        ):
            with column:
                st.markdown(
                    f'<div class="flow-node">'
                    f'<div class="flow-index">STEP {index:02d}</div>'
                    f'<div class="flow-title">{html.escape(title)}</div>'
                    f'<div class="flow-detail">{html.escape(detail)}</div>'
                    f"</div>",
                    unsafe_allow_html=True,
                )

    if prediction:
        st.markdown(
            '<div class="micro-label">TOKEN IDS PASSED TO THE MODEL</div>',
            unsafe_allow_html=True,
        )
        token_labels = [
            f'{tokenizer.index_word.get(token_id, "?")} [{token_id}]'
            for token_id in prediction["model_token_ids"]
        ]
        if len(prediction["token_ids"]) > input_length:
            st.caption(
                f"Showing the last {input_length} recognized tokens; earlier "
                "tokens are truncated to match the model's fixed input."
            )
        st.code("  →  ".join(token_labels), language=None)
        st.markdown(
            f'<div class="micro-label">PRE-PADDED SEQUENCE · {input_length} POSITIONS</div>',
            unsafe_allow_html=True,
        )
        st.code(str(prediction["padded_sequence"]), language="text")
    else:
        st.caption("Run a prediction to inspect the exact token IDs and padded sequence.")


st.markdown('<div class="section-kicker">04 / MODEL BLUEPRINT</div>', unsafe_allow_html=True)
st.markdown('<h2 class="section-title">Architecture explorer</h2>', unsafe_allow_html=True)
st.markdown(
    '<p class="section-copy">Layer names and dimensions are read directly from the saved Keras model.</p>',
    unsafe_allow_html=True,
)

with st.container(key="architecture-panel"):
    with st.container(key="architecture-flow"):
        architecture_columns = st.columns(len(layers))
        for index, (column, (title, details, _layer_type)) in enumerate(
            zip(architecture_columns, layers), start=1
        ):
            with column:
                st.markdown(
                    f'<div class="architecture-node">'
                    f'<div class="flow-index">LAYER {index:02d}</div>'
                    f'<div class="architecture-title">{html.escape(title)}</div>'
                    f'<div class="architecture-detail">{html.escape(details)}</div>'
                    f"</div>",
                    unsafe_allow_html=True,
                )


st.markdown('<div class="section-kicker">05 / THIS SESSION</div>', unsafe_allow_html=True)
st.markdown('<h2 class="section-title">Prediction history</h2>', unsafe_allow_html=True)
st.markdown(
    '<p class="section-copy">Recent results stay in this browser session only.</p>',
    unsafe_allow_html=True,
)

with st.container(key="history-panel"):
    history = st.session_state.prediction_history
    if history:
        history_header, history_action = st.columns([5, 1])
        with history_header:
            st.caption(f"{len(history)} recent prediction{'s' if len(history) != 1 else ''}")
        with history_action:
            st.button(
                "Clear history",
                key="clear-history",
                on_click=clear_history,
                width="stretch",
            )

        for index, item in enumerate(history):
            st.markdown(
                f'<div class="history-entry">'
                f'<div class="micro-label">PREDICTION {len(history) - index:02d} · '
                f'{html.escape(item["timestamp"])}</div>'
                f'<div style="margin-top:.45rem;color:#aab3ca;font-size:.87rem">'
                f'{html.escape(item["input"])}</div>'
                f'<div style="margin-top:.35rem;color:#d7d1ff;font-weight:700">'
                f'{html.escape(item["completed"])}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )
            st.button(
                "Reuse this input",
                key=f"reuse-history-{index}",
                on_click=reuse_history_input,
                args=(item["input"],),
            )
    else:
        st.caption("Your predictions will appear here after the first successful run.")
