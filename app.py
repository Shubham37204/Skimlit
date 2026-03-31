import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import streamlit as st
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import tensorflow_hub as hub

import numpy as np
import pandas as pd
import string
import subprocess


# ─────────────────────────────────────────────────────────────────────────────
# Custom layer to wrap USE — replaces Lambda so it survives save/load
# ─────────────────────────────────────────────────────────────────────────────
class USELayer(tf.keras.layers.Layer):
    """Wraps the Universal Sentence Encoder KerasLayer as a proper sublayer."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.use = hub.KerasLayer(
            "https://tfhub.dev/google/universal-sentence-encoder/4",
            trainable=False,
            name="universal_sentence_encoder"
        )

    def call(self, inputs):
        return self.use(inputs)

    def get_config(self):
        return super().get_config()

# ─────────────────────────────────────────────────────────────────────────────
# Constants  (all taken directly from the notebook)
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR   = "pubmed-rct/PubMed_20k_RCT_numbers_replaced_with_at_sign/"
MODEL_PATH = "skimlit_tribrid_model.keras"

# notebook cell 21 — LabelEncoder sorts alphabetically
CLASS_NAMES = ['BACKGROUND', 'CONCLUSIONS', 'METHODS', 'OBJECTIVE', 'RESULTS']

st.set_page_config(page_title="SkimLit 🔬", page_icon="🔬", layout="wide")

# ─────────────────────────────────────────────────────────────────────────────
# Notebook helper functions  (cells 7, 10, 56)
# ─────────────────────────────────────────────────────────────────────────────

def get_lines(filename):
    """notebook cell 7"""
    with open(filename, "r") as f:
        return f.readlines()


def preprocess_text_with_line_numbers(filename):
    """notebook cell 10"""
    input_lines = get_lines(filename)
    abstract_lines = ""
    abstract_samples = []
    for line in input_lines:
        if line.startswith("###"):
            abstract_lines = ""
        elif line.isspace():
            abstract_line_split = abstract_lines.splitlines()
            for abstract_line_number, abstract_line in enumerate(abstract_line_split):
                line_data = {}
                target_text_split = abstract_line.split("\t")
                line_data["target"]      = target_text_split[0]
                line_data["text"]        = target_text_split[1].lower()
                line_data["line_number"] = abstract_line_number
                line_data["total_lines"] = len(abstract_line_split) - 1
                abstract_samples.append(line_data)
        else:
            abstract_lines += line
    return abstract_samples


def split_chars(text):
    """notebook cell 56"""
    return " ".join(list(text))


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Clone dataset  (notebook cell 2)
# ─────────────────────────────────────────────────────────────────────────────

def clone_dataset(log):
    if os.path.exists(DATA_DIR):
        log.write("Dataset already present, skipping clone.\n")
        return True
    log.write("Cloning pubmed-rct dataset from GitHub…\n")
    result = subprocess.run(
        ["git", "clone", "https://github.com/Franck-Dernoncourt/pubmed-rct.git"],
        capture_output=True, text=True
    )
    log.write(result.stdout)
    if result.returncode != 0:
        log.write(f"ERROR: {result.stderr}\n")
        return False
    log.write("Clone complete.\n")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Full training pipeline  (notebook cells 10-92)
# ─────────────────────────────────────────────────────────────────────────────

def train_and_save(log, epochs=3):
    from tensorflow.keras.layers import (
        TextVectorization, Embedding, Input, Dense,
        Dropout, Concatenate, Bidirectional, LSTM
    )
    from sklearn.preprocessing import OneHotEncoder, LabelEncoder

    # ── Load & preprocess data  (cells 10-11) ────────────────────────────────
    log.write("\n[1/7] Loading and preprocessing data…\n")
    train_samples = preprocess_text_with_line_numbers(DATA_DIR + "train.txt")
    val_samples   = preprocess_text_with_line_numbers(DATA_DIR + "dev.txt")

    train_df = pd.DataFrame(train_samples)
    val_df   = pd.DataFrame(val_samples)

    train_sentences = train_df["text"].tolist()   # cell 15
    val_sentences   = val_df["text"].tolist()

    # ── Encode labels  (cells 17, 19) ────────────────────────────────────────
    log.write("[2/7] Encoding labels…\n")
    one_hot_encoder = OneHotEncoder(sparse_output=False)
    train_labels_one_hot = one_hot_encoder.fit_transform(
        train_df["target"].to_numpy().reshape(-1, 1))
    val_labels_one_hot = one_hot_encoder.transform(
        val_df["target"].to_numpy().reshape(-1, 1))

    label_encoder = LabelEncoder()
    label_encoder.fit_transform(train_df["target"].to_numpy())   # cell 19

    # ── Char-level features  (cells 56-59) ───────────────────────────────────
    log.write("[3/7] Building char vectorizer…\n")
    train_chars = [split_chars(s) for s in train_sentences]   # cell 56
    val_chars   = [split_chars(s) for s in val_sentences]

    char_lens          = [len(s) for s in train_sentences]    # cell 57
    output_seq_char_len = int(np.percentile(char_lens, 95))

    alphabet        = string.ascii_lowercase + string.digits + string.punctuation  # cell 58
    NUM_CHAR_TOKENS = len(alphabet) + 2                                            # cell 59

    char_vectorizer = TextVectorization(
        max_tokens=NUM_CHAR_TOKENS,
        output_sequence_length=output_seq_char_len,
        standardize="lower_and_strip_punctuation",
        name="char_vectorizer"
    )
    char_vectorizer.adapt(train_chars)   # adapt to actual training chars

    # ── Char embedding  (cell 62) ─────────────────────────────────────────────
    char_embed = Embedding(
        input_dim=NUM_CHAR_TOKENS,
        output_dim=25,
        mask_zero=False,
        name="char_embed"
    )

    # ── Positional one-hot features  (cells 76, 78) ───────────────────────────
    log.write("[4/7] Creating positional features…\n")
    train_line_numbers_one_hot  = tf.one_hot(train_df["line_number"].to_numpy(), depth=15)
    val_line_numbers_one_hot    = tf.one_hot(val_df["line_number"].to_numpy(),   depth=15)
    train_total_lines_one_hot   = tf.one_hot(train_df["total_lines"].to_numpy(), depth=20)
    val_total_lines_one_hot     = tf.one_hot(val_df["total_lines"].to_numpy(),   depth=20)

    # ── Universal Sentence Encoder  (cell 80) ────────────────────────────────
    log.write("[5/7] Loading Universal Sentence Encoder…\n")
    # USELayer is a proper sublayer — survives model save/load without Lambda issues

    # ── Build tribrid model  (cell 81) ────────────────────────────────────────
    log.write("[6/7] Building tribrid model…\n")

    # 1. Token branch — use USELayer instead of Lambda(hub.KerasLayer)
    token_inputs     = Input(shape=[], dtype="string", name="token_inputs")
    token_embeddings = USELayer(name="use_layer")(token_inputs)
    token_outputs    = Dense(128, activation="relu")(token_embeddings)
    token_model      = tf.keras.Model(inputs=token_inputs, outputs=token_outputs)

    # 2. Char branch
    char_inputs    = Input(shape=(1,), dtype="string", name="char_inputs")
    char_vectors   = char_vectorizer(char_inputs)
    char_embeddings = char_embed(char_vectors)
    char_bi_lstm   = Bidirectional(LSTM(32))(char_embeddings)
    char_model     = tf.keras.Model(inputs=char_inputs, outputs=char_bi_lstm)

    # 3. Line-number branch
    line_number_inputs = Input(shape=(15,), dtype=tf.int32, name="line_number_input")
    x = Dense(32, activation="relu")(line_number_inputs)
    line_number_model = tf.keras.Model(inputs=line_number_inputs, outputs=x)

    # 4. Total-lines branch
    total_lines_inputs = Input(shape=(20,), dtype=tf.int32, name="total_lines_input")
    y = Dense(32, activation="relu")(total_lines_inputs)
    total_line_model = tf.keras.Model(inputs=total_lines_inputs, outputs=y)

    # 5. Hybrid: token + char
    combined = Concatenate(name="token_char_hybrid_embedding")(
        [token_model.output, char_model.output])
    z = Dense(256, activation="relu")(combined)
    z = Dropout(0.5)(z)

    # 6. Tribrid: add positional
    z = Concatenate(name="token_char_positional_embedding")(
        [line_number_model.output, total_line_model.output, z])

    # 7. Output
    output_layer = Dense(len(CLASS_NAMES), activation="softmax", name="output_layer")(z)

    # 8. Assemble  (cell 81 input order)
    model_5 = tf.keras.Model(
        inputs=[line_number_model.input, total_line_model.input,
                token_model.input, char_model.input],
        outputs=output_layer
    )

    # ── Compile  (cell 83) ────────────────────────────────────────────────────
    model_5.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.2),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=["accuracy"]
    )

    # ── Build tf.data datasets  (cells 84) ───────────────────────────────────
    train_pos_char_token_data = tf.data.Dataset.from_tensor_slices((
        train_line_numbers_one_hot,
        train_total_lines_one_hot,
        train_sentences,
        train_chars
    ))
    train_pos_char_token_labels  = tf.data.Dataset.from_tensor_slices(train_labels_one_hot)
    train_pos_char_token_dataset = tf.data.Dataset.zip(
        (train_pos_char_token_data, train_pos_char_token_labels)
    ).batch(32).prefetch(tf.data.AUTOTUNE)

    val_pos_char_token_data = tf.data.Dataset.from_tensor_slices((
        val_line_numbers_one_hot,
        val_total_lines_one_hot,
        val_sentences,
        val_chars
    ))
    val_pos_char_token_labels  = tf.data.Dataset.from_tensor_slices(val_labels_one_hot)
    val_pos_char_token_dataset = tf.data.Dataset.zip(
        (val_pos_char_token_data, val_pos_char_token_labels)
    ).batch(32).prefetch(tf.data.AUTOTUNE)

    # ── Train  (cell 85) ──────────────────────────────────────────────────────
    log.write(f"[7/7] Training for {epochs} epoch(s)…\n")
    model_5.fit(
        train_pos_char_token_dataset,
        steps_per_epoch=int(0.1 * len(train_pos_char_token_dataset)),
        epochs=epochs,
        validation_data=val_pos_char_token_dataset,
        validation_steps=int(0.1 * len(val_pos_char_token_dataset)),
        verbose=0
    )

    # ── Save  (cell 92) ───────────────────────────────────────────────────────
    model_5.save(MODEL_PATH)
    log.write(f"\nModel saved to '{MODEL_PATH}'\n")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Load trained model  (cell 93)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_trained_model():
    return tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={"USELayer": USELayer},
        safe_mode=False
    )


# ─────────────────────────────────────────────────────────────────────────────
# Sentence splitting  (notebook cell 105 — spaCy sentencizer)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def get_nlp():
    try:
        from spacy.lang.en import English
        nlp = English()
        nlp.add_pipe("sentencizer")
        return nlp
    except ImportError:
        return None


def split_abstract_into_sentences(text):
    """notebook cell 105"""
    nlp = get_nlp()
    if nlp:
        doc = nlp(text)
        return [str(sent).strip() for sent in doc.sents if str(sent).strip()]
    import re
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]


# ─────────────────────────────────────────────────────────────────────────────
# Prepare inference inputs  (notebook cells 106-108)
# ─────────────────────────────────────────────────────────────────────────────

def prepare_inputs(abstract_lines):
    """Mirrors notebook cells 106-108 exactly."""
    total_lines_in_sample = len(abstract_lines)

    # cell 106
    sample_lines = [
        {"text": str(line), "line_number": i, "total_lines": total_lines_in_sample - 1}
        for i, line in enumerate(abstract_lines)
    ]

    # cell 107
    line_numbers_one_hot  = tf.one_hot([d["line_number"]  for d in sample_lines], depth=15)
    total_lines_one_hot   = tf.one_hot([d["total_lines"]  for d in sample_lines], depth=20)

    # cell 108
    abstract_chars = [split_chars(line) for line in abstract_lines]

    # cell 109 — exact tuple order the model expects
    return (
        line_numbers_one_hot,
        total_lines_one_hot,
        tf.constant(abstract_lines),
        tf.constant(abstract_chars),
    )


# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────

COLOUR = {
    "BACKGROUND": "🟦",
    "OBJECTIVE":  "🟨",
    "METHODS":    "🟩",
    "RESULTS":    "🟧",
    "CONCLUSIONS":"🟥",
}


def main():
    st.title("🔬 SkimLit")
    st.markdown("Classify sentences in a medical abstract — powered entirely by the notebook pipeline.")
    st.markdown("---")

    with st.sidebar:
        st.header("🔬 SkimLit Info")

        st.markdown("### 📌 Model")
        st.write("Tribrid NLP (Token + Char + Positional)")

        st.markdown("### 📊 Dataset")
        st.write("PubMed 20K RCT")

        st.markdown("### 🏷️ Labels")
        st.markdown(
            "- 🟦 BACKGROUND\n"
            "- 🟨 OBJECTIVE\n"
            "- 🟩 METHODS\n"
            "- 🟧 RESULTS\n"
            "- 🟥 CONCLUSIONS"
        )

    # ── Training section ──────────────────────────────────────────────────────
    model_exists = os.path.exists(MODEL_PATH)

    if model_exists:
        st.success(f"✅ Trained model found: `{MODEL_PATH}`")
    else:
        st.warning("No trained model found. Click **Train Model** to run the full pipeline.")

    with st.expander("⚙️ Train Model (runs full notebook pipeline)", expanded=not model_exists):
        epochs = st.slider("Training epochs", min_value=1, max_value=5, value=3)
        st.caption(
            "This will: clone the PubMed dataset → preprocess → build the tribrid model → train → save."
        )
        if st.button("🚀 Train Model", type="primary"):
            log_area = st.empty()
            log_text = []

            class StreamLog:
                def write(self, msg):
                    log_text.append(msg)
                    log_area.code("".join(log_text))

            log = StreamLog()
            with st.spinner("Running training pipeline…"):
                try:
                    ok = clone_dataset(log)
                    if ok:
                        ok = train_and_save(log, epochs=epochs)
                    if ok:
                        st.success("✅ Training complete! Refresh the page to classify abstracts.")
                        st.cache_resource.clear()
                except Exception as e:
                    st.error(f"Training failed: {e}")

    st.markdown("---")

    # ── Inference section ─────────────────────────────────────────────────────
    if not os.path.exists(MODEL_PATH):
        st.info("Train the model first, then come back here to classify abstracts.")
        return

    try:
        model = load_trained_model()
    except Exception as e:
        st.error(f"Could not load model: {e}")
        return

    st.markdown("### Paste your abstract")
    abstract_text = st.text_area(
        "Abstract Text:",
        height=250,
        placeholder="Paste a medical research abstract here…"
    )

    if st.button("🔍 Classify Sentences", type="primary"):
        if not abstract_text.strip():
            st.warning("Please enter an abstract first.")
            return

        # cell 105 — sentence splitting
        abstract_lines = split_abstract_into_sentences(abstract_text)
        if not abstract_lines:
            st.error("No sentences detected.")
            return

        with st.spinner(f"Classifying {len(abstract_lines)} sentences…"):
            try:
                inputs = prepare_inputs(abstract_lines)

                # cell 109 — predict
                pred_probs = model.predict(x=inputs, verbose=0)

                # cell 110 — decode
                pred_classes = tf.argmax(pred_probs, axis=1).numpy()
                pred_labels  = [CLASS_NAMES[i] for i in pred_classes]
                confidences  = np.max(pred_probs, axis=1)

            except Exception as e:
                st.error(f"Prediction error: {e}")
                return

        st.success(f"✅ Classified {len(abstract_lines)} sentences")
        st.markdown("---")

        # cell 111 — display
        for i, (line, label, conf) in enumerate(zip(abstract_lines, pred_labels, confidences)):
            st.markdown(f"**{i+1}.** {line}")
            st.caption(f"{COLOUR.get(label,'⬜')} **{label}** — confidence: {conf:.1%}")
            st.markdown("---")

        df = pd.DataFrame({
            "Sentence":   abstract_lines,
            "Label":      pred_labels,
            "Confidence": confidences.round(4),
        })
    st.caption("Powered by TensorFlow · Universal Sentence Encoder · PubMed 20K RCT")


if __name__ == "__main__":
    main()
