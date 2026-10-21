import streamlit as st
import nltk
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from nltk.wsd import lesk
from nltk import pos_tag

st.set_page_config(page_title="WordSense AI Lab", page_icon="🧠", layout="wide")

st.markdown("""
<style>
.stApp {
  background: linear-gradient(135deg, #f3e8ff 0%, #e0f2fe 48%, #fff1f2 100%);
  color: #18213b;
}
.block-container {padding-top: 1.8rem; max-width: 1200px;}
.hero {
  padding: 2rem 2.2rem; border-radius: 24px; color: white;
  background: linear-gradient(120deg, #5b21b6, #2563eb, #db2777);
  box-shadow: 0 12px 35px rgba(58, 45, 130, .22); margin-bottom: 1.2rem;
}
.hero h1 {font-size: 2.5rem; margin: 0; color: white;}
.hero p {font-size: 1.05rem; opacity: .95; margin-bottom: 0;}
div[data-testid="stMetric"] {
  background: rgba(255,255,255,.86); border: 1px solid #ddd6fe;
  border-radius: 16px; padding: 14px; box-shadow: 0 4px 14px rgba(30,41,59,.06);
}
.stButton>button {
  border: 0; border-radius: 12px; color: white; font-weight: 700;
  background: linear-gradient(90deg,#7c3aed,#2563eb,#db2777);
  padding: .65rem 1rem;
}
.stButton>button:hover {filter: brightness(1.08); color: white;}
.panel {
  background: rgba(255,255,255,.88); padding: 1rem 1.2rem; border-radius: 16px;
  border: 1px solid #ddd6fe; margin-bottom: .7rem;
}
.small-note {color:#475569; font-size:.9rem;}
h2, h3 {color:#4c1d95;}
</style>
<div class="hero">
  <h1>🧠 WordSense AI Lab</h1>
  <p>Word Sense Disambiguation using WordNet, lexical relations, and the Simplified Lesk algorithm.</p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def setup_nltk():
    # Download required NLTK resources on first run if they are not already installed.
    for package in ("wordnet", "omw-1.4", "punkt", "punkt_tab", "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng"):
        try:
            nltk.download(package, quiet=True)
        except Exception:
            pass
    return True

setup_nltk()

def get_wordnet_pos(tag):
    if tag.startswith("N"): return wn.NOUN
    if tag.startswith("V"): return wn.VERB
    if tag.startswith("J"): return wn.ADJ
    if tag.startswith("R"): return wn.ADV
    return None

def lexical_info(synset):
    return {
        "Synset": synset.name(),
        "Part of speech": synset.pos(),
        "Definition": synset.definition(),
        "Examples": "; ".join(synset.examples()) or "No example stored in WordNet",
        "Synonyms": ", ".join(sorted({l.name().replace("_", " ") for l in synset.lemmas()})),
        "Hypernyms (broader terms)": ", ".join(s.name().replace("_", " ") for s in synset.hypernyms()) or "None listed",
        "Hyponyms (narrower terms)": ", ".join(s.name().replace("_", " ") for s in synset.hyponyms()[:8]) or "None listed",
    }

with st.sidebar:
    st.markdown("## 🎛️ Workspace")
    st.write("Explore a word's meanings, compare lexical relationships, and run a simple WSD prediction.")
    st.markdown("---")
    st.markdown("### 🧪 Example sentences")
    examples = {
        "Bank (river/finance)": "She sat on the bank of the river and watched the water.",
        "Bat (animal/sports)": "The bat flew out of the cave at night.",
        "Light (illumination/weight)": "This bag is very light and easy to carry.",
        "Plant (living thing/factory)": "The plant needs water and sunlight to grow."
    }
    chosen = st.selectbox("Load a sample", ["Choose one..."] + list(examples.keys()))
    st.caption("Tip: choose a sample, then copy its sentence into the WSD tab.")

tab1, tab2, tab3 = st.tabs(["🔎 Sense Explorer", "🧩 Disambiguation", "📘 About & Report"])

with tab1:
    st.markdown("## Explore WordNet senses")
    c1, c2 = st.columns([2,1])
    with c1:
        word = st.text_input("Enter an ambiguous word", value="bank", placeholder="e.g., bank, bat, plant")
    with c2:
        pos_choice = st.selectbox("Part of speech", ["All", "Noun", "Verb", "Adjective", "Adverb"])
    pos_map = {"Noun": wn.NOUN, "Verb": wn.VERB, "Adjective": wn.ADJ, "Adverb": wn.ADV}
    if st.button("Find meanings", key="find_senses"):
        if not word.strip():
            st.warning("Please enter a word.")
        else:
            try:
                senses = wn.synsets(word.strip().replace(" ", "_"))
                if pos_choice != "All":
                    senses = [s for s in senses if s.pos() == pos_map[pos_choice]]
                st.session_state["senses"] = senses
                st.session_state["sense_word"] = word.strip()
            except LookupError:
                st.error("WordNet is not available. Check your internet connection and restart the app.")
    senses = st.session_state.get("senses", [])
    if senses:
        st.success(f"Found {len(senses)} sense(s) for “{st.session_state.get('sense_word', word)}”.")
        m1, m2, m3 = st.columns(3)
        m1.metric("Senses found", len(senses))
        m2.metric("Unique definitions", len({s.definition() for s in senses}))
        m3.metric("Parts of speech", len({s.pos() for s in senses}))
        for i, syn in enumerate(senses, 1):
            info = lexical_info(syn)
            with st.expander(f"Sense {i}: {info['Synonyms']} — {info['Definition']}", expanded=(i == 1)):
                st.write(f"**Synset:** `{info['Synset']}`  ·  **POS:** `{info['Part of speech']}`")
                st.write("**Definition:**", info["Definition"])
                st.write("**Examples:**", info["Examples"])
                st.write("**Synonyms:**", info["Synonyms"])
                st.write("**Hypernyms:**", info["Hypernyms (broader terms)"])
                st.write("**Hyponyms:**", info["Hyponyms (narrower terms)"])
        report = "\n\n".join([f"{i}. {s.name()} — {s.definition()}" for i,s in enumerate(senses,1)])
        st.download_button("⬇️ Download sense inventory (.txt)", report, file_name=f"{word.strip()}_sense_inventory.txt")
    else:
        st.markdown('<div class="panel">💡 <b>Try it:</b> enter <i>bank</i>, <i>bat</i>, <i>light</i>, or <i>plant</i> and click <b>Find meanings</b>.</div>', unsafe_allow_html=True)

with tab2:
    st.markdown("## Context-aware Word Sense Disambiguation")
    st.write("The Simplified Lesk algorithm selects the WordNet sense whose definition/examples overlap most with the context words.")
    default_sentence = examples.get(chosen, "She deposited money at the bank after walking beside the river.")
    sentence = st.text_area("Enter a sentence containing the ambiguous word", value=default_sentence, height=110)
    target = st.text_input("Target word to disambiguate", value="bank")
    if st.button("🪄 Disambiguate word", key="run_wsd"):
        if not sentence.strip() or not target.strip():
            st.warning("Enter both a sentence and target word.")
        else:
            try:
                tokens = word_tokenize(sentence)
                match = next((t for t in tokens if t.lower().strip(".,!?;:") == target.lower()), None)
                if not match:
                    st.warning(f"The word “{target}” was not found in the sentence. Please check spelling or use the exact word form.")
                else:
                    syn = lesk(tokens, target.lower())
                    st.session_state["wsd_sentence"] = sentence
                    st.session_state["wsd_target"] = target
                    st.session_state["wsd_result"] = syn
                    st.session_state["wsd_tokens"] = tokens
            except LookupError:
                st.error("Tokenizer data is missing. Ensure the app has internet access on first run, then restart.")
    if "wsd_result" in st.session_state:
        syn = st.session_state["wsd_result"]
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        if syn:
            st.markdown(f"### 🎯 Predicted sense: `{syn.name()}`")
            st.write("**Definition:**", syn.definition())
            st.write("**Synonyms:**", ", ".join(sorted({l.name().replace("_"," ") for l in syn.lemmas()})))
            st.write("**Context:**", st.session_state["wsd_sentence"])
            st.caption("This is a lightweight, dictionary-overlap baseline; it may choose an incorrect sense for some sentences.")
        else:
            st.warning("No sense could be selected for this context.")
        st.markdown("</div>", unsafe_allow_html=True)
        if syn:
            result_text = (f"Word Sense Disambiguation Report\nTarget word: {st.session_state['wsd_target']}\n"
                           f"Sentence: {st.session_state['wsd_sentence']}\nPredicted synset: {syn.name()}\n"
                           f"Definition: {syn.definition()}\nSynonyms: {', '.join(l.name() for l in syn.lemmas())}\n"
                           f"Method: NLTK Simplified Lesk\n")
            st.download_button("⬇️ Download WSD report (.txt)", result_text, file_name="wsd_report.txt")

with tab3:
    st.markdown("## Project overview")
    st.markdown("""
    <div class="panel">
    <h3>🎯 Objective</h3>
    Build a Word Sense Disambiguation (WSD) / Word Sense Induction demonstration using WordNet and lexical relations.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("### 🛠️ Methodology")
    st.write("1. Retrieve the word's synsets from WordNet.")
    st.write("2. Display definitions, examples, synonyms, hypernyms, and hyponyms.")
    st.write("3. Apply NLTK's Simplified Lesk algorithm to a sample sentence.")
    st.write("4. Export the sense inventory and prediction as text reports.")
    st.markdown("### 📊 Short analysis report")
    st.write("WordNet stores multiple synsets for polysemous words. Lexical relations help explain how each sense relates to other concepts. The Lesk baseline uses overlap between the context and gloss/examples to select a likely sense. Its output is interpretable but depends on the sentence context and the gloss vocabulary.")
    st.info("Expected deliverables covered: sense inventory/annotation output, WSD results for sample text, and a short analysis report.")
    st.caption("Built with Python, Streamlit, and NLTK WordNet.")
