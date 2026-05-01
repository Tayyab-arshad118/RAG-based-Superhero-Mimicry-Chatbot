"""
extract_dialogues.py
--------------------
Step 1 of the pipeline.
This script reads the movie script PDFs, extracts ONLY the dialogues
of each superhero (using their name + synonyms), and saves them as
clean .txt files inside the dialogues/ folder.

Run this ONCE before starting the chatbot.
Usage: python extract_dialogues.py
"""

import os
import re
import yaml
import pymupdf  # pip install pymupdf
from os.path import join as pjoin

# ─── CONFIG ────────────────────────────────────────────────────────────────────
ROOT = "."
SCRIPTS_FOLDER = pjoin(ROOT, "pdfs")          # put all your PDFs here
DIALOGUES_FOLDER = pjoin(ROOT, "dialogues")   # extracted dialogues saved here
CONFIG_FILE = pjoin(ROOT, "config.yaml")

# How many characters of context to include BEFORE each dialogue line
MAX_CONTEXT_LENGTH = 150

# Max number of other character lines to keep after hero's line (for context)
MAX_EXTRA_DIALOGUES = 3

# ─── LOAD CONFIG ───────────────────────────────────────────────────────────────
with open(CONFIG_FILE, "r") as f:
    config = yaml.safe_load(f)

list_of_superheroes = config["LIST_OF_SUPERHEROES"]
superhero_synonyms = config["SUPERHERO_SYNONYMS"]
movies_list = config["MOVIES_LIST_OF_SUPERHEROES"]

# ─── HELPER FUNCTIONS ──────────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_path):
    """Read all text from a PDF file page by page."""
    try:
        pdf = pymupdf.open(pdf_path)
        text = ""
        for page in pdf:
            text += page.get_text()
        return text
    except Exception as e:
        print(f"    [ERROR] Could not read {pdf_path}: {e}")
        return ""


def get_all_hero_names(superhero):
    """
    Build the full list of name variants for a superhero.
    Example: Ironman → ['IRONMAN', 'TONY STARK', 'TONY', 'STARK']
    All uppercase because movie scripts use CAPS for character names.
    """
    names = [superhero.upper()]  # hero name itself
    synonyms = superhero_synonyms.get(superhero, [])
    for syn in synonyms:
        names.append(syn.upper())
    # Remove duplicates
    return list(set(names))


def extract_hero_dialogues(script_text, hero_names):
    """
    Extract dialogue blocks where the hero is speaking.
    
    In movie scripts, a character's dialogue looks like:
    
        TONY STARK
        I am Iron Man.
    
    So we find lines that EXACTLY match a hero name (uppercase),
    then grab the lines that follow as the dialogue.
    """
    dialogues = []
    lines = script_text.split("\n")

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Check if this line is a character name (matches hero name)
        if line in hero_names:
            # Collect context: a few lines BEFORE this point
            context_start = max(0, i - 5)
            context_lines = lines[context_start:i]
            context = " ".join(l.strip() for l in context_lines if l.strip())

            # Collect the dialogue: lines that follow until next character name or blank
            dialogue_lines = []
            j = i + 1
            extra_char_count = 0

            while j < len(lines):
                next_line = lines[j].strip()

                # If we hit another ALL-CAPS line, it might be another character
                if re.match(r'^[A-Z][A-Z\s\'\-\.]+$', next_line) and len(next_line) > 2:
                    extra_char_count += 1
                    if extra_char_count >= MAX_EXTRA_DIALOGUES:
                        break
                    dialogue_lines.append(next_line)
                elif next_line == "":
                    # Empty lines are okay, keep going
                    dialogue_lines.append("")
                else:
                    dialogue_lines.append(next_line)
                j += 1

            # Clean up the dialogue
            dialogue_text = "\n".join(dialogue_lines).strip()

            if dialogue_text and len(dialogue_text) > 10:  # skip tiny fragments
                full_entry = f"[CONTEXT]: {context}\n[{line}]: {dialogue_text}"
                dialogues.append(full_entry)

            i = j  # jump ahead
        else:
            i += 1

    return dialogues


def save_dialogues(superhero, dialogues):
    """Save all extracted dialogues for a hero into one .txt file."""
    hero_folder = pjoin(DIALOGUES_FOLDER, superhero.replace(" ", "_"))
    os.makedirs(hero_folder, exist_ok=True)
    save_path = pjoin(hero_folder, "dialogues.txt")

    separator = "\n\n---dialogue-separator---\n\n"
    content = separator.join(dialogues)

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(content)

    return save_path


# ─── MAIN EXTRACTION LOOP ──────────────────────────────────────────────────────

def main():
    os.makedirs(DIALOGUES_FOLDER, exist_ok=True)
    print("=" * 60)
    print("  SUPERHERO DIALOGUE EXTRACTOR")
    print("=" * 60)

    for superhero in list_of_superheroes:
        print(f"\n[{superhero.upper()}] Extracting dialogues...")

        hero_names = get_all_hero_names(superhero)
        all_dialogues = []

        # Get list of PDF scripts for this hero
        script_files = movies_list.get(superhero, [])

        if not script_files:
            print(f"  No scripts found for {superhero}. Skipping.")
            continue

        for script_file in script_files:
            script_path = pjoin(SCRIPTS_FOLDER, script_file)

            if not os.path.exists(script_path):
                print(f"  [SKIP] {script_file} not found in pdfs/ folder.")
                continue

            print(f"  Reading: {script_file}")
            script_text = extract_text_from_pdf(script_path)

            if not script_text:
                continue

            dialogues = extract_hero_dialogues(script_text, hero_names)
            print(f"    → Found {len(dialogues)} dialogue blocks.")
            all_dialogues.extend(dialogues)

        if all_dialogues:
            save_path = save_dialogues(superhero, all_dialogues)
            print(f"  ✓ Saved {len(all_dialogues)} dialogues to: {save_path}")
        else:
            print(f"  ✗ No dialogues found for {superhero}.")

    print("\n" + "=" * 60)
    print("  EXTRACTION COMPLETE!")
    print("  Now run: streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
