"""
Pseudo System 2 AI - PDF Query System (wersja z spaCy)
Wymaga: pip install pypdf2 spacy
"""

import re
import argparse
import sys
from typing import List, Dict, Tuple
import spacy

# Załaduj model spaCy
try:
    nlp = spacy.load("en_core_web_sm")
    print("✓ Załadowano model spaCy")
except OSError:
    print("Pobieranie modelu spaCy...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")
    print("✓ Załadowano model spaCy")

class PseudoSystem2AI:
    def __init__(self):
        self.document_words = []  # [(word, pos_tag, sentence_id, word_id)]
        self.sentences = []  # Lista zdań
        self.stop_words = nlp.Defaults.stop_words

    def load_pdf(self, pdf_path: str):
        """Wczytaj PDF i przetwórz na słowa z tagami"""
        try:
            import PyPDF2

            # Wyciągnij tekst z PDF
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()

            self._process_text(text)
            print(f"✓ Wczytano {len(self.sentences)} zdań, {len(self.document_words)} słów")

        except Exception as e:
            print(f"Błąd wczytywania PDF: {e}")
            raise  # Przekaż błąd dalej

    def _process_text(self, text: str):
        """Przetwórz tekst na słowa z tagami gramatycznymi"""
        # Podziel na zdania
        doc = nlp(text)
        self.sentences = [sent.text for sent in doc.sents]

        # Przetwórz każde zdanie
        for sent_id, sentence in enumerate(self.sentences):
            sent_doc = nlp(sentence)

            # Zapisz słowa z tagami
            for word_id, token in enumerate(sent_doc):
                self.document_words.append({
                    'word': token.text.lower(),
                    'original': token.text,
                    'pos': token.pos_,
                    'sentence_id': sent_id,
                    'word_id': word_id
                })

    def _analyze_question(self, question: str) -> Dict:
        """
        Przeanalizuj pytanie - wyciągnij czasownik i rzeczownik
        """
        doc = nlp(question)

        verbs = []
        nouns = []
        adjectives = []

        for token in doc:
            word_lower = token.text.lower()

            # Pomiń stop words
            if word_lower in self.stop_words:
                continue

            if token.pos_ == "VERB":  # Czasownik
                verbs.append(word_lower)
            elif token.pos_ == "NOUN":  # Rzeczownik
                nouns.append(word_lower)
            elif token.pos_ == "ADJ":  # Przymiotnik
                adjectives.append(word_lower)

        return {
            'verbs': verbs,
            'nouns': nouns,
            'adjectives': adjectives,
            'all_keywords': verbs + nouns + adjectives
        }

    def _find_matching_sentences(self, keywords: List[str], min_matches: int = 1) -> List[int]:
        """Znajdź zdania zawierające słowa kluczowe"""
        sentence_matches = {}  # sentence_id: count

        for entry in self.document_words:
            if entry['word'] in keywords:
                sent_id = entry['sentence_id']
                sentence_matches[sent_id] = sentence_matches.get(sent_id, 0) + 1

        # Sortuj według liczby dopasowań (malejąco)
        sorted_sentences = sorted(
            sentence_matches.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Zwróć tylko te z minimum dopasowań
        return [s_id for s_id, count in sorted_sentences if count >= min_matches]

    def query(self, question: str, max_results: int = 3) -> List[str]:
        """
        Zadaj pytanie do systemu

        Args:
            question: Pytanie po angielsku
            max_results: Maksymalna liczba wyników

        Returns:
            Lista znalezionych zdań
        """
        if not self.document_words:
            return ["⚠ Najpierw wczytaj dokument używając load_pdf() lub load_text()"]

        # Analiza pytania
        analysis = self._analyze_question(question)

        print(f"\n🔍 Analiza pytania:")
        print(f"   Czasowniki: {analysis['verbs']}")
        print(f"   Rzeczowniki: {analysis['nouns']}")
        print(f"   Przymiotniki: {analysis['adjectives']}")

        if not analysis['all_keywords']:
            return ["⚠ Nie znaleziono słów kluczowych w pytaniu"]

        # Znajdź pasujące zdania
        matching_sent_ids = self._find_matching_sentences(
            analysis['all_keywords'],
            min_matches=1
        )

        if not matching_sent_ids:
            return ["⚠ Nie znaleziono pasujących fragmentów w dokumencie"]

        # Zwróć zdania
        results = []
        for sent_id in matching_sent_ids[:max_results]:
            results.append(self.sentences[sent_id])

        return results

def main():
    # Konfiguracja argumentów wiersza poleceń
    parser = argparse.ArgumentParser(description='System do odpowiadania na pytania z PDF')
    parser.add_argument('-q', '--question', required=True, help='Pytanie do zadanego użytkownika')
    parser.add_argument('-f', '--file', required=True, help='Ścieżka do pliku PDF')
    parser.add_argument('-r', '--results', type=int, default=3, help='Maksymalna liczba wyników (domyślnie: 3)')

    args = parser.parse_args()

    # Inicjalizacja systemu
    ai = PseudoSystem2AI()

    try:
        # Wczytaj PDF
        ai.load_pdf(args.file)

        # Zadaj pytanie
        print(f"\n❓ Pytanie: {args.question}")
        print("-" * 60)

        results = ai.query(args.question, max_results=args.results)

        # Wyświetl wyniki
        if results:
            for i, result in enumerate(results, 1):
                print(f"\n{i}. {result}")
        else:
            print("\n⚠ Nie znaleziono pasujących odpowiedzi w dokumencie")

    except Exception as e:
        print(f"\n❌ Błąd: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
