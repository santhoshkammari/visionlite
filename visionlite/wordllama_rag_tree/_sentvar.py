import spacy
import en_core_web_sm
from wordllama import WordLlama


class SentenceVariator:
    def __init__(self):
        self.nlp = en_core_web_sm.load()
        self.wl = WordLlama.load()  # Load default 256-dimensional model

    def _analyze(self, text):
        """Analyze text structure with more detail."""
        doc = self.nlp(text)
        components = {
            'subjects': [],
            'verbs': [],
            'objects': [],
            'adverbs': [],
            'preps': [],
            'articles': [],
            'prep_objs': []
        }

        for token in doc:
            if token.dep_ in ('nsubj', 'nsubjpass'):
                components['subjects'].append(str(token))
            elif token.pos_ == 'VERB':
                components['verbs'].append(str(token))
            elif token.dep_ == 'dobj':
                components['objects'].append(str(token))
            elif token.pos_ == 'ADV':
                components['adverbs'].append(str(token))
            elif token.pos_ == 'ADP':
                components['preps'].append(str(token))
            elif token.pos_ == 'DET':
                components['articles'].append(str(token))
            elif token.dep_ == 'pobj':
                components['prep_objs'].append(str(token))

        return components, doc

    def _create_variations(self, components, doc):
        """Generate meaningful variations."""
        variations = set()

        # Add original
        original = ' '.join(token.text for token in doc)
        variations.add(original)

        words = [token.text for token in doc]

        # Move adverbs if present
        for adv in components['adverbs']:
            words_without_adv = [w for w in words if w != adv]

            # Adverb at start
            variations.add(f"{adv} {' '.join(words_without_adv)}")

            # Adverb at end
            variations.add(f"{' '.join(words_without_adv)} {adv}")

            # Adverb after first verb (if exists)
            for i, w in enumerate(words_without_adv):
                if w in components['verbs']:
                    new_words = words_without_adv.copy()
                    new_words.insert(i + 1, adv)
                    variations.add(' '.join(new_words))
                    break

        # Move prepositional phrases if present
        if components['preps'] and components['prep_objs']:
            for prep in components['preps']:
                for obj in components['prep_objs']:
                    prep_phrase = f"{prep} {obj}"
                    base_without_prep = ' '.join([w for w in words
                                                  if w not in [prep, obj]])

                    # Prep phrase at end
                    variations.add(f"{base_without_prep} {prep_phrase}")

                    # Prep phrase after verb
                    for i, w in enumerate(words):
                        if w in components['verbs']:
                            parts = words.copy()
                            prep_idx = parts.index(prep)
                            obj_idx = parts.index(obj)
                            # Remove prep phrase
                            if prep_idx > obj_idx:
                                parts.pop(prep_idx)
                                parts.pop(obj_idx)
                            else:
                                parts.pop(obj_idx)
                                parts.pop(prep_idx)
                            # Insert after verb
                            parts.insert(i + 1, f"{prep} {obj}")
                            variations.add(' '.join(parts))
                            break

        # Try passive voice if possible
        if components['subjects'] and components['verbs'] and components['objects']:
            subj = components['subjects'][0]
            verb = components['verbs'][0]
            obj = components['objects'][0]

            # Simple passive transformation
            passive = f"{obj} was {verb} by {subj}"
            if components['preps'] and components['prep_objs']:
                prep = components['preps'][0]
                prep_obj = components['prep_objs'][0]
                passive += f" {prep} {prep_obj}"
            variations.add(passive)

        return variations

    def gen(self, text):
        """Generate enhanced variations."""
        components, doc = self._analyze(text)
        variations = self._create_variations(components, doc)

        # Clean and format
        cleaned = []
        for var in variations:
            # Basic cleanup
            clean = ' '.join(var.split())
            # Capitalize first letter
            clean = clean[0].upper() + clean[1:]
            # Add period if missing
            if not clean.endswith('.'):
                clean += '.'
            if self.wl.similarity(text,clean)>=0.8:
                cleaned.append(clean)

        return sorted(cleaned, key=len)


if __name__ == '__main__':
    sv = SentenceVariator()
    print(sv.gen("i want to sum two numpy arrays in python"))
