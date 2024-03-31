import argparse
import nltk
from nltk.tag import hmm, brill, brill_trainer

import re
import argparse
import nltk
from nltk.tag import brill, brill_trainer

def train_brill_tagger(train_sents, initial_tagger):
    templates = [
        brill.Template(brill.Pos([-1])),
        brill.Template(brill.Pos([1])),
        brill.Template(brill.Pos([-2])),
        brill.Template(brill.Pos([2])),
        brill.Template(brill.Pos([-2, -1])),
        brill.Template(brill.Pos([1, 2])),
        brill.Template(brill.Pos([-3, -2, -1])),
        brill.Template(brill.Pos([1, 2, 3])),
        brill.Template(brill.Pos([-1]), brill.Pos([1])),
        brill.Template(brill.Word([-1])),
        brill.Template(brill.Word([1])),
        brill.Template(brill.Word([-2])),
        brill.Template(brill.Word([2])),
        brill.Template(brill.Word([-2, -1])),
        brill.Template(brill.Word([1, 2])),
        brill.Template(brill.Word([-3, -2, -1])),
        brill.Template(brill.Word([1, 2, 3])),
        brill.Template(brill.Word([-1]), brill.Word([1])),
        brill.Template(brill.Word([-1]), brill.Pos([1])),
        brill.Template(brill.Pos([-1]), brill.Word([1])),
        brill.Template(brill.Pos([-1]), brill.Pos([1]), brill.Word([2])),
        brill.Template(brill.Pos([1]), brill.Word([-1]), brill.Pos([-2])),
        # Add more templates here
    ]

    unigram_tagger = nltk.UnigramTagger(train_sents, backoff=nltk.DefaultTagger('NN'))
    bigram_tagger = nltk.BigramTagger(train_sents, backoff=unigram_tagger)
    initial_tagger = nltk.TrigramTagger(train_sents, backoff=bigram_tagger)

    trainer = brill_trainer.BrillTaggerTrainer(initial_tagger, templates)
    return trainer.train(train_sents, max_rules=3000, min_score=2)

def read_tagged_sentences(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        sentence = []
        for line in f:
            line = line.strip()
            if line:
                word, tag = line.split()
                sentence.append((word, tag))
            elif sentence:
                yield sentence
                sentence = []
        if sentence:
            yield sentence

def train_hmm_tagger(train_sents, smoothing_param=0.4):
    trainer = hmm.HiddenMarkovModelTrainer()

    # Apply Lidstone smoothing with a custom parameter
    tagger = trainer.train_supervised(train_sents, estimator=lambda fd, bins: nltk.LidstoneProbDist(fd, smoothing_param, bins))
    
    return tagger



def tag_sentences(tagger, test_sents):
    tagged_sentences = []
    for sent in test_sents:
        tagged_sent = tagger.tag([word for word, _ in sent])
        tagged_sentences.append(tagged_sent)
    return tagged_sentences

def write_output(tagged_sentences, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        for sent in tagged_sentences:
            for word, tag in sent:
                f.write(f"{word} {tag}\n")
            f.write('\n')

def main():
    parser = argparse.ArgumentParser(description="POS Tagging with HMM and Brill Tagger")
    parser.add_argument("--tagger", choices=["hmm", "brill"], help="Tagger type: hmm or brill", required=True)
    parser.add_argument("--train", help="Path to the training corpus", required=True)
    parser.add_argument("--test", help="Path to the test corpus", required=True)
    parser.add_argument("--output", help="Path to the output file", required=True)
    args = parser.parse_args()

    # Read and train on the training set
    train_sents = list(read_tagged_sentences(args.train))

    if args.tagger == "hmm":
        tagger = train_hmm_tagger(train_sents)
    elif args.tagger == "brill":
        initial_tagger = nltk.TrigramTagger(train_sents)
        tagger = train_brill_tagger(train_sents, initial_tagger)
    else:
        raise ValueError("Invalid tagger type. Choose between 'hmm' and 'brill'.")

    # Read and tag the test set
    test_sents = list(read_tagged_sentences(args.test))
    tagged_sentences = tag_sentences(tagger, test_sents)
    acc = tagger.evaluate(test_sents)
    print(acc)
    # Write the output
    write_output(tagged_sentences, args.output)

if __name__ == "__main__":
    main()
