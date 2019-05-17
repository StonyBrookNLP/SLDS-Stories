import sys
import csv
import spacy

r_ref_path = sys.argv[1]
w_ref_path1 = "refs/references_2.txt"
w_ref_path2 = "refs/references_4.txt"
w_ref_path3 = "refs/references_1_2.txt"
w_ref_path4 = "refs/references_3_4.txt"

nlp = spacy.load('en')

with open(r_ref_path, 'r') as fr:
    with open(w_ref_path1, 'w') as fw1:
        with open(w_ref_path2, 'w') as fw2:
            with open(w_ref_path3, 'w') as fw3:
                with open(w_ref_path4, 'w') as fw4:
                    csv_reader = csv.reader(fr) 
                    next(csv_reader) # skip the title
                    for line in csv_reader:
                        sents = line[2:7]
                        mod_sents = []
                        for sent in sents:
                            #print(f"sent {sent}")
                            tokens = nlp.tokenizer(sent)
                            mod_sent = " ".join([x.text for x in tokens])
                            mod_sents.append(mod_sent)
                            #print(f"mod_sent {mod_sent}")
                        assert len(sents) == len(mod_sents) == 5, "Must be 5 sentences"

                        # case 1: sent 2 missing
                        ref1 = " ".join([mod_sents[0]] + mod_sents[2:])
                        # case 2: sent 4 missing
                        ref2 = " ".join(mod_sents[0:2] + [mod_sents[4]])
                        # case 3: sent 1 and 2 missing
                        ref3 = " ".join(mod_sents[2:])
                        # case 4: sent 3 and 4 missing
                        ref4 = " ".join(mod_sents[0:1] + [mod_sents[4]])

                        fw1.write(f"{ref1}\n\n")
                        fw2.write(f"{ref2}\n\n")
                        fw3.write(f"{ref3}\n\n")
                        fw4.write(f"{ref4}\n\n")









