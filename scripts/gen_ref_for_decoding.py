import sys
import csv
import spacy

r_ref_path = sys.argv[1]
if "test_ALL_val.csv" in r_ref_path:
    print("VALIDATION DATASET")
    w_ref_path1 = "../refs_val/references_2.txt"
    w_ref_path2 = "../refs_val/references_4.txt"
    w_ref_path3 = "../refs_val/references_1_2.txt"
    w_ref_path4 = "../refs_val/references_3_4.txt"
else:
    print("TEST DATASET")
    w_ref_path1 = "../refs_test/references_2.txt"
    w_ref_path2 = "../refs_test/references_4.txt"
    w_ref_path3 = "../refs_test/references_1_2.txt"
    w_ref_path4 = "../refs_test/references_3_4.txt"   

nlp = spacy.load('en')

with open(r_ref_path, 'r') as fr:
    with open(w_ref_path1, 'w') as fw1:
        with open(w_ref_path2, 'w') as fw2:
            with open(w_ref_path3, 'w') as fw3:
                with open(w_ref_path4, 'w') as fw4:
                    csv_reader = csv.reader(fr) 
                    print("!!SKIPPING THE HEADER!!")
                    next(csv_reader) # skip the title

                    for line in csv_reader:
                        sents = line[1:5]
                        mod_sents = []
                        for sent in sents:
                            #print(f"sent {sent}")
                            tokens = nlp.tokenizer(sent)
                            mod_sent = " ".join([x.text for x in tokens])
                            mod_sents.append(mod_sent)
                            #print(f"mod_sent {mod_sent}")
                        assert len(sents) == len(mod_sents) == 4, "Must be 4 sentences"
                        # sent1, sent2, sent3, sent4

                        # case 1: sent 2 missing
                        ref1 = mod_sents[1]
                        # case 2: sent 4 missing
                        ref2 = mod_sents[3] 
                        # case 3: sent 1 and 2 missing
                        ref3 = " ".join(mod_sents[:2]) 
                        # case 4: sent 3 and 4 missing
                        ref4 = " ".join(mod_sents[2:4]) 

                        fw1.write(f"{ref1}\n\n")
                        fw2.write(f"{ref2}\n\n")
                        fw3.write(f"{ref3}\n\n")
                        fw4.write(f"{ref4}\n\n")

