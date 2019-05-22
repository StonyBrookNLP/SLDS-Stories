import sys
import csv
"""
line1
line2
line3
line4
line5

line11
line12
line13
line14
line15

line21
line22
line23
line24
line25

"""

def get_stories(fpath):
    """
    Add an empty blank line after the 
    very last line for this logic to work.
    """
    stories = [] # list of lists
    story = []
    with open(fpath) as f:
        for line in f:
            line = line.rstrip("\n")
            if line:
                story.append(line)
            else:
                stories.append(story)
                story = []
    return stories

def write_for_turk(slds_path, lm_path, turk_file):

    # read the stories
    slds_stories = get_stories(slds_path)
    lm_stories = get_stories(lm_path)
    assert len(slds_stories) == len(lm_stories), "Systems should have same # stories"
    num_stories = len(slds_stories)
    print("Total stories in each file {}".format(num_stories))
    block_size = 3

    # generate the header string
    header = []
    for n_story in range(1,4):
        h_str = "storyA{}_".format(n_story)
        for n_sent in range(1,6): 
            header.append(h_str + "line{}".format(n_sent))

        h_str = "storyB{}_".format(n_story)
        for n_sent in range(1,6): 
            header.append(h_str + "line{}".format(n_sent))
    assert len(header) == (5 * block_size * 2), "Header must have {} cols.".format(5 * block_size * 2)

    # write into the file
    with open(turk_file, 'w') as fw:
        csv_writer = csv.writer(fw)
        print("Written header")
        csv_writer.writerow(header)
        # iterate both stories in chunks
        for n in range(0, num_stories, block_size):
            ss = slds_stories[n:n+block_size]
            ls = lm_stories[n:n+block_size]
            assert len(ss) == len(ls), "Must be block size of 3"
            #print(len(ss))
            if len(ss) == 3:
                row = []
                # put respective stories together
                for i in range(0, block_size):
                    row.append(ss[i])
                    row.append(ls[i])
                
                # flatten to write into the file
                flat_row = [item for sublist in row for item in sublist]
                assert len(flat_row) == (5 * block_size * 2), "Rows must have {} cols".format(5 * block_size * 2)
                print("Written row")
                csv_writer.writerow(flat_row)        
    
    print("!!File written successfully!!")


slds_path = sys.argv[1]
lm_path = sys.argv[2]
turk_file = "../turk_data_2.csv"
print("SLDS {} LM {} Turd {}".format(slds_path, lm_path, turk_file))
print("MAKE SURE TO ADD A BLANK LINE AT THE END OF THE 2 FILES")
write_for_turk(slds_path, lm_path, turk_file)

