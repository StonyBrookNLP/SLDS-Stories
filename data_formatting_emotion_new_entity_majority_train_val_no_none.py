import pandas as pd
import os
from sklearn.model_selection import train_test_split
import re
from collections import OrderedDict
import pickle


MALE_PRONOUNS = [" he ", " him ", " his ", " himself "]
FEMALE_PRONOUNS = [" she ", " her ", " hers ", " herself "]

FEMALE_NAMES = list()
MALE_NAMES = list()

def get_common_names():
	#function to get the list of the names - both genders
	#save in the corresponding lists
	global MALE_NAMES
	global FEMALE_NAMES
	data_dir = "/home/rad/Documents/sbu_acads/thesis/common_sense/dataset/gender_names/"
	names = ["White-Female-Names.csv", "White-Male-Names.csv"]
	input_prefix = "cleaned-"

	for f in names:
		if "Female" in f: 
			FEMALE_NAMES = list(pd.read_csv(os.path.join(data_dir, input_prefix + f))[" first name"])
		else:
			MALE_NAMES = list(pd.read_csv(os.path.join(data_dir, input_prefix + f))[" first name"]) 
		
def check_pronoun(gender_pronouns, line_lower, ent_type):
	#check for the presence of male/female pronouns in the text
	for pronoun in gender_pronouns:
		if pronoun in line_lower:
			# print "Found character pronoun in sentence:"
			#replace with the string as target entity
			line_lower = line_lower.replace(pronoun, " " + ent_type + "_entity ")
	return line_lower

def char_clean_for_matching(char_list):
	char_clean_list = list()
	for char in char_list:
		char_clean = re.sub("\(\w+\)", "", char)
		char_clean_list.append(char_clean.lower())

	# print char_clean_list
	return char_clean_list

def replace_entity(entity_list, line_lower, ent_type):

	for entity in entity_list:
		#if entity present in line as it is:
		if entity in line_lower:
			# print "Found character in sentence:"
			line_lower = line_lower.replace(entity, ent_type + "_entity")

		#if the character is a common name, the also look for it's pronouns
		elif (entity in MALE_NAMES) or (entity in FEMALE_NAMES):
			# print "Check for presence of pronouns:"
			if entity in MALE_NAMES:
				line_lower = check_pronoun(MALE_PRONOUNS, line_lower, ent_type)

			if entity in FEMALE_NAMES:
				line_lower = check_pronoun(FEMALE_PRONOUNS, line_lower, ent_type)

			# print line_lower
			# print cleaned_char_target
			# print

	return line_lower

def get_data(split):
	#dictionary to hold the data in the format to be input into the encoder
	data_dictonary = dict()  
	#get the correct list of story ids for the training data
	if split == "train":
		storyid_list = storyid_train
	#get the correct list of story ids for the validation data
	else:
		storyid_list = storyid_test
	

	for s_id in storyid_list:
		storyid_true = data['storyid'] == s_id
		#get a subset of the dataframe where the storyid is the one passed
		df_storyid = data[storyid_true]

		#get all the unique characters in the story and lowercase them
		chars_in_story = list(set([c.lower() for c in df_storyid["char"]]))
		chars_in_story_cleaned = char_clean_for_matching(chars_in_story)
		# print "Characters in story:"
		# print chars_in_story
		# print "Characters in story cleaned:"
		# print chars_in_story_cleaned
		# print "Story dataframe"
		# print df_storyid[["char", "sentence"]]
		# print
		# print "All the characters in the story are:"
		# print chars_in_story

		#For combining the emotion annotations by multiple users for each line and character combination
		groupby_char_line = df_storyid.groupby(['linenum', 'char'])

		for name, group in groupby_char_line:
			print name
			print group
			# print "-------------------------------------------------------------------"

			# ####################First part - aggregate labels from multiple users###############################################################################
			
			#get the labels from the "plutchik" column
			labels = group["plutchik"]

			#dictionary to story the list of scores assigned for each emotion label
			local_labels_dict = dict()

			#list to combine the labels added by different annotators if average label score of all the labels >= 2
			local_labels_after_combine = list()

			linenum = name[0]
			char = name[1]

			cleaned_char_target = char_clean_for_matching([char])
			
			cleaned_chars_context = char_clean_for_matching(set(chars_in_story_cleaned) - set(cleaned_char_target))

			for label_row in labels:
			#for labels annotation by each annotator for each character, line pair

				for label in label_row:
					#replace " with an empty string
					#remove any extra space around the strings
					label_clean = re.sub('"', '', label).strip()
					# print label_clean
					#only if label is anything other than none and empty string		
					#only take labels with anger
					if (label_clean != "none") and (len(label_clean.strip()) > 0):
						# print label_clean
						label_key, label_val = label_clean.split(":")
						try:
							#if the particular emotion already annotated by another annotator
							local_labels_dict[label_key] += 1 

						except Exception, e:
							#if the particular emotion not already annotated by another annotator
							#initialize with 1 count
							local_labels_dict[label_key] = 1
						
						# print local_labels_dict

					#if a label of none was added to the character line pair
					#skip adding the non label
					elif label_clean == "none":
						#assign the "none" label to the label list for that particular character line pair
						#label is none, then only that one label is assigned for the character, line pair
						# label_set = [label_clean]
						#whenever none_label - then skip adding the row
						# local_labels_after_combine.append(label_clean)
						continue

			#get the labels which have been annotated by at least 2 users
			for key in local_labels_dict:
				#then add that label to the label list for that particular character line pair
				if local_labels_dict[key] >= 2:
					local_labels_after_combine.append(key)
			
			# print "after averaging"	
			# print local_labels_dict
			# print local_labels_after_combine

			# print 

			####################End of first part###############################################################################

			####################Second part - get context for the target entity###############################################################################
			# linenum = name[0]
			# char = name[1]
			# print
			# print "Line num:"
			# print linenum
			# print "Char"
			# print char
			# print "Annotator"
			# print name[2]
			char_lower = char.lower()

			if linenum == 1:
				# print "No context present"
				context_string = "empty"   #add an empty context string
				# print context_string
				context_string_lower = "empty"
			else:
				# context_string = group["context"]
				#to get the context for the particular character appearing in the line
				context_line_num_list = [i for i in range(1, linenum)]
				# print "context_line_num_list are:"
				# print context_line_num_list
				# print

				#get context lines for the characters
				context_df = df_storyid[(df_storyid['linenum'] >= context_line_num_list[0]) & (df_storyid['linenum'] <= context_line_num_list[-1]) & (df_storyid['char'] == char)]
				# print "context df is:"
				# print context_df[["char", "sentence"]]
				
				#remove repetitions while maintaining order
				context_list = list(OrderedDict.fromkeys(context_df["sentence"].values))
				# print context_list
				#reverse the order of the context so that the oldest is first
				#and the most recent one is last
				context_string = (" ".join(context_list)).strip()

				#replace the names of the specific characters with the string - "target_entity" and "context entity"
				# print "Context string is:"
				# print context_string
				# print
				#lowercase the context string
				context_string_lower = context_string.lower()
				# context_string_lower = replace_entity()
				# print "Target entities:", cleaned_char_target
				# print "Context entities:", cleaned_chars_context
				# print char
				# print "Before replacement"
				# print context_string_lower
				context_string_lower = replace_entity(cleaned_char_target, context_string_lower, "target") #getting the target entities
				context_string_lower = replace_entity(cleaned_chars_context, context_string_lower, "context") #getting the context entities
				# print "After replacement"
				# print context_string_lower
				# print



			####################Third part - mark the target entity and other entity in each line###############################################################################
			###### Whatever missed, do that manually############################################################################################################################
			line_lower = (group['sentence'].values[0]).lower()
			# print "Line is:"
			# print line_lower
			# print "Chars in story are:"
			# print chars_in_story
			# print "Char at this sentence:"
			# print char
			# print "Cleaning the char:"

			# print "Target enitities:", cleaned_char_target
			# print "Context entities:", cleaned_chars_context
			# print "Before replacement line:"
			# print line_lower
			line_lower = replace_entity(cleaned_char_target, line_lower, "target") #getting the target entities
			line_lower = replace_entity(cleaned_chars_context, line_lower, "context") #getting the context entities
			# print "After replacement line:"
			# print line_lower
			# print

			try:	

				# "Entry to be added is:"
				#putting this additional condition because when we skipped the none label, the labels list remained empty
				if len(local_labels_after_combine) > 0:
					print {(" ".join(local_labels_after_combine)).strip() : [line_lower, context_string_lower]}
					print
					data_dictonary[s_id].append({(" ".join(local_labels_after_combine)).strip() : [line_lower, context_string_lower]})

			except Exception, e:
				#putting this additional condition because when we skipped the none label, the labels list remained empty
				if len(local_labels_after_combine) > 0:
					data_dictonary[s_id] = [{(" ".join(local_labels_after_combine)).strip() : [line_lower, context_string_lower]}]	

	return data_dictonary

#will have one big vocabulary for sentences, contexts and labels - from training data
# training_data_sentences, training_data_char_contexts, training_data_labels = get_dataset(training_data)
# val_data_sentences, val_data_char_contexts, val_data_labels = get_dataset(val_data)

#get commonly used names for males and females in English
get_common_names()

print "The male names are:"
print MALE_NAMES
print

#specify directory for the input data
data_dir = "/home/rad/Documents/sbu_acads/thesis/common_sense/dataset/storycommonsense_data/csv_version/dev/emotion/"
#read the entire file into a dataframe
data = pd.read_csv(os.path.join(data_dir, 'allcharlinepairs.csv'))   

#convert the string of plutchik column into an actual list, by splitting on ", "
data["plutchik"] = data["plutchik"].apply(lambda x : x[1:-1].split(", "))

# #get unique storyids
storyids = list(set(data['storyid']))

# #splitting data into training and test set based on the storyids
storyid_train, storyid_test = train_test_split(storyids, test_size = 0.2, train_size = 0.8, shuffle=True)

#dictionary to hold the data in the format to be input into the encoder
train_data_dictonary = dict()
val_data_dictionary = dict()

# print "Getting the validation data:"
train_data_dictonary  = get_data("train") #get data from the training split
val_data_dictionary = get_data("validation") #get data from the validation split

#storing the training and the validation data
data_dir = "../../../data_for_code_new_entities/"
pickle.dump(train_data_dictonary, open(data_dir + "training_data.p", "wb"))
pickle.dump(val_data_dictionary, open(data_dir + "validation_data.p", "wb"))

# pickle.dump(test_data_dictionary, open(data_dir + "test_data.p", "wb"))



