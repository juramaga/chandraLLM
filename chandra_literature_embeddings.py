import random
import json
import openai
from openai import OpenAI
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Initialize the OpenAI client with your API key
client = OpenAI(api_key='')

# Model names
generation_model = "gpt-4o-mini"  # or the model you are using for text generation
embedding_model = "text-embedding-ada-002"  # Use this for generating embeddings

def getGPTresponse(prompt):
    response = client.chat.completions.create(
        model=generation_model,
        messages=[
            {"role": "system", "content": "You are an expert astronomer specializing in the Chandra Data Archive."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=500  # Adjust based on your needs
    )
    
    try:
        result = response.choices[0].message.content
    except Exception as e:
        print(f"Error generating text: {e}")
        result = None
    
    return result

def getEmbeddings(text):
    response = client.embeddings.create(
        model=embedding_model,
        input=text
    )
    
    try:
        embedding = response.data[0].embedding
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        embedding = None
    
    return embedding


def perform_tsne(embeddings, perplexity=2):
    # Perform t-SNE to reduce the dimensionality to 2D
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    return embeddings_2d

# This us the file that contains the bibliographic data and the association with Chandra obsids
file_path = 'chandrabib_all.jsonl'

# Set the seed for reproducibility
random.seed(13)  # You can use any integer as the seed value

# Define the range and the number of random lines you want to read
total_lines = 17377
num_random_lines = 20  # Set this to how many random lines you want to read

# Generate a set of unique random line numbers
random_line_numbers = set(random.sample(range(1, total_lines + 1), num_random_lines))


# List to hold all the embeddings
all_embeddings = []
#titles = []
obsids = []
answers = []

with open(file_path, 'r') as file:
    for current_line_number, line in enumerate(file, start=1):
        try:
            if current_line_number in random_line_numbers:
                # Load the JSON line
                data = json.loads(line)
                
                #for i in range(len(data['papers'])):
                    
                prompt_question1 = f"""Please answer the following question:
                                    What is the astrophysical object being targeted by the observational proposal 
                                    with title {data['proposal']['title']}?
                                    """ 
                context = data['proposal']['abstract']
                full_prompt = f"{context}\n\n{prompt_question1}"

                # Generate the response using GPT
                answer = getGPTresponse(full_prompt)

                if answer:
                    print(f"Proposal Title: {data['proposal']['title']}")
                    print(f"Answer: {answer}\n")
                    
                    prompt_question2 = f"""Given the text provided, answer the following questions:
                                    1) What is the main scientific hypothesis being examined in the text?
                                    2) If specifically discussed, what properties of the ligth curve and spectrum 
                                    are reported for the target(s) of observation {data['obsid']}? {answer}. Please
                                    do not infer if the information is not specifically discussed.
                                    3) How is the data of the target obtained in Chandra observation (obsID 
                                    {data['obsid']}) being used to test the hypothesis?
                                    """ 
                    
                    
                    all_answers2 = ""
                    
                    for i in range(len(data['papers'])):
                        context2 = data['papers'][i]['body']
                        full_prompt2 = f"{context2}\n\n{prompt_question2}"
                    
                        # Generate the response using GPT
                        answer2 = getGPTresponse(full_prompt2)
                    
                        # Concatenate the valid answer2 to the accumulating string
                        all_answers2 += f"{answer2}\n"
                            
                    
                    if (len(all_answers2)>0):
                        # Print the final concatenated string after the loop completes
                        #print("All concatenated answers2:")
                        #print(all_answers2)
                        
                        # Generate embeddings for the answer
                        embedding = getEmbeddings(all_answers2)

                        if embedding:
                            all_embeddings.append(embedding)
                            #titles.append(data['paper']['title'])
                            obsids.append(data['obsid'])
                            answers.append(all_answers2)
                    
        except Exception as e:
            print(f"Error processing line {current_line_number}: {e}")
            continue
            
# Convert the list of embeddings to a numpy array
embeddings_array = np.array(all_embeddings)

print('Shape of embeddings: ',np.shape(embeddings_array))

# Perform t-SNE to reduce to 2D
embeddings_2d = perform_tsne(embeddings_array)


# Save the full embeddings in an HDF5, with corresponding obsid

# Check that all lists are of the same length
assert len(obsids) == len(all_embeddings) == len(answers), \
    "All input lists must have the same length."

# Build the 'data' structure
data = []
for i in range(len(obsids)):
    entry = {
        'obsid': obsids[i],
        'embedding': all_embeddings[i],
        'answer': answers[i]
    }
    data.append(entry)


import h5py
import numpy as np

# Save data to an HDF5 file
with h5py.File('text_embeddings_new.h5', 'w') as f:
    for i, entry in enumerate(data):
        group = f.create_group(f'entry_{i}')
        group.create_dataset('obsid', data=entry['obsid'])
        group.create_dataset('embedding', data=entry['embedding'])
        group.create_dataset('answer', data=entry['answer'].encode('utf-8'))

print("Data saved to 'text_embeddings_new.h5'.")


# Finally, save 2D embeddings (from tSNE) to a CSV file

import csv
#embeddings

# Assuming embeddings_2d, obsids, and titles are already defined

# Open a CSV file to write the 2D t-SNE embeddings and association with the papers
with open('paper_embs2d_new.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    
    # Write the header row
    writer.writerow(["Embedding X", "Embedding Y", "ObsID", "Answer"])
    
    # Loop through your data and write each row to the CSV
    for i in range(len(embeddings_2d)):
        writer.writerow([embeddings_2d[i, 0], embeddings_2d[i, 1], obsids[i], answers[i]])

print("CSV file 'paper_embs2d_new.csv' created successfully.")
