import random
import json
import openai
from openai import OpenAI
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pyvo as vo
from astroquery.simbad import Simbad
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.table import Table
import pandas as pd

# This file contains the targets names and the target coordinates
# for each obsid. This is from the Chandra Chaser
df = pd.read_csv('target_obsid_coordinates.csv')

# Here I pick a particular obsid. 
obsid_target = 7924
target = df['target'].values[df['obsid'].values == obsid_target][0]
print(df['obsid'].values[df['obsid'].values == obsid_target][0])
print(df['target'].values[df['obsid'].values == obsid_target][0])

# CSC 2.1 TAP service
tap = vo.dal.TAPService('http://cda.cfa.harvard.edu/csc21tap') # For CSC 2.1

# Construct the query that takes each ACIS obsID, and gets coordinates for all the sources
# in the obsid. 

qry = """
SELECT DISTINCT m.name,o.obsid,m.ra,m.dec FROM csc21.master_source m , csc21.master_stack_assoc a , csc21.observation_source o , 
csc21.stack_observation_assoc b , csc21.stack_source s 
WHERE ((m.name NOT LIKE '%X') AND (o.instrument = 'ACIS') AND (o.obsid = """+str(obsid_target)+""")
AND (o.theta <= 5) AND (o.flux_significance_b >= 5)) AND (m.name = a.name)
AND (s.detect_stack_id = a.detect_stack_id and s.region_id = a.region_id) 
AND (s.detect_stack_id = b.detect_stack_id and s.region_id = b.region_id) 
AND (o.obsid = b.obsid and o.obi = b.obi and o.region_id = b.region_id)ORDER BY name ASC
"""

results = tap.search(qry)


# The SIMBAD API is used here to find all the catalog sources
# from the previous query that have identifiers (e.g., that
# have been given a name other than the CSC name)
simbad = Simbad()
simbad.add_votable_fields("otype")

# Loop through the results and query the region for each
idents = []
tipos = []
for i in range(len(results)):
        # Perform the query for the given coordinates
        print('Coordinates: ', results[i]['ra'], results[i]['dec'])
        print('CSC ID: ', results[i]['name'], '     Obsid: ', results[i]['obsid'])
        
        simby = simbad.query_region(
            SkyCoord(results[i]['ra'], results[i]['dec'], unit=(u.deg, u.deg), frame='fk5'),
            radius=3 * u.arcsec
        )

        # Check if the query returns a table
        if simby is not None:
            # Print only the desired column (e.g., 'MAIN_ID') for each result
        
            print('ID: ', simby['MAIN_ID'][0])  # Main SIMBAD ID of the source
            idents.append(simby['MAIN_ID'][0])  
            print('TYPE: ',simby['OTYPE'][0])   # SIMBAD type of the source
            tipos.append(simby['OTYPE'][0])
        else:
            print(f"No results found for index {i}.")
        print(' ')


# Test using SIMBAD source IDs
import random
import json
import openai
from openai import OpenAI
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pyvo as vo
import requests
import argparse
import h5py
import numpy as np

from astroquery.simbad import Simbad
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.table import Table


# We now feed GPT with the papers for the same obsid as before
# and prompt the summary generation for each source with a SIMBAD
# name, asking specific questions


def estimate_token_count(text):
    # Simple token estimation based on word count; use more accurate methods like tiktoken if needed
    return len(text.split())


obsid_list = [obsid_target]


# Initialize the OpenAI client with your API key
client = OpenAI(api_key='my-key')

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
        max_tokens=600  # Adjust based on your needs
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

# Open the file and count the number of lines
with open(file_path, 'r') as file:
    num_lines = sum(1 for line in file)

print('Num lines: ',num_lines)

# Set the seed for reproducibility
#random.seed(13)  # You can use any integer as the seed value


# List to hold all the embeddings
all_embeddings = []
#titles = []
obsids = []
answers = []

with open(file_path, 'r') as file:
    for current_line_number, line in enumerate(file, start=1):
        try:
            # Load the JSON line
            data = json.loads(line)
            
            # Check if the current obsid is in the list of obsids you want to process
            if data['obsid'] in obsid_list:
                
                prompt_question1 = f"""The target of the observation is {target}. Is this confirmed by the proposal 
                                    abstract provided, with title {data['proposal']['title']}? If so, please provide
                                    a short context about this observation.
                                    """  
                context = data['proposal']['abstract']
                full_prompt = f"{context}\n\n{prompt_question1}"

                # Generate the response using GPT
                answer = getGPTresponse(full_prompt)

                if answer:
                    print(f" ")
                    print(f"Obsid: {data['obsid']}")
                    print(f"Proposal Title: {data['proposal']['title']}")
                    print(f"Answer: {answer}\n")

                    for j,ido in enumerate(idents):

                        identifiers = Simbad.query_objectids(ido)

                        name_ids = []
                        for idito in identifiers['ID'].data: 
                            idito = idito.decode('utf-8')
                            name_ids.append(idito)
                        name_ids = list(name_ids)
                        

                        prompt_question2 = f""" 
                                        Given the text provided, search for information about the source identified 
                                        with any of the following names:
                                        
                                        {', '.join(repr(item) for item in name_ids)}. 
                                        
                                        The source is a source of type {tipos[j]}.
 
                                        Again based on the text provided, answer the following questions regarding the 
                                        source in question, without mentioning the name of the source or the target:

                                        Is the source specifically mentioned in the text, or is the source the target
                                        of the observation? If the answer is 'yes' to any of these questions, do the following. 
                                        If not, say only "Not discussed".
                                            A) Summarize the X-ray properties of the source in question, as inferred 
                                        directly from the data. Focus on variability (transient behavior, periodicity, etc.), 
                                        and spectral features (models fitted, hardness ratios, n_h, etc.), but provide values 
                                        of any relevant measured quantities if measured directly from the X-ray data.
                                            B) Describe how these properties or other X-ray data from the source is used to 
                                        test the scientific hypotheses being examined in the text provided.
                                        """

                        print(prompt_question2)

                        #=============================================================

                        # Initialize an empty string to hold the concatenated context
                        concatenated_context2 = answer
                        max_tokens_allowed = 80000  # Maximum tokens allowed by the model
                        current_token_count = 0

                        # Concatenate the context2 structures for each index i
                        for i in range(len(data['papers'])):
                            try:
                                body_text = data['papers'][i]['body'] + "\n"
                                body_token_count = estimate_token_count(body_text)


                                # Check if adding this body would exceed the max token limit
                                if current_token_count + body_token_count > max_tokens_allowed:
                                    #print(f"Reached token limit with paper {i}")
                                    continue

                                else:
                                    concatenated_context2 += body_text
                                    current_token_count += body_token_count
                                    #print(f"Length of body for paper {i}: {len(data['papers'][i]['body'])}, current token count: {current_token_count}")
        
                            except:
                                continue

                        # Create the full prompt using the concatenated context
                        full_prompt2 = f"{concatenated_context2}\n\n{prompt_question2}"

                        # Generate the response using GPT
                        answer2 = getGPTresponse(full_prompt2)

                        # Add the response to all_answers2 if it is valid
                        if answer2:                    
        
                            print(answer2)
                            # Generate embeddings for the answer
                            embedding = getEmbeddings(answer2)

                            if embedding:
                                all_embeddings.append(embedding)
                                obsids.append(data['obsid'])
                                answers.append(answer2)
                        else:
                            print("No answer for obsid: ",{data['obsid']})
                    
        except Exception as e:
            print(f"Error processing line {current_line_number}: {e}")
            continue
            
