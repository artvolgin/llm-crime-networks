
import os
import openai
import pandas as pd
import re
import time
import numpy as np
import json
import spacy
import fitz
from nltk.tokenize import sent_tokenize
from nltk.tokenize import RegexpTokenizer
import itertools
import warnings
pd.options.mode.chained_assignment = None
warnings.simplefilter(action='ignore', category=FutureWarning)


# -----------------------------------------------------------------------------
# ----- FUNCTIONS
# -----------------------------------------------------------------------------

def detectOutliers(numbers):
    """
    Detects outliers in a list of numbers.

    Args:
        numbers (list): A list of numbers.

    Returns:
        list: A list of indices where outliers are found.
    """

    outliers = []
    for i in range(1, len(numbers)):
        if numbers[i] != numbers[i - 1] + 1:
            # Check if the current number is not the start of a new sequence
            if i < len(numbers) - 1 and numbers[i + 1] == numbers[i] + 1:
                continue

            outliers.append(i)
    return outliers


def findBetween(text, start, end):
    """
    Finds and returns the substring between two given strings in a larger text.

    Args:
        text (str): The larger text in which to search for the substring.
        start (str): The starting string of the substring.
        end (str): The ending string of the substring.

    Returns:
        str: The substring between the start and end strings.

    """
    pattern = f'{re.escape(start)}(.*?){re.escape(end)}'
    matches = re.findall(pattern, text, re.DOTALL)[0]
    matches = start + matches 
    
    return matches


def containsYear(string):
    # The regular expression pattern for a 4-digit year
    pattern = r'\b\d{4}\b'

    # Search for the pattern in the given string
    result = re.search(pattern, string)

    # If a match is found, the function returns True, otherwise False
    return result is not None



def extractPars(case):
    """
    Extracts paragraphs from a PDF document based on the given case.

    Parameters:
    - case (str): The case name.

    Returns:
    - list: A list of extracted paragraphs.
    """

    if case != 'Airbus':
    
        par_pattern = r'(^\n\d+\.\n)|(^\d+\. )'
        doc = fitz.open(WD + "{}/sof.pdf".format(case))
        
        # Get combined text
        remove_text = 'VIII. SUPPLEMENTAL AGREEMENT NO. 2 (“SA2”) – BOSE HEADSETS \n \n'
        text = ''
        for i in range(len(doc)):
        
            # Get the text from this page
            page_text = doc[i].get_text("text").replace(remove_text, '')
            page_text = re.split('(\d+)', page_text,maxsplit=1)[2]
            # Append the text from this page to the overall string
            text += page_text
        
        # Split by paragraphs
        list_blocks = []
        for i in range(len(doc)):
            
            page_blocks = doc[i].get_text("blocks")
            
            for block in page_blocks:
                list_blocks.append(block[4].strip().replace(remove_text, ''))
                
        list_blocks = list(filter(lambda x: bool(re.match(par_pattern, x)), list_blocks))
        par_number = list(map(lambda x: ''.join(list(re.findall(par_pattern, x)[0])), list_blocks)) # Might be different for Airbus
        df_pars = pd.DataFrame({'number':par_number,
                                'text':list_blocks})
        df_pars['number'] = df_pars['number'].apply(lambda x: int(re.match(r'^(\d+)\. ', x).group(1)))
        df_pars = df_pars.drop(index=detectOutliers(df_pars['number'])).reset_index(drop=True)
        
        
        # Extract full paragraphs
        list_pars = []
        for i in range(len(df_pars)-1):
            
            start_substring = df_pars.iloc[i]['text']
            end_substring = df_pars.iloc[i+1]['text']
            result = findBetween(text, start_substring, end_substring)
            list_pars.append(result)
        list_pars.append(df_pars.iloc[len(df_pars)-1]['text'])
        list_pars = [re.sub(par_pattern, '', x) for x in list_pars]
        df_pars['text'] = list_pars
    
        if case == 'Sarclad':
        
            # Fix an issue with duplicated paragraphs in Sarclad
            df_pars = df_pars.groupby('number')['text'].apply(lambda x: ' '.join(x)).reset_index()    
            df_pars['has_year'] = df_pars['text'].apply(lambda x: containsYear(x))
            merged = []
            current_group = []
            for i, (num, binary) in enumerate(zip((df_pars['number']-1).values, df_pars['has_year'].values)):
                if binary:
                    if current_group:  # If the current_group is not empty, add it to the merged list
                        merged.append(current_group)
                    current_group = [num]  # Start a new group with the current number
                else:
                    current_group.append(num)
            
            if current_group:  # Add the last group to the merged list
                merged.append(current_group)
                
            # Add grouped paragraphs to the dataset
            list_dfs = []
            for group in merged:
                df_group = df_pars.iloc[group]
                df_group['group'] = str(group)
                list_dfs.append(df_group)
            df_pars = pd.concat(list_dfs)
            df_pars = df_pars.groupby('group')['text'].apply(
                lambda x: ' '.join(x)).reset_index()
        
        # List with paragraphs
        list_pars = list(df_pars['text'].values)
        
        # Remove \n 
        list_pars = list(map(lambda x: x.replace('\n', ''), list_pars))
        
        if case == 'Sarclad':
            list_pars = [re.sub(': “.*?”','', x, flags=re.DOTALL) for x in list_pars]
            list_pars = [re.sub(':  “.*?”','', x, flags=re.DOTALL) for x in list_pars]
            
        if case == 'StdBank':
            list_pars = [re.sub(': ".*?"','', x, flags=re.DOTALL) for x in list_pars]
        
    else:
        
        doc = fitz.open(WD + "{}/sof.pdf".format(case))
        par_pattern = r'(^\n\d+\.\n)|(^ \d+\. )|(^  \d+\. )|(^   \d+\. )' 
        min_font_size = 11
        
        # Get combined text
        text = ''
        for i in range(len(doc)):
            
            page = doc[i]
            
            page_blocks = page.get_text("dict")['blocks']
            
            for block in page_blocks:
            
                block_lines = block['lines']
                block_clean = ''
                
                for line in block_lines:
                    
                    line_size = line['spans'][0]['size']
                    
                    if line_size >= min_font_size:
                        
                        line_text = line['spans'][0]['text']
                        line_text = line_text.strip()
                        # text += '\n' + line_text
                        text += ' ' + line_text
        
        
        # Split by paragraphs
        list_blocks = []
        for i in range(len(doc)):
            
            page = doc[i]
            
            page_blocks = page.get_text("dict")['blocks']
            
            for block in page_blocks:
            
                block_lines = block['lines']
                block_clean = ''
                
                for line in block_lines:
                    
                    line_size = line['spans'][0]['size']
                    
                    if line_size >= min_font_size:
                        
                        line_text = line['spans'][0]['text']
                        line_text = line_text.strip()
                        # block_clean += '\n' + line_text
                        block_clean += ' ' + line_text
                        
                list_blocks.append(block_clean)
        
        # ---
        list_blocks = list(filter(lambda x: bool(re.match(par_pattern, x)), list_blocks))
        par_number = list(map(lambda x: ''.join(list(re.findall(par_pattern, x)[0])), list_blocks)) 
        df_pars = pd.DataFrame({'number':par_number,
                                'text':list_blocks})
        df_pars['number'] = df_pars['number'].apply(lambda x: int(re.sub(r"\D", "", x)))
        df_pars = df_pars.drop(index=detectOutliers(df_pars['number'])).reset_index(drop=True)
        
        # Extract full paragraphs
        list_pars = []
        for i in range(len(df_pars)-1):
            
            start_substring = df_pars.iloc[i]['text']
            end_substring = df_pars.iloc[i+1]['text']
            result = findBetween(text, start_substring, end_substring)
            list_pars.append(result)
        list_pars.append(df_pars.iloc[len(df_pars)-1]['text'])
        
        list_pars = [re.sub(par_pattern, '', x) for x in list_pars]
        df_pars['text'] = list_pars
        df_pars['text'] = df_pars['text'].str.replace('\n',' ')
        
        # --- List with paragraphs
        list_pars = list(df_pars['text'].values)
    
    return list_pars


def parseEntitiesInteractions(list_pars, model):
    """
    Parses the given list of paragraphs using ChatGPT model to extract entity interactions.

    Args:
        list_pars (list): A list of paragraphs to be parsed.
        model: The Chat GPT model used for parsing.

    Returns:
        list: A list of results containing the extracted entity interactions for each paragraph.
    """
    
    list_results = []
    for i in range(len(list_pars)):
        print(i)
        time.sleep(0.3)
        paragraph = list_pars[i]
        for j in range(3):
            try:
                list_result = queryGptRelations(paragraph, model)            
                list_results.append(list_result)
                break
            except:
                time.sleep(1)
                list_result = queryGptRelations(paragraph, model)
                list_results.append(list_result)
    return list_results


def preprocessInteractions(list_result):
    """
    Preprocesses a list of results by parsing interactions and creating a DataFrame.

    Args:
        list_result (list): A list of results.

    Returns:
        pandas.DataFrame: A DataFrame containing the parsed interactions.
    """

    list_interactions = []
    for i in range(len(list_result)):
        
        try:
            dict_result = json.loads(list_result[i])
            
            if len(dict_result) == 0:
                dict_result["interactions"] = []
            
        except:
            dict_result = {"interactions": []}
        
        list_interactions.append(dict_result["interactions"])

    df_interactions = list()
    for i in range(len(list_interactions)):
        
        df_temp = pd.DataFrame(list_interactions[i])
        df_temp['paragraph'] = i + 1
        df_interactions.append( df_temp )
    
    df_interactions = pd.concat(df_interactions)
        
    return(df_interactions)
    

def preprocessPersons(list_result):
    """
    Preprocesses a list of results and extracts persons/entities of type 'person'.

    Args:
        list_result (list): A list of the results.

    Returns:
        set: A set of unique persons/entities of type 'person'.
    """

    list_entities = []
    for i in range(len(list_result)):
        
        try:
            dict_result = json.loads(list_result[i])
            
            if len(dict_result) == 0:
                dict_result["entities"] = []
            
        except:
            dict_result = {"entities": []}
        
        list_entities.append(dict_result["entities"])
    
    list_persons = list()
    for i in range(len(list_entities)):
        # df_temp = pd.DataFrame(list_entities[i])
        if len(list_entities[i]) > 0:
            list_persons.append(list({k for k, v in list_entities[i].items() if v == 'person'}))
    set_persons = set(np.concatenate(list_persons))
    
    return set_persons



def replaceAbbreviations(list_pars, dict_syn):
    """
    Replaces abbreviations in a list of paragraphs with their corresponding full forms.

    Args:
        list_pars (list): A list of paragraphs containing abbreviations.
        dict_syn (dict): A dictionary mapping abbreviations to their full forms.

    Returns:
        list: A list of paragraphs with abbreviations replaced by their full forms.
    """
    list_result = []
    
    for par in list_pars:
        
        par_new = par
        for short, full in dict_syn.items():
            par_new = par_new.replace(short, full)
        
        list_result.append(par_new)
            
    return(list_result)


def replaceAll(text, dic):
    for i, j in dic.items():
        text = text.replace(i, j)
    return text


def removeDuplicatedLabels(sent):
    """
    Removes duplicated labels from the input sentence.

    Args:
        sent (str): The input sentence containing labels.

    Returns:
        str: The modified sentence with duplicated labels removed.
    """

    sent = sent.split(' ')
    sent = [t.replace("_PERSON", "", 1) if t.count('_PERSON')==2 else t for t in sent]
    sent = [t.replace("_ORG", "", 1) if t.count('_ORG')==2 else t for t in sent]
    sent = [t.replace("_LAW", "", 1) if t.count('_LAW')==2 else t for t in sent]
    sent = ' '.join(sent)
    
    return sent


def queryGptRelations(text, model):
    """
    Extract pairwise interactions between named entities in the given text using a relation extraction model.
    
    Args:
        text (str): The input text containing named entities and their interactions.
        model (str): The name or ID of the relation extraction model to use.
    
    Returns:
        str: The extracted interactions between named entities in JSON format.
    """
    
    task = ''' 
    
    You are a model for relation extractions. 
    Extract pairwise interactions between named entities in the text. Provide output in JSON format.
    If there are no interactions output {}
    
    Example:
        
    Input text:  
    Smith regularly received money from Jane Black. In order to secure business from the airlines, 
    most sales personnel used Adidas agents. McDonalds employee 1 sent an email 
    to Adidas employee 2 on September 23, 2021. James Snyder was employed at McDonalds. 
    Jane Black discussed it with his company. 
    
    Output: {
        
        "enities": {"Smith":"person",
                    "Jane Black":"person",
                    "Mcdonalds employee 1":"person",
                    "Adidas employee 2":"person",
                    "James Snyder":"person",
                    "McDonalds":"company"},

        "interactions": [
            {
                "from": "Smith",
                "to": "Jane Black",
                "interaction": "recieved money from",
                "date": "nan"
            },
            {
                "from": "Mcdonalds employee 1",
                "to": "Adidas employee 2",
                "interaction": "emailed to",
                "date": "September 23, 2021"
            },
            {
                "from": "James Snyder",
                "to": "McDonalds",
                "interaction": "was employed at",
                "date": "2010"
            }
        ]
    }
    
    Here is the text:
    '''
    
    gpt_input = (task + text)
    
    # NER
    completion = openai.ChatCompletion.create(model=model, 
                                              messages=[{"role": "user", "content": gpt_input}],
                                              temperature=0.0)
    
    gpt_output = completion.choices[0].message.content
    
    return gpt_output


def createAbbEntityDict(case):
    """
    Create a dictionary mapping abbreviations to corresponding entities.

    Args:
        case (str): The case name used to select the abbreviation style pattern.

    Returns:
        dict: A dictionary mapping abbreviations to corresponding entities.
    """
    # ----- 3. Full pipeline    

    NER = spacy.load("en_core_web_trf")    

    # Select abbreviation style pattern based on case
    patterns = {
        'Sarclad': r'\[“[^\]]+\”]',
        'Gurlap': r'\(“.*?”\)',
        'Airbus': r'\(“.*?”\)',
        'ASL': r'\[‘[^\]]+\’]',
        'StdBank': r'\[[^\]]+\]' 
    }
    pattern = patterns.get(case)
    
    # Load pdf
    doc = fitz.open("{}/sof.pdf".format(case))
    
    # Get combined text
    text = ''
    for i in range(len(doc)):
    
        # Get the text from this page
        page_text = doc[i].get_text("text")
    
        # Append the text from this page to the overall string
        text += page_text
    
    # Split to sentences
    sentences_full = sent_tokenize(text)
    
    # Select only sentences with abbreviations
    sentences = []
    for sentence in sentences_full:
        if re.search(pattern, sentence):
            sentences.append(sentence)
    
    # Clean
    sentences = [replaceAll(sent, {',':'', '\n':'', '.':''}) for sent in sentences]
    sentences = [ sent if ('”)' in sent) else replaceAll(sent, {')':'', '(':''}) for sent in sentences ]
    
    # Extract entities
    list_entities = []
    for i in range(len(sentences)):
    
        sentence = sentences[i]
        spacy_obj = NER(sentence)
        entities = {token.text:token.text.replace(' ', '_') + '_' + token.label_ 
                    for token in spacy_obj.ents if token.label_ in {'PERSON', 'ORG', 'LAW'}}
        list_entities.append(entities)
    
    # Replace entities with entities_LABEL 
    sentences = [ replaceAll(sentences[i], list_entities[i]) for i in range(len(sentences)) ]
    
    # Remove duplicated labels   
    sentences = [removeDuplicatedLabels(sent) for sent in sentences]
    
    # Create dictionary with abbreviation:entity
    dict_abb_ent = {}
    for i in range(len(sentences)):
        
        sent = sentences[i]
        
        for abb in re.findall(pattern, sent):
            
            ent = sent.split(abb)[0].strip().split(' ')[-1]
            if abb not in dict_abb_ent:
                dict_abb_ent[abb] = ent
    
    # Clean dictionary
    dict_abb_ent = {k:v for k, v in dict_abb_ent.items() if any([x in v for x in ['_ORG', '_PERSON', 'LAW']])}
    
    dict_abb_ent = {k:v for k, v in dict_abb_ent.items() if re.search(pattern, k)}
    dict_abb_ent = {replaceAll(k, {'_ORG':'', '_PERSON':'', '_LAW':'', '_':' ',}):
                    replaceAll(v, {'_ORG':'', '_PERSON':'', '_LAW':'', '_':' '}) 
     for k, v in dict_abb_ent.items()}
    dict_abb_ent = {re.sub("[^a-zA-Z0-9- ]+", "", k):re.sub("[^a-zA-Z0-9- ]+", "", v) 
     for k, v in dict_abb_ent.items()}
    dict_abb_ent = {k:v for k, v in dict_abb_ent.items() if k[0].isupper()}
    
    # Add manually detected abb:ent
    dict_abb_ent = dict_abb_ent | {'SMO':'Strategy and Marketing Organisation',
     # 'BP':'Business Partner',
     'Fu Fu':'Intermediary 2',
     'Fu Funien':'Intermediary 2',
     'Van Gogh':'TNA Parent Executive 3',
     'GSL':'Guralp Systems Limited',
     'Brightwell':'Roshfor Corporation',
     'CKS':'Cheong Kum Steel Co Ltd',
     'Handran':'Handan Iron & Steel Stock Co. Ltd',
     'Benxi':'Benxi Iron & Steel (Group) International Economic & Trading co. Ltd',
     'Sunag':'Sunag Engineering',
     'IMD':'International Market Development'
     }
    
    return dict_abb_ent


def matchNamesGptCoders(df, case, dict_gpt_coder):
    
    if case!='ASL':
        
        dict_case = dict_gpt_coder.get(case)
             
        df['from'] = df['from'].replace(dict_case)
        df['to'] = df['to'].replace(dict_case)
    
    return df


def removeLowerCasedEntities(df):
    
    df = df[df['from'].str[0].str.isupper()]
    df = df[df['to'].str[0].str.isupper()]
    df = df[df['from'].apply(lambda x: x not in {'IMD', 'Sunag', 'Benxi', 'Chief Engineer'})]
    df = df[df['to'].apply(lambda x: x not in {'IMD', 'Sunag', 'Benxi', 'Chief Engineer'})]

    return df


def extractEntities(df):
    
    return np.unique(np.concatenate([df['from'].values, df['to'].values]))


def queryGptDeduplication(text, model):
    """
    Perform GPT-based deduplication on a list of names.

    Args:
        text (str): The list of names to be deduplicated.
        model (str): The GPT model to be used for deduplication.

    Returns:
        str: The deduplicated list of names as a string.
    """
    
    task = ''' 
    You are given a list of names. 
    Group the names if they are referring to the same person. 
    
    Example:
        
    Input: ['John Black', 'James Snyder', 'Black', 'Dr. Black']
    
    Output:  {'John Black':['Black',  'Dr. Black'], 'James Snyder':[]}
    
    Output only dictionary without additional comments. 
    
    Here is the list of names:
    '''
    
    gpt_input = (task + text)
    completion = openai.ChatCompletion.create(model=model, 
                                              messages=[{"role": "user", "content": gpt_input}],
                                              temperature=0.0)
    gpt_output = completion.choices[0].message.content
    
    return gpt_output


def deduplicateEntities(df):
    """
    Deduplicates entities in the given DataFrame by replacing duplicate values in the 'from' and 'to' columns.
    
    Args:
        df (pandas.DataFrame): The DataFrame containing the entities to be deduplicated.
        
    Returns:
        pandas.DataFrame: The DataFrame with deduplicated entities.
    """
    list_persons = extractEntities(df)
    
    gpt_output = queryGptDeduplication(str(list_persons), 'gpt-4')
    
    dict_groups = eval(gpt_output)
    dict_groups = {k:v for k, v in dict_groups.items() if len(v)>0}
    dict_groups = {v: k for k, vs in dict_groups.items() for v in vs}
    
    df['from'] = df['from'].replace(dict_groups)
    df['to'] = df['to'].replace(dict_groups)
    
    return df


# -----------------------------------------------------------------------------
# ----- Constants
# -----------------------------------------------------------------------------

openai.api_key = os.getenv('OPENAI_API_KEY')

# Working directory
WD = '/data'

# NER model
NER = spacy.load("en_core_web_trf")

# Cases and models
LIST_CASES = ['Airbus', 'ASL', 'Gurlap', 'Sarclad', 'StdBank']
LIST_MODELS = ['gpt-3.5-turbo', 'gpt-4']
LIST_PARAMS = list(itertools.product(LIST_CASES, LIST_MODELS))

# Match entities between GPT and coders
dict_gpt_coder = {
    
    'Sarclad': {
        
        'Telinen of Brightwell':'Telinen',
        'Pat': 'Jones',
        'Adrian Leek': 'Leek',
        'Guang Jiang': 'Jiang',
        'David Justice': 'Justice',
        'Michael Sorby': 'Sorby',
        'Hansam': 'Hansam',
        'Wiggans': 'Wiggans',
        'Kathy': 'Wiggans',
        'Mr Chen': 'Chen',
        'Shillam': 'Shillam',
        'Rao': 'Rao',
        'Teruo Maehata': 'Maehata',
        'Kathy Wiggans': 'Wiggans',
        'Peter Telinen': 'Telinen',
        'Neumeyer': 'Neumeyer',
        'Kim': 'Kim',
        'Mr Koo': 'Koo',
        'Mike Michael Sorby': 'Sorby',
        'Mike': 'Sorby',
        'Dawtry': 'Dawtry',
        'Guang': 'Jiang',
        'Mr Rao': 'Rao',
        'M.R. Rao': 'Rao',
        'H.D. Vasa': 'Vasa',
        'Pat Jones': 'Jones',
        'Jennifer Kim': 'Kim',
        'Charlie Park': 'Park_C',
        'Judy Park': 'Park',
        'M. R. Rao': 'Rao',
        'Anand Kumar': 'Kumar',
        'Lin': 'Lin',
        'Suzanne Roberts': 'Roberts',
        'Mr Wang': 'Chen',
        'Taewan Kim': 'Kim',
        'Evans': 'Evans',
        'Rajib': 'Rajib',
        'Y S Choi': 'Choi',
        'Steve Shillito': 'Shilito',
        'Park': 'Park',
        'Maehata': 'Maehata',
        'Adrian Adrian Leek': 'Leek',
        'Hansam Mulsang': 'Hansam',
        'HD Vasa': 'Vasa'},
    
    'Gurlap': {'Heon-Cheol Chi': 'Dr Chi',
        'Natalie Pearce': 'Natalie Pearce',
        'Dr Güralp': 'Dr Guralp',
        'Andrew Bell': 'Andrew Bell',
        'Dr Cansun Güralp': 'Dr Guralp',
        'Dr Cho': 'Dr Cho',
        'Head of Sales, Natalie Pearce': 'Natalie Pearce',
        'Dr. Güralp': 'Dr Guralp'},
    
    'Airbus': {'Airbus employee 1': 'Airbus employee 1 [senior]',
        'Airbus employee 16': 'Airbus employee 16',
        'Airbus employee 15': 'Airbus employee 15 [senior]',
        'Intermediary 5': 'Intermediary 5',
        'Airbus employee 2': 'Airbus employee 2 [senior]',
        'Intermediary 4': 'Intermediary 4',
        'AirAsia Executive 1': 'AirAsia Executive 1',
        'Airbus employee 21': 'Airbus employee 21 [senior]',
        'Airbus employee 22': 'Airbus employee 22 [senior]',
        'Government Official 1': 'Government Official 1',
        'Intermediary 6': 'Intermediary 6',
        'Intermediary 2': 'Intermediary 2',
        'Airbus employee 17': 'Airbus employee 17 [senior]',
        'Airbus employee 10': 'Airbus employee 10',
        'Airbus employee 18': 'Airbus employee 18',
        'Airbus employee 13': 'Airbus employee 13',
        'PT Garuda Indonesia Persero Tbk Executive 2': 'Garuda Executive 2',
        'Intermediary 3': 'Intermediary 4',
        'Airbus employee 1 [senior]': 'Airbus employee 1 [senior]',
        'Airbus employee 4': 'Airbus employee 4 [very senior]',
        'PT Garuda Indonesia Persero Tbk Executive 3': 'Garuda Executive 3',
        'Airbus employee 5': 'Airbus employee 5 [very senior]',
        'Intermediary 7': 'Intermediary 7',
        'Airbus employee 20': 'Airbus employee 20',
        'Airbus employee 8': 'Airbus employee 8 [senior]',
        'Airbus employee 2 [senior]': 'Airbus employee 2 [senior]',
        'Airbus employee 19': 'Airbus employee 19',
        'Airbus employee 12': 'Airbus employee 12',
        'Minister of Finance': 'Minister of Finance',
        'Airbus employee 21 [senior]': 'Airbus employee 21 [senior]',
        'Airbus employee 14': 'Airbus employee 14',
        'Airbus employee 9 [senior]': 'Airbus employee 9 [senior]',
        'TransAsia Airways Parent Executive 1': 'TNA Parent Executive 3',
        'UK Export Finance employee': 'UKEF employee',
        'SriLankan Airlines Executive 1': 'SLA Executive 1',
        'SMO International employee': 'SMO International employee',
        'Airbus employee 11': 'Airbus employee 11',
        'Airbus employee 7': 'Airbus employee 8 [senior]',
        'Airbus employee 15 [senior]': 'Airbus employee 15 [senior]',
        'Airbus employee 6 [very senior]': 'Airbus employee 6 [very senior]',
        'Airbus employee 22 [senior]': 'Airbus employee 22 [senior]',
        'Airbus employee 9': 'Airbus employee 9 [senior]',
        'Airbus employee 4 [very senior]': 'Airbus employee 4 [very senior]',
        'Airbus employee 5 [very senior]': 'Airbus employee 5 [very senior]',
        'PT Garuda Indonesia Persero Tbk Executive 1':'Garuda Executive 1',
        'Airbus employee 6':'Airbus employee 6 [very senior]',
        'Strategy and Marketing Organisation International employee':'SMO International employee',
        'UK Export Finance personnel':'UKEF employee',
        'TransAsia Airways Parent Executive 3':'TNA Parent Executive 3'
        },
    
    'StdBank': {'Shose Sinare': 'Shose Sinare',
        
        'Mr Hartig':'Florian Von Hartig',
        'CEO [Bashir Awale]': 'Bashir Awale',
        'Standard Bank plcG Head [Florian von Hartig]':'Florian Von Hartig',
        'Shose':'Shose Sinare',
        'M JACKSON':'Mike Jackson',
        'Secondee J':'SB Secondee J',
        'Mboya':'Dr Fratern Mboya',
        'Florian von Hartig': 'Florian Von Hartig',
        'Bashir Awale': 'Bashir Awale',
        'Standard Bank plc Employee H': 'SB Employee H',
        'Standard Bank plc Employee I': 'SB Employee I',
        'Minister B': 'Minister B',
        'Stanbic Bank Tanzania Limited Employee K': 'ST Employee K',
        'Standard Bank plc Secondee J': 'SB Secondee J',
        'Harry Kitilya': 'Harry Kitilya',
        'Florian Hartig': 'Florian Von Hartig',
        'Minister A': 'Minister A',
        'Stanbic Bank Tanzania Limited Employee Z': 'ST Employee Z',
        'Public Official X': 'Public Official X',
        'Dr Mboya': 'Dr Fratern Mboya',
        'Standard Bank plc Employee G': 'SB Employee G',
        'Bashir': 'Bashir Awale',
        'Mr Kitiliya': 'Harry Kitilya',
        'Mr Kitilya': 'Harry Kitilya',
        'Nyabuti': 'Peter Nyabuti',
        'Minister A': 'Minister A',
        'Kitilya': 'Harry Kitilya',
        'Public Official E. C. Bashir Awale': 'Public Official E',
        'Dr. Fratern Mboya': 'Dr Fratern Mboya',
        'Mr. H. KITILYA': 'Harry Kitilya',
        'Dr Fratern Mboya': 'Dr Fratern Mboya',
        'Public Official N': 'Public Official N',
        
        'Employee K':'ST Employee K'
        
    }
}

# -----------------------------------------------------------------------------
# ----- 1. PDF PRERPOCESSING
# -----------------------------------------------------------------------------

for case in LIST_CASES:
    
    print(case)    

    # --- 1. Extract paragraphs
    list_pars = extractPars(case)
    
    # --- 2. Replace Abbreviations with full names
    dict_abb_ent = createAbbEntityDict(case)
    list_pars = [replaceAll(par, dict_abb_ent) for par in list_pars]
    print(len(list_pars))
        
    # ~ Save 
    pd.Series(list_pars).to_pickle(WD + case + '/list_pars.obj')



# Count the number of words in each paragraph
tokenizer = RegexpTokenizer(r'\w+')

list_nwords = []
for case in LIST_CASES:
    
    # Count the number of words in each paragraph
    list_pars = extractPars(case)
    list_nwords_case = [len(tokenizer.tokenize(par)) for par in list_pars]
    list_nwords.append(list_nwords_case)

# Average number of words in paragraph by case
list(map(lambda x: np.mean(x), list_nwords))


# -----------------------------------------------------------------------------
# ----- 2. EXTRACT NETWORKS WITH CHATGPT
# -----------------------------------------------------------------------------

for params in LIST_PARAMS:

    case = params[0]
    model = params[1]
    print([case, model])
    list_pars = list(pd.read_pickle(WD + case + '/list_pars.obj'))
    list_jsons = parseEntitiesInteractions(list_pars, model)
    
    # Save 
    pd.Series(list_jsons).to_pickle(WD + case + '/list_jsons_' + model + '.obj')
    

# -----------------------------------------------------------------------------
# ----- 3. PREPROCESS NETWORKS
# -----------------------------------------------------------------------------

for params in LIST_PARAMS:

    case = params[0]
    model = params[1]
    
    # --- 1. Preprocess interactions
    list_jsons = list(pd.read_pickle(WD + case + '/list_jsons_' + model + '.obj'))
    df_interactions = preprocessInteractions(list_jsons)
    
    # --- 2. Select only persons
    set_persons = preprocessPersons(list_jsons)
    df_interactions = df_interactions[
         (df_interactions['from'].isin(set_persons)) & (df_interactions['to'].isin(set_persons))]
    
    # --- 3. Remove lowercased entities
    df_interactions = removeLowerCasedEntities(df_interactions)
    
    # --- 4. Deduplicate entities with ChatGPT
    df_interactions = deduplicateEntities(df_interactions)
    
    # --- 5. Match entities with coders
    df_interactions = matchNamesGptCoders(df_interactions, case, dict_gpt_coder)
    
    print(params)
    
    # ~ Save
    df_interactions.to_pickle(WD + case + '/df_interactions_' + model + '.obj')

    
