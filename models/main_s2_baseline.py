import os
import json
import orjson
import requests
import time

from tqdm import tqdm

from datetime import datetime

from ExtractionUtils import *

# Threading
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

# Thread lock for saving the cache
from threading import Lock
THREAD_LOCK_CACHE = Lock()


#
#   Semantic Scholar Wrapper
#

FILENAME_S2_CACHE = "s2_cache.json"

global_s2cache = None
global_num_cache_writes_since_save = 0

class SemanticScholar():
    def __init__(self, api_key=None):
        self.api_key = api_key

    def _snippet_search(self, query, limit=10, latest_year=None, latest_month=None):
        # Rate limit
        #time.sleep(1.5)
        time.sleep(0.1)

        # Set the fields to return, which are the above, plus the paper title and publication date
        #fields = "paper.title,paper.publicationDate,snippet.text,snippet.snippetKind,snippet.section,snippet.snippetOffset,snippet.annotations.refMentions,snippet.annotations.sentences"

        url = "https://api.semanticscholar.org/graph/v1/snippet/search"
        params = {
            "query": query,
            "limit": limit,
            #"fields": fields,
        }
        if (latest_year is not None):
            if (latest_month is not None):
                # Add the year and month to the query (up to latest_year and latest_month)
                params["publicationDateOrYear"] = f":{latest_year}-{latest_month:02d}"
            else:
                # Add the year range to the query (up to latest_year)
                params["publicationDateOrYear"] = f":{latest_year}"

        headers = {}
        if self.api_key:
            # Use the API key string, not the object
            headers["x-api-key"] = self.api_key
        response = requests.get(url, params=params, headers=headers)

        if response.status_code == 200:
            data = response.json()
            return data
        else:
            print("Error:", response.status_code)
            print(response.text)
            return None

    # Bulk paper search, with abstracts
    def _bulk_abstract_search(self, query, limit=10, latest_year=None, latest_month=None):
        # Rate limit
        #time.sleep(1.5)
        time.sleep(0.1)

        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": query,
            "limit": limit,
            "fields": "title,abstract,corpusId,publicationDate"
        }
        if (latest_year is not None):
            if (latest_month is not None):
                # Add the year and month to the query (up to latest_year and latest_month)
                params["publicationDateOrYear"] = f":{latest_year}-{latest_month:02d}"
            else:
                # Add the year range to the query (up to latest_year)
                params["publicationDateOrYear"] = f":{latest_year}"

        headers = {}
        if self.api_key:
            # Use the API key string, not the object
            headers["x-api-key"] = self.api_key
        response = requests.get(url, params=params, headers=headers)

        if response.status_code == 200:
            data = response.json()
            return data
        else:
            print("Error:", response.status_code)
            print(response.text)
            return None



    #
    #   Caching
    #
    def query_cache(self, query_type:str, query, limit, date_cutoff_yyyy=None, date_cutoff_mm=None):
        global global_s2cache
        # Check if the cache file exists
        if os.path.exists(FILENAME_S2_CACHE):
            # Make a key for this query
            try:
                query_key = str(query_type) + "-" + str(query) + "-limit" + str(limit) + "-datecutoff" + str(date_cutoff_yyyy) + "-" + str(date_cutoff_mm)

                # Load the cache file, if we haven't already
                if (global_s2cache is None):
                    print("Loading cache file from disk: " + FILENAME_S2_CACHE)
                    with open(FILENAME_S2_CACHE, 'r') as f:
                        # Faster JSON loading
                        import orjson
                        global_s2cache = orjson.loads(f.read())
                        #data = json.load(f)
                # Check if the query is in the cache
                if (query_key in global_s2cache):
                    return global_s2cache[query_key]

            except Exception as e:
                print(f"Error loading cache file {FILENAME_S2_CACHE}: {e}")
                import traceback
                traceback.print_exc()
                return None

        return None

    def save_cache(self, query_type:str, query, limit, date_cutoff_yyyy=None, date_cutoff_mm=None, data_to_store=None, force_save=False):
        # Check if the cache file exists
        global global_s2cache
        global global_num_cache_writes_since_save
        with THREAD_LOCK_CACHE:
            data = {}
            query_key = str(query_type) + "-" + str(query) + "-limit" + str(limit) + "-datecutoff" + str(date_cutoff_yyyy) + "-" + str(date_cutoff_mm)
            # Set the key
            global_s2cache[query_key] = data_to_store
            global_num_cache_writes_since_save += 1

            # Save the data to the cache file
            if ((global_num_cache_writes_since_save > 0) and (global_num_cache_writes_since_save % 25 == 0)) or (force_save == True):
                # Save the cache file
                print("Saving cache file " + FILENAME_S2_CACHE)
                with open(FILENAME_S2_CACHE, 'w') as f:
                    json.dump(global_s2cache, f, indent=4)
                global_num_cache_writes_since_save = 0

            # with open(FILENAME_S2_CACHE, 'w') as f:
            #     print("Saving cache file " + FILENAME_S2_CACHE)
            #     json.dump(global_s2cache, f, indent=4)


    #
    #   Snippet search (with date cutoffs, and with caching)
    #
    def search_snippets(self, query, limit=10, date_cutoff_yyyy=None, date_cutoff_mm=None):
        # Check if there's a cache hit -- if so, return it
        cached_data = self.query_cache("snippet", query, limit, date_cutoff_yyyy, date_cutoff_mm)
        if (cached_data is not None):
            print("Cache hit for query: " + str(query))
            return cached_data

        # Do the regular snippet search
        MAX_RETRIES = 5
        for i in range(MAX_RETRIES):
            data = self._snippet_search(query, limit, latest_year=date_cutoff_yyyy, latest_month=date_cutoff_mm)
            if (data is not None):
                break
            else:
                delay_time = 5*(i+1)
                print("Error: No data found for query " + str(query) + ". Retrying... (" + str(i+1) + "/" + str(MAX_RETRIES) + ")")
                time.sleep(delay_time)

        #print("Data: ")
        #print(json.dumps(data, indent=4))

        if data is None:
            return None
        if ("data" in data):
            data = data["data"]
        else:
            return None

        # Pack it
        snippets = {
            "query": query,
            "limit": limit,
            "date_cutoff_yyyy": date_cutoff_yyyy,
            "date_cutoff_mm": date_cutoff_mm,
            "snippets": data
        }

        # Save the data to the cache file
        self.save_cache("snippet", query, limit, date_cutoff_yyyy, date_cutoff_mm, data_to_store=snippets)

        return snippets


    #
    #   S2 Bulk Paper Abstract Search based on query (with date cutoffs, and with caching)
    #
    def search_bulk_abstracts(self, query, limit=10, date_cutoff_yyyy=None, date_cutoff_mm=None):
        # Check if there's a cache hit -- if so, return it
        cached_data = self.query_cache("bulk_abstract", query, limit, date_cutoff_yyyy, date_cutoff_mm)
        if (cached_data is not None):
            print("Cache hit for query: " + str(query))
            return cached_data

        # Do the regular bulk abstract search
        MAX_RETRIES = 5
        for i in range(MAX_RETRIES):
            data = self._bulk_abstract_search(query, limit, latest_year=date_cutoff_yyyy, latest_month=date_cutoff_mm)
            if (data is not None):
                break
            else:
                delay_time = 5*(i+1)
                print("Error: No data found for query " + str(query) + ". Retrying... (" + str(i+1) + "/" + str(MAX_RETRIES) + ")")
                time.sleep(delay_time)

        #print("Data: ")
        #print(json.dumps(data, indent=4))
        if data is None:
            return None
        if ("data" in data):
            data = data["data"]
        else:
            return None

        # Pack it
        abstracts = {
            "query": query,
            "limit": limit,
            "date_cutoff_yyyy": date_cutoff_yyyy,
            "date_cutoff_mm": date_cutoff_mm,
            "abstracts": data,
        }
        # Save the data to the cache file
        self.save_cache("bulk_abstract", query, limit, date_cutoff_yyyy, date_cutoff_mm, data_to_store=abstracts)
        return abstracts




#
#   Load the benchmark
#
def load_benchmark(filename_in:str):
    # Load the JSON file
    data = None
    print("Loading file: " + filename_in)
    try:
        with open(filename_in, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading file {filename_in}: {e}")
        return None

    print("Loaded " + str(len(data)) + " claims from file " + filename_in)

    return data



#
#   Pre-cache S2 queries
#
def run_precache_s2_queries(filename_in:str, path_out:str, retrieval_mode:str, num_to_retrieve:int, DEBUG_LIMIT:int=-1, NUM_WORKERS:int=6):
    s2_key_file = "s2_key.donotcommit.txt"

    VALID_RETRIEVAL_MODES = ["snippet", "snippet-nodatelimit", "abstract", "abstract-nodatelimit"]
    if (retrieval_mode not in VALID_RETRIEVAL_MODES):
        print("Error: Invalid retrieval mode: " + str(retrieval_mode))
        print("Valid retrieval modes are: " + str(VALID_RETRIEVAL_MODES))
        return

    # Load the claims
    claims = load_benchmark(filename_in)
    if (DEBUG_LIMIT >= 0):
        print("DEBUG_LIMIT is set to " + str(DEBUG_LIMIT) + ", so only processing the first " + str(DEBUG_LIMIT) + " claims.")
        claims = claims[0:DEBUG_LIMIT]
        time.sleep(3)

    print("Loaded " + str(len(claims)) + " claims from file " + filename_in)

    # Initialize the SemanticScholar API
    with open(s2_key_file, 'r') as f:
        s2_api_key = f.read()
    s2 = SemanticScholar(s2_api_key)


    # Process each claim
    start_time = time.time()

    # Threaded
    num_processed = 0
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = []
        for claim_problem in claims:
            claim_text = claim_problem["claim_text"]
            retrieval_limit_yyyy = claim_problem["metadata"]["exclude_date"]["year"]
            retrieval_limit_mm = claim_problem["metadata"]["exclude_date"]["month"]

            # Submit the task
            if (retrieval_mode == "snippet"):
                # Perform the snippet search
                future = executor.submit(s2.search_snippets, claim_text, limit=num_to_retrieve, date_cutoff_yyyy=retrieval_limit_yyyy, date_cutoff_mm=retrieval_limit_mm)
                futures.append(future)
            elif (retrieval_mode == "snippet-nodatelimit"):
                # Perform the snippet search with no date limit
                future = executor.submit(s2.search_snippets, claim_text, limit=num_to_retrieve, date_cutoff_yyyy=None, date_cutoff_mm=None)
                futures.append(future)


        # Process tasks as they complete.
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            # Save the output file
            result = future.result()
            if (result == None):
                print("Error: result is None.")
                continue
            try:
                num_processed += 1
                # Update the progress bar
                with tqdm(total=len(futures)) as pbar:
                    pbar.update(num_processed)
                    pbar.set_postfix({"Completed": num_processed, "Time": str(time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))), "Remaining": str(time.strftime("%H:%M:%S", time.gmtime((time.time() - start_time)/num_processed*(len(futures)-num_processed))))})


            except Exception as e:
                print("Error processing claim: " + str(e))
                continue

    s2.save_cache("snippet", "", 0, date_cutoff_yyyy=None, date_cutoff_mm=None, data_to_store=None, force_save=True)

    print("Precaching complete...")



#
#   S2 Retrieval Model
#

def answer_claim_s2_baseline(claim_id:str, claim:str, claim_gold_label:bool, retrieval_mode:str="snippet", num_to_retrieve:int=10, retrieval_limit_yyyy:int=2025, retrieval_limit_mm:int=12, model_str:str="gpt-4o-mini", temperature:float=0.0, max_tokens:int=4000, sch:SemanticScholar=None):

    def mkPrompt(retrieval_data:list=None):
        prompt = "You are ScientistGPT, the most advanced automated scientific discovery and inference system in the world. You can use your enormous intellect to solve any scientific problem, and always do so by breaking the problem down into its finest details.  You are accurate, faithful, and act with the highest scientific integrity.\n"
        prompt += "\n"
        prompt += "# Task\n"
        prompt += "Your task is to consider a scientific claim, and determine whether how it is true/feasible, or false/infeasible. You must also provide a brief explanation as to why you consider the claim more likely to be true/feasible or false/infeasible.\n"
        prompt += "If you consider the claim more likely to be true/feasible, then you should output True. If you consider the claim more likely to be false/infeasible, then you should output False.\n"
        prompt += "To help you perform this, a search was performed on Semantic Scholar for paper abstracts and/or paper snippets that are related to the claim.  These are provided below.\n"
        prompt += "The claim is provided below.\n"
        prompt += "\n"
        prompt += "# Claim\n"
        prompt += "The scientific claim for you to evaluate is:\n"
        prompt += "```\n"
        prompt += claim + "\n"
        prompt += "```\n"
        prompt += "\n"


        # N-shot examples
        filename_nshot = "20-shot-prompt.trainingset.json"
        with open(filename_nshot, 'r') as f:
            nshot_data = json.load(f)
        prompt += "# Examples\n"
        prompt += "Below are " + str(len(nshot_data)) + " examples of claims and their evaluations.\n"
        prompt += "```\n"
        prompt += json.dumps(nshot_data, indent=4) + "\n"
        prompt += "```\n"
        prompt += "\n"

        prompt += "# Paper Abstracts / Paper Snippets\n"
        prompt += "The following paper abstracts and/or snippets were retrieved, which may contain information that is helpful for you to evaluate the claim.\n"
        prompt += "```\n"
        prompt += json.dumps(retrieval_data, indent=4) + "\n"
        prompt += "```\n"
        prompt += "\n"

        prompt += "# Why you are doing this\n"
        prompt += "You are part of an automated scientific discovery system, that aims to perform scientific discovery tasks that could provide large positive benefits to humanity, through the discovery of new knowledge and the creation of new scientific artifacts.\n"
        prompt += "This is a very important task, and you must do it with the utmost care, attention to detail, accuracy, and scientific integrity.  Please do not hallucinate.\n"
        prompt += "Performing poorly on this task is a critical failure, and may result in highly negative consequences.\n"
        prompt += "\n"
        prompt += "# Output format\n"
        prompt += "Please produce your output in JSON format, between a single set of codeblocks (```).  Your output must be valid JSON, as it will be automatically parsed, and if it is not valid JSON then this will be a critical failure.\n"
        prompt += "You can (and are encouraged to) think and plan before writing the JSON, and can write text before the codeblock to help you think, to produce deductions/answers of the highest quality.  But the text inside the codeblock must be valid JSON.\n"
        prompt += "\n"
        prompt += "# Output description\n"
        prompt += "Your output must be in the following format:\n"
        prompt += "A dictionary with 2 main keys: `claim_true_or_false` and `explanation`.\n"
        prompt += "- `claim_true_or_false` is a boolean, that is True if you believe the claim is more likely true/feasible, or False if you believe the claim is more likely false/infeasible.\n"
        prompt += "- `explanation` is a string, that is a brief explanation as to why the claim is true or false.\n"
        prompt += "\n"
        prompt += "# Example output\n"
        prompt += "Here is an example of the output JSON format:\n"
        prompt += "```\n"
        prompt += "{\n"
        prompt += "    \"claim_true_or_false\": True or False,\n"
        prompt += "    \"explanation\": \"The claim is {true/feasible, false/infeasible} because... (be detailed, specific, and scientific, but try to limit your response to (at most) 5-6 sentences.\"\n"
        prompt += "}\n"
        prompt += "```\n"
        prompt += "\n"
        prompt += "Please produce your output now.\n"

        return prompt

    print("* Started S2 model claim (" + claim_id + "), claim text: " + str(claim) )

    # Perform the retrieval
    retrieval_data = None
    if (retrieval_mode == "snippet") or (retrieval_mode == "snippet-nodatelimit"):
        # Perform the snippet search
        snippet_list = None
        if (retrieval_mode == "snippet"):
            snippet_list = sch.search_snippets(claim, limit=num_to_retrieve, date_cutoff_yyyy=retrieval_limit_yyyy, date_cutoff_mm=retrieval_limit_mm)
        elif (retrieval_mode == "snippet-nodatelimit"):
            # No date limit
            snippet_list = sch.search_snippets(claim, limit=num_to_retrieve, date_cutoff_yyyy=None, date_cutoff_mm=None)

        if (snippet_list is not None):
            snippets = snippet_list["snippets"]
            # Extract the relevant fields from the snippet results
            retrieval_data = []
            for snippet in snippets:
                packed = {
                    "title": snippet["paper"].get("title", None),
                    "snippet": snippet["snippet"].get("text", None),
                    "snippetKind": snippet["snippet"].get("snippetKind", None),
                    "section": snippet["snippet"].get("section", None),
                }
                retrieval_data.append(packed)

    elif (retrieval_mode == "abstract") or (retrieval_mode == "abstract-nodatelimit"):
        # Perform the bulk abstract search
        abstract_list = None
        if (retrieval_mode == "abstract"):
            abstract_list = sch.search_bulk_abstracts(claim, limit=num_to_retrieve, date_cutoff_yyyy=retrieval_limit_yyyy, date_cutoff_mm=retrieval_limit_mm)
        elif (retrieval_mode == "abstract-nodatelimit"):
            # No date limit
            abstract_list = sch.search_bulk_abstracts(claim, limit=num_to_retrieve, date_cutoff_yyyy=None, date_cutoff_mm=None)

        if (abstract_list is not None):
            abstracts = abstract_list["abstracts"]
            # Extract the relevant fields from the abstract results
            retrieval_data = []
            for abstract in abstracts:
                packed = {
                    "title": abstract.get("title", None),
                    "abstract": abstract.get("abstract", None),
                }
                retrieval_data.append(packed)

    if (retrieval_data is None) or ((isinstance(retrieval_data, list) == True) and (len(retrieval_data) == 0)):
        print("Error: No retrieval data found for claim " + claim_id)
        return None

    # Keep track of total cost
    total_cost = 0.0
    # Create the prompt
    prompt = mkPrompt(retrieval_data=retrieval_data)

    metadata_all = []

    # Get the response
    responseJSON, responseText, cost, metadata = getLLMResponseJSONWithMetadata(promptStr=prompt, model=model_str, maxTokens=max_tokens, temperature=temperature, jsonOut=True)
    total_cost += cost
    metadata_all.append(metadata)

    final_response = responseJSON

    # Check whether the model response is correct
    correct = 0
    errors = []
    model_response = None
    if (final_response != None):
        try:
            model_response = final_response["claim_true_or_false"]
            if (model_response == claim_gold_label):
                correct = 1
            else:
                correct = 0
        except Exception as e:
            print("Error parsing claim correctness: " + str(e))
            correct = 0
            errors.append("Error parsing claim correctness: " + str(e))


    packed = {
        "claim_id": claim_id,
        "claim": claim,
        "response": final_response,
        "claim_gold_label": claim_gold_label,
        "model_label": model_response,
        "correct": correct,
        "responseText": responseText,
        "retrieval_info": {
            "retrieval_data": retrieval_data,
            "retrieval_mode": retrieval_mode,
            "retrieval_limit_yyyy": retrieval_limit_yyyy,
            "retrieval_limit_mm": retrieval_limit_mm,
            "num_to_retrieve": num_to_retrieve,
        },
        "initial_response": responseJSON,
        "model": model_str,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "cost": total_cost,
        "metadata": metadata_all,
    }

    print("* Completed S2 model claim (" + claim_id + ").  Cost: " + str(total_cost) + ", Correct: " + str(correct) + ", Claim: " + str(claim) + ", Model response length: " + str(len(responseText)))

    return packed



def run_s2_claim_llm_baseline(filename_in:str, path_out:str, mode:str, num_to_retrieve:int, model_str:str, temperature:float=0.0, max_tokens:int=4000, MAX_WORKERS:int=5, DEBUG_LIMIT:int=-1, restart_from_file:str=None):
    timestamp_str = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    s2_key_file = "s2_key.donotcommit.txt"

    VALID_RETRIEVAL_MODES = ["snippet", "snippet-nodatelimit", "abstract", "abstract-nodatelimit"]
    if (mode not in VALID_RETRIEVAL_MODES):
        print("Error: Invalid retrieval mode: " + str(mode))
        print("Valid retrieval modes are: " + str(VALID_RETRIEVAL_MODES))
        return

    # Load the claims
    claims = load_benchmark(filename_in)
    if (DEBUG_LIMIT >= 0):
        print("DEBUG_LIMIT is set to " + str(DEBUG_LIMIT) + ", so only processing the first " + str(DEBUG_LIMIT) + " claims.")
        claims = claims[0:DEBUG_LIMIT]
        time.sleep(3)

    print("Loaded " + str(len(claims)) + " claims from file " + filename_in)

    # Create the output directory if it doesn't exist
    os.makedirs(path_out, exist_ok=True)
    # Also make a subpath for the individual files
    results_path_out = os.path.join(path_out, "results-s2-" + mode + "-" + str(model_str) + "-" + timestamp_str)
    os.makedirs(results_path_out, exist_ok=True)

    # Initialize the SemanticScholar API
    with open(s2_key_file, 'r') as f:
        s2_api_key = f.read()
    s2 = SemanticScholar(s2_api_key)

    # Load the cache (by doing a faux query, which will cause it to load)
    print("Loading cache file from disk: " + FILENAME_S2_CACHE)
    s2.query_cache(query_type="", query="", limit=10, date_cutoff_yyyy=None, date_cutoff_mm=None)
    print("Cache file loaded.")

    # Create the output
    out = []

    # Implicitly add an index to the claims, so we can put them back in order after processing.  (Also create a claim_id to index mapping)
    claim_id_to_index = {}
    for idx in range(len(claims)):
        claims[idx]["index"] = idx
        claim_id = claims[idx]["claim_id"]
        claim_id_to_index[claim_id] = idx

    # Process each claim
    start_time = time.time()
    total_cost = 0.0
    num_processed_claims = 0
    all_results = []

    # Load any past results
    claim_ids_already_processed = set()
    if (restart_from_file is not None):
        # Load this JSON file
        print("Loading previous results from file: " + restart_from_file)
        try:
            with open(restart_from_file, 'r') as f:
                raw_restart_data = json.load(f)
                all_results = raw_restart_data["raw_results"]
            print("* Continuing from previous results, loaded " + str(len(all_results)) + " results.")
        except Exception as e:
            print(f"Error loading file {restart_from_file}: {e}")
            exit(1)
        # Get the claim IDs that have already been processed
        for result in all_results:
            claim_id = result["claim_id"]
            claim_ids_already_processed.add(claim_id)

        # Wait briefly
        time.sleep(5)




    # Set up the output filename
    timestamp_str = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    filename_out = "results_out.s2." + mode + "." + str(model_str) + "." + timestamp_str + ".json"
    filename_out = os.path.join(path_out, filename_out)
    # Do a test save so there aren't any surprises later, and we crash here if we can't save (instead of later)
    print("Performing a test save to: " + filename_out)
    with open(filename_out, 'w') as f:
        json.dump(all_results, f, indent=4)



    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        num_skipped = 0
        for claim in claims:
            claim_id = claim["claim_id"]
            claim_text = claim["claim_text"]
            claim_gold_label = claim["gold_label"]
            # Date cutoff
            exclude_date = claim["metadata"]["exclude_date"]
            cutoff_date_yyyy = exclude_date["year"]
            cutoff_date_mm = exclude_date["month"]

            # If this is a restart, skip any claims that have already been processed
            if (claim_id in claim_ids_already_processed):
                print("Skipping claim " + claim_id + " because it has already been processed.")
                num_skipped += 1
                continue
            # Submit the task
            future = executor.submit(answer_claim_s2_baseline, claim_id, claim_text, claim_gold_label, retrieval_mode=mode, num_to_retrieve=num_to_retrieve, retrieval_limit_yyyy=cutoff_date_yyyy, retrieval_limit_mm=cutoff_date_mm, model_str=model_str, temperature=temperature, max_tokens=max_tokens, sch=s2)
            futures.append(future)
        print("Submitted " + str(len(futures)) + " tasks to the executor.")
        print("(Skipped " + str(num_skipped) + " claims that were already processed.)")
        # Wait for all tasks to complete
        print("Waiting for all tasks to complete...")
        time.sleep(5)

        # Process tasks as they complete.
        for future in concurrent.futures.as_completed(futures):
            # Save the output file
            result = future.result()
            if (result == None):
                print("* Future: ERROR: result is None.")
                continue
            try:
                print("* Future: Recieved results for claim " + str(result["claim_id"]) + ": " + str(result))
                # Get the ID of the claim
                claim_id = result["claim_id"]
                # Get the original claim
                original_claim = None
                for claim in claims:
                    if (claim["claim_id"] == claim_id):
                        original_claim = claim
                        break
                # Add the original claim to the result
                result["original_claim"] = original_claim
                # Add the model parameters to the result
                result["model_info"] = {
                    "model": model_str,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }

                filename_out_paper = os.path.join(results_path_out, result["claim_id"] + ".result." + str(model_str) + ".json")

                print("Saving output to: " + filename_out_paper)
                with open(filename_out_paper, 'w') as f:
                    json.dump(result, f, indent=4)

                num_processed_claims += 1
                total_cost += result["cost"]
                print("Total cost: " + str(total_cost) + ", num claims processed: " + str(num_processed_claims), ", cost per claim: " + str(round(total_cost / num_processed_claims, 3)))

                # Keep track of the results
                all_results.append(result)

                # Periodically save the results to the output file
                if (num_processed_claims % 100 == 0) and (num_processed_claims > 0):
                    # Sort the results by claim_idx
                    all_results.sort(key=lambda x: x["original_claim"]["index"])

                    # Tally the scores
                    total_correct = 0
                    total = 0
                    performance_by_category = {}
                    total_counts = {}
                    for result in all_results:
                        if (result["correct"] == 1):
                            total_correct += 1
                        total += 1

                        # True/False
                        score = result["correct"]
                        label = str(result["original_claim"]["gold_label"])
                        qual_or_quant = str(result["original_claim"]["metadata"]["quant_or_qual"])

                        # True/false key
                        key_tf = label
                        if (key_tf not in performance_by_category):
                            performance_by_category[key_tf] = 0
                            total_counts[key_tf] = 0
                        performance_by_category[key_tf] += score
                        total_counts[key_tf] += 1

                        # Qual/quant key
                        key_qq = qual_or_quant
                        if (key_qq not in performance_by_category):
                            performance_by_category[key_qq] = 0
                            total_counts[key_qq] = 0
                        performance_by_category[key_qq] += score
                        total_counts[key_qq] += 1

                        # combined key
                        key_combined = qual_or_quant + "-" + label
                        if (key_combined not in performance_by_category):
                            performance_by_category[key_combined] = 0
                            total_counts[key_combined] = 0
                        performance_by_category[key_combined] += score
                        total_counts[key_combined] += 1

                    # Calculate the averages
                    averages = {}
                    for key in performance_by_category:
                        if (key in total_counts) and (total_counts[key] > 0):
                            averages[key] = performance_by_category[key] / total_counts[key]
                        else:
                            averages[key] = 0.0
                    print("Total correct: " + str(total_correct) + ", total: " + str(total) + ", accuracy: " + str(round(total_correct / total, 3)))


                    packed = {
                        "run_info": {
                            "filename": filename_in,
                            "info": "This is a run of the S2 retrieval model",
                            "retrieval_mode": mode,
                            "num_to_retrieve": num_to_retrieve,
                            "model_info": {
                                "model": model_str,
                                "temperature": temperature,
                                "max_tokens": max_tokens
                            },
                            "num_claims_processed": num_processed_claims,
                            "total_cost": total_cost,
                            "cost_per_claim": round(total_cost / num_processed_claims, 3),
                            "accuracy": round(total_correct / total, 3),
                            "timestamp": timestamp_str,
                            "num_claims_total": len(claims),
                            "num_claims_processed": num_processed_claims,
                            "num_claims_remaining": len(claims) - num_processed_claims,
                        },
                        "scores": {
                            "total_correct": total_correct,
                            "total": total,
                            "accuracy": round(total_correct / total, 4),
                            "performance_by_category": performance_by_category,
                            "total_counts": total_counts,
                            "averages": averages,
                        },
                        "raw_results": all_results
                    }

                    # Save the all_results list to a file
                    print("Saving results to: " + filename_out)
                    with open(filename_out, 'w') as f:
                        json.dump(packed, f, indent=4)
                    print("Saved results to: " + filename_out)

                # Update the progress bar
                from tqdm import tqdm
                with tqdm(total=len(futures)) as pbar:
                    pbar.update(num_processed_claims)
                    pbar.set_postfix({"Completed": num_processed_claims, "Time": str(time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))), "Remaining": str(time.strftime("%H:%M:%S", time.gmtime((time.time() - start_time)/num_processed_claims*(len(futures)-num_processed_claims))))})

            except Exception as e:
                print("Error saving output: " + str(e))
                continue


        # If we reach here, the model is completed -- perform a final save
        # Sort the results by claim_idx
        all_results.sort(key=lambda x: x["original_claim"]["index"])

        # Tally the scores
        total_correct = 0
        total = 0
        performance_by_category = {}
        total_counts = {}
        for result in all_results:
            if (result["correct"] == 1):
                total_correct += 1
            total += 1

            # True/False
            score = result["correct"]
            label = str(result["original_claim"]["gold_label"])
            qual_or_quant = str(result["original_claim"]["metadata"]["quant_or_qual"])

            # True/false key
            key_tf = label
            if (key_tf not in performance_by_category):
                performance_by_category[key_tf] = 0
                total_counts[key_tf] = 0
            performance_by_category[key_tf] += score
            total_counts[key_tf] += 1

            # Qual/quant key
            key_qq = qual_or_quant
            if (key_qq not in performance_by_category):
                performance_by_category[key_qq] = 0
                total_counts[key_qq] = 0
            performance_by_category[key_qq] += score
            total_counts[key_qq] += 1

            # combined key
            key_combined = qual_or_quant + "-" + label
            if (key_combined not in performance_by_category):
                performance_by_category[key_combined] = 0
                total_counts[key_combined] = 0
            performance_by_category[key_combined] += score
            total_counts[key_combined] += 1

        # Calculate the averages
        averages = {}
        for key in performance_by_category:
            if (key in total_counts) and (total_counts[key] > 0):
                averages[key] = performance_by_category[key] / total_counts[key]
            else:
                averages[key] = 0.0
        print("Total correct: " + str(total_correct) + ", total: " + str(total) + ", accuracy: " + str(round(total_correct / total, 3)))


        packed = {
            "run_info": {
                "filename": filename_in,
                "info": "This is a run of the S2 retrieval model",
                "retrieval_mode": mode,
                "num_to_retrieve": num_to_retrieve,
                "model_info": {
                    "model": model_str,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                },
                "num_claims_processed": num_processed_claims,
                "total_cost": total_cost,
                "cost_per_claim": round(total_cost / num_processed_claims, 3),
                "accuracy": round(total_correct / total, 3),
                "timestamp": timestamp_str,
                "num_claims_total": len(claims),
                "num_claims_processed": num_processed_claims,
                "num_claims_remaining": len(claims) - num_processed_claims,
            },
            "scores": {
                "total_correct": total_correct,
                "total": total,
                "accuracy": round(total_correct / total, 4),
                "performance_by_category": performance_by_category,
                "total_counts": total_counts,
                "averages": averages,
            },
            "raw_results": all_results
        }

        # Save the all_results list to a file
        print("Saving results to: " + filename_out)
        with open(filename_out, 'w') as f:
            json.dump(packed, f, indent=4)
        print("Saved results to: " + filename_out)






if __name__ == "__main__":
    # Load the API keys
    loadAPIKeys()

    # Load the evaluation set
    filename_test = "benchmark/matteroffact.test.20242025.4446.json"
    path_out = "results-claim"

    # Retrieve the top-N snippets for each claim
    NUM_TO_RETRIEVE = 20

    # Pre-cache S2 queries
    # If you plan to run multiple models, it's helpful to pre-cache the semantic scholar searches, so that you don't have to continually re-query for the same queries,
    # but also this then lets one run the calls in parallel later without having to wait for the semantic scholar rate limit.
    #run_precache_s2_queries(filename_test, path_out, retrieval_mode="snippet", num_to_retrieve=NUM_TO_RETRIEVE, DEBUG_LIMIT=-1)    # For the 'feasibility' mode
    #run_precache_s2_queries(filename_test, path_out, retrieval_mode="snippet-nodatelimit", num_to_retrieve=NUM_TO_RETRIEVE, DEBUG_LIMIT=-1)   # For the 'claim verification' mode (no date limit on S2 searches)

    # S2 Retrieval Model
    # GPT-4o-mini (s2, snippet)
    run_s2_claim_llm_baseline(filename_test, path_out, mode="snippet", num_to_retrieve=NUM_TO_RETRIEVE, model_str="gpt-4o-mini", temperature=0.0, max_tokens=1000, MAX_WORKERS=2, DEBUG_LIMIT=-1)
    #run_s2_claim_llm_baseline(filename_test, path_out, mode="snippet-nodatelimit", num_to_retrieve=NUM_TO_RETRIEVE, model_str="gpt-4o-mini", temperature=0.0, max_tokens=1000, MAX_WORKERS=25, DEBUG_LIMIT=-1)

    # Claude 3.7 (s2, snippet)
    #run_s2_claim_llm_baseline(filename_test, path_out, mode="snippet", num_to_retrieve=NUM_TO_RETRIEVE, model_str="claude-3-7-sonnet-20250219", temperature=0.0, max_tokens=1000, MAX_WORKERS=4, DEBUG_LIMIT=-1)
    #run_s2_claim_llm_baseline(filename_test, path_out, mode="snippet-nodatelimit", num_to_retrieve=NUM_TO_RETRIEVE, model_str="claude-3-7-sonnet-20250219", temperature=0.0, max_tokens=1000, MAX_WORKERS=4, DEBUG_LIMIT=-1)

    # GPT o4-mini-2025-04-16 (s2, snippet)
    #run_s2_claim_llm_baseline(filename_test, path_out, mode="snippet", num_to_retrieve=NUM_TO_RETRIEVE, model_str="o4-mini-2025-04-16", temperature=0.0, max_tokens=10000, MAX_WORKERS=50, DEBUG_LIMIT=-1)
    #run_s2_claim_llm_baseline(filename_test, path_out, mode="snippet-nodatelimit", num_to_retrieve=NUM_TO_RETRIEVE, model_str="o4-mini-2025-04-16", temperature=0.0, max_tokens=10000, MAX_WORKERS=25, DEBUG_LIMIT=-1)

    print("Completed.")
