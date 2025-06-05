# Part of this code-generation model is adapted from CodeScientist ("https://github.com/allenai/codescientist/").

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

# Maximum number of characters to display for a given log element
MAX_CHARS = 5000

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
#   Code Model
#

# Wrapper for running code in a Modal container
def run_code_in_container(code:str, requirements:str, path_out:str, timeout_sec:int=60*5):
    from modules.ModuleRunPythonInModal import ModuleRunPythonInModal
    moduleModal = ModuleRunPythonInModal()

    # Create the payload
    payload = {
        "input": {
            "base_path": path_out,
            "max_runtime_seconds": timeout_sec,
            "python_version": "3.12",
            "requirements.txt": requirements,
            "code": code,
        }
    }

    # Run the action
    result = moduleModal.runAction(moduleModal.ACTION_RUN_PROGRAM["name"], payload)
    print("run_code_in_container result:")
    print(json.dumps(result, indent=4))

    def trim_to_length(obj, max_char_length:int=10000):
        if isinstance(obj, str):
            if (len(obj) > max_char_length):
                obj = obj[:max_char_length] + "\n[TRUNCATED TO " + str(max_char_length) + " CHARACTERS]\n"
                return obj
            else:
                return obj
        else:
            # Turn it into a JSON string, then trim it
            obj_str = json.dumps(obj)
            if len(obj_str) > max_char_length:
                obj_str = obj_str[:max_char_length] + "\n[TRUNCATED TO " + str(max_char_length) + " CHARACTERS]\n"
                return obj_str
            else:
                return obj

    # Make a sanitized version of the result
    results_sanitized = None
    try:
        results_sanitized = {
            "pip.stdout": trim_to_length(result["output"]["pip.stdout"], max_char_length=MAX_CHARS),
            "pip.stderr": trim_to_length(result["output"]["pip.stderr"], max_char_length=MAX_CHARS),
            "python.stdout": trim_to_length(result["output"]["python.stdout"], max_char_length=MAX_CHARS),
            "python.stderr": trim_to_length(result["output"]["python.stderr"], max_char_length=MAX_CHARS),
            "log": trim_to_length(result["output"]["log"], max_char_length=MAX_CHARS),
            "results_json": trim_to_length(result["output"]["results_json"], max_char_length=MAX_CHARS),
            "return_code": result["output"]["return_code"],
            "other_errors": result["output"]["other_errors"],
            "sandbox_errors": result["output"]["sandbox_errors"],
            "modal_container_completed": result["output"]["modal_container_completed"],
            "statistics": result["output"]["statistics"],
        }
    except Exception as e:
        import traceback
        print("Error sanitizing result: " + str(e))
        traceback.print_exc()

    return result, results_sanitized

# Finds all the codeblocks in a string, and returns them as a list of lists of strings.
# Very useful when providing a format prompt to an LLM, as you can ask it to provide specific structured responses within a codeblock, then extract these.
# e.g. "Please respond in JSON format, as a dictionary with a single key, `answer', which is a number. Place your response between codeblocks (```)"
# Expected input_str:
# ```
# {
#    "answer": 42
# }
# ```
# Returns: ["{\n\"answer\": 42\n}"]
# Will handle multiple codeblocks in the input string.
# NOTE: This function is used in the LLM proxy code, and is critical for extracting structured data from LLM responses.
def find_codeblocks(input_str):
    # Find all codeblocks in the input string
    codeblocks = []
    lines = input_str.split("\n")
    current_codeblock = []
    active = False

    for idx, line in enumerate(lines):
        if line.startswith("```"):
            if (active == True):
                # Finish off the current codeblock
                codeblocks.append(current_codeblock)
                current_codeblock = []
                active = False
            else:
                # Start a new codeblock
                active = True
        else:
            # If we're currently in the middle of a codeblock, add the line to the current codeblock (we never want the ``` to be included in the codeblock)
            if (active == True):
                current_codeblock.append(line)

    # Note: turn each of the lines in the codeblock into a single string, so we can parse it as JSON later
    for idx in range(len(codeblocks)):
        codeblocks[idx] = "\n".join(codeblocks[idx])

    return codeblocks


# LLM Code-generation baseline
def answer_code_baseline(claim_id:str, claim:str, claim_gold_label:bool, code_max_runtime_secs:int=60*5, path_out_code:str="generated-code-claims/", model_str:str="gpt-4o-mini", temperature:float=0.0, max_tokens:int=4000):

    def mkPrompt(generate_code_or_answer:str, past_requirements=None, past_code=None, code_result=None, code_max_runtime_secs:int=60*5):
        if (generate_code_or_answer not in ["code", "answer"]):
            print("Error: generate_code_or_answer must be 'code' or 'answer'")
            return None

        prompt = "You are ScientistGPT, the most advanced automated scientific discovery and inference system in the world. You can use your enormous intellect to solve any scientific problem, and always do so by breaking the problem down into its finest details.  You are accurate, faithful, and act with the highest scientific integrity.\n"
        prompt += "\n"
        prompt += "# Task\n"
        prompt += "Your task is to consider a scientific claim, and determine whether how it is true/feasible, or false/infeasible. You must also provide a brief explanation as to why you consider the claim more likely to be true/feasible or false/infeasible.\n"
        if (past_code is not None):
            prompt += "If you consider the claim more likely to be true/feasible, then you should output True. If you consider the claim more likely to be false/infeasible, then you should output False.\n"
        prompt += "To help you perform this, you will be allowed to author and run a Python program to generate results relevant to this feasibility claim, and observe its results.\n"
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


        if (past_code is not None):
            prompt += "# Past code\n"
            prompt += "You previously generated the following code to help provide evidence for this claim.  The code and the results of running it are below.\n"
            prompt += "\n"
            prompt += "## Requirements\n"
            prompt += "requirements.txt:\n"
            prompt += "```\n"
            prompt += past_requirements + "\n"
            prompt += "```\n"
            prompt += "\n"
            prompt += "## Code\n"
            prompt += "```\n"
            prompt += past_code + "\n"
            prompt += "```\n"
            prompt += "\n"
            prompt += "## Code Results\n"
            prompt += "```\n"
            prompt += json.dumps(code_result, indent=4) + "\n"
            prompt += "```\n"
            prompt += "\n"

        prompt += "# Why you are doing this\n"
        prompt += "You are part of an automated scientific discovery system, that aims to perform scientific discovery tasks that could provide large positive benefits to humanity, through the discovery of new knowledge and the creation of new scientific artifacts.\n"
        prompt += "This is a very important task, and you must do it with the utmost care, attention to detail, accuracy, and scientific integrity.  Please do not hallucinate.\n"
        prompt += "Performing poorly on this task is a critical failure, and may result in highly negative consequences.\n"
        prompt += "\n"

        if (generate_code_or_answer == "answer"):
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

        elif (generate_code_or_answer == "code"):
            prompt += "# Output format\n"
            prompt += "Your task is to now produce output code.  You must produce exactly two codeblocks (```).  The first will be copy/pasted into your `requirements.txt` file, and the second will be copy/pasted into your `main.py`.  They will then be run, and the output returned to you.\n"
            prompt += "## Special notes\n"
            prompt += "- Code output: The following output will be captured: stdout/stderr, the contents of a file called `log.json`, and the contents of a file called `results.json`.  All other files will not be returned, so you must use one of these three methods to report your findings.\n"
            prompt += "- The code will run in a container with a maximum runtime of " + str(round(code_max_runtime_secs/60, 1)) + " minutes, and will not be able to access special resources (like GPUs).\n"
            prompt += "- The code can install Python packages (through requirements.txt), but cannot install system packages (like apt-get).\n"
            prompt += "\n"
            prompt += "## Example of what you should generate:\n"
            prompt += "You must generate exactly two codeblocks.\n"
            prompt += "The first codeblock will be taken as the contents of a `requirements.txt` file, such as the one below:\n"
            prompt += "```\n"
            prompt += "numpy==1.26.4\n"
            prompt += "```\n"
            prompt += "The second codeblock will be taken as the contents of a `main.py` file, such as the one below:\n"
            prompt += "```\n"
            prompt += "import json\n"
            prompt += "import numpy as np\n"
            prompt += "\n"
            prompt += "def save_json(contents, filename_out):\n"
            prompt += "    with open(filename_out, \"w\") as f:\n"
            prompt += "        json.dump(contents, f, indent=4)\n"
            prompt += "\n"
            prompt += "print(\"hello world\")\n"
            # Perform some example computation with numpy in a loop 5 times
            prompt += "log = []\n"
            prompt += "sum = 0\n"
            prompt += "# Calculate the sum of the first MAX_NUM integers\n"
            prompt += "MAX_NUM = 5\n"
            prompt += "for i in range(MAX_NUM):\n"
            prompt += "    print(\"hello world\", i)\n"
            prompt += "    log.append(\"At iteration: \" + str(i))\n"
            prompt += "    sum += i\n"
            prompt += "    print(\"Sum: \" + str(sum))\n"
            prompt += "    # Save the log to a file\n"
            prompt += "    save_json(log, \"log.json\")\n"
            prompt += "\n"
            prompt += "# Save a final results.json file\n"
            prompt += "results = {\n"
            prompt += "    \"MAX_NUM\": MAX_NUM,\n"
            prompt += "    \"sum\": sum\n"
            prompt += "}\n"
            prompt += "save_json(results, \"results.json\")\n"
            prompt += "```\n"
            prompt += "\n"
            prompt += "## Important notes\n"
            prompt += "- You must generate exactly two codeblocks. The first codeblock will be taken as the contents of a `requirements.txt` file, and the second codeblock will be taken as the contents of a `main.py` file.\n"
            prompt += "- The contents of the requirements.txt and main.py will be automatically copied verbatim into the files, so you must ensure that they are valid and perfectly formatted. If they are invalid, this will be seen as a critical error.\n"
            prompt += "- The code will be run in a container with a maximum runtime of " + str(round(code_max_runtime_secs/60, 1)) + " minutes, and will not be able to access special resources (like GPUs).\n"
            prompt += "- Note that logs, results, or stdout/stderr greater than " + str(MAX_CHARS) + " characters will be truncated, for space.\n"
            prompt += "- You can write any text before the codeblocks that you would like, to help you think or plan.\n"
            prompt += "\n"
            prompt += "Please produce your output now.\n"

        return prompt


    # Keep track of total cost
    total_cost = 0.0
    # Create the first prompt to generate code
    prompt = mkPrompt(generate_code_or_answer="code", past_requirements=None, past_code=None, code_result=None, code_max_runtime_secs=code_max_runtime_secs)

    metadata_all = []

    # Get the response
    responseJSON, responseText, cost, metadata = getLLMResponseJSONWithMetadata(promptStr=prompt, model=model_str, maxTokens=max_tokens, temperature=temperature, jsonOut=False)
    total_cost += cost
    metadata_all.append(metadata)

    # Try to parse out the two codeblocks
    codeblocks = find_codeblocks(responseText)
    if (len(codeblocks) != 2):
        print("Error: Expected 2 codeblocks, but got " + str(len(codeblocks)) + ".  Response text: " + responseText)
        # Return a failure, so it's still counted (just as incorrect)
        packed = {
            "claim_id": claim_id,
            "claim": claim,
            "response": responseJSON,
            "claim_gold_label": claim_gold_label,
            "model_label": None,
            "correct": 0,
            "responseText": responseText,
            "code": {
                "generated_requirements_txt": None,
                "generated_main_py": None,
                "code_result": None,
                "code_max_runtime_secs": code_max_runtime_secs,
            },
            "model": model_str,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "cost": total_cost,
            "metadata": metadata_all,
        }
        return packed

    requirements_txt = codeblocks[0]
    main_py = codeblocks[1]

    # Now, try to run these in a container
    path_out = path_out_code + "/" + claim_id + "/"
    code_result, code_result_sanitized = run_code_in_container(code=main_py, requirements=requirements_txt, path_out=path_out, timeout_sec=code_max_runtime_secs)

    # Now, build a prompt to answer the claim
    prompt2 = mkPrompt(generate_code_or_answer="answer", past_requirements=requirements_txt, past_code=main_py, code_result=code_result_sanitized, code_max_runtime_secs=code_max_runtime_secs)
    # Get the response
    responseJSON, responseText, cost, metadata = getLLMResponseJSONWithMetadata(promptStr=prompt2, model=model_str, maxTokens=max_tokens, temperature=temperature, jsonOut=True)
    total_cost += cost

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
        "code": {
            "generated_requirements_txt": requirements_txt,
            "generated_main_py": main_py,
            "code_result": code_result,
            "code_max_runtime_secs": code_max_runtime_secs,
        },
        "model": model_str,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "cost": total_cost,
        "metadata": metadata_all,
    }

    return packed



def run_code_llm_baseline(filename_in:str, path_out:str, model_str:str, temperature:float=0.0, max_tokens:int=4000, MAX_WORKERS:int=5, DEBUG_LIMIT:int=-1, filename_in_continuation:str=None):
    timestamp_str = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

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
    results_path_out = os.path.join(path_out, "results-code-" + str(model_str) + "-" + timestamp_str)
    os.makedirs(results_path_out, exist_ok=True)

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

    # If we are continuing from a previous run, load the results
    claim_ids_already_completed = set()
    if (filename_in_continuation is not None):
        print("Loading previous results from: " + filename_in_continuation)
        with open(filename_in_continuation, 'r') as f:
            past_results = json.load(f)
        print("Loaded " + str(len(all_results)) + " results from file " + filename_in_continuation)

        num_with_errors = 0
        # Get the raw results
        for result in past_results["raw_results"]:
            # Get the claim ID
            claim_id = result["claim_id"]
            # Check to see if there was a failure/error
            model_label = result["model_label"]
            if (model_label is not None):
                # Add it
                all_results.append(result)
                # Add the claim ID to the set of completed claims
                claim_ids_already_completed.add(claim_id)
            else:
                num_with_errors += 1

        # Print the number of claims already completed
        print("Already completed " + str(len(claim_ids_already_completed)) + " claims.")
        print("Number of claims with errors (that we'll attempt to re-process): " + str(num_with_errors))
        print("Number of claims remaining: " + str(len(claims) - len(claim_ids_already_completed)))
        time.sleep(5)



    # Set up the output filename
    timestamp_str = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    filename_out = "results_out.code." + str(model_str) + "." + timestamp_str + ".json"
    filename_out = os.path.join(path_out, filename_out)
    # Do a test save so there aren't any surprises later, and we crash here if we can't save (instead of later)
    print("Performing a test save to: " + filename_out)
    with open(filename_out, 'w') as f:
        json.dump(all_results, f, indent=4)


    MAX_RUNTIME_SECS = 60*10
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []

        for claim in claims:
            claim_id = claim["claim_id"]
            claim_text = claim["claim_text"]
            claim_gold_label = claim["gold_label"]

            # Submit the task
            if (claim_id in claim_ids_already_completed):
                print("Skipping claim " + claim_id + " because it has already been completed.")
                continue

            future = executor.submit(answer_code_baseline, claim_id, claim_text, claim_gold_label, code_max_runtime_secs=MAX_RUNTIME_SECS, path_out_code=results_path_out, model_str=model_str, temperature=temperature, max_tokens=max_tokens)
            futures.append(future)

        # Process tasks as they complete.
        for future in concurrent.futures.as_completed(futures):
            # Save the output file
            result = future.result()
            if (result == None):
                print("Error: result is None.")
                continue
            try:
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
                            "model": model_str,
                            "temperature": temperature,
                            "max_tokens": max_tokens,
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
                            "total_evaluated": total,
                            "total_claims": len(claims),
                            "accuracy_by_evaluated": round(total_correct / total, 4),
                            "accuracy": round(total_correct / len(claims), 4),
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
                "model": model_str,
                "temperature": temperature,
                "max_tokens": max_tokens,
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
                "total_evaluated": total,
                "total_claims": len(claims),
                "accuracy_by_evaluated": round(total_correct / total, 4),
                "accuracy": round(total_correct / len(claims), 4),
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

    # Code LLM Baseline
    # GPT-4o-mini
    run_code_llm_baseline(filename_test, path_out, model_str="gpt-4o-mini", temperature=0.0, max_tokens=8000, MAX_WORKERS=2, DEBUG_LIMIT=-1)

    # Claude 3.7
    # NOTE: Token limit needs to be raised? (Claude likes to generate long code)
    #run_code_llm_baseline(filename_test, path_out, model_str="claude-3-7-sonnet-20250219", temperature=0.0, max_tokens=8000, MAX_WORKERS=8, DEBUG_LIMIT=-1)

    # GPT o4-mini-2025-04-16
    #run_code_llm_baseline(filename_test, path_out, model_str="o4-mini-2025-04-16", temperature=0.0, max_tokens=15000, MAX_WORKERS=50, DEBUG_LIMIT=-1)

    print("Completed.")
