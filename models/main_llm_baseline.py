# Main baselines

import os
import json
import random
import time

from ExtractionUtils import *
from tqdm import tqdm

# Threading
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

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
#   Chain-of-thought baseline
#
def answer_claim_llm_baseline(claim_id:str, claim:str, claim_gold_label:bool, model_str:str, temperature:float=0.0, max_tokens:int=4000, reflection:bool=True, nshot:bool=True):

    def mkPrompt(reflection=None, nshot=True):

        prompt = "You are ScientistGPT, the most advanced automated scientific discovery and inference system in the world. You can use your enormous intellect to solve any scientific problem, and always do so by breaking the problem down into its finest details.  You are accurate, faithful, and act with the highest scientific integrity.\n"
        prompt += "\n"
        prompt += "# Task\n"
        prompt += "Your task is to consider a scientific claim, and determine whether how it is true/feasible, or false/infeasible. You must also provide a brief explanation as to why you consider the claim more likely to be true/feasible or false/infeasible.\n"
        prompt += "If you consider the claim more likely to be true/feasible, then you should output True. If you consider the claim more likely to be false/infeasible, then you should output False.\n"
        prompt += "The claim is provided below.\n"
        prompt += "\n"
        prompt += "# Claim\n"
        prompt += "The scientific claim for you to evaluate is:\n"
        prompt += "```\n"
        prompt += claim + "\n"
        prompt += "```\n"
        prompt += "\n"

        # N-shot examples
        if (nshot == True):
            filename_nshot = "20-shot-prompt.trainingset.json"
            with open(filename_nshot, 'r') as f:
                nshot_data = json.load(f)
            prompt += "# Examples\n"
            prompt += "Below are " + str(len(nshot_data)) + " examples of claims and their evaluations.\n"
            prompt += "```\n"
            prompt += json.dumps(nshot_data, indent=4) + "\n"
            prompt += "```\n"
            prompt += "\n"

        if (reflection != None):
            prompt += "# Reflection\n"
            prompt += "This is a reflection step.  Previously, you generated output (below) for this task, and now you will reflect on that output, identify any issues with it (particularly with respect to accuracy, scientific integrity, completeness, quality, etc.), and fix all issues so that the data is of exceptional quality.\n"
            prompt += "The output from your initial step is below:\n"
            prompt += "--- START OF PREVIOUS OUTPUT ---\n"
            prompt += reflection + "\n"
            prompt += "--- END OF PREVIOUS OUTPUT ---\n"
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

    # Keep track of total cost
    total_cost = 0.0
    # Create the prompt
    prompt = mkPrompt(reflection=None, nshot=nshot)

    metadata_all = []

    # Get the response
    responseJSON, responseText, cost, metadata = getLLMResponseJSONWithMetadata(promptStr=prompt, model=model_str, maxTokens=max_tokens, temperature=temperature, jsonOut=True)
    total_cost += cost
    metadata_all.append(metadata)

    responseJSONReflection = None
    if (reflection == True):
        # Create the prompt
        prompt = mkPrompt(reflection=responseText, nshot=nshot)
        responseJSONReflection, responseText, cost, metadata = getLLMResponseJSONWithMetadata(promptStr=prompt, model=model_str, maxTokens=max_tokens, temperature=temperature, jsonOut=True)
        total_cost += cost
        metadata_all.append(metadata)

    # Pack the response

    final_response = responseJSON
    if (reflection == True) and (responseJSONReflection != None):
        final_response = responseJSONReflection

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
        "nshot": nshot,
        "reflection": reflection,
        "initial_response": responseJSON,
        "model": model_str,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "cost": total_cost,
        "metadata": metadata_all,
    }

    return packed


def run_answer_claim_llm_baseline(filename_in:str, path_out:str, model_str:str, temperature:float=0.0, max_tokens:int=4000, reflection:bool=True, nshot:bool=True, MAX_WORKERS:int=5, DEBUG_LIMIT:int=-1, filename_in_continuation:str=None):
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
    results_path_out = os.path.join(path_out,  "results-" + str(model_str) + "-nshot" + str(nshot) + "-" + timestamp_str)
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
    filename_out = "results_out." + str(model_str) + "-nshot" + str(nshot) + "-" + timestamp_str + ".json"
    filename_out = os.path.join(path_out, filename_out)
    # Do a test save so there aren't any surprises later, and we crash here if we can't save (instead of later)
    print("Performing a test save to: " + filename_out)
    with open(filename_out, 'w') as f:
        json.dump(all_results, f, indent=4)


    performance_by_category = {}
    total_counts = {}
    averages = {}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for claim in claims:
            claim_id = claim["claim_id"]
            claim_text = claim["claim_text"]
            claim_gold_label = claim["gold_label"]
            # Submit the task
            if (claim_id in claim_ids_already_completed):
                print("Claim " + str(claim_id) + " already completed, skipping.")
                continue

            future = executor.submit(answer_claim_llm_baseline, claim_id, claim_text, claim_gold_label, model_str, temperature=temperature, max_tokens=max_tokens, reflection=reflection, nshot=nshot)
            futures.append(future)

        # Process tasks as they complete.
        for future in concurrent.futures.as_completed(futures):
            # Save the output file
            result = future.result()
            if (result == None):
                print("* Future: Error: result is None.")
                continue
            try:
                print("* Future: Processing results for claim " + str(result["claim_id"]) + ": " + str(result))

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

    # GPT-4o-mini
    run_answer_claim_llm_baseline(filename_test, path_out, model_str="gpt-4o-mini", temperature=0.0, max_tokens=1000, reflection=True, nshot=True, MAX_WORKERS=2, DEBUG_LIMIT=-1)

    # Claude 3.7
    #run_answer_claim_llm_baseline(filename_test, path_out, model_str="claude-3-7-sonnet-20250219", temperature=0.0, max_tokens=1000, reflection=True, nshot=True, MAX_WORKERS=4, DEBUG_LIMIT=-1)

    # GPT o4-mini-2025-04-16
    #run_answer_claim_llm_baseline(filename_test, path_out, model_str="o4-mini-2025-04-16", temperature=0.0, max_tokens=10000, reflection=True, nshot=True, MAX_WORKERS=50, DEBUG_LIMIT=-1)

    # Baselines w/o reflection or n-shot
    #run_answer_claim_llm_baseline(filename_test, path_out, model_str="gpt-4o-mini", temperature=0.0, max_tokens=1000, reflection=False, nshot=False, MAX_WORKERS=150, DEBUG_LIMIT=-1)
    #run_answer_claim_llm_baseline(filename_test, path_out, model_str="o4-mini-2025-04-16", temperature=0.0, max_tokens=1000, reflection=False, nshot=False, MAX_WORKERS=100, DEBUG_LIMIT=-1)
    #run_answer_claim_llm_baseline(filename_test, path_out, model_str="claude-3-7-sonnet-20250219", temperature=0.0, max_tokens=1000, reflection=False, nshot=False, MAX_WORKERS=20, DEBUG_LIMIT=-1)


    print("Completed.")
