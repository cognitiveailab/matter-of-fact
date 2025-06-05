# ExtractionUtils.py

import os
import json
import time
import traceback
import requests
import random

from litellm import completion
from litellm import embedding

import tiktoken

from func_timeout import func_timeout, FunctionTimedOut

# For AWS Bedrock
#os.environ["AWS_REGION_NAME"] = "us-east-1"

TOTAL_LLM_COST = 0.0

DEFAULT_MAX_TOKENS = 8000


#
#   Helper: Counting tokens
#
# Use TikToken to measure the number of tokens in an input string
tiktokenEncoder = tiktoken.encoding_for_model("gpt-4")
def countTokens(inputStr:str):
    try:
        tokens = tiktokenEncoder.encode(inputStr)
        return len(tokens)
    except Exception as e:
        print("ERROR: Failed counting tokens on the following input string:")
        print(inputStr)
        print("ERROR MESSAGE:")
        print(e)
        return None


def tokenize(inputStr:str):
    tokens = tiktokenEncoder.encode(inputStr)
    # Convert each token to it's respective string, keeping the output as a list
    tokens = [tiktokenEncoder.decode([token]) for token in tokens]
    return tokens


#
#   Helper: Load the API keys
#

def loadAPIKeys():
    print("Loading API keys...")

    API_KEY_FILE = "api_keys.donotcommit.json"
    with open(API_KEY_FILE, 'r') as fileIn:
        apiKeys = json.load(fileIn)
        # Set the OpenAI environment variable
        if ("openai" in apiKeys):
            os.environ["OPENAI_API_KEY"] = apiKeys["openai"]
        # Set the Anthropic environment variable
        if ("anthropic" in apiKeys):
            os.environ["ANTHROPIC_API_KEY"] = apiKeys["anthropic"]
        # Set the deepseek environment variable
        if ("deepseek" in apiKeys):
            os.environ["DEEPSEEK_API_KEY"] = apiKeys["deepseek"]
        # Together AI
        if ("togetherai" in apiKeys):
            os.environ["TOGETHERAI_API_KEY"] = apiKeys["togetherai"]



#
#   Helper: Get the total cost of all LLM queries
#
def getTotalLLMCost():
    global TOTAL_LLM_COST
    return TOTAL_LLM_COST

#
#   Helper: Query the local Deepseek server
#

def chat_completion_deepseek_local(user_input):
    # Unsupported
    return None


#
#   Helper: Get a response from an LLM model
#


# Get a response from an LLM model using litellm
def getLLMResponseJSON(promptStr:str, model:str, temperature:float=0, maxTokens:int=DEFAULT_MAX_TOKENS, jsonOut:bool=True):
    MAX_RETRIES = 10
    MAX_GENERATION_TIME_SECONDS = 60 * 5        # Maximum of 5 minutes per generation (guard against long hangs)
    count_too_long_errors = 0

    for retryIdx in range(MAX_RETRIES):
        try:
            if model != "deepseek_local":
                # Use timeout
                responseJSON, responseText, cost = func_timeout(MAX_GENERATION_TIME_SECONDS, _getLLMResponseJSON, args=(promptStr, model, temperature, maxTokens, jsonOut))
            else:
                # for local deepseek, remove the timeout for debugging
                # promptStr = ' '.join(promptStr.split()[:800])
                responseJSON, responseText, cost = _getLLMResponseJSON(promptStr, model, temperature, maxTokens, jsonOut)
            return responseJSON, responseText, cost

        # timeout
        except FunctionTimedOut:
            errorInfo = "Time: " + str(time.strftime("%Y%m%d-%H%M%S")) + "  count_too_long_errors: " + str(count_too_long_errors) + "  model: " + model + "  prompt length: " + str(len(promptStr))
            print("ERROR: LLM Generation timed out. " + str(errorInfo))
            count_too_long_errors += 1
            if (count_too_long_errors >= 3):
                print("ERROR: LLM Generation time out: Too many timeouts. Exiting.")
                #exit(1)
                return None, "", 0

        # Keyboard exception
        except KeyboardInterrupt:
            exit(1)
        except Exception as e:
            print("ERROR: Could not get LLM response. Retrying... ")
            print("ERROR MESSAGE:")
            print(e)
            print(traceback.format_exc())

            # Check for some known kinds of errors
            errorStr = str(e)
            if ("prompt is too long" in errorStr):
                print("ERROR: Prompt is too long. Exiting.")
                return None, "", 0

        # Short delay before retrying
        print("Delaying for a few seconds before retrying...")
        print("Attempt " + str(retryIdx) + " of " + str(MAX_RETRIES))
        time.sleep(retryIdx * 15)

    # If we reach here, something terrible happened
    print("ERROR: Could not get LLM response. Exiting.")
    #exit(1)
    return None, "", 0


# Get a response from an LLM model using litellm
def getLLMResponseJSONWithMetadata(promptStr:str, model:str, temperature:float=0, maxTokens:int=DEFAULT_MAX_TOKENS, jsonOut:bool=True):
    MAX_RETRIES = 10
    MAX_GENERATION_TIME_SECONDS = 60 * 5        # Maximum of 5 minutes per generation (guard against long hangs)
    count_too_long_errors = 0

    for retryIdx in range(MAX_RETRIES):
        start_time = time.time()
        try:
            print("* getLLMResponseJSONWithMetadata(): Starting LLM query at " + str(time.strftime("%Y%m%d-%H%M%S")) + "... (Retry " + str(retryIdx) + " of " + str(MAX_RETRIES) + ")")
            if model != "deepseek_local":
                # Use timeout
                responseJSON, responseText, cost, metadata = func_timeout(MAX_GENERATION_TIME_SECONDS, _getLLMResponseJSONWithMetadata, args=(promptStr, model, temperature, maxTokens, jsonOut))
            else:
                # for local deepseek, remove the timeout for debugging
                # promptStr = ' '.join(promptStr.split()[:800])
                responseJSON, responseText, cost, metadata = _getLLMResponseJSONWithMetadata(promptStr, model, temperature, maxTokens, jsonOut)
            print("* getLLMResponseJSONWithMetadata(): Completed LLM query started at " + str(time.strftime("%Y%m%d-%H%M%S")) + " in " + str(time.time() - start_time) + " seconds.")
            return responseJSON, responseText, cost, metadata


        # timeout
        except FunctionTimedOut:
            errorInfo = "Time: " + str(time.strftime("%Y%m%d-%H%M%S")) + "  count_too_long_errors: " + str(count_too_long_errors) + "  model: " + model + "  prompt length: " + str(len(promptStr))
            print("ERROR: LLM Generation timed out. " + str(errorInfo))
            count_too_long_errors += 1
            if (count_too_long_errors >= 3):
                print("ERROR: LLM Generation time out: Too many timeouts. Exiting.")
                #exit(1)
                return None, "", 0, {}

        # Keyboard exception
        except KeyboardInterrupt:
            exit(1)
        except Exception as e:
            print("* getLLMResponseJSONWithMetadata(): ERROR: For LLM query started at " + str(time.strftime("%Y%m%d-%H%M%S")) + ", the following error occurred after " + str(time.time() - start_time) + " seconds: " + str(e))
            #print("ERROR: Could not get LLM response. Retrying... ")
            #print("ERROR MESSAGE:")
            #print(e)
            print(traceback.format_exc())

            # Check for some known kinds of errors
            errorStr = str(e)
            if ("prompt is too long" in errorStr):
                print("ERROR: Prompt is too long. Exiting.")
                return None, "", 0, {}

        # Short delay before retrying
        print("Delaying for a few seconds before retrying...")
        print("Attempt " + str(retryIdx) + " of " + str(MAX_RETRIES))
        time.sleep(retryIdx * 15)

    # If we reach here, something terrible happened
    print("ERROR: Could not get LLM response. Exiting.")
    #exit(1)
    return None, "", 0, {}


# Get a response from an LLM model using litellm
def getLLMResponseJSON_ManualMessages(messages:list, model:str, temperature:float=0, maxTokens:int=DEFAULT_MAX_TOKENS, jsonOut:bool=True):
    MAX_RETRIES = 10
    MAX_GENERATION_TIME_SECONDS = 60 * 5        # Maximum of 5 minutes per generation (guard against long hangs)
    count_too_long_errors = 0

    for retryIdx in range(MAX_RETRIES):
        try:
            if model != "deepseek_local":
                # Use timeout
                #responseJSON, responseText, cost = func_timeout(MAX_GENERATION_TIME_SECONDS, _getLLMResponseJSON, args=(promptStr, model, temperature, maxTokens, jsonOut))
                responseJSON, responseText, cost = _getLLMResponseJSON(promptStr=None, model=model, temperature=temperature, maxTokens=maxTokens, jsonOut=jsonOut, manualMessages=messages)
            else:
                # for local deepseek, remove the timeout for debugging
                promptStr = ' '.join(promptStr.split()[:800])
                responseJSON, responseText, cost = _getLLMResponseJSON(promptStr=None, model=model, temperature=temperature, maxTokens=maxTokens, jsonOut=jsonOut, manualMessages=messages)
            return responseJSON, responseText, cost

        # timeout
        except FunctionTimedOut:
            errorInfo = "Time: " + str(time.strftime("%Y%m%d-%H%M%S")) + "  count_too_long_errors: " + str(count_too_long_errors) + "  model: " + model + "  prompt length: " + str(len(promptStr))
            print("ERROR: LLM Generation timed out. " + str(errorInfo))
            count_too_long_errors += 1
            if (count_too_long_errors >= 3):
                print("ERROR: LLM Generation time out: Too many timeouts. Exiting.")
                #exit(1)
                return None, "", 0

        # Keyboard exception
        except KeyboardInterrupt:
            exit(1)
        except Exception as e:
            print("ERROR: Could not get LLM response. Retrying... ")
            print("ERROR MESSAGE:")
            print(e)
            print(traceback.format_exc())

            # Check for some known kinds of errors
            errorStr = str(e)
            if ("prompt is too long" in errorStr):
                print("ERROR: Prompt is too long. Exiting.")
                return None, "", 0

        # Short delay before retrying
        print("Delaying for a few seconds before retrying...")
        print("Attempt " + str(retryIdx) + " of " + str(MAX_RETRIES))
        time.sleep(retryIdx * 15)

    # If we reach here, something terrible happened
    print("ERROR: Could not get LLM response. Exiting.")
    #exit(1)
    return None, "", 0

def _getLLMResponseJSON(promptStr:str, model:str, temperature:float=0, maxTokens:int=DEFAULT_MAX_TOKENS, jsonOut:bool=True, manualMessages:list=None):
    global TOTAL_LLM_COST
    print("Querying LLM model (" + str(model) + ")... ")

    # Note the running cost of all LLM queries
    print("(Running cost of all LLM generations so far: " + str(round(TOTAL_LLM_COST, 2)) + ")")

    # Measure the number of tokens in the prompt
    if (promptStr != None):
        print("Prompt tokens: " + str(countTokens(promptStr)))

    messages = []
    if manualMessages != None:
        messages = manualMessages
    else:
        messages=[
            {"role": "user",
            "content": [
                {
                    "type": "text",
                    "text": promptStr
                },
            ]
            }
        ]

    # Dump the whole thing to a file called (prompt-debug.txt)
    # Check 'prompts' directory exists, and make it otherwise
    if not os.path.exists("prompts"):
        os.makedirs("prompts")
    print("Writing prompt to prompt-debug.txt...")
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    # Also add a random 4 digit number to the filename
    timestamp += "-" + str(random.randint(1000, 9999))
    if (manualMessages != None):
        filenameOut = "prompts/prompt-debug." + timestamp + ".manual.json"
        with open(filenameOut, "w") as f:
            json.dump(messages, f, indent=4)
        # But also save a text version
        outStr = ""
        for message in messages:
            role = message["role"]
            content = message["content"]
            outStr += "-" * 80 + "\n"
            outStr += "Role: " + role + "\n"
            outStr += "-" * 80 + "\n"
            outStr += content + "\n"

        filenameOut = "prompts/prompt-debug." + timestamp + ".manual.txt"
        with open(filenameOut, "w") as f:
            f.write(outStr)


    else:
        filenameOut = "prompts/prompt-debug." + timestamp + ".txt"
        with open(filenameOut, "w") as f:
            f.write(promptStr)

    #litellm.drop_params=True
    cost = 0
    if model == "deepseek_local":
        responseText = chat_completion_deepseek_local(promptStr)
        print("Completed.")
    else:
        # Set litellm to drop params
        import litellm
        litellm.drop_params = True

        extra_headers = None
        if (model == "claude-3-5-sonnet-20240620") and (maxTokens > 4096):
            extra_headers={"anthropic-beta": "max-tokens-3-5-sonnet-2024-07-15"}

        response = None


        # New (special handling for different models)
        if (model == "deepseek/deepseek-reasoner"):
            response = completion(model=model, messages=messages, temperature=temperature, max_tokens=maxTokens, response_format={"type": "json_object"}, extra_headers=extra_headers)

        elif ("o3-mini" in model):
            # needs max_completion_tokens
            #print("** SPECIAL HANDLING FOR O3-MINI **")
            #reasoning_effort = None
            #reasoning_effort = "low"
            #reasoning_effort = "medium"
            reasoning_effort = "high"

            if (jsonOut):
                if (reasoning_effort == None):
                    response = completion(model=model, messages=messages, max_completion_tokens=maxTokens, response_format={"type": "json_object"}, extra_headers=extra_headers)
                else:
                    response = completion(model=model, messages=messages, max_completion_tokens=maxTokens, response_format={"type": "json_object"}, extra_headers=extra_headers, reasoning_effort=reasoning_effort)
            else:
                if (reasoning_effort == None):
                    response = completion(model=model, messages=messages, max_completion_tokens=maxTokens, extra_headers=extra_headers)
                else:
                    response = completion(model=model, messages=messages, max_completion_tokens=maxTokens, extra_headers=extra_headers, reasoning_effort=reasoning_effort)

        elif ("claude" in model):
            #print("** SPECIAL HANDLING FOR CLAUDE **")
            # Do not use special JSON model for claude -- it seems to do better at adhering to the schema without it
            response = completion(model=model, messages=messages, temperature=temperature, max_tokens=maxTokens, extra_headers=extra_headers)

        else:
            if (jsonOut):
                response = completion(model=model, messages=messages, temperature=temperature, max_tokens=maxTokens, response_format={"type": "json_object"}, extra_headers=extra_headers)
            else:
                response = completion(model=model, messages=messages, temperature=temperature, max_tokens=maxTokens, extra_headers=extra_headers)


        print(response._hidden_params["response_cost"])

        # Get the response text

        # DEBUG: PRINT FULL RESPONSE
        print("FULL RESPONSE:")
        print(response)

        responseText = response["choices"][0]["message"]["content"]

        if ("response_cost" in response._hidden_params) and (response._hidden_params["response_cost"] != None):
            cost = response._hidden_params["response_cost"]
            print("Found cost in response: " + str(cost))
            TOTAL_LLM_COST += cost
        else:
            # Try to calculate the cost based on the number of tokens
            # Structure:
            # ModelResponse(id='8625a86b-f301-46db-98be-8185a910b096', choices=[Choices(finish_reason='stop', index=0, message=Message(content="The capital of France is **Paris**. As the country's largest city and cultural, political, and economic hub, Paris has long been the seat of France's government. Key institutions like the Élysée Palace (residence of the president), the National Assembly, and the Senate are located there. Landmarks such as the Eiffel Tower and the Louvre Museum further underscore its significance. While temporary relocations occurred historically (e.g., to Vichy during WWII), Paris remains the undisputed capital.", role='assistant', tool_calls=None, function_call=None))], created=1737498098, model='deepseek/deepseek-reasoner', object='chat.completion', system_fingerprint='fp_1c5d8833bc', usage=Usage(completion_tokens=563, prompt_tokens=12, total_tokens=575, completion_tokens_details={'audio_tokens': None, 'reasoning_tokens': 459}, prompt_tokens_details={'audio_tokens': None, 'cached_tokens': 0}, prompt_cache_hit_tokens=0, prompt_cache_miss_tokens=12), service_tier=None)
            prompt_tokens = response["usage"].get("prompt_tokens", 0)
            completion_tokens = response["usage"].get("completion_tokens", 0)
            reasoning_tokens = 0
            if ("completion_tokens_details" in response["usage"]):
                reasoning_tokens = None
                try:
                    reasoning_tokens = response["usage"]["completion_tokens_details"].get("reasoning_tokens", 0)
                except Exception as e:
                    pass
                if (reasoning_tokens == None):
                    # Try to parse from the new CompletionTokensDetailsWrapper structure
                    completion_details = response["usage"]["completion_tokens_details"]
                    reasoning_tokens = completion_details.reasoning_tokens
                if (reasoning_tokens == None):
                    reasoning_tokens = 0
            print("Reasoning tokens: " + str(reasoning_tokens))
            completion_tokens += reasoning_tokens   # Usually reasoning tokens count as completion tokens

            # Calculate the cost
            cost_prompt_tokens_per_million = 0.0
            cost_completion_tokens_per_million = 0.0
            costs_dict = {
                "default": {
                    "prompt_tokens_per_million": 5.00,              # Use some high cost estimate
                    "completion_tokens_per_million": 20.00
                },
                "o3-mini-2025-01-31": {
                    "prompt_tokens_per_million": 1.10,
                    "completion_tokens_per_million": 4.40
                },
                "o3-mini": {
                    "prompt_tokens_per_million": 1.10,
                    "completion_tokens_per_million": 4.40
                },
                "openai/o3-mini-2025-01-31": {
                    "prompt_tokens_per_million": 1.10,
                    "completion_tokens_per_million": 4.40
                }
            }

            # Get keys in the costs_dict
            cost_model_keys = costs_dict.keys()
            if (model in costs_dict):
                cost_prompt_tokens_per_million = costs_dict[model]["prompt_tokens_per_million"]
                cost_completion_tokens_per_million = costs_dict[model]["completion_tokens_per_million"]
            else:
                # Check if it's a fine-tuned model
                if (model.startswith("ft:")):
                    # Fine tuned models have unique names, but might start with one of the same prefixes
                    # Check if the model name is in the costs_dict
                    found = False
                    for key in cost_model_keys:
                        if (model.startswith(key)):
                            cost_prompt_tokens_per_million = costs_dict[key]["prompt_tokens_per_million"]
                            cost_completion_tokens_per_million = costs_dict[key]["completion_tokens_per_million"]
                            found = True
                            print("INFO: Found fine-tuned model in costs_dict. Using costs of " + str(cost_prompt_tokens_per_million) + " and " + str(cost_completion_tokens_per_million) + " per million tokens.")
                            break
                    if (not found):
                        cost_prompt_tokens_per_million = costs_dict["default"]["prompt_tokens_per_million"]
                        cost_completion_tokens_per_million = costs_dict["default"]["completion_tokens_per_million"]
                        print("WARNING: Model costs not found. Using default (high) costs of " + str(cost_prompt_tokens_per_million) + " and " + str(cost_completion_tokens_per_million) + " per million tokens.")
                else:
                    cost_prompt_tokens_per_million = costs_dict["default"]["prompt_tokens_per_million"]
                    cost_completion_tokens_per_million = costs_dict["default"]["completion_tokens_per_million"]
                    print("WARNING: Model costs not found. Using default (high) costs of " + str(cost_prompt_tokens_per_million) + " and " + str(cost_completion_tokens_per_million) + " per million tokens.")

            # Calculate the cost
            cost = (prompt_tokens * cost_prompt_tokens_per_million / 1000000) + (completion_tokens * cost_completion_tokens_per_million / 1000000)

        print("Completed.  Cost: " + str(round(cost, 2)) + "  (Total Cost: " + str(round(TOTAL_LLM_COST, 2)) + ")")


    # Also dump the response (timestamped)
    filenameOut = "prompts/prompt-debug." + timestamp + ".response.txt"
    with open(filenameOut, "w") as f:
        f.write(responseText)

    responseOutJSON = None
    # Convert the response text to JSON
    try:
        responseOutJSON = json.loads(responseText)
    # Keyboard exception
    except KeyboardInterrupt:
        exit(1)
    except Exception as e:
        print("ERROR: Could not convert response to JSON on first pass. ")

    # Second pass for converting to JSON
    if (responseOutJSON == None):
        # Try to find the last JSON block in the response, which starts with "```"
        lines = responseText.split("\n")
        startIdx = 0
        endIdx = 0
        for idx, line in enumerate(lines):
            if (line.startswith("```")):
                startIdx = endIdx
                endIdx = idx

        if (startIdx >= 0) and (endIdx > 0):
            # Exclude the start and end line
            linesJSON = lines[startIdx+1:endIdx]
            # Join the lines
            linesJSONStr = "\n".join(linesJSON)
            # Try to convert to JSON
            try:
                responseOutJSON = json.loads(linesJSONStr)
            except Exception as e:
                print("ERROR: Could not convert response to JSON on second pass. ")
        else:
            print("ERROR: Could not find JSON block in response. ")

    # Return
    return responseOutJSON, responseText, cost



def _getLLMResponseJSONWithMetadata(promptStr:str, model:str, temperature:float=0, maxTokens:int=DEFAULT_MAX_TOKENS, jsonOut:bool=True, manualMessages:list=None):
    global TOTAL_LLM_COST
    start_time = time.time()
    print("getLLMResponseJSONWithMetadata(): Querying LLM model (" + str(model) + ")... (started at " + str(time.strftime("%Y%m%d-%H%M%S")) + ")")
    #print("Querying LLM model (" + str(model) + ")... ")

    # Note the running cost of all LLM queries
    print("(Running cost of all LLM generations so far: " + str(round(TOTAL_LLM_COST, 2)) + ")")

    # Measure the number of tokens in the prompt
    if (promptStr != None):
        print("Prompt tokens: " + str(countTokens(promptStr)))

    messages = []
    if manualMessages != None:
        messages = manualMessages
    else:
        messages=[
            {"role": "user",
            "content": [
                {
                    "type": "text",
                    "text": promptStr
                },
            ]
            }
        ]

    # Dump the whole thing to a file called (prompt-debug.txt)
    # Check 'prompts' directory exists, and make it otherwise
    if not os.path.exists("prompts"):
        os.makedirs("prompts")
    print("Writing prompt to prompt-debug.txt...")
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    # Also add a random 4 digit number to the filename
    timestamp += "-" + str(random.randint(1000, 9999))
    if (manualMessages != None):
        filenameOut = "prompts/prompt-debug." + timestamp + ".manual.json"
        with open(filenameOut, "w") as f:
            json.dump(messages, f, indent=4)
        # But also save a text version
        outStr = ""
        for message in messages:
            role = message["role"]
            content = message["content"]
            outStr += "-" * 80 + "\n"
            outStr += "Role: " + role + "\n"
            outStr += "-" * 80 + "\n"
            outStr += content + "\n"

        filenameOut = "prompts/prompt-debug." + timestamp + ".manual.txt"
        with open(filenameOut, "w") as f:
            f.write(outStr)


    else:
        filenameOut = "prompts/prompt-debug." + timestamp + ".txt"
        with open(filenameOut, "w") as f:
            f.write(promptStr)

    #litellm.drop_params=True
    cost = 0
    if model == "deepseek_local":
        responseText = chat_completion_deepseek_local(promptStr)
        print("Completed.")
    else:
        # Set litellm to drop params
        import litellm
        litellm.drop_params = True

        extra_headers = None
        if (model == "claude-3-5-sonnet-20240620") and (maxTokens > 4096):
            extra_headers={"anthropic-beta": "max-tokens-3-5-sonnet-2024-07-15"}

        response = None


        # New (special handling for different models)
        if (model == "deepseek/deepseek-reasoner"):
            response = completion(model=model, messages=messages, temperature=temperature, max_tokens=maxTokens, response_format={"type": "json_object"}, extra_headers=extra_headers)

        elif ("o3-mini" in model) or ("o4-mini" in model):
            # needs max_completion_tokens
            #print("** SPECIAL HANDLING FOR O3-MINI **")
            reasoning_effort = None
            #reasoning_effort = "low"
            #reasoning_effort = "medium"
            #reasoning_effort = "high"

            if (jsonOut):
                if (reasoning_effort == None):
                    response = completion(model=model, messages=messages, max_completion_tokens=maxTokens, response_format={"type": "json_object"}, extra_headers=extra_headers)
                else:
                    response = completion(model=model, messages=messages, max_completion_tokens=maxTokens, response_format={"type": "json_object"}, extra_headers=extra_headers, reasoning_effort=reasoning_effort)
            else:
                if (reasoning_effort == None):
                    response = completion(model=model, messages=messages, max_completion_tokens=maxTokens, extra_headers=extra_headers)
                else:
                    response = completion(model=model, messages=messages, max_completion_tokens=maxTokens, extra_headers=extra_headers, reasoning_effort=reasoning_effort)


        elif ("claude" in model):
            #print("** SPECIAL HANDLING FOR CLAUDE **")
            # Do not use special JSON model for claude -- it seems to do better at adhering to the schema without it
            response = completion(model=model, messages=messages, temperature=temperature, max_tokens=maxTokens, extra_headers=extra_headers)

        else:
            if (jsonOut):
                response = completion(model=model, messages=messages, temperature=temperature, max_tokens=maxTokens, response_format={"type": "json_object"}, extra_headers=extra_headers)
            else:
                response = completion(model=model, messages=messages, temperature=temperature, max_tokens=maxTokens, extra_headers=extra_headers)


        print(response._hidden_params["response_cost"])

        # Get the response text

        # DEBUG: PRINT FULL RESPONSE
        print("FULL RESPONSE:")
        print(response)
        metadata = {}

        responseText = response["choices"][0]["message"]["content"]

        if ("response_cost" in response._hidden_params) and (response._hidden_params["response_cost"] != None):
            cost = response._hidden_params["response_cost"]
            print("Found cost in response: " + str(cost))
            TOTAL_LLM_COST += cost
        else:
            # Try to calculate the cost based on the number of tokens
            prompt_tokens = response["usage"].get("prompt_tokens", 0)
            completion_tokens = response["usage"].get("completion_tokens", 0)
            reasoning_tokens = 0
            if ("completion_tokens_details" in response["usage"]):
                reasoning_tokens = None
                try:
                    reasoning_tokens = response["usage"]["completion_tokens_details"].get("reasoning_tokens", 0)
                except Exception as e:
                    pass
                if (reasoning_tokens == None):
                    # Try to parse from the new CompletionTokensDetailsWrapper structure
                    completion_details = response["usage"]["completion_tokens_details"]
                    reasoning_tokens = completion_details.reasoning_tokens
                if (reasoning_tokens == None):
                    reasoning_tokens = 0
            print("Reasoning tokens: " + str(reasoning_tokens))
            completion_tokens += reasoning_tokens   # Usually reasoning tokens count as completion tokens

            # Calculate the cost
            cost_prompt_tokens_per_million = 0.0
            cost_completion_tokens_per_million = 0.0
            costs_dict = {
                "default": {
                    "prompt_tokens_per_million": 5.00,              # Use some high cost estimate
                    "completion_tokens_per_million": 20.00
                },
                "o3-mini-2025-01-31": {
                    "prompt_tokens_per_million": 1.10,
                    "completion_tokens_per_million": 4.40
                },
                "o3-mini": {
                    "prompt_tokens_per_million": 1.10,
                    "completion_tokens_per_million": 4.40
                },
                "openai/o3-mini-2025-01-31": {
                    "prompt_tokens_per_million": 1.10,
                    "completion_tokens_per_million": 4.40
                }
            }

            # Get keys in the costs_dict
            cost_model_keys = costs_dict.keys()
            if (model in costs_dict):
                cost_prompt_tokens_per_million = costs_dict[model]["prompt_tokens_per_million"]
                cost_completion_tokens_per_million = costs_dict[model]["completion_tokens_per_million"]
            else:
                # Check if it's a fine-tuned model
                if (model.startswith("ft:")):
                    # Fine tuned models have unique names, but might start with one of the same prefixes
                    # Check if the model name is in the costs_dict
                    found = False
                    for key in cost_model_keys:
                        if (model.startswith(key)):
                            cost_prompt_tokens_per_million = costs_dict[key]["prompt_tokens_per_million"]
                            cost_completion_tokens_per_million = costs_dict[key]["completion_tokens_per_million"]
                            found = True
                            print("INFO: Found fine-tuned model in costs_dict. Using costs of " + str(cost_prompt_tokens_per_million) + " and " + str(cost_completion_tokens_per_million) + " per million tokens.")
                            break
                    if (not found):
                        cost_prompt_tokens_per_million = costs_dict["default"]["prompt_tokens_per_million"]
                        cost_completion_tokens_per_million = costs_dict["default"]["completion_tokens_per_million"]
                        print("WARNING: Model costs not found. Using default (high) costs of " + str(cost_prompt_tokens_per_million) + " and " + str(cost_completion_tokens_per_million) + " per million tokens.")
                else:
                    cost_prompt_tokens_per_million = costs_dict["default"]["prompt_tokens_per_million"]
                    cost_completion_tokens_per_million = costs_dict["default"]["completion_tokens_per_million"]
                    print("WARNING: Model costs not found. Using default (high) costs of " + str(cost_prompt_tokens_per_million) + " and " + str(cost_completion_tokens_per_million) + " per million tokens.")

            # Calculate the cost
            cost = (prompt_tokens * cost_prompt_tokens_per_million / 1000000) + (completion_tokens * cost_completion_tokens_per_million / 1000000)

        print("Completed.  Cost: " + str(round(cost, 2)) + "  (Total Cost: " + str(round(TOTAL_LLM_COST, 2)) + ")")


    # Metadata
    try:
        prompt_tokens = response["usage"].get("prompt_tokens", 0)
        completion_tokens = response["usage"].get("completion_tokens", 0)
        reasoning_tokens = 0
        if ("completion_tokens_details" in response["usage"]):
            reasoning_tokens = None
            try:
                reasoning_tokens = response["usage"]["completion_tokens_details"].get("reasoning_tokens", 0)
            except Exception as e:
                pass
            if (reasoning_tokens == None):
                # Try to parse from the new CompletionTokensDetailsWrapper structure
                completion_details = response["usage"]["completion_tokens_details"]
                reasoning_tokens = completion_details.reasoning_tokens
            if (reasoning_tokens == None):
                reasoning_tokens = 0
        print("Reasoning tokens: " + str(reasoning_tokens))

        metadata['tokens_prompt'] = prompt_tokens
        metadata['tokens_completion'] = completion_tokens
        metadata['tokens_reasoning'] = reasoning_tokens
        metadata['tokens_total'] = prompt_tokens + completion_tokens + reasoning_tokens
        metadata['cost'] = cost
        metadata['model'] = model
        metadata['temperature'] = temperature
        metadata['max_tokens'] = maxTokens
    except Exception as e:
        print("ERROR: Could not extract metadata from response. ")


    # Also dump the response (timestamped)
    filenameOut = "prompts/prompt-debug." + timestamp + ".response.txt"
    with open(filenameOut, "w") as f:
        f.write(responseText)

    responseOutJSON = None
    # Convert the response text to JSON
    try:
        responseOutJSON = json.loads(responseText)
    # Keyboard exception
    except KeyboardInterrupt:
        exit(1)
    except Exception as e:
        print("ERROR: Could not convert response to JSON on first pass. ")

    # Second pass for converting to JSON
    if (responseOutJSON == None):
        # Try to find the last JSON block in the response, which starts with "```"
        lines = responseText.split("\n")
        startIdx = 0
        endIdx = 0
        for idx, line in enumerate(lines):
            if (line.startswith("```")):
                startIdx = endIdx
                endIdx = idx

        if (startIdx >= 0) and (endIdx > 0):
            # Exclude the start and end line
            linesJSON = lines[startIdx+1:endIdx]
            # Join the lines
            linesJSONStr = "\n".join(linesJSON)
            # Try to convert to JSON
            try:
                responseOutJSON = json.loads(linesJSONStr)
            except Exception as e:
                print("ERROR: Could not convert response to JSON on second pass. ")
        else:
            print("ERROR: Could not find JSON block in response. ")

    # Return
    print("getLLMResponseJSONWithMetadata(): Returning successfully: LLM model (" + str(model) + ")... (started at " + str(time.strftime("%Y%m%d-%H%M%S")) + ")")
    return responseOutJSON, responseText, cost, metadata




# OpenAI Embeddings
def getEmbedding(textStr:str, model:str = "text-embedding-3-small"):
    # Get the embedding from the model
    response = embedding(
        model=model,
        input=textStr,
    )

    #print(response)
    # Return just the embedding, as a list of floats.
    try:
        vectorOut = response["data"][0]["embedding"]
        return vectorOut
    except Exception as e:
        print("ERROR: Could not extract embedding from response. ")
        return None


def cosineSimilarity(vec1, vec2):
    from numpy import dot
    from numpy.linalg import norm
    return dot(vec1, vec2)/(norm(vec1)*norm(vec2))




# Quick test
if __name__ == "__main__":
    # Set keys
    loadAPIKeys()

    # Test of deepseek
    #modelStr = "claude-3-5-sonnet-20241022"
    #modelStr = "o1-mini"
    modelStr = "openai/o3-mini-2025-01-31"
    promptStr = "What is the capital of France? Respond in JSON."
    responseJSON, responseText, cost = getLLMResponseJSON(promptStr, modelStr, temperature=0.0, maxTokens=DEFAULT_MAX_TOKENS, jsonOut=True)

    print("RESPONSE (JSON):")
    print(json.dumps(responseJSON, indent=4))
    print("RESPONSE:")
    print(responseText)
    print("COST: " + str(cost))
