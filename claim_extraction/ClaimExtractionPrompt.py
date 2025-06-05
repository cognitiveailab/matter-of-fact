# ClaimExtractionPrompt.py

import os
import json
import time

from ExtractionUtils import *



def extract_claims_from_paper(paperstore, paper_id:str, topics:list, paper_license:str, model_str:str, temperature:float=0.0, max_tokens:int=24000, reflection:bool=True):

    def mkPrompt(paper_text, reflection=None):
        prompt = "You are ScientistGPT, the most advanced automated scientific discovery and inference system in the world. You can use your enormous intellect to solve any scientific problem, and always do so by breaking the problem down into its finest details.  You are accurate, faithful, and act with the highest scientific integrity.\n"
        prompt += "\n"
        prompt += "# Task\n"
        prompt += "Your task is to extract all of the central claims from a scientific paper. For each claim, you must also provide whether the claim is true or false, and write a brief explanation as to why the claim is true or false (directly supported by the paper text).\n"
        prompt += "The text of the paper is provided below.\n"
        prompt += "\n"

        prompt += "# Extracting claims\n"
        NUM_CLAIMS = 16
        prompt += "Your goal is to extract " + str(NUM_CLAIMS) + " scientific claims (" + str(NUM_CLAIMS/2) + " positive, " + str(NUM_CLAIMS/2) + " negative) from the paper text.\n"
        prompt += "The claims should nominally be about the results of the paper, whether those be experimental results, code/simulation results, theoretical results, other results, or integrating multiple types.\n"
        prompt += "\n"

        prompt += "## Positive and Negative claims\n"
        prompt += "For each claim that you generate, you will also be asked to generate a `negative` version of it, i.e. one that is not true (and, in particular, is particularly infeasible).\n"
        prompt += "This is a challenging task -- you must balance your language so that there are not cues in the text that indicate whether the claim is true or false.\n"
        prompt += "For example, if the positive claim is \"Model X achieves specific result Y\", and the negative claim is \"Model X does NOT achieve specific result Y\", the fact that the first claim is framed in the affirmative and the second claim is framed in the negative is a cue that the first claim is true and the second claim is false.\n"
        prompt += "A different -- better -- way of framing the claims would be \"Model X achieves specific result Y\" (true), and \"Model X achieves specific result Z\" (false), where Y and Z are different results, one that is supported by the data, and one that isn't.  That way, the language is controlled between positive and negative claims, and it is the content of the claim rather than its presentation that must be evaluated.\n"
        prompt += "(This data is to train and evaluate machine learning models, so you must be careful to avoid any cues that would allow a model to determine whether the claim is true or false in a way other than faithfully evaluating the evidence from the papers, etc.)\n"
        prompt += "\n"
        prompt += "## Additional specifications for negative claims\n"
        prompt += "- Negative claims should be false, or clearly infesible (but not wildly so).  They should not be claims for which no evidence is available, and that could go either way with more information -- rather, their false (or infeasibile) nature should be directly supported by the source paper text.\n"
        prompt += "\n"
        prompt += "## Stand-alone claims\n"
        prompt += "The user of these claims will never see the original paper text, they will only see the claim text that you generate.  You must make sure that each claim is completely stand-alone, and that it contains all of the information needed to understand the claim.\n"
        prompt += "Here are negative examples:\n"
        prompt += "- \"The paper claims that...\" (this is not a stand-alone claim, because it references the paper)\n"
        prompt += "- \"The analysis shows that...\" (this is not a stand-alone claim, because it references the analysis)\n"
        prompt += "- \"X has property Y because Chart 5 suggests that...\" (this is not a stand-alone claim, because it references a chart)\n"
        prompt += "- \"Specific Model X was used with hyperparameters Y and Z\" (this is not a stand-alone claim because it has information that is highly specific to a particular implementation, and is likely impossible to infer without the original source paper, which the user will not have).\n"
        prompt += "\n"
        prompt += "## Source of knowledge for the claims\n"
        prompt += "- Your claims should be based on the knowledge that's available in your source input -- nominally, the *paper text* and the *tables*.\n"
        prompt += "- You should NOT use the *figures* for generating claims, because you're generally unable to see the content of these diagrams, so making high-quality, accurate, verifiable claims from them would be difficult.\n"
        prompt += "\n"
        prompt += "## Qualitative and Quantitative claims\n"
        prompt += "Some claims are qualitative, and some claims are quantitative.  You should consider making about half of your claims quantitative, and the other half qualitative.  Here are examples that help tell the two types apart:\n"
        prompt += "###  Qualitative claim\n"
        prompt += "Example: \"In type-II superconductors, magnetic flux penetrates the material above the lower critical field in the form of quantized vortices, allowing them to remain superconducting even in high applied magnetic fields.\"\n"
        prompt += "Why it's quantitative: it describes how the material behaves (flux‐vortex formation and flux pinning) without assigning particular numerical values.\n"
        prompt += "###  Quantitative claim\n"
        prompt += "Example: \"The critical temperature (Tc) of YBa₂Cu₃O₇₋δ (a high-Tc cuprate superconductor) is 92 K, and at 77 K it can sustain a critical current density Jc of approximately 3 × 10⁶ A/cm² in self-field.\"\n"
        prompt += "Why it's quantitative: it describes specific numerical properties of the material.\n"
        prompt += "### Summary\n"
        prompt += "You can see how (in the above examples) the quantitative statement tells you (for example) how much or how high, whereas the qualitative one tells you what happens or why it matters -- though this is only one way to think about it.\n"
        prompt += "In general, qualitative claims are more about the behavior of the material (e.g. how/why) while quantitative claims are more about the specific numbers and measurements (e.g. what/how much).\n"
        prompt += "\n"

        prompt += "# Additional sub-tasks\n"
        prompt += "In addition, please perform the following sub-tasks while you're performing the main task of claim generation:\n"
        prompt += "## Subtask 1: Topic Identification\n"
        prompt += "Is the paper about one of these 4 specific topics of interest?  If so, please list that topic exactly as it appears in these:\n"
        prompt += "1. Topic 1: \"A15 Superconductors\", or (more broadly) \"superconductors\"\n"
        prompt += "2. Topic 2: \"Room-temperature lithium-ion batteries\", or (more broadly) \"batteries\"\n"
        prompt += "3. Topic 3: \"light-weight aerospace metal alloys\", or (more broadly) \"aerospace materials\"\n"
        prompt += "4. Topic 4: \"Thin-film semi-conductors\", or (more broadly) \"semi-conductors\".\n"
        prompt += "i.e. if the paper is about superconductors, please list \"superconductors\" as the topic.  But if it's specifically about A15 superconductors, please list \"A15 superconductors\" as the topic.\n"
        prompt += "If the paper is not about any of these topics, please list any specific topics that describe it.\n"
        prompt += "\n"
        prompt += "## Subtask 2: Code repositories\n"
        prompt += "If the paper mentions any code repositories or experimental result repositories that accompany the paper, please list them in the output.  These should take the form of valid URLS.\n"
        prompt += "\n"
        prompt += "# Paper Text\n"
        prompt += "The text of the paper to extract claims from is below:\n"
        prompt += "```\n"
        prompt += paper_text + "\n"
        prompt += "```\n"
        prompt += "\n"

        if (reflection != None):
            prompt += "# Reflection\n"
            prompt += "This is a reflection step.  Previously, you generated output from the paper text, and now you will reflect on that output, identify any issues with it (particularly with respect to accuracy, scientific integrity, completeness, quality, etc.), and fix all issues so that the data is of exceptional quality.\n"
            prompt += "The output from your initial step is below:\n"
            prompt += "```\n"
            prompt += json.dumps(reflection, indent=4) + "\n"
            prompt += "```\n"
            prompt += "\n"

        prompt += "# Why you are doing this\n"
        prompt += "You are part of an automated scientific discovery system, that aims to perform scientific discovery tasks that could provide large positive benefits to humanity, through the discovery of new knowledge and the creation of new scientific artifacts.\n"
        prompt += "This is a very important task, and you must do it with the utmost care, attention to detail, accuracy, and scientific integrity.  Please do not hallucinate.\n"
        prompt += "Performing poorly on this task is a critical failure, and may result in highly negative consequences.\n"
        prompt += "\n"

        prompt += "# Output format\n"
        prompt += "Please produce your output in JSON format, between a single set of codeblocks (```).  Your output must be valid JSON, as it will be automatically parsed, and if it is not valid JSON, it will cause a critical failure.\n"
        prompt += "You can (and are encouraged to) think and plan before writing the JSON, and can write text before the codeblock to help you think, to produce the claims of the highest quality.  But the text inside the codeblock must be valid JSON.\n"
        prompt += "\n"
        prompt += "## Output description\n"
        prompt += "Your output must be in the following format:\n"
        prompt += "A dictionary with 2 main keys: `central_claims`, `topics`, and `code_repositories`."
        prompt += "- `central_claims` is a list of claim groups, that contain three keys: `claim_group`:str, `quant_or_qual`:str (possible values are only 'quantitative' or 'qualitative'), `main_type`:str (one of: `experimental`, `code/simulation`, `theoretical`, `other`, or `integrative`), and `claims`:list.  The claims are a list of dictionaries, with `claim_text`, `claim_true_or_false`, `supporting_facts_from_paper`, and `explanation` keys. NOTE: Please tag individual supporting facts with `(experiment)`, `(code)`, `(theory)`, if they come from one of those sources, and no tag otherwise.\n"
        prompt += "- `topics` is a list of strings, that are the topics of the paper. They should contain the exact strings listed above (e.g. `\"A15 superconductors\"`, `\"superconductors\"`, etc.) if the paper includes those topics, but otherwise will be authored by you.\n"
        prompt += "- `code_repositories` is a list of strings, that are any URLs of code repositories or experimental result repositories that are mentioned in the paper.  These should be URLs, and should be valid URLs from the paper.\n"
        prompt += "\n"
        prompt += "## Example output\n"
        prompt += "Here is an example of the output JSON format:\n"
        prompt += "```\n"
        prompt += "{\n"
        prompt += "    \"central_claims\": [\n"
        prompt += "         {\"claim_group\": \"What are these 2 claims about?\", \"quant_or_qual\": \"quantitative\", \"main_type\": \"experimental\", \"claims\": [\n"
        prompt += "             {\"claim_text\": \"This is the positive version of the claim\", \"claim_true_or_false\": True, \"supporting_facts_from_paper\": [\"fact1\", \"fact2\", ...], \"explanation\": \"A detailed explanation, with reference to the supporting facts from the paper, as to why the claim is true\"},\n"
        prompt += "             {\"claim_text\": \"This is the negative/substantially infeasible version of the claim\", \"claim_true_or_false\": False, \"supporting_facts_from_paper\": [\"fact1\", \"fact2\", ...], \"explanation\": \"A detailed explanation, with reference to the supporting facts from the paper, as to why the claim is false/infeasible.\"},\n"
        prompt += "         ]},\n"
        prompt += "         {\"claim_group\": \"What are these 2 claims about?\", \"quant_or_qual\": \"qualitative\", \"main_type\": \"integrative\", \"claims\": [\n"
        prompt += "             {\"claim_text\": \"This is the positive version of the claim\", \"claim_true_or_false\": True, \"supporting_facts_from_paper\": [\"fact1\", \"fact2\", ...], \"explanation\": \"A detailed explanation, with reference to the supporting facts from the paper, as to why the claim is true\"},\n"
        prompt += "             {\"claim_text\": \"This is the negative/substantially infeasible version of the claim\", \"claim_true_or_false\": False, \"supporting_facts_from_paper\": [\"fact1\", \"fact2\", ...], \"explanation\": \"A detailed explanation, with reference to the supporting facts from the paper, as to why the claim is false/infeasible.\"},\n"
        prompt += "         ]},\n"
        prompt += "         {\"claim_group\": \"Cartoon example about the boiling point of water\", \"quant_or_qual\": \"quantitative\", \"main_type\": \"experimental\", \"claims\": [\n"
        prompt += "             {\"claim_text\": \"The boiling point of water is 100 degrees Celsius at standard atmospheric pressure\", \"claim_true_or_false\": True, \"supporting_facts_from_paper\": [\"fact1\", \"fact2\", ...], \"explanation\": \"A detailed explanation, with reference to the supporting facts from the paper, as to why the claim is true\"},\n"
        prompt += "             {\"claim_text\": \"The boiling point of water is 110 degrees Celsius at standard atmospheric pressure\", \"claim_true_or_false\": False, \"supporting_facts_from_paper\": [\"fact1\", \"fact2\", ...], \"explanation\": \"A detailed explanation, with reference to the supporting facts from the paper, as to why the claim is false/infeasible.\"},\n"
        prompt += "         ]},\n"
        prompt += "         # And so forth, for all core claims, until reaching " + str(NUM_CLAIMS/2) + " claim groups (for a total of " + str(NUM_CLAIMS) + " claims)\n"
        prompt += "    ],\n"
        prompt += "    \"topics\": [\"superconductors\", \"Niobium-Tin superconductors\"],\n"
        prompt += "    \"code_repositories\": [\"github.com/user/repo1\", \"github.com/user/repo2\"]\n"
        prompt += "}\n"
        prompt += "```\n"
        prompt += "\n"
        prompt += "Please produce your output now.\n"
        prompt += ""
        prompt += "Don't forget to try and make " + str(NUM_CLAIMS/2) + " of the claim groups quantitative claims, and the other " + str(NUM_CLAIMS/2) + " qualitative.\n"
        prompt += "TONE: The claims should be written independent of referencing the paper specifically -- i.e. they shouldn't say `The paper says X`, or `The approach will cause X`.  They should be written as if they are independent of the paper (i.e. under condition X, we will observe Y; or A will cause B), where the paper could be seen as a source of evidence supporting or refuting that claim. Ambiguous references like `the paper`, `the approach`, `Smith et. al`, or anything that doesn't stand alone without external information is considered a failure.\n"
        if (reflection != None):
            prompt += "Please remember that the output must meet the requirements specified above, including making " + str(NUM_CLAIMS/2) + " of the claim groups quantitative claims, and the other " + str(NUM_CLAIMS/2) + " qualitative.\n"

        return prompt

    # Keep track of total cost
    total_cost = 0.0

    print ("Started processing paper: " + str(paper_id))

    # Get the paper text
    paper_text = paperstore.get_paper_text(paper_id)
    if (paper_text == None):
        print("Error: Could not get paper text for paper: " + str(paper_id))
        return None


    # Create the prompt
    prompt = mkPrompt(paper_text)
    # Make sure the prompt isn't too big
    prompt_tokens = countTokens(prompt)
    MAX_PROMPT_TOKENS = 100000
    if (prompt_tokens > MAX_PROMPT_TOKENS):
        print("Prompt is too long: " + str(prompt_tokens) + " tokens.  Max: " + str(MAX_PROMPT_TOKENS) + " (for paper: " + str(paper_id) + ")")
        return None
    responseJSON, responseText, cost = getLLMResponseJSON(promptStr=prompt, model=model_str, maxTokens=max_tokens, temperature=temperature, jsonOut=True)
    total_cost += cost

    initial_response = responseJSON
    print("Reflection for paper: " + str(paper_id))
    if (reflection == True):
        # Create the prompt
        prompt = mkPrompt(paper_text, responseJSON)
        responseJSON, responseText, cost = getLLMResponseJSON(promptStr=prompt, model=model_str, maxTokens=max_tokens, temperature=temperature, jsonOut=True)
        total_cost += cost

    print("Returning from paper: " + str(paper_id))

    packed = {
        "paper_id": paper_id,
        "topics": topics,
        "license": paper_license,
        "response": responseJSON,
        "responseText": responseText,
        "reflection": reflection,
        "initial_response": initial_response,
        "model": model_str,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "cost": total_cost
    }
    return packed
